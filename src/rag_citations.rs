//! Module 24 — Цитаты, источники и анти-галлюцинации.
//!
//! Доработанный RAG:
//! - структурированный ответ: `answer` + `sources` + `quotes`
//! - обязательный режим "не знаю", когда верхний relevance ниже порога
//! - валидация: каждый пункт выдачи проверяется на наличие источников/цитат
//!
//! Цитаты берутся напрямую из текста чанков, чтобы нельзя было "выдумать".
//! LLM подключается через тот же `LlmClient`, что и в модуле 22.

use crate::rag_indexing::{Embedder, RagIndex};
use crate::rag_query::{LlmClient, ScoredChunk, format_context, search_top_k};
use crate::rag_reranking::{RerankConfig, retrieve_filter_rerank};
use serde::{Deserialize, Serialize};

/// Один использованный источник.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct SourceRef {
    pub source: String,
    pub section: String,
    pub chunk_id: String,
}

impl From<&ScoredChunk> for SourceRef {
    fn from(c: &ScoredChunk) -> Self {
        Self {
            source: c.chunk.meta.source.clone(),
            section: c.chunk.meta.section.clone(),
            chunk_id: c.chunk.meta.chunk_id.clone(),
        }
    }
}

/// Цитата (фрагмент) из чанка.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Quote {
    pub chunk_id: String,
    pub text: String,
}

/// Структурированный ответ.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CitedAnswer {
    pub answer: String,
    pub sources: Vec<SourceRef>,
    pub quotes: Vec<Quote>,
    pub i_dont_know: bool,
    pub top_score: f32,
}

impl CitedAnswer {
    pub fn idk(reason: &str, top_score: f32) -> Self {
        Self {
            answer: format!("Не знаю: {reason}. Уточните, пожалуйста."),
            sources: vec![],
            quotes: vec![],
            i_dont_know: true,
            top_score,
        }
    }
}

/// Конфигурация анти-галлюцинаций.
#[derive(Debug, Clone)]
pub struct CitationConfig {
    /// Если top-score ниже — отвечаем "не знаю".
    pub min_top_score: f32,
    /// Максимальная длина одной цитаты (символы).
    pub max_quote_chars: usize,
    /// Сколько чанков максимум берём в контекст.
    pub top_k: usize,
    /// Настройки реранкинга/фильтра.
    pub rerank: RerankConfig,
    /// Начальный k до реранкинга.
    pub initial_k: usize,
}

impl Default for CitationConfig {
    fn default() -> Self {
        Self {
            min_top_score: 0.1,
            max_quote_chars: 300,
            top_k: 3,
            rerank: RerankConfig::default(),
            initial_k: 10,
        }
    }
}

pub const SYSTEM_CITED: &str =
    "Ты — помощник с обязательной цитируемостью.\n\
     Используй ТОЛЬКО предоставленный контекст.\n\
     В ответе ссылайся на блоки контекста через [1], [2], ...\n\
     Если информации в контексте нет — скажи 'не знаю'.";

/// Формирует обрезанную цитату из чанка.
fn make_quote(chunk: &ScoredChunk, max_chars: usize) -> Quote {
    let text = chunk.chunk.text.trim();
    let clipped: String = text.chars().take(max_chars).collect();
    let clipped = if text.chars().count() > max_chars {
        format!("{clipped}…")
    } else {
        clipped
    };
    Quote {
        chunk_id: chunk.chunk.meta.chunk_id.clone(),
        text: clipped,
    }
}

/// Основная функция RAG-ответа с цитатами и режимом "не знаю".
pub fn answer_with_citations<E: Embedder, L: LlmClient>(
    index: &RagIndex,
    embedder: &E,
    llm: &L,
    question: &str,
    cfg: &CitationConfig,
) -> Result<CitedAnswer, Box<dyn std::error::Error>> {
    let chunks = retrieve_filter_rerank(
        index,
        embedder,
        question,
        cfg.initial_k,
        cfg.top_k,
        &cfg.rerank,
    );

    let top_score = chunks.first().map(|c| c.score).unwrap_or(0.0);

    if chunks.is_empty() {
        return Ok(CitedAnswer::idk(
            "релевантных чанков не найдено",
            top_score,
        ));
    }
    if top_score < cfg.min_top_score {
        return Ok(CitedAnswer::idk(
            &format!(
                "максимальная релевантность {:.3} ниже порога {:.3}",
                top_score, cfg.min_top_score
            ),
            top_score,
        ));
    }

    let context = format_context(&chunks);
    let user_prompt = format!(
        "Контекст (фрагменты с номерами):\n{context}\nВопрос: {question}\n\
         Ответь коротко, ссылайся на блоки как [1], [2].",
    );
    let raw = llm.complete(SYSTEM_CITED, &user_prompt)?;

    // Собираем источники и цитаты.
    let sources: Vec<SourceRef> = chunks.iter().map(SourceRef::from).collect();
    let quotes: Vec<Quote> = chunks
        .iter()
        .map(|c| make_quote(c, cfg.max_quote_chars))
        .collect();

    // Если LLM всё-таки сказала "не знаю" — прокидываем флаг.
    let is_idk = raw.to_lowercase().contains("не знаю") || raw.to_lowercase().contains("i don't know");

    Ok(CitedAnswer {
        answer: raw,
        sources,
        quotes,
        i_dont_know: is_idk,
        top_score,
    })
}

/// Результат валидации ответа.
#[derive(Debug, Clone)]
pub struct CitationValidation {
    pub has_sources: bool,
    pub has_quotes: bool,
    /// Ответ упоминает хотя бы один токен из цитат (семантическая сверка).
    pub answer_matches_quotes: bool,
}

pub fn validate(answer: &CitedAnswer) -> CitationValidation {
    let answer_lc = answer.answer.to_lowercase();
    let mut matches = false;
    for q in &answer.quotes {
        // Ищем хотя бы одно осмысленное совпадение (4+ символов) между
        // цитатой и ответом. Это грубая, но полезная проверка.
        for token in q.text.split_whitespace() {
            let token = token.trim_matches(|c: char| !c.is_alphanumeric()).to_lowercase();
            if token.chars().count() >= 4 && answer_lc.contains(&token) {
                matches = true;
                break;
            }
        }
        if matches {
            break;
        }
    }
    CitationValidation {
        has_sources: !answer.sources.is_empty(),
        has_quotes: !answer.quotes.is_empty(),
        answer_matches_quotes: matches,
    }
}

/// "LLM", склеивающая первые предложения цитат — для тестов анти-галлюцинаций.
/// Она всегда опирается на переданный user-промпт, имитируя аккуратное
/// поведение RAG-помощника.
pub struct QuoteComposerLlm;

impl LlmClient for QuoteComposerLlm {
    fn complete(&self, _system: &str, user: &str) -> Result<String, Box<dyn std::error::Error>> {
        // Ищем блоки "[1] ... \n ... \n\n" и склеиваем первые строки.
        let mut bits = Vec::new();
        for block in user.split("\n\n") {
            if block.trim_start().starts_with('[') {
                // Берём последнюю строку блока, исключая заголовок.
                if let Some(body) = block.splitn(2, '\n').nth(1) {
                    bits.push(body.trim().to_string());
                }
            }
        }
        if bits.is_empty() {
            Ok("не знаю".to_string())
        } else {
            Ok(bits.join(" "))
        }
    }
}

/// Прогон 10 контрольных вопросов для проверки наличия источников/цитат и
/// режима "не знаю". Используем детерминированную LLM из `QuoteComposerLlm`.
pub fn run_validation_suite<E: Embedder, L: LlmClient>(
    index: &RagIndex,
    embedder: &E,
    llm: &L,
    cfg: &CitationConfig,
) -> Vec<(String, CitedAnswer, CitationValidation)> {
    let qs = crate::rag_query::control_questions();
    let mut out = Vec::new();
    for q in qs {
        let answer = answer_with_citations(index, embedder, llm, q.question, cfg).unwrap();
        let validation = validate(&answer);
        out.push((q.question.to_string(), answer, validation));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rag_query::default_index;

    #[test]
    fn answer_contains_sources_and_quotes() {
        let (index, embedder) = default_index();
        let cfg = CitationConfig::default();
        let llm = QuoteComposerLlm;
        let a = answer_with_citations(&index, &embedder, &llm, "что такое RAG?", &cfg).unwrap();
        assert!(!a.sources.is_empty());
        assert!(!a.quotes.is_empty());
        assert!(!a.i_dont_know);
    }

    #[test]
    fn low_relevance_triggers_dont_know() {
        let (index, embedder) = default_index();
        let mut cfg = CitationConfig::default();
        cfg.min_top_score = 0.99; // Порог, которому ничто не удовлетворяет
        let llm = QuoteComposerLlm;
        let a = answer_with_citations(&index, &embedder, &llm, "что такое квантовое запутывание?", &cfg)
            .unwrap();
        assert!(a.i_dont_know);
        assert!(a.sources.is_empty());
    }

    #[test]
    fn validation_detects_quote_match() {
        let (index, embedder) = default_index();
        let cfg = CitationConfig::default();
        let llm = QuoteComposerLlm;
        let a = answer_with_citations(&index, &embedder, &llm, "что такое MCP?", &cfg).unwrap();
        let v = validate(&a);
        assert!(v.has_sources);
        assert!(v.has_quotes);
        assert!(v.answer_matches_quotes);
    }

    #[test]
    fn quote_is_truncated() {
        let (index, embedder) = default_index();
        let cfg = CitationConfig {
            max_quote_chars: 10,
            ..Default::default()
        };
        let llm = QuoteComposerLlm;
        let a = answer_with_citations(&index, &embedder, &llm, "MCP", &cfg).unwrap();
        for q in &a.quotes {
            assert!(q.text.chars().count() <= 11); // 10 + ellipsis
        }
    }

    #[test]
    fn suite_runs_on_all_ten_questions() {
        let (index, embedder) = default_index();
        let cfg = CitationConfig::default();
        let llm = QuoteComposerLlm;
        let results = run_validation_suite(&index, &embedder, &llm, &cfg);
        assert_eq!(results.len(), 10);
        // Как минимум половина вопросов должна давать цитируемый ответ
        let with_sources = results.iter().filter(|(_, _, v)| v.has_sources).count();
        assert!(with_sources >= 5);
    }

    #[test]
    fn source_ref_serializes() {
        let s = SourceRef {
            source: "docs/a.md".into(),
            section: "Intro".into(),
            chunk_id: "a#0".into(),
        };
        let js = serde_json::to_string(&s).unwrap();
        assert!(js.contains("docs/a.md"));
    }
}
