//! Module 22 — Первый RAG-запрос.
//!
//! Пайплайн:
//! - принимает вопрос
//! - ищет релевантные чанки в `RagIndex` по косинусной близости
//! - собирает prompt "контекст + вопрос"
//! - вызывает LLM (через инжектируемый `LlmClient`) и получает ответ
//!
//! Два режима: с RAG и без RAG (baseline). В комплекте — 10 контрольных
//! вопросов с ожидаемыми ключевыми словами и источниками.

use crate::rag_indexing::{Chunk, Embedder, HashEmbedder, RagIndex, cosine_similarity};

/// Оценка чанка при поиске.
#[derive(Debug, Clone)]
pub struct ScoredChunk {
    pub chunk: Chunk,
    pub score: f32,
}

/// Поиск top-k чанков по косинусной близости.
pub fn search_top_k<E: Embedder>(
    index: &RagIndex,
    embedder: &E,
    query: &str,
    k: usize,
) -> Vec<ScoredChunk> {
    let q = embedder.embed(query);
    let mut scored: Vec<ScoredChunk> = index
        .chunks
        .iter()
        .map(|c| ScoredChunk {
            chunk: c.clone(),
            score: cosine_similarity(&c.embedding, &q),
        })
        .collect();
    scored.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
    scored.truncate(k);
    scored
}

/// Собирает блок контекста для prompt — с нумерацией источников.
pub fn format_context(chunks: &[ScoredChunk]) -> String {
    let mut out = String::new();
    for (i, sc) in chunks.iter().enumerate() {
        out.push_str(&format!(
            "[{}] source={} section={} chunk_id={}\n{}\n\n",
            i + 1,
            sc.chunk.meta.source,
            sc.chunk.meta.section,
            sc.chunk.meta.chunk_id,
            sc.chunk.text.trim()
        ));
    }
    out
}

/// Абстракция LLM-клиента. Позволяет тестировать RAG без обращения к сети.
pub trait LlmClient {
    fn complete(&self, system: &str, user: &str) -> Result<String, Box<dyn std::error::Error>>;
}

/// Детерминированный "LLM": просто эхо вопроса и списка найденных
/// заголовков секций. Используется в тестах, чтобы проверять, что
/// контекст дошёл до модели.
pub struct StubLlm;

impl LlmClient for StubLlm {
    fn complete(&self, system: &str, user: &str) -> Result<String, Box<dyn std::error::Error>> {
        Ok(format!("STUB[sys_len={}]: {user}", system.len()))
    }
}

/// Ответ RAG-пайплайна.
#[derive(Debug, Clone)]
pub struct RagAnswer {
    pub answer: String,
    pub used_chunks: Vec<ScoredChunk>,
    pub mode: &'static str,
}

/// Режим вызова LLM.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QueryMode {
    NoRag,
    WithRag { top_k: usize },
}

pub const SYSTEM_WITHOUT_RAG: &str =
    "Ты — помощник. Отвечай на вопросы опираясь только на свои знания.";

pub const SYSTEM_WITH_RAG: &str =
    "Ты — помощник. Используй ТОЛЬКО переданный контекст для ответа.\n\
     Если информации нет — скажи 'не знаю'.";

/// Выполняет запрос в одном из режимов.
pub fn ask<E: Embedder, L: LlmClient>(
    index: &RagIndex,
    embedder: &E,
    llm: &L,
    question: &str,
    mode: QueryMode,
) -> Result<RagAnswer, Box<dyn std::error::Error>> {
    match mode {
        QueryMode::NoRag => {
            let answer = llm.complete(SYSTEM_WITHOUT_RAG, question)?;
            Ok(RagAnswer {
                answer,
                used_chunks: vec![],
                mode: "no-rag",
            })
        }
        QueryMode::WithRag { top_k } => {
            let chunks = search_top_k(index, embedder, question, top_k);
            let context = format_context(&chunks);
            let prompt = format!("Контекст:\n{context}\nВопрос: {question}\nОтвет:");
            let answer = llm.complete(SYSTEM_WITH_RAG, &prompt)?;
            Ok(RagAnswer {
                answer,
                used_chunks: chunks,
                mode: "with-rag",
            })
        }
    }
}

/// Контрольный вопрос + ожидаемые ключевые слова + ожидаемые источники.
#[derive(Debug, Clone)]
pub struct ControlQuestion {
    pub question: &'static str,
    pub expected_keywords: &'static [&'static str],
    pub expected_sources: &'static [&'static str],
}

/// 10 контрольных вопросов по `sample_corpus()` из `rag_indexing`.
pub fn control_questions() -> Vec<ControlQuestion> {
    vec![
        ControlQuestion {
            question: "Что заменяет сборщик мусора в Rust?",
            expected_keywords: &["ownership"],
            expected_sources: &["docs/rust.md"],
        },
        ControlQuestion {
            question: "Какую задачу решает Rust в concurrency?",
            expected_keywords: &["fearless", "concurrency"],
            expected_sources: &["docs/rust.md"],
        },
        ControlQuestion {
            question: "Что стандартизирует MCP?",
            expected_keywords: &["tool", "access"],
            expected_sources: &["docs/mcp.md"],
        },
        ControlQuestion {
            question: "Какие атрибуты у инструмента MCP?",
            expected_keywords: &["name", "description", "schema"],
            expected_sources: &["docs/mcp.md"],
        },
        ControlQuestion {
            question: "Как MCP-сервер отдаёт инструменты клиенту?",
            expected_keywords: &["stdio", "sse"],
            expected_sources: &["docs/mcp.md"],
        },
        ControlQuestion {
            question: "Что такое RAG?",
            expected_keywords: &["retrieval", "augmented", "context"],
            expected_sources: &["docs/rag.md"],
        },
        ControlQuestion {
            question: "Из чего собирается retrieval-корпус?",
            expected_keywords: &["chunk", "embedding"],
            expected_sources: &["docs/rag.md"],
        },
        ControlQuestion {
            question: "Как выбираются top-k чанки?",
            expected_keywords: &["cosine", "similarity"],
            expected_sources: &["docs/rag.md"],
        },
        ControlQuestion {
            question: "Зачем цитаты в ответах RAG?",
            expected_keywords: &["sources", "hallucination"],
            expected_sources: &["docs/rag.md"],
        },
        ControlQuestion {
            question: "Что такое data race в Rust?",
            expected_keywords: &["data", "race"],
            expected_sources: &["docs/rust.md"],
        },
    ]
}

/// Метрики качества одного ответа.
#[derive(Debug, Clone)]
pub struct AnswerMetrics {
    pub keyword_hits: usize,
    pub keyword_total: usize,
    pub source_hits: usize,
    pub source_total: usize,
}

impl AnswerMetrics {
    pub fn keyword_recall(&self) -> f32 {
        if self.keyword_total == 0 {
            1.0
        } else {
            self.keyword_hits as f32 / self.keyword_total as f32
        }
    }

    pub fn source_recall(&self) -> f32 {
        if self.source_total == 0 {
            1.0
        } else {
            self.source_hits as f32 / self.source_total as f32
        }
    }
}

/// Оценивает ответ против ожиданий: считает попадания по ключевым словам
/// (в тексте ответа + контексте) и по источникам (только RAG).
pub fn evaluate(answer: &RagAnswer, question: &ControlQuestion) -> AnswerMetrics {
    let haystack = {
        let mut s = answer.answer.to_lowercase();
        for c in &answer.used_chunks {
            s.push(' ');
            s.push_str(&c.chunk.text.to_lowercase());
        }
        s
    };

    let keyword_hits = question
        .expected_keywords
        .iter()
        .filter(|kw| haystack.contains(&kw.to_lowercase()))
        .count();

    let source_hits = question
        .expected_sources
        .iter()
        .filter(|src| {
            answer
                .used_chunks
                .iter()
                .any(|c| c.chunk.meta.source == **src)
        })
        .count();

    AnswerMetrics {
        keyword_hits,
        keyword_total: question.expected_keywords.len(),
        source_hits,
        source_total: question.expected_sources.len(),
    }
}

/// Прогоняет все 10 вопросов в двух режимах и возвращает результаты.
pub fn compare_modes<E: Embedder, L: LlmClient>(
    index: &RagIndex,
    embedder: &E,
    llm: &L,
) -> Vec<ComparisonRow> {
    let questions = control_questions();
    let mut rows = Vec::new();
    for q in questions {
        let no = ask(index, embedder, llm, q.question, QueryMode::NoRag).unwrap();
        let with = ask(
            index,
            embedder,
            llm,
            q.question,
            QueryMode::WithRag { top_k: 3 },
        )
        .unwrap();
        rows.push(ComparisonRow {
            question: q.question,
            no_rag: evaluate(&no, &q),
            with_rag: evaluate(&with, &q),
        });
    }
    rows
}

/// Одна строка сравнения.
#[derive(Debug, Clone)]
pub struct ComparisonRow {
    pub question: &'static str,
    pub no_rag: AnswerMetrics,
    pub with_rag: AnswerMetrics,
}

/// "LLM", который возвращает первый чанк как ответ — нужен для тестов,
/// чтобы чётко видеть разницу `NoRag` vs `WithRag`.
pub struct ContextEchoLlm;

impl LlmClient for ContextEchoLlm {
    fn complete(&self, _system: &str, user: &str) -> Result<String, Box<dyn std::error::Error>> {
        // Возвращаем всё, что было передано как контекст — для оценки
        // по keyword_recall этого достаточно.
        Ok(user.to_string())
    }
}

/// Конструирует индекс из sample_corpus — единая точка для модулей 22-25.
pub fn default_index() -> (RagIndex, HashEmbedder) {
    let embedder = HashEmbedder::new();
    let corpus = crate::rag_indexing::sample_corpus();
    let index = crate::rag_indexing::build_index(
        &corpus,
        crate::rag_indexing::ChunkStrategy::StructureAware,
        &embedder,
    );
    (index, embedder)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn search_returns_top_k_sorted() {
        let (index, embedder) = default_index();
        let top = search_top_k(&index, &embedder, "MCP servers", 3);
        assert_eq!(top.len(), 3);
        for w in top.windows(2) {
            assert!(w[0].score >= w[1].score);
        }
    }

    #[test]
    fn ten_control_questions_defined() {
        let qs = control_questions();
        assert_eq!(qs.len(), 10);
        for q in &qs {
            assert!(!q.expected_keywords.is_empty());
            assert!(!q.expected_sources.is_empty());
        }
    }

    #[test]
    fn with_rag_provides_chunks() {
        let (index, embedder) = default_index();
        let llm = StubLlm;
        let a = ask(&index, &embedder, &llm, "что такое RAG?", QueryMode::WithRag { top_k: 2 })
            .unwrap();
        assert_eq!(a.used_chunks.len(), 2);
        assert_eq!(a.mode, "with-rag");
    }

    #[test]
    fn no_rag_has_no_chunks() {
        let (index, embedder) = default_index();
        let llm = StubLlm;
        let a = ask(&index, &embedder, &llm, "что такое RAG?", QueryMode::NoRag).unwrap();
        assert!(a.used_chunks.is_empty());
    }

    #[test]
    fn with_rag_beats_no_rag_on_keyword_recall() {
        let (index, embedder) = default_index();
        let llm = ContextEchoLlm;
        let rows = compare_modes(&index, &embedder, &llm);
        let no_avg: f32 =
            rows.iter().map(|r| r.no_rag.keyword_recall()).sum::<f32>() / rows.len() as f32;
        let with_avg: f32 =
            rows.iter().map(|r| r.with_rag.keyword_recall()).sum::<f32>() / rows.len() as f32;
        // С RAG эхо-LLM видит контекст и keyword_recall заведомо выше
        assert!(with_avg > no_avg);
    }

    #[test]
    fn evaluation_counts_hits() {
        let q = ControlQuestion {
            question: "x",
            expected_keywords: &["alpha", "beta"],
            expected_sources: &["src"],
        };
        let a = RagAnswer {
            answer: "Alpha and gamma.".into(),
            used_chunks: vec![],
            mode: "with-rag",
        };
        let m = evaluate(&a, &q);
        assert_eq!(m.keyword_hits, 1);
        assert_eq!(m.keyword_total, 2);
    }
}
