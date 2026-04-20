//! Module 23 — Реранкинг и фильтрация.
//!
//! Второй этап после retrieval:
//! - фильтрация по порогу косинусной близости
//! - эвристический реранкер, учитывающий совпадение токенов запроса
//! - query rewrite: расширение запроса синонимами/вариантами
//!
//! Сравнение режимов:
//! - baseline (top-k из ретривера)
//! - with filter
//! - with filter + rerank
//! - with query rewrite + filter + rerank

use crate::rag_indexing::{Embedder, RagIndex, tokenize};
use crate::rag_query::{ScoredChunk, search_top_k};

/// Конфигурация реранкера/фильтра.
#[derive(Debug, Clone)]
pub struct RerankConfig {
    /// Минимальный cosine-score, ниже которого чанк считается нерелевантным.
    pub min_similarity: f32,
    /// Вес, на который увеличивается скор за каждое совпадение токена запроса.
    pub keyword_boost: f32,
    /// Вес исходной косинусной близости при смешивании скорингов.
    pub vector_weight: f32,
}

impl Default for RerankConfig {
    fn default() -> Self {
        Self {
            min_similarity: 0.05,
            keyword_boost: 0.15,
            vector_weight: 1.0,
        }
    }
}

/// Отбрасывает чанки со score ниже `min_similarity`.
pub fn filter_by_threshold(chunks: Vec<ScoredChunk>, threshold: f32) -> Vec<ScoredChunk> {
    chunks.into_iter().filter(|c| c.score >= threshold).collect()
}

/// Эвристический реранкер: линейная комбинация косинус-скоринга и количества
/// совпавших токенов запроса в тексте чанка.
pub fn rerank_keyword(
    mut chunks: Vec<ScoredChunk>,
    query: &str,
    cfg: &RerankConfig,
) -> Vec<ScoredChunk> {
    let q_tokens: std::collections::HashSet<String> = tokenize(query).into_iter().collect();
    for c in chunks.iter_mut() {
        let chunk_tokens: std::collections::HashSet<String> =
            tokenize(&c.chunk.text).into_iter().collect();
        let overlap = q_tokens.intersection(&chunk_tokens).count() as f32;
        c.score = cfg.vector_weight * c.score + cfg.keyword_boost * overlap;
    }
    chunks.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
    chunks
}

/// Query rewrite: простая эвристика, добавляющая к запросу ключевые
/// синонимы/термины. Для теста достаточно словаря; в реале — LLM.
pub fn rewrite_query(query: &str) -> Vec<String> {
    let lower = query.to_lowercase();
    let mut variants = vec![query.to_string()];

    let pairs = [
        ("сборщик мусора", "garbage collector gc ownership"),
        ("rag", "retrieval augmented generation"),
        ("mcp", "model context protocol tools"),
        ("чанк", "chunk embedding"),
        ("источник", "source citation"),
        ("цитата", "citation quote"),
        ("поиск", "retrieval search"),
        ("concurrency", "fearless concurrency data race"),
    ];
    for (ru, extra) in pairs {
        if lower.contains(ru) {
            variants.push(format!("{query} {extra}"));
        }
    }
    variants
}

/// Полный retrieval-pipeline: ретривер → filter → rerank → truncate.
pub fn retrieve_filter_rerank<E: Embedder>(
    index: &RagIndex,
    embedder: &E,
    query: &str,
    initial_k: usize,
    final_k: usize,
    cfg: &RerankConfig,
) -> Vec<ScoredChunk> {
    let raw = search_top_k(index, embedder, query, initial_k);
    let filtered = filter_by_threshold(raw, cfg.min_similarity);
    let mut reranked = rerank_keyword(filtered, query, cfg);
    reranked.truncate(final_k);
    reranked
}

/// Выполняет несколько rewrite-вариантов и объединяет результаты,
/// дедуплицируя по `chunk_id` и сохраняя максимальный score.
pub fn retrieve_with_rewrite<E: Embedder>(
    index: &RagIndex,
    embedder: &E,
    query: &str,
    initial_k: usize,
    final_k: usize,
    cfg: &RerankConfig,
) -> Vec<ScoredChunk> {
    use std::collections::HashMap;
    let variants = rewrite_query(query);
    let mut best: HashMap<String, ScoredChunk> = HashMap::new();
    for v in &variants {
        let hits = retrieve_filter_rerank(index, embedder, v, initial_k, initial_k, cfg);
        for h in hits {
            best.entry(h.chunk.meta.chunk_id.clone())
                .and_modify(|existing| {
                    if h.score > existing.score {
                        *existing = h.clone();
                    }
                })
                .or_insert(h);
        }
    }
    let mut out: Vec<ScoredChunk> = best.into_values().collect();
    out.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
    out.truncate(final_k);
    out
}

/// Режим retrieval для сравнения.
#[derive(Debug, Clone, Copy)]
pub enum RetrievalMode {
    Baseline,
    Filter,
    FilterAndRerank,
    Rewrite,
}

pub fn retrieve_mode<E: Embedder>(
    mode: RetrievalMode,
    index: &RagIndex,
    embedder: &E,
    query: &str,
    initial_k: usize,
    final_k: usize,
    cfg: &RerankConfig,
) -> Vec<ScoredChunk> {
    match mode {
        RetrievalMode::Baseline => {
            let mut raw = search_top_k(index, embedder, query, initial_k);
            raw.truncate(final_k);
            raw
        }
        RetrievalMode::Filter => {
            let raw = search_top_k(index, embedder, query, initial_k);
            let mut filtered = filter_by_threshold(raw, cfg.min_similarity);
            filtered.truncate(final_k);
            filtered
        }
        RetrievalMode::FilterAndRerank => {
            retrieve_filter_rerank(index, embedder, query, initial_k, final_k, cfg)
        }
        RetrievalMode::Rewrite => {
            retrieve_with_rewrite(index, embedder, query, initial_k, final_k, cfg)
        }
    }
}

/// Метрики качества retrieval для одной пары (вопрос, ожидаемый источник).
#[derive(Debug, Clone)]
pub struct RetrievalMetrics {
    pub precision: f32,
    pub hit_at_k: bool,
    pub size: usize,
}

pub fn evaluate_retrieval(
    chunks: &[ScoredChunk],
    expected_sources: &[&str],
) -> RetrievalMetrics {
    if chunks.is_empty() {
        return RetrievalMetrics {
            precision: 0.0,
            hit_at_k: false,
            size: 0,
        };
    }
    let mut relevant = 0;
    for c in chunks {
        if expected_sources.iter().any(|s| c.chunk.meta.source == *s) {
            relevant += 1;
        }
    }
    RetrievalMetrics {
        precision: relevant as f32 / chunks.len() as f32,
        hit_at_k: relevant > 0,
        size: chunks.len(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rag_query::{control_questions, default_index};

    #[test]
    fn threshold_filters_below() {
        let (index, embedder) = default_index();
        let raw = search_top_k(&index, &embedder, "MCP", 20);
        let filtered = filter_by_threshold(raw.clone(), 0.1);
        assert!(filtered.len() <= raw.len());
        assert!(filtered.iter().all(|c| c.score >= 0.1));
    }

    #[test]
    fn rerank_keeps_relevant_top() {
        let (index, embedder) = default_index();
        let raw = search_top_k(&index, &embedder, "mcp tools", 5);
        let ranked = rerank_keyword(raw, "mcp tools", &RerankConfig::default());
        assert!(!ranked.is_empty());
        // После реранкинга первый чанк обязан быть из mcp-документа
        assert!(ranked[0].chunk.meta.source.contains("mcp"));
    }

    #[test]
    fn rewrite_adds_variants_for_mcp() {
        let variants = rewrite_query("что такое MCP?");
        assert!(variants.len() >= 2);
        assert!(variants.iter().any(|v| v.contains("model context protocol")));
    }

    #[test]
    fn rewrite_returns_original_when_unknown() {
        let variants = rewrite_query("ничего особенного");
        assert_eq!(variants.len(), 1);
    }

    #[test]
    fn filter_rerank_pipeline_returns_final_k() {
        let (index, embedder) = default_index();
        let cfg = RerankConfig::default();
        let out = retrieve_filter_rerank(&index, &embedder, "RAG", 10, 3, &cfg);
        assert!(out.len() <= 3);
    }

    #[test]
    fn rewrite_pipeline_dedupes_by_chunk_id() {
        let (index, embedder) = default_index();
        let cfg = RerankConfig::default();
        let out = retrieve_with_rewrite(&index, &embedder, "MCP", 10, 5, &cfg);
        let ids: std::collections::HashSet<_> =
            out.iter().map(|c| c.chunk.meta.chunk_id.clone()).collect();
        assert_eq!(ids.len(), out.len());
    }

    #[test]
    fn modes_can_be_compared() {
        let (index, embedder) = default_index();
        let cfg = RerankConfig::default();
        let qs = control_questions();
        let modes = [
            RetrievalMode::Baseline,
            RetrievalMode::Filter,
            RetrievalMode::FilterAndRerank,
            RetrievalMode::Rewrite,
        ];
        let mut hits: std::collections::HashMap<&str, usize> = std::collections::HashMap::new();
        for q in &qs {
            for m in modes {
                let chunks = retrieve_mode(m, &index, &embedder, q.question, 10, 3, &cfg);
                let metrics = evaluate_retrieval(&chunks, q.expected_sources);
                let key = match m {
                    RetrievalMode::Baseline => "baseline",
                    RetrievalMode::Filter => "filter",
                    RetrievalMode::FilterAndRerank => "rerank",
                    RetrievalMode::Rewrite => "rewrite",
                };
                if metrics.hit_at_k {
                    *hits.entry(key).or_default() += 1;
                }
            }
        }
        // Каждый режим должен давать хотя бы одно попадание
        for k in ["baseline", "filter", "rerank", "rewrite"] {
            assert!(*hits.get(k).unwrap_or(&0) > 0, "mode {k} had no hits");
        }
    }
}
