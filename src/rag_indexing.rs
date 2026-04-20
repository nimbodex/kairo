//! Module 21 — Индексация документов.
//!
//! Пайплайн:
//! - чтение документов (или подача готовым корпусом)
//! - chunking — две стратегии:
//!     * `FixedSize` — по фиксированному числу символов с overlap
//!     * `StructureAware` — по заголовкам Markdown (`#`, `##`, ...)
//! - embeddings — детерминированный hash-based vectorizer, 128-мерный
//!   (без внешних зависимостей; реальные эмбеддинги подключаются через
//!   интерфейс `Embedder`)
//! - метаданные — source, title, section, chunk_id
//! - сохранение индекса в JSON и обратная загрузка
//! - сравнение двух стратегий chunking по базовым метрикам

use serde::{Deserialize, Serialize};
use std::path::Path;

/// Размерность эмбеддингов по умолчанию.
pub const EMBEDDING_DIM: usize = 128;

/// Метаданные чанка.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ChunkMeta {
    pub source: String,
    pub title: String,
    pub section: String,
    pub chunk_id: String,
}

/// Чанк + вектор.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Chunk {
    pub meta: ChunkMeta,
    pub text: String,
    pub embedding: Vec<f32>,
}

/// Стратегия chunking.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChunkStrategy {
    FixedSize { size: usize, overlap: usize },
    StructureAware,
}

/// Входной документ: исходный текст + идентифицирующие метаданные.
#[derive(Debug, Clone)]
pub struct SourceDocument {
    pub source: String,
    pub title: String,
    pub content: String,
}

impl SourceDocument {
    pub fn new(source: &str, title: &str, content: &str) -> Self {
        Self {
            source: source.to_string(),
            title: title.to_string(),
            content: content.to_string(),
        }
    }

    pub fn from_file(path: &Path) -> std::io::Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let source = path.to_string_lossy().to_string();
        let title = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("document")
            .to_string();
        Ok(Self {
            source,
            title,
            content,
        })
    }
}

/// Трейт эмбеддера — подменяется на любой backend (OpenAI, локальный ST и т.п.).
pub trait Embedder: Send + Sync {
    fn embed(&self, text: &str) -> Vec<f32>;
    fn dim(&self) -> usize {
        EMBEDDING_DIM
    }
}

/// Детерминированный эмбеддер: токенизует по словам, каждому слову
/// назначает бакет по FNV-хэшу, потом нормализует вектор. Этого достаточно
/// для экспериментов с RAG без выхода в сеть.
pub struct HashEmbedder {
    dim: usize,
}

impl HashEmbedder {
    pub fn new() -> Self {
        Self { dim: EMBEDDING_DIM }
    }

    pub fn with_dim(dim: usize) -> Self {
        Self { dim }
    }

    fn hash(s: &str) -> u64 {
        // FNV-1a
        let mut h: u64 = 0xcbf29ce484222325;
        for b in s.bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
        h
    }
}

impl Default for HashEmbedder {
    fn default() -> Self {
        Self::new()
    }
}

impl Embedder for HashEmbedder {
    fn embed(&self, text: &str) -> Vec<f32> {
        let mut v = vec![0.0f32; self.dim];
        for tok in tokenize(text) {
            let idx = (Self::hash(&tok) as usize) % self.dim;
            v[idx] += 1.0;
            // "знаковый" бит — помогает различать похожие токены
            let sign_idx = (Self::hash(&format!("~{tok}")) as usize) % self.dim;
            v[sign_idx] -= 0.5;
        }
        normalize(&mut v);
        v
    }

    fn dim(&self) -> usize {
        self.dim
    }
}

pub fn tokenize(text: &str) -> Vec<String> {
    text.to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|s| !s.is_empty())
        .map(|s| s.to_string())
        .collect()
}

pub fn normalize(v: &mut [f32]) {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    let mut dot = 0.0f32;
    let mut na = 0.0f32;
    let mut nb = 0.0f32;
    for i in 0..n {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    if na == 0.0 || nb == 0.0 {
        0.0
    } else {
        dot / (na.sqrt() * nb.sqrt())
    }
}

/// Chunking по фиксированному размеру символов с overlap.
pub fn chunk_fixed_size(doc: &SourceDocument, size: usize, overlap: usize) -> Vec<(ChunkMeta, String)> {
    assert!(size > 0, "size must be > 0");
    let overlap = overlap.min(size.saturating_sub(1));
    let chars: Vec<char> = doc.content.chars().collect();
    if chars.is_empty() {
        return Vec::new();
    }
    let mut out = Vec::new();
    let mut start = 0usize;
    let mut idx = 0usize;
    while start < chars.len() {
        let end = (start + size).min(chars.len());
        let text: String = chars[start..end].iter().collect();
        let meta = ChunkMeta {
            source: doc.source.clone(),
            title: doc.title.clone(),
            section: format!("chars[{start}..{end}]"),
            chunk_id: format!("{}#fixed#{idx}", doc.title),
        };
        out.push((meta, text));
        if end == chars.len() {
            break;
        }
        start = end - overlap;
        idx += 1;
    }
    out
}

/// Chunking по структуре Markdown: режем по заголовкам `#`/`##`/...
/// Каждый чанк — заголовок + следующий за ним текст до следующего заголовка.
pub fn chunk_structure_aware(doc: &SourceDocument) -> Vec<(ChunkMeta, String)> {
    let mut sections: Vec<(String, String)> = Vec::new();
    let mut current_title = "Intro".to_string();
    let mut current_body = String::new();

    for line in doc.content.lines() {
        if let Some(rest) = line.trim_start().strip_prefix('#') {
            if !current_body.trim().is_empty() {
                sections.push((current_title.clone(), current_body.trim().to_string()));
            }
            // ведущие '#' уже съедены один раз; уберём остальные
            let title = rest.trim_start_matches('#').trim().to_string();
            current_title = if title.is_empty() {
                "Section".to_string()
            } else {
                title
            };
            current_body.clear();
        } else {
            current_body.push_str(line);
            current_body.push('\n');
        }
    }
    if !current_body.trim().is_empty() {
        sections.push((current_title, current_body.trim().to_string()));
    }

    sections
        .into_iter()
        .enumerate()
        .map(|(i, (section, body))| {
            let text = format!("{section}\n\n{body}");
            let meta = ChunkMeta {
                source: doc.source.clone(),
                title: doc.title.clone(),
                section,
                chunk_id: format!("{}#struct#{i}", doc.title),
            };
            (meta, text)
        })
        .collect()
}

/// Применяет выбранную стратегию к документу и возвращает чанки без
/// эмбеддингов. Эмбеддинги добавляются `build_index`.
pub fn chunk_document(doc: &SourceDocument, strategy: ChunkStrategy) -> Vec<(ChunkMeta, String)> {
    match strategy {
        ChunkStrategy::FixedSize { size, overlap } => chunk_fixed_size(doc, size, overlap),
        ChunkStrategy::StructureAware => chunk_structure_aware(doc),
    }
}

/// Локальный индекс.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RagIndex {
    pub dim: usize,
    pub chunks: Vec<Chunk>,
}

impl RagIndex {
    pub fn new(dim: usize) -> Self {
        Self {
            dim,
            chunks: Vec::new(),
        }
    }

    pub fn len(&self) -> usize {
        self.chunks.len()
    }

    pub fn is_empty(&self) -> bool {
        self.chunks.is_empty()
    }

    pub fn save_json(&self, path: &Path) -> std::io::Result<()> {
        let data = serde_json::to_string_pretty(self).unwrap();
        std::fs::write(path, data)
    }

    pub fn load_json(path: &Path) -> std::io::Result<Self> {
        let raw = std::fs::read_to_string(path)?;
        let idx = serde_json::from_str(&raw).map_err(std::io::Error::other)?;
        Ok(idx)
    }
}

/// Собирает индекс из документов по выбранной стратегии.
pub fn build_index<E: Embedder>(
    docs: &[SourceDocument],
    strategy: ChunkStrategy,
    embedder: &E,
) -> RagIndex {
    let mut index = RagIndex::new(embedder.dim());
    for doc in docs {
        for (meta, text) in chunk_document(doc, strategy) {
            let embedding = embedder.embed(&text);
            index.chunks.push(Chunk {
                meta,
                text,
                embedding,
            });
        }
    }
    index
}

/// Простая статистика по чанкам — для сравнения двух стратегий.
#[derive(Debug, Clone)]
pub struct ChunkingStats {
    pub strategy: &'static str,
    pub chunk_count: usize,
    pub avg_len: f32,
    pub min_len: usize,
    pub max_len: usize,
}

pub fn chunking_stats(label: &'static str, chunks: &[(ChunkMeta, String)]) -> ChunkingStats {
    let lens: Vec<usize> = chunks.iter().map(|(_, t)| t.chars().count()).collect();
    let total: usize = lens.iter().sum();
    ChunkingStats {
        strategy: label,
        chunk_count: chunks.len(),
        avg_len: if chunks.is_empty() {
            0.0
        } else {
            total as f32 / chunks.len() as f32
        },
        min_len: *lens.iter().min().unwrap_or(&0),
        max_len: *lens.iter().max().unwrap_or(&0),
    }
}

/// Небольшой тестовый корпус (≈ раздел документации) — хватает на
/// демонстрацию chunking/RAG без внешних файлов.
pub fn sample_corpus() -> Vec<SourceDocument> {
    vec![
        SourceDocument::new(
            "docs/rust.md",
            "rust",
            "# Rust\n\nRust is a systems programming language.\n\n## Memory\n\nOwnership replaces garbage collection.\n\n## Concurrency\n\nFearless concurrency without data races.\n",
        ),
        SourceDocument::new(
            "docs/mcp.md",
            "mcp",
            "# MCP\n\nModel Context Protocol standardises tool access.\n\n## Tools\n\nTools have names, descriptions, input schemas.\n\n## Servers\n\nServers expose tools to clients over stdio or SSE.\n",
        ),
        SourceDocument::new(
            "docs/rag.md",
            "rag",
            "# RAG\n\nRetrieval augmented generation grounds answers in context.\n\n## Indexing\n\nChunking + embeddings create the retrieval corpus.\n\n## Retrieval\n\nCosine similarity scores pick top-k chunks.\n\n## Citations\n\nAnswers must cite their sources to reduce hallucinations.\n",
        ),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixed_size_chunking_produces_overlap() {
        let doc = SourceDocument::new("s", "t", "abcdefghij");
        let chunks = chunk_fixed_size(&doc, 4, 1);
        // [0..4][3..7][6..10]
        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0].1, "abcd");
        assert_eq!(chunks[1].1, "defg");
        assert_eq!(chunks[2].1, "ghij");
    }

    #[test]
    fn structure_aware_splits_by_headings() {
        let doc = SourceDocument::new(
            "s",
            "t",
            "# A\nalpha\n## B\nbeta\n## C\ngamma\n",
        );
        let chunks = chunk_structure_aware(&doc);
        assert_eq!(chunks.len(), 3);
        let sections: Vec<_> = chunks.iter().map(|(m, _)| m.section.as_str()).collect();
        assert_eq!(sections, vec!["A", "B", "C"]);
    }

    #[test]
    fn hash_embedder_is_deterministic() {
        let e = HashEmbedder::new();
        let v1 = e.embed("hello world");
        let v2 = e.embed("hello world");
        assert_eq!(v1, v2);
    }

    #[test]
    fn cosine_similarity_bounds() {
        let e = HashEmbedder::new();
        let a = e.embed("rust ownership memory");
        let b = e.embed("rust ownership memory");
        let c = e.embed("completely different tokens here");
        assert!((cosine_similarity(&a, &b) - 1.0).abs() < 1e-5);
        assert!(cosine_similarity(&a, &c) < cosine_similarity(&a, &b));
    }

    #[test]
    fn build_index_includes_metadata() {
        let embedder = HashEmbedder::new();
        let idx = build_index(
            &sample_corpus(),
            ChunkStrategy::StructureAware,
            &embedder,
        );
        assert!(idx.len() > 3);
        let first = &idx.chunks[0];
        assert!(!first.meta.chunk_id.is_empty());
        assert!(!first.meta.source.is_empty());
        assert_eq!(first.embedding.len(), EMBEDDING_DIM);
    }

    #[test]
    fn index_json_roundtrip() {
        let embedder = HashEmbedder::new();
        let idx = build_index(
            &sample_corpus(),
            ChunkStrategy::FixedSize {
                size: 100,
                overlap: 20,
            },
            &embedder,
        );
        let tmp = std::env::temp_dir().join("kairo_index.json");
        idx.save_json(&tmp).unwrap();
        let loaded = RagIndex::load_json(&tmp).unwrap();
        assert_eq!(idx.len(), loaded.len());
        assert_eq!(idx.dim, loaded.dim);
        let _ = std::fs::remove_file(&tmp);
    }

    #[test]
    fn compare_strategies_stats() {
        let corpus = sample_corpus();
        let mut fixed = Vec::new();
        let mut structured = Vec::new();
        for doc in &corpus {
            fixed.extend(chunk_fixed_size(doc, 80, 20));
            structured.extend(chunk_structure_aware(doc));
        }
        let f = chunking_stats("fixed", &fixed);
        let s = chunking_stats("structured", &structured);
        assert!(f.chunk_count >= s.chunk_count);
        // size=80 с overlap=20 даёт среднее не больше 80
        assert!(f.avg_len <= 80.0);
    }
}
