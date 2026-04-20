//! Module 19 — Композиция MCP-инструментов.
//!
//! Три инструмента, которые работают как конвейер:
//! - `search` — ищет документы по ключевому слову
//! - `summarize` — делает краткое резюме найденных результатов
//! - `save_to_file` — сохраняет итог на диск (JSON)
//!
//! Pipeline: `search → summarize → save_to_file`. Результат каждого шага
//! передаётся в следующий, ошибки обрываются прерыванием цепочки.

use crate::mcp_connection::{
    InProcessMcpServer, McpClient, McpError, McpServer, McpToolDescriptor, ToolCallResult,
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

/// Мини-документ с текстовым содержимым.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Document {
    pub id: String,
    pub title: String,
    pub body: String,
}

/// Простейший поиск по подстроке — достаточно для тестов конвейера.
pub fn naive_search(corpus: &[Document], query: &str) -> Vec<Document> {
    let q = query.to_lowercase();
    corpus
        .iter()
        .filter(|d| {
            d.title.to_lowercase().contains(&q) || d.body.to_lowercase().contains(&q)
        })
        .cloned()
        .collect()
}

/// Экстрактивный саммаризатор: берёт первое предложение каждого документа.
pub fn extractive_summary(docs: &[Document]) -> String {
    let mut parts = Vec::new();
    for d in docs {
        let first = d
            .body
            .split_terminator(['.', '!', '?'])
            .next()
            .unwrap_or(&d.body)
            .trim();
        parts.push(format!("- {}: {}.", d.title, first));
    }
    parts.join("\n")
}

/// Шаг в истории исполнения пайплайна.
#[derive(Debug, Clone)]
pub struct PipelineStep {
    pub tool: String,
    pub input: Value,
    pub output: Value,
}

/// Собирает сервер с тремя инструментами. Поиск идёт по статическому
/// корпусу, переданному в замыкание; файл сохраняется в переданный
/// временный каталог, чтобы тесты не трогали cwd.
pub fn build_pipeline_server(
    corpus: Arc<Vec<Document>>,
    save_dir: PathBuf,
) -> Arc<InProcessMcpServer> {
    let server = Arc::new(InProcessMcpServer::new("pipeline-server"));

    let c = Arc::clone(&corpus);
    server.register(
        McpToolDescriptor::new("search", "Поиск документов по ключевому слову")
            .with_param("query", "string", "поисковый запрос", true),
        move |args| {
            let q = args
                .get("query")
                .and_then(|v| v.as_str())
                .ok_or_else(|| McpError::InvalidArguments("missing 'query'".into()))?;
            let results = naive_search(&c, q);
            Ok(ToolCallResult {
                content: serde_json::to_string(&results).unwrap(),
                is_error: false,
            })
        },
    );

    server.register(
        McpToolDescriptor::new("summarize", "Краткое резюме набора документов")
            .with_param("documents", "array", "список документов из search", true),
        |args| {
            let docs_v = args.get("documents").ok_or_else(|| {
                McpError::InvalidArguments("missing 'documents'".into())
            })?;
            let docs: Vec<Document> = serde_json::from_value(docs_v.clone())
                .map_err(|e| McpError::InvalidArguments(format!("bad documents: {e}")))?;
            let summary = extractive_summary(&docs);
            Ok(ToolCallResult {
                content: json!({ "summary": summary, "count": docs.len() }).to_string(),
                is_error: false,
            })
        },
    );

    let save_dir = save_dir;
    server.register(
        McpToolDescriptor::new("save_to_file", "Сохраняет полезную нагрузку в файл")
            .with_param("filename", "string", "имя файла", true)
            .with_param("payload", "object", "что сохранить", true),
        move |args| {
            let filename = args
                .get("filename")
                .and_then(|v| v.as_str())
                .ok_or_else(|| McpError::InvalidArguments("missing 'filename'".into()))?;
            let payload = args
                .get("payload")
                .ok_or_else(|| McpError::InvalidArguments("missing 'payload'".into()))?;
            let path = save_dir.join(filename);
            let pretty = serde_json::to_string_pretty(payload).unwrap();
            std::fs::create_dir_all(&save_dir)
                .map_err(|e| McpError::ExecutionFailed(format!("mkdir: {e}")))?;
            std::fs::write(&path, &pretty)
                .map_err(|e| McpError::ExecutionFailed(format!("write: {e}")))?;
            Ok(ToolCallResult {
                content: json!({ "path": path.to_string_lossy(), "bytes": pretty.len() })
                    .to_string(),
                is_error: false,
            })
        },
    );

    server
}

/// Выполняет линейный пайплайн `search → summarize → save_to_file`.
/// История шагов записывается в `trace`, чтобы было видно передачу данных.
pub struct Pipeline {
    client: McpClient,
    pub trace: Mutex<Vec<PipelineStep>>,
}

impl Pipeline {
    pub fn new(server: Arc<dyn McpServer>) -> Self {
        let mut client = McpClient::new();
        client.connect(server);
        Self {
            client,
            trace: Mutex::new(Vec::new()),
        }
    }

    fn run_step(&self, tool: &str, input: Value) -> Result<Value, McpError> {
        let result = self.client.call_tool(tool, &input)?;
        if result.is_error {
            return Err(McpError::ExecutionFailed(result.content));
        }
        let parsed: Value = serde_json::from_str(&result.content)
            .map_err(|e| McpError::ExecutionFailed(format!("parse {tool}: {e}")))?;
        self.trace.lock().unwrap().push(PipelineStep {
            tool: tool.to_string(),
            input,
            output: parsed.clone(),
        });
        Ok(parsed)
    }

    pub fn run(&self, query: &str, filename: &str) -> Result<Value, McpError> {
        let docs = self.run_step("search", json!({ "query": query }))?;
        let summary = self.run_step("summarize", json!({ "documents": docs }))?;
        let saved = self.run_step(
            "save_to_file",
            json!({ "filename": filename, "payload": summary }),
        )?;
        Ok(saved)
    }

    pub fn trace_snapshot(&self) -> Vec<PipelineStep> {
        self.trace.lock().unwrap().clone()
    }
}

/// Тестовый корпус: несколько документов про Rust/LLM/MCP.
pub fn sample_corpus() -> Vec<Document> {
    vec![
        Document {
            id: "d1".into(),
            title: "Intro to Rust".into(),
            body: "Rust is a systems language focused on safety. It has no garbage collector."
                .into(),
        },
        Document {
            id: "d2".into(),
            title: "LLM basics".into(),
            body: "LLMs predict next tokens. Sampling is controlled by temperature.".into(),
        },
        Document {
            id: "d3".into(),
            title: "MCP protocol".into(),
            body: "MCP lets agents call tools through a standard interface.".into(),
        },
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp_dir(name: &str) -> PathBuf {
        let p = std::env::temp_dir().join(name);
        let _ = std::fs::remove_dir_all(&p);
        p
    }

    #[test]
    fn search_filters_by_query() {
        let docs = sample_corpus();
        let hits = naive_search(&docs, "rust");
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].id, "d1");
    }

    #[test]
    fn summary_includes_all_documents() {
        let docs = sample_corpus();
        let s = extractive_summary(&docs);
        assert!(s.contains("Intro to Rust"));
        assert!(s.contains("LLM basics"));
        assert!(s.contains("MCP protocol"));
    }

    #[test]
    fn full_pipeline_writes_file() {
        let dir = tmp_dir("kairo_pipeline_full");
        let server = build_pipeline_server(Arc::new(sample_corpus()), dir.clone());
        let pipeline = Pipeline::new(server);
        let saved = pipeline.run("MCP", "result.json").unwrap();
        let path = saved.get("path").unwrap().as_str().unwrap().to_string();
        let content = std::fs::read_to_string(&path).unwrap();
        assert!(content.contains("MCP protocol"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn trace_records_all_three_steps() {
        let dir = tmp_dir("kairo_pipeline_trace");
        let server = build_pipeline_server(Arc::new(sample_corpus()), dir.clone());
        let pipeline = Pipeline::new(server);
        pipeline.run("rust", "r.json").unwrap();
        let trace = pipeline.trace_snapshot();
        assert_eq!(trace.len(), 3);
        assert_eq!(trace[0].tool, "search");
        assert_eq!(trace[1].tool, "summarize");
        assert_eq!(trace[2].tool, "save_to_file");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn pipeline_passes_data_between_steps() {
        let dir = tmp_dir("kairo_pipeline_pass");
        let server = build_pipeline_server(Arc::new(sample_corpus()), dir.clone());
        let pipeline = Pipeline::new(server);
        pipeline.run("rust", "r.json").unwrap();
        let trace = pipeline.trace_snapshot();

        let search_out = &trace[0].output;
        let summarize_in = trace[1].input.get("documents").unwrap();
        assert_eq!(search_out, summarize_in);

        let summarize_out = &trace[1].output;
        let save_payload = trace[2].input.get("payload").unwrap();
        assert_eq!(summarize_out, save_payload);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn empty_search_produces_empty_summary() {
        let dir = tmp_dir("kairo_pipeline_empty");
        let server = build_pipeline_server(Arc::new(sample_corpus()), dir.clone());
        let pipeline = Pipeline::new(server);
        pipeline.run("nonexistent", "e.json").unwrap();
        let trace = pipeline.trace_snapshot();
        let summary_count = trace[1].output.get("count").unwrap().as_u64().unwrap();
        assert_eq!(summary_count, 0);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
