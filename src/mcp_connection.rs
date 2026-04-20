//! Module 16 — Подключение MCP.
//!
//! Минимальная реализация MCP-подобного клиента/сервера в процессе:
//! - устанавливает соединение с MCP-сервером
//! - получает список доступных инструментов (name + description + schema)
//!
//! Настоящий MCP SDK общается по stdio/SSE; здесь — трейты и in-process
//! имплементация, которой достаточно для разработки и тестов без внешних
//! зависимостей. Та же абстракция переносится на реальный транспорт.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// Схема входных параметров инструмента (упрощённый JSON Schema).
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ToolInputSchema {
    #[serde(rename = "type")]
    pub kind: String,
    pub properties: HashMap<String, ToolProperty>,
    #[serde(default)]
    pub required: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolProperty {
    #[serde(rename = "type")]
    pub kind: String,
    pub description: String,
}

/// Описание инструмента, отдаваемое MCP-сервером.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpToolDescriptor {
    pub name: String,
    pub description: String,
    pub input_schema: ToolInputSchema,
}

impl McpToolDescriptor {
    pub fn new(name: &str, description: &str) -> Self {
        Self {
            name: name.to_string(),
            description: description.to_string(),
            input_schema: ToolInputSchema {
                kind: "object".to_string(),
                properties: HashMap::new(),
                required: Vec::new(),
            },
        }
    }

    pub fn with_param(mut self, name: &str, kind: &str, description: &str, required: bool) -> Self {
        self.input_schema.properties.insert(
            name.to_string(),
            ToolProperty {
                kind: kind.to_string(),
                description: description.to_string(),
            },
        );
        if required {
            self.input_schema.required.push(name.to_string());
        }
        self
    }
}

/// Ошибка MCP-слоя.
#[derive(Debug, Clone)]
pub enum McpError {
    NotConnected,
    UnknownTool(String),
    InvalidArguments(String),
    ExecutionFailed(String),
}

impl std::fmt::Display for McpError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            McpError::NotConnected => write!(f, "MCP: not connected"),
            McpError::UnknownTool(t) => write!(f, "MCP: unknown tool '{t}'"),
            McpError::InvalidArguments(m) => write!(f, "MCP: invalid arguments: {m}"),
            McpError::ExecutionFailed(m) => write!(f, "MCP: execution failed: {m}"),
        }
    }
}

impl std::error::Error for McpError {}

/// Результат вызова инструмента.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallResult {
    pub content: String,
    pub is_error: bool,
}

/// Трейт MCP-сервера: регистрирует инструменты, исполняет вызовы.
pub trait McpServer: Send + Sync {
    fn name(&self) -> &str;
    fn list_tools(&self) -> Vec<McpToolDescriptor>;
    fn call_tool(
        &self,
        tool: &str,
        args: &serde_json::Value,
    ) -> Result<ToolCallResult, McpError>;
}

/// In-process MCP-сервер: хранит набор инструментов и их обработчиков.
type ToolHandler =
    Arc<dyn Fn(&serde_json::Value) -> Result<ToolCallResult, McpError> + Send + Sync>;

pub struct InProcessMcpServer {
    name: String,
    tools: Mutex<HashMap<String, (McpToolDescriptor, ToolHandler)>>,
}

impl InProcessMcpServer {
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            tools: Mutex::new(HashMap::new()),
        }
    }

    pub fn register<F>(&self, descriptor: McpToolDescriptor, handler: F)
    where
        F: Fn(&serde_json::Value) -> Result<ToolCallResult, McpError> + Send + Sync + 'static,
    {
        let mut tools = self.tools.lock().unwrap();
        tools.insert(descriptor.name.clone(), (descriptor, Arc::new(handler)));
    }
}

impl McpServer for InProcessMcpServer {
    fn name(&self) -> &str {
        &self.name
    }

    fn list_tools(&self) -> Vec<McpToolDescriptor> {
        self.tools
            .lock()
            .unwrap()
            .values()
            .map(|(d, _)| d.clone())
            .collect()
    }

    fn call_tool(
        &self,
        tool: &str,
        args: &serde_json::Value,
    ) -> Result<ToolCallResult, McpError> {
        let tools = self.tools.lock().unwrap();
        let (_, handler) = tools
            .get(tool)
            .ok_or_else(|| McpError::UnknownTool(tool.to_string()))?;
        handler(args)
    }
}

/// MCP-клиент: устанавливает соединение с сервером и запрашивает список
/// инструментов. Эквивалент `initialize` + `tools/list` в реальном MCP.
pub struct McpClient {
    server: Option<Arc<dyn McpServer>>,
}

impl McpClient {
    pub fn new() -> Self {
        Self { server: None }
    }

    /// Устанавливает "соединение" с MCP-сервером (in-process).
    pub fn connect(&mut self, server: Arc<dyn McpServer>) {
        self.server = Some(server);
    }

    pub fn is_connected(&self) -> bool {
        self.server.is_some()
    }

    pub fn server_name(&self) -> Option<&str> {
        self.server.as_ref().map(|s| s.name())
    }

    pub fn list_tools(&self) -> Result<Vec<McpToolDescriptor>, McpError> {
        let server = self.server.as_ref().ok_or(McpError::NotConnected)?;
        Ok(server.list_tools())
    }

    pub fn call_tool(
        &self,
        tool: &str,
        args: &serde_json::Value,
    ) -> Result<ToolCallResult, McpError> {
        let server = self.server.as_ref().ok_or(McpError::NotConnected)?;
        server.call_tool(tool, args)
    }
}

impl Default for McpClient {
    fn default() -> Self {
        Self::new()
    }
}

/// Демо-сервер с двумя инструментами — удобно для быстрой проверки
/// "соединился → получил список".
pub fn build_demo_server() -> Arc<InProcessMcpServer> {
    let server = Arc::new(InProcessMcpServer::new("demo-server"));

    server.register(
        McpToolDescriptor::new("echo", "Возвращает переданный текст обратно")
            .with_param("text", "string", "текст для эха", true),
        |args| {
            let text = args
                .get("text")
                .and_then(|v| v.as_str())
                .ok_or_else(|| McpError::InvalidArguments("missing 'text'".into()))?;
            Ok(ToolCallResult {
                content: text.to_string(),
                is_error: false,
            })
        },
    );

    server.register(
        McpToolDescriptor::new("add", "Складывает два числа")
            .with_param("a", "number", "первое число", true)
            .with_param("b", "number", "второе число", true),
        |args| {
            let a = args
                .get("a")
                .and_then(|v| v.as_f64())
                .ok_or_else(|| McpError::InvalidArguments("missing 'a'".into()))?;
            let b = args
                .get("b")
                .and_then(|v| v.as_f64())
                .ok_or_else(|| McpError::InvalidArguments("missing 'b'".into()))?;
            Ok(ToolCallResult {
                content: format!("{}", a + b),
                is_error: false,
            })
        },
    );

    server
}

/// Человекочитаемый листинг инструментов — полезно для CLI-проверки.
pub fn format_tool_list(tools: &[McpToolDescriptor]) -> String {
    let mut out = String::new();
    out.push_str(&format!("Доступно инструментов: {}\n", tools.len()));
    for (i, t) in tools.iter().enumerate() {
        out.push_str(&format!("{}. {} — {}\n", i + 1, t.name, t.description));
        if !t.input_schema.properties.is_empty() {
            out.push_str("   параметры:\n");
            for (name, prop) in &t.input_schema.properties {
                let req = if t.input_schema.required.contains(name) {
                    " (required)"
                } else {
                    ""
                };
                out.push_str(&format!("   - {name}: {} — {}{req}\n", prop.kind, prop.description));
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn client_rejects_calls_before_connect() {
        let client = McpClient::new();
        assert!(!client.is_connected());
        assert!(matches!(client.list_tools(), Err(McpError::NotConnected)));
    }

    #[test]
    fn client_lists_tools_after_connect() {
        let mut client = McpClient::new();
        client.connect(build_demo_server());
        assert!(client.is_connected());
        let tools = client.list_tools().unwrap();
        assert_eq!(tools.len(), 2);
        let names: Vec<_> = tools.iter().map(|t| t.name.as_str()).collect();
        assert!(names.contains(&"echo"));
        assert!(names.contains(&"add"));
    }

    #[test]
    fn echo_tool_returns_text() {
        let mut client = McpClient::new();
        client.connect(build_demo_server());
        let result = client
            .call_tool("echo", &json!({ "text": "hello" }))
            .unwrap();
        assert_eq!(result.content, "hello");
        assert!(!result.is_error);
    }

    #[test]
    fn add_tool_sums_numbers() {
        let mut client = McpClient::new();
        client.connect(build_demo_server());
        let result = client
            .call_tool("add", &json!({ "a": 2.0, "b": 3.5 }))
            .unwrap();
        assert_eq!(result.content, "5.5");
    }

    #[test]
    fn unknown_tool_errors() {
        let mut client = McpClient::new();
        client.connect(build_demo_server());
        assert!(matches!(
            client.call_tool("nope", &json!({})),
            Err(McpError::UnknownTool(_))
        ));
    }

    #[test]
    fn format_list_contains_names() {
        let mut client = McpClient::new();
        client.connect(build_demo_server());
        let out = format_tool_list(&client.list_tools().unwrap());
        assert!(out.contains("echo"));
        assert!(out.contains("add"));
    }
}
