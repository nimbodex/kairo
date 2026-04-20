//! Module 17 — Первый MCP-инструмент.
//!
//! MCP-сервер вокруг mock-API issue-трекера. Содержит:
//! - регистрацию инструментов (create_issue, get_issue, list_issues)
//! - описание входных параметров
//! - возврат результата
//! - агент, который делает вызов и использует результат
//!
//! Цель: показать полный путь "агент → MCP tool call → API → результат".

use crate::mcp_connection::{
    InProcessMcpServer, McpClient, McpError, McpToolDescriptor, ToolCallResult,
};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::sync::{Arc, Mutex};

/// Запись issue-трекера.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Issue {
    pub id: u64,
    pub title: String,
    pub description: String,
    pub status: String,
}

/// Минимальный in-memory API трекера; потокобезопасный.
#[derive(Debug, Default)]
pub struct TrackerApi {
    next_id: Mutex<u64>,
    issues: Mutex<Vec<Issue>>,
}

impl TrackerApi {
    pub fn new() -> Self {
        Self {
            next_id: Mutex::new(1),
            issues: Mutex::new(Vec::new()),
        }
    }

    pub fn create(&self, title: &str, description: &str) -> Issue {
        let mut next_id = self.next_id.lock().unwrap();
        let id = *next_id;
        *next_id += 1;
        let issue = Issue {
            id,
            title: title.to_string(),
            description: description.to_string(),
            status: "open".to_string(),
        };
        self.issues.lock().unwrap().push(issue.clone());
        issue
    }

    pub fn get(&self, id: u64) -> Option<Issue> {
        self.issues
            .lock()
            .unwrap()
            .iter()
            .find(|i| i.id == id)
            .cloned()
    }

    pub fn list(&self) -> Vec<Issue> {
        self.issues.lock().unwrap().clone()
    }
}

/// Собирает готовый MCP-сервер поверх `TrackerApi`, регистрируя 3 инструмента.
pub fn build_tracker_server(api: Arc<TrackerApi>) -> Arc<InProcessMcpServer> {
    let server = Arc::new(InProcessMcpServer::new("tracker-server"));

    // create_issue
    let api_c = Arc::clone(&api);
    server.register(
        McpToolDescriptor::new("create_issue", "Создаёт issue в трекере")
            .with_param("title", "string", "заголовок issue", true)
            .with_param("description", "string", "описание issue", true),
        move |args| {
            let title = args
                .get("title")
                .and_then(|v| v.as_str())
                .ok_or_else(|| McpError::InvalidArguments("missing 'title'".into()))?;
            let description = args
                .get("description")
                .and_then(|v| v.as_str())
                .ok_or_else(|| McpError::InvalidArguments("missing 'description'".into()))?;
            let issue = api_c.create(title, description);
            Ok(ToolCallResult {
                content: serde_json::to_string(&issue).unwrap(),
                is_error: false,
            })
        },
    );

    // get_issue
    let api_g = Arc::clone(&api);
    server.register(
        McpToolDescriptor::new("get_issue", "Возвращает issue по id")
            .with_param("id", "number", "идентификатор issue", true),
        move |args| {
            let id = args
                .get("id")
                .and_then(|v| v.as_u64())
                .ok_or_else(|| McpError::InvalidArguments("missing 'id'".into()))?;
            match api_g.get(id) {
                Some(issue) => Ok(ToolCallResult {
                    content: serde_json::to_string(&issue).unwrap(),
                    is_error: false,
                }),
                None => Ok(ToolCallResult {
                    content: format!("issue {id} not found"),
                    is_error: true,
                }),
            }
        },
    );

    // list_issues
    let api_l = Arc::clone(&api);
    server.register(
        McpToolDescriptor::new("list_issues", "Возвращает список всех issues"),
        move |_args| {
            let list = api_l.list();
            Ok(ToolCallResult {
                content: serde_json::to_string(&list).unwrap(),
                is_error: false,
            })
        },
    );

    server
}

/// Агент с прикреплённым MCP-клиентом: выполняет простую логику
/// "сформировать вызов → выполнить → распарсить".
pub struct ToolUsingAgent {
    client: McpClient,
}

impl ToolUsingAgent {
    pub fn new() -> Self {
        Self {
            client: McpClient::new(),
        }
    }

    pub fn attach(&mut self, server: Arc<dyn crate::mcp_connection::McpServer>) {
        self.client.connect(server);
    }

    pub fn create_issue(&self, title: &str, description: &str) -> Result<Issue, McpError> {
        let result = self.client.call_tool(
            "create_issue",
            &json!({ "title": title, "description": description }),
        )?;
        serde_json::from_str(&result.content)
            .map_err(|e| McpError::ExecutionFailed(format!("parse issue: {e}")))
    }

    pub fn get_issue(&self, id: u64) -> Result<Option<Issue>, McpError> {
        let result = self.client.call_tool("get_issue", &json!({ "id": id }))?;
        if result.is_error {
            return Ok(None);
        }
        let issue = serde_json::from_str(&result.content)
            .map_err(|e| McpError::ExecutionFailed(format!("parse issue: {e}")))?;
        Ok(Some(issue))
    }

    pub fn list_issues(&self) -> Result<Vec<Issue>, McpError> {
        let result = self.client.call_tool("list_issues", &json!({}))?;
        serde_json::from_str(&result.content)
            .map_err(|e| McpError::ExecutionFailed(format!("parse list: {e}")))
    }

    /// Сценарий использования: агент создаёт issue и читает его обратно.
    pub fn demo_scenario(&self) -> Result<String, McpError> {
        let created = self.create_issue("Исправить логин", "Форма не принимает email с плюсом")?;
        let fetched = self
            .get_issue(created.id)?
            .ok_or_else(|| McpError::ExecutionFailed("issue lost after create".into()))?;
        Ok(format!(
            "создано #{} ({}), прочитано обратно: статус={}",
            fetched.id, fetched.title, fetched.status
        ))
    }
}

impl Default for ToolUsingAgent {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tracker_api_roundtrip() {
        let api = TrackerApi::new();
        let issue = api.create("bug", "broken");
        assert_eq!(issue.id, 1);
        let fetched = api.get(1).unwrap();
        assert_eq!(fetched.title, "bug");
    }

    #[test]
    fn agent_can_create_and_fetch_issue() {
        let api = Arc::new(TrackerApi::new());
        let server = build_tracker_server(Arc::clone(&api));
        let mut agent = ToolUsingAgent::new();
        agent.attach(server);

        let created = agent.create_issue("bug", "broken").unwrap();
        assert_eq!(created.status, "open");

        let fetched = agent.get_issue(created.id).unwrap().unwrap();
        assert_eq!(fetched.id, created.id);
    }

    #[test]
    fn agent_lists_multiple_issues() {
        let api = Arc::new(TrackerApi::new());
        let server = build_tracker_server(Arc::clone(&api));
        let mut agent = ToolUsingAgent::new();
        agent.attach(server);

        agent.create_issue("a", "x").unwrap();
        agent.create_issue("b", "y").unwrap();
        let list = agent.list_issues().unwrap();
        assert_eq!(list.len(), 2);
    }

    #[test]
    fn get_missing_issue_returns_none() {
        let api = Arc::new(TrackerApi::new());
        let server = build_tracker_server(Arc::clone(&api));
        let mut agent = ToolUsingAgent::new();
        agent.attach(server);
        assert!(agent.get_issue(999).unwrap().is_none());
    }

    #[test]
    fn demo_scenario_runs_end_to_end() {
        let api = Arc::new(TrackerApi::new());
        let server = build_tracker_server(Arc::clone(&api));
        let mut agent = ToolUsingAgent::new();
        agent.attach(server);
        let summary = agent.demo_scenario().unwrap();
        assert!(summary.contains("статус=open"));
    }

    #[test]
    fn missing_title_is_invalid_args() {
        let api = Arc::new(TrackerApi::new());
        let server = build_tracker_server(api);
        let mut client = McpClient::new();
        client.connect(server);
        let err = client.call_tool("create_issue", &json!({})).unwrap_err();
        assert!(matches!(err, McpError::InvalidArguments(_)));
    }
}
