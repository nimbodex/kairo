//! Module 20 — Оркестрация нескольких MCP-серверов.
//!
//! Registry нескольких MCP-серверов: tracker, scheduler, pipeline.
//! Оркестратор:
//! - выбирает нужный сервер по namespace (`server::tool`) либо по имени
//!   инструмента (если оно уникально);
//! - маршрутизирует вызовы;
//! - выполняет длинный сценарий через инструменты с разных серверов.
//!
//! Пример сценария: "найти документы → засуммаризировать → сохранить →
//! завести issue с результатом → расписать периодическое напоминание".

use crate::mcp_composition::{build_pipeline_server, sample_corpus};
use crate::mcp_connection::{McpError, McpServer, McpToolDescriptor, ToolCallResult};
use crate::mcp_scheduler::{Scheduler, build_scheduler_server};
use crate::mcp_tool::{TrackerApi, build_tracker_server};
use serde_json::{Value, json};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

/// Запись в реестре серверов.
pub struct ServerEntry {
    pub name: String,
    pub server: Arc<dyn McpServer>,
    pub tool_names: Vec<String>,
}

/// Агрегирующий реестр нескольких MCP-серверов.
pub struct McpRegistry {
    entries: Vec<ServerEntry>,
}

impl McpRegistry {
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    pub fn register(&mut self, server: Arc<dyn McpServer>) {
        let name = server.name().to_string();
        let tool_names = server.list_tools().into_iter().map(|t| t.name).collect();
        self.entries.push(ServerEntry {
            name,
            server,
            tool_names,
        });
    }

    pub fn servers(&self) -> Vec<&str> {
        self.entries.iter().map(|e| e.name.as_str()).collect()
    }

    pub fn all_tools(&self) -> Vec<(String, McpToolDescriptor)> {
        let mut out = Vec::new();
        for entry in &self.entries {
            for t in entry.server.list_tools() {
                out.push((entry.name.clone(), t));
            }
        }
        out
    }

    /// Выбирает сервер по `server::tool` либо по уникальному имени
    /// инструмента. Возвращает кортеж `(server, tool_name)`.
    pub fn route<'a>(
        &'a self,
        target: &str,
    ) -> Result<(&'a Arc<dyn McpServer>, String), McpError> {
        if let Some((srv, tool)) = target.split_once("::") {
            let entry = self
                .entries
                .iter()
                .find(|e| e.name == srv)
                .ok_or_else(|| McpError::UnknownTool(format!("server '{srv}' not registered")))?;
            if !entry.tool_names.contains(&tool.to_string()) {
                return Err(McpError::UnknownTool(format!(
                    "tool '{tool}' not in '{srv}'"
                )));
            }
            return Ok((&entry.server, tool.to_string()));
        }
        let candidates: Vec<&ServerEntry> = self
            .entries
            .iter()
            .filter(|e| e.tool_names.iter().any(|n| n == target))
            .collect();
        match candidates.as_slice() {
            [] => Err(McpError::UnknownTool(target.to_string())),
            [only] => Ok((&only.server, target.to_string())),
            _ => {
                let names: Vec<_> = candidates.iter().map(|e| e.name.as_str()).collect();
                Err(McpError::InvalidArguments(format!(
                    "tool '{target}' ambiguous: {}",
                    names.join(", ")
                )))
            }
        }
    }

    pub fn call(&self, target: &str, args: &Value) -> Result<ToolCallResult, McpError> {
        let (server, tool) = self.route(target)?;
        server.call_tool(&tool, args)
    }
}

impl Default for McpRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Запись шага оркестрации — удобно для отладки сценария.
#[derive(Debug, Clone)]
pub struct OrchestrationStep {
    pub target: String,
    pub args: Value,
    pub output: Value,
}

/// Высокоуровневый "оркестратор" поверх реестра. Хранит историю шагов.
pub struct Orchestrator {
    registry: McpRegistry,
    trace: Vec<OrchestrationStep>,
}

impl Orchestrator {
    pub fn new(registry: McpRegistry) -> Self {
        Self {
            registry,
            trace: Vec::new(),
        }
    }

    pub fn invoke(&mut self, target: &str, args: Value) -> Result<Value, McpError> {
        let result = self.registry.call(target, &args)?;
        if result.is_error {
            return Err(McpError::ExecutionFailed(result.content));
        }
        let parsed: Value = serde_json::from_str(&result.content).unwrap_or(Value::Null);
        self.trace.push(OrchestrationStep {
            target: target.to_string(),
            args,
            output: parsed.clone(),
        });
        Ok(parsed)
    }

    pub fn trace(&self) -> &[OrchestrationStep] {
        &self.trace
    }

    pub fn registry(&self) -> &McpRegistry {
        &self.registry
    }

    /// Длинный сценарий, использующий все три сервера:
    /// 1. pipeline::search — найти документы по теме
    /// 2. pipeline::summarize — собрать резюме
    /// 3. tracker::create_issue — завести тикет с резюме
    /// 4. scheduler::remind_every — расписать проверку тикета
    /// 5. scheduler::tick — продвинуть часы, чтобы задачи сработали
    /// 6. tracker::list_issues — подтвердить наличие issue
    pub fn research_and_schedule(
        &mut self,
        topic: &str,
        tick_now: u64,
    ) -> Result<Value, McpError> {
        let docs = self.invoke("pipeline::search", json!({ "query": topic }))?;
        let summary = self.invoke("pipeline::summarize", json!({ "documents": docs }))?;
        let summary_text = summary
            .get("summary")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();

        let issue = self.invoke(
            "tracker::create_issue",
            json!({
                "title": format!("Research: {topic}"),
                "description": summary_text,
            }),
        )?;

        self.invoke(
            "scheduler::remind_every",
            json!({ "first_at": 1, "interval": 5, "message": "review issue" }),
        )?;
        self.invoke("scheduler::tick", json!({ "now": tick_now }))?;

        let listed = self.invoke("tracker::list_issues", json!({}))?;
        Ok(json!({
            "issue": issue,
            "issues_total": listed.as_array().map(|a| a.len()).unwrap_or(0),
            "steps": self.trace.len(),
        }))
    }
}

/// Собирает типовой реестр из трёх серверов — удобно для демо/тестов.
pub fn build_demo_registry(save_dir: PathBuf) -> McpRegistry {
    let mut reg = McpRegistry::new();
    reg.register(build_tracker_server(Arc::new(TrackerApi::new())));
    reg.register(build_scheduler_server(Arc::new(Scheduler::new())));
    reg.register(build_pipeline_server(
        Arc::new(sample_corpus()),
        save_dir,
    ));
    reg
}

/// Вспомогательное: собирает человекочитаемый каталог "сервер → инструменты".
pub fn catalog_map(registry: &McpRegistry) -> HashMap<String, Vec<String>> {
    let mut out: HashMap<String, Vec<String>> = HashMap::new();
    for (server, tool) in registry.all_tools() {
        out.entry(server).or_default().push(tool.name);
    }
    out
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
    fn registry_lists_all_servers() {
        let reg = build_demo_registry(tmp_dir("kairo_orch_list"));
        let names = reg.servers();
        assert!(names.contains(&"tracker-server"));
        assert!(names.contains(&"scheduler-server"));
        assert!(names.contains(&"pipeline-server"));
    }

    #[test]
    fn route_by_namespace() {
        let reg = build_demo_registry(tmp_dir("kairo_orch_route"));
        let (_, tool) = reg.route("tracker-server::list_issues").unwrap();
        assert_eq!(tool, "list_issues");
    }

    #[test]
    fn route_by_unique_tool_name() {
        let reg = build_demo_registry(tmp_dir("kairo_orch_unique"));
        let (_, tool) = reg.route("create_issue").unwrap();
        assert_eq!(tool, "create_issue");
    }

    #[test]
    fn unknown_tool_errors() {
        let reg = build_demo_registry(tmp_dir("kairo_orch_unknown"));
        assert!(matches!(reg.route("nope"), Err(McpError::UnknownTool(_))));
    }

    #[test]
    fn unknown_server_errors() {
        let reg = build_demo_registry(tmp_dir("kairo_orch_nosrv"));
        assert!(matches!(
            reg.route("ghost::create_issue"),
            Err(McpError::UnknownTool(_))
        ));
    }

    #[test]
    fn long_scenario_uses_multiple_servers() {
        let dir = tmp_dir("kairo_orch_scenario");
        let reg = build_demo_registry(dir.clone());
        let mut orch = Orchestrator::new(reg);
        let result = orch.research_and_schedule("MCP", 11).unwrap();

        assert!(result.get("issue").is_some());
        assert_eq!(result.get("issues_total").unwrap().as_u64().unwrap(), 1);

        let servers_used: std::collections::HashSet<_> = orch
            .trace()
            .iter()
            .map(|s| s.target.split("::").next().unwrap_or(&s.target).to_string())
            .collect();
        assert!(servers_used.contains("pipeline-server"));
        assert!(servers_used.contains("tracker-server"));
        assert!(servers_used.contains("scheduler-server"));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn catalog_groups_tools_by_server() {
        let reg = build_demo_registry(tmp_dir("kairo_orch_catalog"));
        let cat = catalog_map(&reg);
        assert!(cat.get("tracker-server").unwrap().contains(&"get_issue".to_string()));
        assert!(cat.get("pipeline-server").unwrap().contains(&"search".to_string()));
    }
}
