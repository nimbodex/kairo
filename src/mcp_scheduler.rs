//! Module 18 — Планировщик и фоновые задачи.
//!
//! MCP-инструмент с отложенным/периодическим выполнением:
//! - `remind_at` — однократное напоминание в момент T
//! - `remind_every` — периодическое срабатывание с интервалом
//! - `summary` — агрегированный результат (сколько сработало, что запланировано)
//!
//! Состояние (задачи + лог срабатываний) сохраняется в JSON, поэтому агент
//! может "работать 24/7": перезагрузка не теряет расписание.

use crate::mcp_connection::{
    InProcessMcpServer, McpError, McpToolDescriptor, ToolCallResult,
};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

/// Тип задачи по расписанию.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum TaskKind {
    /// Один раз в момент `at_seconds` (unix-секунды условного времени).
    Once,
    /// Каждые `interval_seconds`, начиная с `at_seconds`.
    Periodic,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScheduledTask {
    pub id: u64,
    pub kind: TaskKind,
    pub message: String,
    /// Следующее время запуска (секунды логических часов).
    pub next_at: u64,
    /// Интервал для периодических задач.
    pub interval_seconds: u64,
    /// Сколько раз сработала.
    pub fire_count: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FireLogEntry {
    pub task_id: u64,
    pub fired_at: u64,
    pub message: String,
}

/// Сериализуемое состояние планировщика.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SchedulerState {
    pub next_id: u64,
    pub tasks: Vec<ScheduledTask>,
    pub log: Vec<FireLogEntry>,
}

/// Планировщик задач с JSON-персистентностью.
///
/// Часы у нас логические (не wall clock) — удобно для тестов и для
/// детерминированной агрегации. `tick(now)` сдвигает время вперёд и
/// запускает все созревшие задачи.
pub struct Scheduler {
    state: Mutex<SchedulerState>,
    path: Option<PathBuf>,
}

impl Scheduler {
    pub fn new() -> Self {
        Self {
            state: Mutex::new(SchedulerState {
                next_id: 1,
                ..Default::default()
            }),
            path: None,
        }
    }

    /// Создаёт планировщик с привязкой к файлу для персистентности.
    pub fn with_persistence(path: &Path) -> Self {
        let state = match std::fs::read_to_string(path) {
            Ok(s) => serde_json::from_str(&s).unwrap_or_else(|_| SchedulerState {
                next_id: 1,
                ..Default::default()
            }),
            Err(_) => SchedulerState {
                next_id: 1,
                ..Default::default()
            },
        };
        Self {
            state: Mutex::new(state),
            path: Some(path.to_path_buf()),
        }
    }

    fn save(&self, state: &SchedulerState) {
        if let Some(path) = &self.path {
            if let Ok(json) = serde_json::to_string_pretty(state) {
                let _ = std::fs::write(path, json);
            }
        }
    }

    pub fn schedule_once(&self, at: u64, message: &str) -> u64 {
        let mut state = self.state.lock().unwrap();
        let id = state.next_id;
        state.next_id += 1;
        state.tasks.push(ScheduledTask {
            id,
            kind: TaskKind::Once,
            message: message.to_string(),
            next_at: at,
            interval_seconds: 0,
            fire_count: 0,
        });
        self.save(&state);
        id
    }

    pub fn schedule_periodic(&self, first_at: u64, interval: u64, message: &str) -> u64 {
        let mut state = self.state.lock().unwrap();
        let id = state.next_id;
        state.next_id += 1;
        state.tasks.push(ScheduledTask {
            id,
            kind: TaskKind::Periodic,
            message: message.to_string(),
            next_at: first_at,
            interval_seconds: interval.max(1),
            fire_count: 0,
        });
        self.save(&state);
        id
    }

    pub fn cancel(&self, id: u64) -> bool {
        let mut state = self.state.lock().unwrap();
        let before = state.tasks.len();
        state.tasks.retain(|t| t.id != id);
        let removed = state.tasks.len() < before;
        if removed {
            self.save(&state);
        }
        removed
    }

    /// Сдвигает логическое время к `now`. Запускает все созревшие задачи,
    /// логирует каждое срабатывание, для периодических планирует следующий.
    /// Возвращает количество сработавших задач.
    pub fn tick(&self, now: u64) -> usize {
        let mut state = self.state.lock().unwrap();
        let mut fired = 0usize;

        loop {
            let Some(idx) = state
                .tasks
                .iter()
                .enumerate()
                .filter(|(_, t)| t.next_at <= now)
                .min_by_key(|(_, t)| t.next_at)
                .map(|(i, _)| i)
            else {
                break;
            };

            let (kind, next_at, interval, message, id) = {
                let t = &state.tasks[idx];
                (
                    t.kind.clone(),
                    t.next_at,
                    t.interval_seconds,
                    t.message.clone(),
                    t.id,
                )
            };
            state.log.push(FireLogEntry {
                task_id: id,
                fired_at: next_at,
                message: message.clone(),
            });
            fired += 1;

            match kind {
                TaskKind::Once => {
                    state.tasks.remove(idx);
                }
                TaskKind::Periodic => {
                    let t = &mut state.tasks[idx];
                    t.fire_count += 1;
                    t.next_at += interval;
                }
            }
        }

        // Корректируем fire_count для "Once"-задач — у них мы уже удалили
        // запись, но в логах сохранилась история.
        self.save(&state);
        fired
    }

    pub fn summary(&self) -> SchedulerSummary {
        let state = self.state.lock().unwrap();
        SchedulerSummary {
            active_tasks: state.tasks.len(),
            periodic_tasks: state
                .tasks
                .iter()
                .filter(|t| t.kind == TaskKind::Periodic)
                .count(),
            once_tasks: state
                .tasks
                .iter()
                .filter(|t| t.kind == TaskKind::Once)
                .count(),
            total_fires: state.log.len(),
            last_events: state.log.iter().rev().take(5).cloned().collect(),
        }
    }

    pub fn tasks(&self) -> Vec<ScheduledTask> {
        self.state.lock().unwrap().tasks.clone()
    }

    pub fn log(&self) -> Vec<FireLogEntry> {
        self.state.lock().unwrap().log.clone()
    }
}

impl Default for Scheduler {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchedulerSummary {
    pub active_tasks: usize,
    pub periodic_tasks: usize,
    pub once_tasks: usize,
    pub total_fires: usize,
    pub last_events: Vec<FireLogEntry>,
}

/// Собирает MCP-сервер с инструментами планировщика.
pub fn build_scheduler_server(scheduler: Arc<Scheduler>) -> Arc<InProcessMcpServer> {
    let server = Arc::new(InProcessMcpServer::new("scheduler-server"));

    let s = Arc::clone(&scheduler);
    server.register(
        McpToolDescriptor::new("remind_at", "Однократное напоминание в момент T")
            .with_param("at", "number", "unix-секунды логических часов", true)
            .with_param("message", "string", "текст напоминания", true),
        move |args| {
            let at = args
                .get("at")
                .and_then(|v| v.as_u64())
                .ok_or_else(|| McpError::InvalidArguments("missing 'at'".into()))?;
            let message = args
                .get("message")
                .and_then(|v| v.as_str())
                .ok_or_else(|| McpError::InvalidArguments("missing 'message'".into()))?;
            let id = s.schedule_once(at, message);
            Ok(ToolCallResult {
                content: json!({ "id": id }).to_string(),
                is_error: false,
            })
        },
    );

    let s = Arc::clone(&scheduler);
    server.register(
        McpToolDescriptor::new("remind_every", "Периодическое напоминание")
            .with_param("first_at", "number", "когда сработать впервые", true)
            .with_param("interval", "number", "интервал, секунды", true)
            .with_param("message", "string", "текст", true),
        move |args| {
            let first_at = args
                .get("first_at")
                .and_then(|v| v.as_u64())
                .ok_or_else(|| McpError::InvalidArguments("missing 'first_at'".into()))?;
            let interval = args
                .get("interval")
                .and_then(|v| v.as_u64())
                .ok_or_else(|| McpError::InvalidArguments("missing 'interval'".into()))?;
            let message = args
                .get("message")
                .and_then(|v| v.as_str())
                .ok_or_else(|| McpError::InvalidArguments("missing 'message'".into()))?;
            let id = s.schedule_periodic(first_at, interval, message);
            Ok(ToolCallResult {
                content: json!({ "id": id }).to_string(),
                is_error: false,
            })
        },
    );

    let s = Arc::clone(&scheduler);
    server.register(
        McpToolDescriptor::new("tick", "Сдвигает часы и запускает созревшие задачи")
            .with_param("now", "number", "текущее логическое время", true),
        move |args| {
            let now = args
                .get("now")
                .and_then(|v| v.as_u64())
                .ok_or_else(|| McpError::InvalidArguments("missing 'now'".into()))?;
            let fired = s.tick(now);
            Ok(ToolCallResult {
                content: json!({ "fired": fired }).to_string(),
                is_error: false,
            })
        },
    );

    let s = Arc::clone(&scheduler);
    server.register(
        McpToolDescriptor::new("summary", "Агрегированная сводка"),
        move |_args| {
            let summary = s.summary();
            Ok(ToolCallResult {
                content: serde_json::to_string(&summary).unwrap(),
                is_error: false,
            })
        },
    );

    server
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn once_task_fires_and_disappears() {
        let sched = Scheduler::new();
        sched.schedule_once(10, "ping");
        assert_eq!(sched.tick(5), 0);
        assert_eq!(sched.tick(10), 1);
        assert!(sched.tasks().is_empty());
        assert_eq!(sched.log().len(), 1);
    }

    #[test]
    fn periodic_task_fires_multiple_times() {
        let sched = Scheduler::new();
        sched.schedule_periodic(5, 3, "poll");
        // 5, 8, 11, 14 → tick(15) должен дать 4 срабатывания
        assert_eq!(sched.tick(15), 4);
        let task = &sched.tasks()[0];
        assert_eq!(task.fire_count, 4);
        assert_eq!(task.next_at, 17);
    }

    #[test]
    fn cancel_removes_task() {
        let sched = Scheduler::new();
        let id = sched.schedule_once(100, "ping");
        assert!(sched.cancel(id));
        assert!(sched.tasks().is_empty());
        assert!(!sched.cancel(id));
    }

    #[test]
    fn summary_counts_types() {
        let sched = Scheduler::new();
        sched.schedule_once(10, "a");
        sched.schedule_periodic(1, 1, "b");
        sched.tick(0);
        let summary = sched.summary();
        assert_eq!(summary.active_tasks, 2);
        assert_eq!(summary.once_tasks, 1);
        assert_eq!(summary.periodic_tasks, 1);
    }

    #[test]
    fn persistence_roundtrip() {
        let tmp = std::env::temp_dir().join("kairo_scheduler_test.json");
        let _ = std::fs::remove_file(&tmp);
        {
            let sched = Scheduler::with_persistence(&tmp);
            sched.schedule_once(10, "ping");
            sched.schedule_periodic(5, 2, "poll");
        }
        let sched2 = Scheduler::with_persistence(&tmp);
        assert_eq!(sched2.tasks().len(), 2);
        let _ = std::fs::remove_file(&tmp);
    }

    #[test]
    fn mcp_tools_round_trip() {
        let sched = Arc::new(Scheduler::new());
        let server = build_scheduler_server(Arc::clone(&sched));
        let mut client = crate::mcp_connection::McpClient::new();
        client.connect(server);

        client
            .call_tool("remind_at", &json!({ "at": 5, "message": "hi" }))
            .unwrap();
        client
            .call_tool(
                "remind_every",
                &json!({ "first_at": 2, "interval": 3, "message": "poll" }),
            )
            .unwrap();

        let fired = client.call_tool("tick", &json!({ "now": 10 })).unwrap();
        // 1 once + 3 periodic (2, 5, 8) = 4
        assert!(fired.content.contains("\"fired\":4"));

        let summary = client.call_tool("summary", &json!({})).unwrap();
        assert!(summary.content.contains("\"total_fires\":4"));
    }
}
