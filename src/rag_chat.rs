//! Module 25 — Мини-чат с RAG + памятью (production-like).
//!
//! Чат:
//! - хранит историю диалога
//! - при каждом новом запросе ищет контекст в индексе (RAG)
//! - формирует ответ с обязательными источниками
//! - ведёт "task state": уточнения, ограничения, цель диалога
//!
//! Поведение отлаживается через абстракцию `LlmClient`, что позволяет
//! гонять длинные сценарии (10–15 реплик) без внешнего API.

use crate::rag_citations::{
    CitationConfig, CitedAnswer, SourceRef, answer_with_citations,
};
use crate::rag_indexing::{Embedder, RagIndex};
use crate::rag_query::LlmClient;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

/// Роль участника чата.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum ChatRole {
    User,
    Assistant,
    System,
}

/// Сообщение в истории чата.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: ChatRole,
    pub content: String,
    pub sources: Vec<SourceRef>,
}

impl ChatMessage {
    pub fn user(text: &str) -> Self {
        Self {
            role: ChatRole::User,
            content: text.to_string(),
            sources: vec![],
        }
    }

    pub fn assistant(text: &str, sources: Vec<SourceRef>) -> Self {
        Self {
            role: ChatRole::Assistant,
            content: text.to_string(),
            sources,
        }
    }
}

/// Память задачи: цель, зафиксированные ограничения и уточнения.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TaskState {
    pub goal: Option<String>,
    pub constraints: Vec<String>,
    pub clarifications: Vec<String>,
    pub terms: Vec<String>,
}

impl TaskState {
    pub fn set_goal(&mut self, goal: &str) {
        self.goal = Some(goal.trim().to_string());
    }

    pub fn add_constraint(&mut self, c: &str) {
        let c = c.trim().to_string();
        if !c.is_empty() && !self.constraints.contains(&c) {
            self.constraints.push(c);
        }
    }

    pub fn add_clarification(&mut self, c: &str) {
        let c = c.trim().to_string();
        if !c.is_empty() && !self.clarifications.contains(&c) {
            self.clarifications.push(c);
        }
    }

    pub fn add_term(&mut self, t: &str) {
        let t = t.trim().to_string();
        if !t.is_empty() && !self.terms.contains(&t) {
            self.terms.push(t);
        }
    }

    pub fn render(&self) -> String {
        let mut parts = Vec::new();
        if let Some(g) = &self.goal {
            parts.push(format!("Цель: {g}"));
        }
        if !self.constraints.is_empty() {
            parts.push(format!("Ограничения: {}", self.constraints.join("; ")));
        }
        if !self.clarifications.is_empty() {
            parts.push(format!("Уточнения: {}", self.clarifications.join("; ")));
        }
        if !self.terms.is_empty() {
            parts.push(format!("Термины: {}", self.terms.join(", ")));
        }
        parts.join("\n")
    }

    pub fn is_empty(&self) -> bool {
        self.goal.is_none()
            && self.constraints.is_empty()
            && self.clarifications.is_empty()
            && self.terms.is_empty()
    }
}

/// Эвристический экстрактор состояния из реплики пользователя. В реальном
/// production-боте это делает LLM-classifier; здесь — простые паттерны,
/// достаточные для сценарного тестирования.
pub fn extract_state_updates(state: &mut TaskState, user_message: &str) {
    let text = user_message.trim();
    let lower = text.to_lowercase();

    for marker in ["моя цель", "цель:", "хочу"] {
        if let Some(pos) = lower.find(marker) {
            let after = &text[pos + marker.len()..];
            let goal = after.trim_start_matches([':', ' ']).trim().to_string();
            if !goal.is_empty() {
                state.set_goal(&goal);
                break;
            }
        }
    }
    for marker in ["ограничение:", "нельзя", "только "] {
        if let Some(pos) = lower.find(marker) {
            let after = &text[pos + marker.len()..];
            state.add_constraint(after.trim_start_matches([':', ' ']).trim());
        }
    }
    for marker in ["уточнение:", "уточняю:"] {
        if let Some(pos) = lower.find(marker) {
            let after = &text[pos + marker.len()..];
            state.add_clarification(after.trim_start_matches([':', ' ']).trim());
        }
    }
    for marker in ["термин:"] {
        if let Some(pos) = lower.find(marker) {
            let after = &text[pos + marker.len()..];
            state.add_term(after.trim_start_matches([':', ' ']).trim());
        }
    }
}

/// Мини-чат с RAG.
pub struct RagChat<E: Embedder, L: LlmClient> {
    index: RagIndex,
    embedder: E,
    llm: L,
    history: Vec<ChatMessage>,
    pub state: TaskState,
    cfg: CitationConfig,
    max_history: usize,
}

impl<E: Embedder, L: LlmClient> RagChat<E, L> {
    pub fn new(index: RagIndex, embedder: E, llm: L) -> Self {
        Self {
            index,
            embedder,
            llm,
            history: Vec::new(),
            state: TaskState::default(),
            cfg: CitationConfig::default(),
            max_history: 20,
        }
    }

    pub fn with_config(mut self, cfg: CitationConfig) -> Self {
        self.cfg = cfg;
        self
    }

    pub fn with_max_history(mut self, n: usize) -> Self {
        self.max_history = n;
        self
    }

    pub fn history(&self) -> &[ChatMessage] {
        &self.history
    }

    pub fn reset(&mut self) {
        self.history.clear();
        self.state = TaskState::default();
    }

    fn enriched_query(&self, user_message: &str) -> String {
        let mut parts = vec![user_message.to_string()];
        if let Some(g) = &self.state.goal {
            parts.push(format!("(цель: {g})"));
        }
        if !self.state.terms.is_empty() {
            parts.push(format!("(термины: {})", self.state.terms.join(", ")));
        }
        parts.join(" ")
    }

    /// Отправляет реплику, получает ответ. Обновляет историю и task state.
    pub fn send(&mut self, user_message: &str) -> Result<CitedAnswer, Box<dyn std::error::Error>> {
        extract_state_updates(&mut self.state, user_message);
        self.history.push(ChatMessage::user(user_message));

        let retrieval_query = self.enriched_query(user_message);
        let answer = answer_with_citations(
            &self.index,
            &self.embedder,
            &self.llm,
            &retrieval_query,
            &self.cfg,
        )?;

        self.history
            .push(ChatMessage::assistant(&answer.answer, answer.sources.clone()));

        if self.history.len() > self.max_history {
            let drop = self.history.len() - self.max_history;
            self.history.drain(0..drop);
        }

        Ok(answer)
    }

    /// Проверка: все ответы ассистента имеют источники (кроме "не знаю").
    pub fn all_answers_cited(&self) -> bool {
        self.history.iter().all(|m| match m.role {
            ChatRole::Assistant => {
                !m.sources.is_empty()
                    || m.content.to_lowercase().contains("не знаю")
            }
            _ => true,
        })
    }

    /// Проверка: ассистент не теряет цель (цель фиксируется и не стирается).
    pub fn goal_preserved(&self) -> bool {
        self.state.goal.is_some()
    }

    /// Сколько уникальных источников использовано за весь диалог.
    pub fn unique_sources(&self) -> usize {
        let mut set: HashSet<String> = HashSet::new();
        for m in &self.history {
            for s in &m.sources {
                set.insert(format!("{}#{}", s.source, s.chunk_id));
            }
        }
        set.len()
    }
}

/// Запускает сценарий — серию пользовательских реплик — и возвращает
/// финальное состояние чата.
pub fn run_scenario<E: Embedder, L: LlmClient>(
    chat: &mut RagChat<E, L>,
    messages: &[&str],
) -> Vec<CitedAnswer> {
    let mut out = Vec::new();
    for m in messages {
        if let Ok(a) = chat.send(m) {
            out.push(a);
        }
    }
    out
}

/// Сценарий 1: обучение RAG (13 реплик).
pub const SCENARIO_RAG_LEARNING: &[&str] = &[
    "Моя цель: разобраться как работает RAG.",
    "Что такое RAG?",
    "Из чего собирается retrieval-корпус?",
    "Как выбираются top-k чанки?",
    "Уточнение: мне важен именно текстовый корпус.",
    "Зачем нужны цитаты в ответе?",
    "Термин: ownership",
    "А при чём тут Rust? Объясни про ownership.",
    "Что про concurrency пишут в документации Rust?",
    "Ограничение: ответ должен опираться на документы.",
    "Вернёмся к RAG — как уменьшить галлюцинации?",
    "Какие атрибуты имеют инструменты MCP?",
    "Суммируй: какая у меня цель?",
];

/// Сценарий 2: сравнение MCP и RAG (12 реплик).
pub const SCENARIO_MCP_VS_RAG: &[&str] = &[
    "Моя цель: понять разницу между MCP и RAG.",
    "Что стандартизирует MCP?",
    "Как MCP-сервер отдаёт инструменты клиенту?",
    "Какие у инструмента MCP атрибуты?",
    "А что такое RAG в этом контексте?",
    "Термин: chunking",
    "Как устроен chunking?",
    "Уточнение: меня интересует именно retrieval, не chat.",
    "Зачем нужны цитаты в RAG-ответе?",
    "Ограничение: опирайся только на документы.",
    "Может ли MCP вызывать RAG-инструмент?",
    "Напомни мою цель и зафиксированные ограничения.",
];

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rag_citations::QuoteComposerLlm;
    use crate::rag_query::default_index;

    fn make_chat() -> RagChat<crate::rag_indexing::HashEmbedder, QuoteComposerLlm> {
        let (index, embedder) = default_index();
        RagChat::new(index, embedder, QuoteComposerLlm)
    }

    #[test]
    fn state_extracts_goal_and_constraint() {
        let mut state = TaskState::default();
        extract_state_updates(&mut state, "Моя цель: разобраться с RAG");
        extract_state_updates(&mut state, "Ограничение: только документация");
        extract_state_updates(&mut state, "Термин: chunking");
        assert!(state.goal.as_deref().unwrap().contains("разобраться"));
        assert_eq!(state.constraints.len(), 1);
        assert_eq!(state.terms, vec!["chunking".to_string()]);
    }

    #[test]
    fn single_message_produces_cited_answer() {
        let mut chat = make_chat();
        let a = chat.send("Что такое RAG?").unwrap();
        assert!(!a.sources.is_empty() || a.i_dont_know);
        assert_eq!(chat.history().len(), 2);
    }

    #[test]
    fn max_history_truncates_old_messages() {
        let mut chat = make_chat().with_max_history(4);
        for q in [
            "привет",
            "Что такое RAG?",
            "Что такое MCP?",
            "Что такое chunking?",
            "Что такое ownership?",
        ] {
            chat.send(q).unwrap();
        }
        assert!(chat.history().len() <= 4);
    }

    #[test]
    fn scenario_one_preserves_goal() {
        let mut chat = make_chat().with_max_history(100);
        let answers = run_scenario(&mut chat, SCENARIO_RAG_LEARNING);
        assert_eq!(answers.len(), SCENARIO_RAG_LEARNING.len());
        assert!(chat.goal_preserved());
        assert!(chat.state.goal.as_deref().unwrap().contains("RAG"));
    }

    #[test]
    fn scenario_two_uses_multiple_sources() {
        let mut chat = make_chat().with_max_history(100);
        run_scenario(&mut chat, SCENARIO_MCP_VS_RAG);
        assert!(chat.unique_sources() >= 2);
        assert!(chat.goal_preserved());
    }

    #[test]
    fn all_non_idk_answers_cite_sources() {
        let mut chat = make_chat().with_max_history(100);
        run_scenario(&mut chat, SCENARIO_RAG_LEARNING);
        assert!(chat.all_answers_cited());
    }

    #[test]
    fn reset_clears_history_and_state() {
        let mut chat = make_chat();
        chat.send("Моя цель: тест").unwrap();
        chat.reset();
        assert!(chat.history().is_empty());
        assert!(chat.state.is_empty());
    }

    #[test]
    fn task_state_renders_human_readable() {
        let mut st = TaskState::default();
        st.set_goal("разобраться с RAG");
        st.add_constraint("только документация");
        st.add_term("chunking");
        let rendered = st.render();
        assert!(rendered.contains("Цель:"));
        assert!(rendered.contains("Ограничения:"));
        assert!(rendered.contains("Термины:"));
    }
}
