//! Assistant memory model module.
//!
//! Implements three distinct memory layers:
//! - Short-term: current dialog messages (ephemeral, cleared between sessions)
//! - Working: current task data (plans, decisions, intermediate results)
//! - Long-term: persistent knowledge (user profile, past decisions, domain knowledge)
//!
//! Each layer is stored separately with explicit save/retrieve operations.

use std::collections::HashMap;

/// A single memory entry with metadata.
#[derive(Debug, Clone)]
pub struct MemoryEntry {
    pub key: String,
    pub value: String,
    pub timestamp: u64,
}

impl MemoryEntry {
    pub fn new(key: &str, value: &str, timestamp: u64) -> Self {
        Self {
            key: key.to_string(),
            value: value.to_string(),
            timestamp,
        }
    }
}

/// A message in the short-term dialog memory.
#[derive(Debug, Clone)]
pub struct DialogMessage {
    pub role: String,
    pub content: String,
}

impl DialogMessage {
    pub fn user(content: &str) -> Self {
        Self {
            role: "user".to_string(),
            content: content.to_string(),
        }
    }

    pub fn assistant(content: &str) -> Self {
        Self {
            role: "assistant".to_string(),
            content: content.to_string(),
        }
    }

    pub fn system(content: &str) -> Self {
        Self {
            role: "system".to_string(),
            content: content.to_string(),
        }
    }
}

/// Short-term memory: current dialog messages.
#[derive(Debug, Default)]
pub struct ShortTermMemory {
    messages: Vec<DialogMessage>,
}

impl ShortTermMemory {
    pub fn new() -> Self {
        Self {
            messages: Vec::new(),
        }
    }

    pub fn add(&mut self, message: DialogMessage) {
        self.messages.push(message);
    }

    pub fn messages(&self) -> &[DialogMessage] {
        &self.messages
    }

    pub fn len(&self) -> usize {
        self.messages.len()
    }

    pub fn is_empty(&self) -> bool {
        self.messages.is_empty()
    }

    pub fn clear(&mut self) {
        self.messages.clear();
    }

    pub fn last_n(&self, n: usize) -> &[DialogMessage] {
        let start = self.messages.len().saturating_sub(n);
        &self.messages[start..]
    }
}

/// Working memory: data relevant to the current task.
#[derive(Debug, Default)]
pub struct WorkingMemory {
    entries: HashMap<String, MemoryEntry>,
}

impl WorkingMemory {
    pub fn new() -> Self {
        Self {
            entries: HashMap::new(),
        }
    }

    pub fn set(&mut self, key: &str, value: &str, timestamp: u64) {
        self.entries
            .insert(key.to_string(), MemoryEntry::new(key, value, timestamp));
    }

    pub fn get(&self, key: &str) -> Option<&MemoryEntry> {
        self.entries.get(key)
    }

    pub fn remove(&mut self, key: &str) -> bool {
        self.entries.remove(key).is_some()
    }

    pub fn all(&self) -> Vec<&MemoryEntry> {
        self.entries.values().collect()
    }

    pub fn keys(&self) -> Vec<&String> {
        self.entries.keys().collect()
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn clear(&mut self) {
        self.entries.clear();
    }
}

/// Long-term memory: persistent knowledge across sessions.
#[derive(Debug, Default)]
pub struct LongTermMemory {
    entries: HashMap<String, MemoryEntry>,
}

impl LongTermMemory {
    pub fn new() -> Self {
        Self {
            entries: HashMap::new(),
        }
    }

    pub fn store(&mut self, key: &str, value: &str, timestamp: u64) {
        self.entries
            .insert(key.to_string(), MemoryEntry::new(key, value, timestamp));
    }

    pub fn recall(&self, key: &str) -> Option<&MemoryEntry> {
        self.entries.get(key)
    }

    pub fn remove(&mut self, key: &str) -> bool {
        self.entries.remove(key).is_some()
    }

    pub fn search(&self, query: &str) -> Vec<&MemoryEntry> {
        let query_lower = query.to_lowercase();
        self.entries
            .values()
            .filter(|e| {
                e.key.to_lowercase().contains(&query_lower)
                    || e.value.to_lowercase().contains(&query_lower)
            })
            .collect()
    }

    pub fn all(&self) -> Vec<&MemoryEntry> {
        self.entries.values().collect()
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

/// The unified memory model combining all three layers.
#[derive(Debug)]
pub struct AssistantMemory {
    pub short_term: ShortTermMemory,
    pub working: WorkingMemory,
    pub long_term: LongTermMemory,
}

impl AssistantMemory {
    pub fn new() -> Self {
        Self {
            short_term: ShortTermMemory::new(),
            working: WorkingMemory::new(),
            long_term: LongTermMemory::new(),
        }
    }

    /// Promotes a working memory entry to long-term memory.
    pub fn promote_to_long_term(&mut self, key: &str) -> bool {
        if let Some(entry) = self.working.get(key) {
            self.long_term
                .store(&entry.key, &entry.value, entry.timestamp);
            true
        } else {
            false
        }
    }

    /// Builds a context block for the assistant prompt with all memory layers.
    pub fn context_prompt(&self, recent_messages: usize) -> String {
        let mut sections = Vec::new();

        // Long-term memory
        if !self.long_term.is_empty() {
            let mut ltm = vec!["[Long-term memory]".to_string()];
            for entry in self.long_term.all() {
                ltm.push(format!("  {}: {}", entry.key, entry.value));
            }
            sections.push(ltm.join("\n"));
        }

        // Working memory
        if !self.working.is_empty() {
            let mut wm = vec!["[Working memory (current task)]".to_string()];
            for entry in self.working.all() {
                wm.push(format!("  {}: {}", entry.key, entry.value));
            }
            sections.push(wm.join("\n"));
        }

        // Short-term: last N messages
        let recent = self.short_term.last_n(recent_messages);
        if !recent.is_empty() {
            let mut stm = vec!["[Recent dialog]".to_string()];
            for msg in recent {
                stm.push(format!("  {}: {}", msg.role, msg.content));
            }
            sections.push(stm.join("\n"));
        }

        if sections.is_empty() {
            "No memory data available.".to_string()
        } else {
            sections.join("\n\n")
        }
    }
}

impl Default for AssistantMemory {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_short_term_memory() {
        let mut stm = ShortTermMemory::new();
        assert!(stm.is_empty());

        stm.add(DialogMessage::user("Hello"));
        stm.add(DialogMessage::assistant("Hi there!"));
        assert_eq!(stm.len(), 2);

        let last = stm.last_n(1);
        assert_eq!(last.len(), 1);
        assert_eq!(last[0].role, "assistant");

        stm.clear();
        assert!(stm.is_empty());
    }

    #[test]
    fn test_working_memory() {
        let mut wm = WorkingMemory::new();
        wm.set("goal", "Build a REST API", 1000);
        wm.set("constraint", "Use PostgreSQL", 1001);

        assert_eq!(wm.len(), 2);
        assert_eq!(wm.get("goal").unwrap().value, "Build a REST API");

        assert!(wm.remove("goal"));
        assert_eq!(wm.len(), 1);
        assert!(!wm.remove("nonexistent"));
    }

    #[test]
    fn test_long_term_memory() {
        let mut ltm = LongTermMemory::new();
        ltm.store("user_preference", "Prefers concise answers", 500);
        ltm.store("tech_decision", "Using Rust for backend", 600);

        assert_eq!(ltm.len(), 2);
        assert_eq!(
            ltm.recall("user_preference").unwrap().value,
            "Prefers concise answers"
        );

        let results = ltm.search("rust");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].key, "tech_decision");
    }

    #[test]
    fn test_promote_to_long_term() {
        let mut memory = AssistantMemory::new();
        memory.working.set("decision", "Use microservices", 1000);

        assert!(memory.promote_to_long_term("decision"));
        assert!(memory.long_term.recall("decision").is_some());
        assert_eq!(
            memory.long_term.recall("decision").unwrap().value,
            "Use microservices"
        );
    }

    #[test]
    fn test_promote_nonexistent_fails() {
        let mut memory = AssistantMemory::new();
        assert!(!memory.promote_to_long_term("nonexistent"));
    }

    #[test]
    fn test_context_prompt_all_layers() {
        let mut memory = AssistantMemory::new();
        memory
            .long_term
            .store("profile", "Senior developer", 100);
        memory
            .working
            .set("current_task", "Implement auth module", 200);
        memory
            .short_term
            .add(DialogMessage::user("How should we handle auth?"));
        memory
            .short_term
            .add(DialogMessage::assistant("I suggest JWT tokens."));

        let prompt = memory.context_prompt(5);
        assert!(prompt.contains("Long-term memory"));
        assert!(prompt.contains("Senior developer"));
        assert!(prompt.contains("Working memory"));
        assert!(prompt.contains("auth module"));
        assert!(prompt.contains("Recent dialog"));
        assert!(prompt.contains("JWT tokens"));
    }

    #[test]
    fn test_context_prompt_empty() {
        let memory = AssistantMemory::new();
        assert_eq!(memory.context_prompt(5), "No memory data available.");
    }

    #[test]
    fn test_last_n_more_than_available() {
        let mut stm = ShortTermMemory::new();
        stm.add(DialogMessage::user("Only one"));
        let last = stm.last_n(10);
        assert_eq!(last.len(), 1);
    }

    #[test]
    fn test_working_memory_overwrite() {
        let mut wm = WorkingMemory::new();
        wm.set("key", "value1", 100);
        wm.set("key", "value2", 200);
        assert_eq!(wm.len(), 1);
        assert_eq!(wm.get("key").unwrap().value, "value2");
        assert_eq!(wm.get("key").unwrap().timestamp, 200);
    }

    #[test]
    fn test_long_term_search_case_insensitive() {
        let mut ltm = LongTermMemory::new();
        ltm.store("UPPERCASE_KEY", "Some Value", 100);
        let results = ltm.search("uppercase");
        assert_eq!(results.len(), 1);
    }
}
