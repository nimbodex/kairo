//! Context management strategies module (without summary).
//!
//! Implements three strategies for managing conversation context:
//! 1. Sliding Window — keep only the last N messages
//! 2. Sticky Facts — key-value facts + last N messages
//! 3. Branching — checkpoint/branch/switch dialog branches
//!
//! All strategies implement a common `ContextStrategy` trait.

use std::collections::HashMap;

/// A chat message.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Message {
    pub role: String,
    pub content: String,
}

impl Message {
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

/// Statistics for comparing strategies.
#[derive(Debug, Clone)]
pub struct ContextStats {
    pub total_messages_stored: usize,
    pub messages_in_context: usize,
    pub estimated_tokens: usize,
}

/// Common trait for context strategies.
pub trait ContextStrategy {
    /// Adds a message to the history.
    fn add_message(&mut self, message: Message);

    /// Returns the messages that should be sent as context to the model.
    fn build_context(&self) -> Vec<Message>;

    /// Returns statistics about the current context.
    fn stats(&self) -> ContextStats;

    /// Clears all stored data.
    fn clear(&mut self);
}

// ---------------------------------------------------------------------------
// Strategy 1: Sliding Window
// ---------------------------------------------------------------------------

/// Keeps only the last N messages. Everything older is discarded.
#[derive(Debug)]
pub struct SlidingWindow {
    messages: Vec<Message>,
    window_size: usize,
}

impl SlidingWindow {
    pub fn new(window_size: usize) -> Self {
        Self {
            messages: Vec::new(),
            window_size,
        }
    }
}

impl ContextStrategy for SlidingWindow {
    fn add_message(&mut self, message: Message) {
        self.messages.push(message);
    }

    fn build_context(&self) -> Vec<Message> {
        let start = self.messages.len().saturating_sub(self.window_size);
        self.messages[start..].to_vec()
    }

    fn stats(&self) -> ContextStats {
        let context = self.build_context();
        let tokens: usize = context.iter().map(|m| m.content.len() / 4 + 1).sum();
        ContextStats {
            total_messages_stored: self.messages.len(),
            messages_in_context: context.len(),
            estimated_tokens: tokens,
        }
    }

    fn clear(&mut self) {
        self.messages.clear();
    }
}

// ---------------------------------------------------------------------------
// Strategy 2: Sticky Facts (Key-Value Memory)
// ---------------------------------------------------------------------------

/// Maintains a set of key-value "facts" extracted from the conversation,
/// plus the last N messages. Facts persist even when messages scroll off.
#[derive(Debug)]
pub struct StickyFacts {
    messages: Vec<Message>,
    window_size: usize,
    facts: HashMap<String, String>,
}

impl StickyFacts {
    pub fn new(window_size: usize) -> Self {
        Self {
            messages: Vec::new(),
            window_size,
            facts: HashMap::new(),
        }
    }

    /// Sets a fact (key-value pair).
    pub fn set_fact(&mut self, key: &str, value: &str) {
        self.facts.insert(key.to_string(), value.to_string());
    }

    /// Removes a fact.
    pub fn remove_fact(&mut self, key: &str) -> bool {
        self.facts.remove(key).is_some()
    }

    /// Returns all facts.
    pub fn facts(&self) -> &HashMap<String, String> {
        &self.facts
    }

    /// Builds the facts block as a system message.
    fn facts_message(&self) -> Option<Message> {
        if self.facts.is_empty() {
            return None;
        }
        let mut lines = vec!["Known facts:".to_string()];
        let mut keys: Vec<_> = self.facts.keys().collect();
        keys.sort();
        for key in keys {
            lines.push(format!("  {}: {}", key, self.facts[key]));
        }
        Some(Message::system(&lines.join("\n")))
    }
}

impl ContextStrategy for StickyFacts {
    fn add_message(&mut self, message: Message) {
        self.messages.push(message);
    }

    fn build_context(&self) -> Vec<Message> {
        let mut context = Vec::new();

        // Facts go first as a system message
        if let Some(facts_msg) = self.facts_message() {
            context.push(facts_msg);
        }

        // Then the last N messages
        let start = self.messages.len().saturating_sub(self.window_size);
        context.extend_from_slice(&self.messages[start..]);

        context
    }

    fn stats(&self) -> ContextStats {
        let context = self.build_context();
        let tokens: usize = context.iter().map(|m| m.content.len() / 4 + 1).sum();
        ContextStats {
            total_messages_stored: self.messages.len(),
            messages_in_context: context.len(),
            estimated_tokens: tokens,
        }
    }

    fn clear(&mut self) {
        self.messages.clear();
        self.facts.clear();
    }
}

// ---------------------------------------------------------------------------
// Strategy 3: Branching (dialog branches)
// ---------------------------------------------------------------------------

/// A named branch of the conversation.
#[derive(Debug, Clone)]
struct Branch {
    messages: Vec<Message>,
}

/// Supports creating checkpoints and branching the conversation into
/// independent paths that can be switched between.
#[derive(Debug)]
pub struct Branching {
    branches: HashMap<String, Branch>,
    active_branch: String,
}

impl Branching {
    pub fn new() -> Self {
        let mut branches = HashMap::new();
        branches.insert(
            "main".to_string(),
            Branch {
                messages: Vec::new(),
            },
        );
        Self {
            branches,
            active_branch: "main".to_string(),
        }
    }

    /// Creates a checkpoint (new branch) from the current state of the active branch.
    pub fn create_branch(&mut self, name: &str) -> Result<(), String> {
        if self.branches.contains_key(name) {
            return Err(format!("Branch '{}' already exists", name));
        }
        let current_messages = self.branches[&self.active_branch].messages.clone();
        self.branches.insert(
            name.to_string(),
            Branch {
                messages: current_messages,
            },
        );
        Ok(())
    }

    /// Switches to a different branch.
    pub fn switch_branch(&mut self, name: &str) -> Result<(), String> {
        if !self.branches.contains_key(name) {
            return Err(format!("Branch '{}' does not exist", name));
        }
        self.active_branch = name.to_string();
        Ok(())
    }

    /// Returns the name of the active branch.
    pub fn active_branch(&self) -> &str {
        &self.active_branch
    }

    /// Returns the names of all branches.
    pub fn branch_names(&self) -> Vec<&String> {
        self.branches.keys().collect()
    }

    /// Returns the message count in a specific branch.
    pub fn branch_len(&self, name: &str) -> Option<usize> {
        self.branches.get(name).map(|b| b.messages.len())
    }
}

impl Default for Branching {
    fn default() -> Self {
        Self::new()
    }
}

impl ContextStrategy for Branching {
    fn add_message(&mut self, message: Message) {
        if let Some(branch) = self.branches.get_mut(&self.active_branch) {
            branch.messages.push(message);
        }
    }

    fn build_context(&self) -> Vec<Message> {
        self.branches
            .get(&self.active_branch)
            .map(|b| b.messages.clone())
            .unwrap_or_default()
    }

    fn stats(&self) -> ContextStats {
        let context = self.build_context();
        let tokens: usize = context.iter().map(|m| m.content.len() / 4 + 1).sum();
        let total: usize = self.branches.values().map(|b| b.messages.len()).sum();
        ContextStats {
            total_messages_stored: total,
            messages_in_context: context.len(),
            estimated_tokens: tokens,
        }
    }

    fn clear(&mut self) {
        self.branches.clear();
        self.branches.insert(
            "main".to_string(),
            Branch {
                messages: Vec::new(),
            },
        );
        self.active_branch = "main".to_string();
    }
}

/// Runs the same scenario through all three strategies and returns comparative stats.
pub fn compare_strategies(
    messages: &[(Message, Option<(&str, &str)>)],
    window_size: usize,
) -> (ContextStats, ContextStats, ContextStats) {
    let mut sliding = SlidingWindow::new(window_size);
    let mut facts = StickyFacts::new(window_size);
    let mut branching = Branching::new();

    for (msg, fact) in messages {
        sliding.add_message(msg.clone());
        facts.add_message(msg.clone());
        branching.add_message(msg.clone());

        if let Some((key, value)) = fact {
            facts.set_fact(key, value);
        }
    }

    (sliding.stats(), facts.stats(), branching.stats())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_messages(n: usize) -> Vec<Message> {
        (0..n)
            .map(|i| {
                if i % 2 == 0 {
                    Message::user(&format!("User message {}", i))
                } else {
                    Message::assistant(&format!("Assistant response {}", i))
                }
            })
            .collect()
    }

    // --- Sliding Window tests ---

    #[test]
    fn test_sliding_window_within_limit() {
        let mut sw = SlidingWindow::new(10);
        for msg in sample_messages(5) {
            sw.add_message(msg);
        }
        assert_eq!(sw.build_context().len(), 5);
    }

    #[test]
    fn test_sliding_window_truncates() {
        let mut sw = SlidingWindow::new(3);
        for msg in sample_messages(10) {
            sw.add_message(msg);
        }
        let context = sw.build_context();
        assert_eq!(context.len(), 3);
        // Should be the last 3 messages
        assert!(context[0].content.contains("7"));
        assert!(context[1].content.contains("8"));
        assert!(context[2].content.contains("9"));
    }

    #[test]
    fn test_sliding_window_stats() {
        let mut sw = SlidingWindow::new(3);
        for msg in sample_messages(10) {
            sw.add_message(msg);
        }
        let stats = sw.stats();
        assert_eq!(stats.total_messages_stored, 10);
        assert_eq!(stats.messages_in_context, 3);
    }

    // --- Sticky Facts tests ---

    #[test]
    fn test_facts_persist_beyond_window() {
        let mut sf = StickyFacts::new(2);
        sf.set_fact("goal", "Build an API");
        sf.set_fact("constraint", "Use PostgreSQL");

        for msg in sample_messages(10) {
            sf.add_message(msg);
        }

        let context = sf.build_context();
        // 1 facts message + 2 recent messages
        assert_eq!(context.len(), 3);
        assert!(context[0].content.contains("goal"));
        assert!(context[0].content.contains("Build an API"));
    }

    #[test]
    fn test_facts_update() {
        let mut sf = StickyFacts::new(5);
        sf.set_fact("status", "planning");
        assert_eq!(sf.facts()["status"], "planning");

        sf.set_fact("status", "in_progress");
        assert_eq!(sf.facts()["status"], "in_progress");
    }

    #[test]
    fn test_facts_remove() {
        let mut sf = StickyFacts::new(5);
        sf.set_fact("temp", "value");
        assert!(sf.remove_fact("temp"));
        assert!(!sf.remove_fact("temp"));
        assert!(sf.facts().is_empty());
    }

    #[test]
    fn test_facts_no_system_message_when_empty() {
        let mut sf = StickyFacts::new(5);
        sf.add_message(Message::user("Hello"));
        let context = sf.build_context();
        assert_eq!(context.len(), 1);
        assert_eq!(context[0].role, "user");
    }

    // --- Branching tests ---

    #[test]
    fn test_branching_default_main() {
        let branching = Branching::new();
        assert_eq!(branching.active_branch(), "main");
    }

    #[test]
    fn test_branching_create_and_switch() {
        let mut br = Branching::new();
        br.add_message(Message::user("Shared message"));

        br.create_branch("experiment-a").unwrap();
        br.switch_branch("experiment-a").unwrap();
        br.add_message(Message::user("Branch A message"));

        br.switch_branch("main").unwrap();
        br.add_message(Message::user("Main continues"));

        // Main has 2 messages (shared + main continues)
        assert_eq!(br.branch_len("main").unwrap(), 2);
        // Branch A has 2 messages (shared + branch A message)
        assert_eq!(br.branch_len("experiment-a").unwrap(), 2);

        // Contexts differ
        br.switch_branch("main").unwrap();
        let main_ctx = br.build_context();
        br.switch_branch("experiment-a").unwrap();
        let branch_ctx = br.build_context();

        assert_ne!(main_ctx.last(), branch_ctx.last());
    }

    #[test]
    fn test_branching_duplicate_name_error() {
        let mut br = Branching::new();
        br.create_branch("test").unwrap();
        let result = br.create_branch("test");
        assert!(result.is_err());
    }

    #[test]
    fn test_branching_switch_nonexistent_error() {
        let mut br = Branching::new();
        let result = br.switch_branch("nonexistent");
        assert!(result.is_err());
    }

    #[test]
    fn test_branching_independent_branches() {
        let mut br = Branching::new();
        // Add 3 shared messages
        for msg in sample_messages(3) {
            br.add_message(msg);
        }

        // Create two branches
        br.create_branch("a").unwrap();
        br.create_branch("b").unwrap();

        // Add different messages to each
        br.switch_branch("a").unwrap();
        br.add_message(Message::user("Path A choice"));

        br.switch_branch("b").unwrap();
        br.add_message(Message::user("Path B choice"));

        // Verify independence
        assert_eq!(br.branch_len("a").unwrap(), 4);
        assert_eq!(br.branch_len("b").unwrap(), 4);
        assert_eq!(br.branch_len("main").unwrap(), 3);
    }

    // --- Comparison test ---

    #[test]
    fn test_compare_strategies() {
        let messages: Vec<(Message, Option<(&str, &str)>)> = vec![
            (Message::user("I want to build a REST API"), Some(("goal", "Build REST API"))),
            (Message::assistant("Sure, what language?"), None),
            (Message::user("Use Rust with Actix"), Some(("stack", "Rust + Actix"))),
            (Message::assistant("Good choice. Database?"), None),
            (Message::user("PostgreSQL"), Some(("database", "PostgreSQL"))),
            (Message::assistant("Got it."), None),
            (Message::user("Add authentication"), Some(("feature", "Auth required"))),
            (Message::assistant("JWT or session-based?"), None),
            (Message::user("JWT"), Some(("auth", "JWT tokens"))),
            (Message::assistant("Understood."), None),
        ];

        let (sliding, facts, branching) = compare_strategies(&messages, 4);

        // Sliding window: only 4 messages in context
        assert_eq!(sliding.messages_in_context, 4);
        // Facts: 4 messages + 1 facts system message = 5
        assert_eq!(facts.messages_in_context, 5);
        // Branching: all 10 messages
        assert_eq!(branching.messages_in_context, 10);

        // All stored 10 messages
        assert_eq!(sliding.total_messages_stored, 10);
        assert_eq!(facts.total_messages_stored, 10);
        assert_eq!(branching.total_messages_stored, 10);

        // Sliding window uses fewest tokens (only last N messages)
        assert!(sliding.estimated_tokens < branching.estimated_tokens);
    }

    #[test]
    fn test_clear_all_strategies() {
        let mut sw = SlidingWindow::new(5);
        let mut sf = StickyFacts::new(5);
        let mut br = Branching::new();

        sw.add_message(Message::user("test"));
        sf.add_message(Message::user("test"));
        sf.set_fact("k", "v");
        br.add_message(Message::user("test"));

        sw.clear();
        sf.clear();
        br.clear();

        assert_eq!(sw.build_context().len(), 0);
        assert_eq!(sf.build_context().len(), 0);
        assert_eq!(br.build_context().len(), 0);
        assert!(sf.facts().is_empty());
    }
}
