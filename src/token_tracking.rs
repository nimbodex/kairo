//! Token tracking module.
//!
//! Provides token estimation, per-message and cumulative usage tracking,
//! and detection of context window overflow scenarios.

use std::fmt;

/// Rough token estimate: ~4 characters per token (common heuristic for English/mixed text).
const CHARS_PER_TOKEN: usize = 4;

/// Estimates the number of tokens in a string.
pub fn estimate_tokens(text: &str) -> usize {
    let chars = text.chars().count();
    (chars + CHARS_PER_TOKEN - 1) / CHARS_PER_TOKEN
}

/// Token usage snapshot for a single API exchange.
#[derive(Debug, Clone)]
pub struct ExchangeUsage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
}

impl fmt::Display for ExchangeUsage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "prompt={}, completion={}, total={}",
            self.prompt_tokens, self.completion_tokens, self.total_tokens
        )
    }
}

/// Tracks token usage across a dialog session.
#[derive(Debug)]
pub struct TokenTracker {
    /// Maximum context window size in tokens.
    context_limit: usize,
    /// Per-exchange usage history.
    exchanges: Vec<ExchangeUsage>,
    /// Running total of prompt tokens sent.
    cumulative_prompt: usize,
    /// Running total of completion tokens received.
    cumulative_completion: usize,
}

/// What happens when the context limit is exceeded.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OverflowStatus {
    /// Within limits. Contains remaining capacity.
    Ok { remaining: usize },
    /// Approaching the limit (less than 10% remaining).
    Warning { remaining: usize },
    /// Context window exceeded.
    Overflow { excess: usize },
}

impl fmt::Display for OverflowStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            OverflowStatus::Ok { remaining } => {
                write!(f, "OK ({} tokens remaining)", remaining)
            }
            OverflowStatus::Warning { remaining } => {
                write!(f, "WARNING: only {} tokens remaining", remaining)
            }
            OverflowStatus::Overflow { excess } => {
                write!(f, "OVERFLOW: exceeded limit by {} tokens", excess)
            }
        }
    }
}

impl TokenTracker {
    /// Creates a new tracker with the given context window limit.
    pub fn new(context_limit: usize) -> Self {
        Self {
            context_limit,
            exchanges: Vec::new(),
            cumulative_prompt: 0,
            cumulative_completion: 0,
        }
    }

    /// Records a single request/response exchange.
    pub fn record_exchange(&mut self, prompt_tokens: usize, completion_tokens: usize) {
        let usage = ExchangeUsage {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens + completion_tokens,
        };
        self.cumulative_prompt += prompt_tokens;
        self.cumulative_completion += completion_tokens;
        self.exchanges.push(usage);
    }

    /// Estimates token usage from raw message texts and records the exchange.
    pub fn record_from_text(&mut self, prompt_text: &str, completion_text: &str) {
        let prompt_tokens = estimate_tokens(prompt_text);
        let completion_tokens = estimate_tokens(completion_text);
        self.record_exchange(prompt_tokens, completion_tokens);
    }

    /// Returns the current estimated context size (what the next prompt would cost).
    /// This is the sum of all messages in the dialog so far.
    pub fn current_context_size(&self) -> usize {
        // Each exchange's prompt already includes history; the last exchange's prompt_tokens
        // is the best estimate, but for a growing dialog we track the latest prompt size
        // plus the latest completion (since that gets added to history).
        self.exchanges
            .last()
            .map(|e| e.prompt_tokens + e.completion_tokens)
            .unwrap_or(0)
    }

    /// Checks overflow status given an estimated next prompt size.
    pub fn check_overflow(&self, next_prompt_tokens: usize) -> OverflowStatus {
        if next_prompt_tokens > self.context_limit {
            OverflowStatus::Overflow {
                excess: next_prompt_tokens - self.context_limit,
            }
        } else {
            let remaining = self.context_limit - next_prompt_tokens;
            let threshold = self.context_limit / 10;
            if remaining < threshold {
                OverflowStatus::Warning { remaining }
            } else {
                OverflowStatus::Ok { remaining }
            }
        }
    }

    /// Returns cumulative prompt tokens across all exchanges.
    pub fn total_prompt_tokens(&self) -> usize {
        self.cumulative_prompt
    }

    /// Returns cumulative completion tokens across all exchanges.
    pub fn total_completion_tokens(&self) -> usize {
        self.cumulative_completion
    }

    /// Returns the number of exchanges recorded.
    pub fn exchange_count(&self) -> usize {
        self.exchanges.len()
    }

    /// Returns a reference to all recorded exchanges.
    pub fn exchanges(&self) -> &[ExchangeUsage] {
        &self.exchanges
    }

    /// Returns the context limit.
    pub fn context_limit(&self) -> usize {
        self.context_limit
    }

    /// Simulates a growing dialog and returns per-step usage and overflow status.
    /// Each message pair is (user_message, assistant_response).
    pub fn simulate_dialog(
        context_limit: usize,
        messages: &[(&str, &str)],
    ) -> Vec<(ExchangeUsage, OverflowStatus)> {
        let mut tracker = TokenTracker::new(context_limit);
        let mut history_tokens: usize = 0;
        let mut results = Vec::new();

        for (user_msg, assistant_msg) in messages {
            let user_tokens = estimate_tokens(user_msg);
            let assistant_tokens = estimate_tokens(assistant_msg);

            // Prompt = all history + new user message
            let prompt_tokens = history_tokens + user_tokens;
            tracker.record_exchange(prompt_tokens, assistant_tokens);

            let overflow = tracker.check_overflow(prompt_tokens + assistant_tokens);

            results.push((
                ExchangeUsage {
                    prompt_tokens,
                    completion_tokens: assistant_tokens,
                    total_tokens: prompt_tokens + assistant_tokens,
                },
                overflow,
            ));

            // After this exchange, history grows by user + assistant messages
            history_tokens += user_tokens + assistant_tokens;
        }

        results
    }

    /// Prints a summary table of token growth across exchanges.
    pub fn print_growth_report(&self) {
        println!("=== Token Growth Report ===\n");
        println!(
            "{:<10} {:>12} {:>12} {:>12}",
            "Exchange", "Prompt", "Completion", "Total"
        );
        println!("{}", "-".repeat(48));
        for (i, ex) in self.exchanges.iter().enumerate() {
            println!(
                "{:<10} {:>12} {:>12} {:>12}",
                i + 1,
                ex.prompt_tokens,
                ex.completion_tokens,
                ex.total_tokens
            );
        }
        println!("{}", "-".repeat(48));
        println!(
            "{:<10} {:>12} {:>12} {:>12}",
            "TOTAL",
            self.cumulative_prompt,
            self.cumulative_completion,
            self.cumulative_prompt + self.cumulative_completion
        );
        println!();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_estimate_tokens() {
        assert_eq!(estimate_tokens(""), 0);
        assert_eq!(estimate_tokens("hi"), 1); // 2 chars / 4 = 0.5 → rounds up to 1
        assert_eq!(estimate_tokens("hello world"), 3); // 11 chars / 4 = 2.75 → 3
        assert_eq!(estimate_tokens("abcd"), 1); // exactly 4 chars → 1 token
        assert_eq!(estimate_tokens("abcde"), 2); // 5 chars → 2 tokens
    }

    #[test]
    fn test_record_exchange() {
        let mut tracker = TokenTracker::new(1000);
        tracker.record_exchange(100, 50);
        assert_eq!(tracker.exchange_count(), 1);
        assert_eq!(tracker.total_prompt_tokens(), 100);
        assert_eq!(tracker.total_completion_tokens(), 50);
    }

    #[test]
    fn test_record_from_text() {
        let mut tracker = TokenTracker::new(1000);
        tracker.record_from_text("hello world", "goodbye world");
        assert_eq!(tracker.exchange_count(), 1);
        // "hello world" = 11 chars → 3 tokens
        assert_eq!(tracker.total_prompt_tokens(), 3);
        // "goodbye world" = 13 chars → 4 tokens
        assert_eq!(tracker.total_completion_tokens(), 4);
    }

    #[test]
    fn test_cumulative_tracking() {
        let mut tracker = TokenTracker::new(1000);
        tracker.record_exchange(100, 50);
        tracker.record_exchange(200, 80);
        tracker.record_exchange(350, 120);

        assert_eq!(tracker.exchange_count(), 3);
        assert_eq!(tracker.total_prompt_tokens(), 650);
        assert_eq!(tracker.total_completion_tokens(), 250);
    }

    #[test]
    fn test_overflow_ok() {
        let tracker = TokenTracker::new(1000);
        let status = tracker.check_overflow(500);
        assert_eq!(status, OverflowStatus::Ok { remaining: 500 });
    }

    #[test]
    fn test_overflow_warning() {
        let tracker = TokenTracker::new(1000);
        // 10% threshold = 100, so 950 leaves only 50 remaining
        let status = tracker.check_overflow(950);
        assert_eq!(status, OverflowStatus::Warning { remaining: 50 });
    }

    #[test]
    fn test_overflow_exceeded() {
        let tracker = TokenTracker::new(1000);
        let status = tracker.check_overflow(1200);
        assert_eq!(status, OverflowStatus::Overflow { excess: 200 });
    }

    #[test]
    fn test_simulate_short_dialog() {
        let messages = vec![
            ("What is Rust?", "Rust is a systems programming language."),
            ("Tell me more.", "It focuses on safety and performance."),
        ];
        let results = TokenTracker::simulate_dialog(1000, &messages);
        assert_eq!(results.len(), 2);

        // First exchange: no history yet
        assert!(matches!(results[0].1, OverflowStatus::Ok { .. }));
        // Second exchange: history includes first exchange
        assert!(results[1].0.prompt_tokens > results[0].0.prompt_tokens);
    }

    #[test]
    fn test_simulate_overflow_dialog() {
        // Tiny context limit to force overflow
        let messages = vec![
            ("Hello, how are you today?", "I am fine, thank you for asking!"),
            (
                "Can you explain quantum computing?",
                "Quantum computing uses qubits that can exist in superposition.",
            ),
            (
                "What about quantum entanglement and its implications?",
                "Entanglement is a phenomenon where particles become correlated.",
            ),
        ];
        let results = TokenTracker::simulate_dialog(20, &messages);
        assert_eq!(results.len(), 3);

        // The last exchange should overflow a 20-token limit
        let last_status = &results[2].1;
        assert!(matches!(last_status, OverflowStatus::Overflow { .. }));
    }

    #[test]
    fn test_token_growth_is_monotonic() {
        let messages: Vec<(&str, &str)> = vec![
            ("Hi", "Hello!"),
            ("How are you?", "I'm good."),
            ("What's the weather?", "It's sunny today."),
            ("Thanks!", "You're welcome!"),
        ];
        let results = TokenTracker::simulate_dialog(10000, &messages);

        // Prompt tokens should grow monotonically (history accumulates)
        for i in 1..results.len() {
            assert!(
                results[i].0.prompt_tokens >= results[i - 1].0.prompt_tokens,
                "Prompt tokens should grow: exchange {} ({}) < exchange {} ({})",
                i,
                results[i].0.prompt_tokens,
                i - 1,
                results[i - 1].0.prompt_tokens,
            );
        }
    }
}
