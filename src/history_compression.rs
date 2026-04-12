//! Context management: history compression module.
//!
//! Keeps the last N messages verbatim, replaces older messages with summaries.
//! Summaries are stored separately and injected into the context instead of
//! the full history. Supports comparing token usage with and without compression.

use std::fmt;

/// A chat message.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Message {
    pub role: String,
    pub content: String,
}

impl Message {
    pub fn user(content: &str) -> Self {
        Self { role: "user".to_string(), content: content.to_string() }
    }

    pub fn assistant(content: &str) -> Self {
        Self { role: "assistant".to_string(), content: content.to_string() }
    }

    pub fn system(content: &str) -> Self {
        Self { role: "system".to_string(), content: content.to_string() }
    }
}

/// A summary block replacing a chunk of older messages.
#[derive(Debug, Clone)]
pub struct Summary {
    pub original_count: usize,
    pub text: String,
}

impl fmt::Display for Summary {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "[Summary of {} messages] {}",
            self.original_count, self.text
        )
    }
}

/// Token usage comparison between compressed and uncompressed modes.
#[derive(Debug, Clone)]
pub struct CompressionStats {
    pub total_messages: usize,
    pub recent_messages_kept: usize,
    pub messages_summarized: usize,
    pub summaries_count: usize,
    pub uncompressed_tokens: usize,
    pub compressed_tokens: usize,
    pub savings_percent: f64,
}

impl fmt::Display for CompressionStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Messages: {} total ({} recent, {} summarized into {} summaries)\n\
             Tokens: {} uncompressed, {} compressed ({:.1}% savings)",
            self.total_messages,
            self.recent_messages_kept,
            self.messages_summarized,
            self.summaries_count,
            self.uncompressed_tokens,
            self.compressed_tokens,
            self.savings_percent,
        )
    }
}

/// A function that produces a summary from a batch of messages.
/// In production this would call an LLM; here we use a pluggable function.
pub type SummarizerFn = Box<dyn Fn(&[Message]) -> String>;

/// Default summarizer: concatenates the first sentence of each message.
pub fn default_summarizer(messages: &[Message]) -> String {
    let parts: Vec<String> = messages
        .iter()
        .map(|m| {
            let first_sentence = m
                .content
                .split('.')
                .next()
                .unwrap_or(&m.content)
                .trim();
            format!("{}: {}", m.role, first_sentence)
        })
        .collect();
    parts.join(". ") + "."
}

/// History compressor that manages recent messages and summaries.
#[derive(Debug)]
pub struct HistoryCompressor {
    /// All raw messages (full history).
    all_messages: Vec<Message>,
    /// Number of recent messages to keep verbatim.
    recent_window: usize,
    /// How many messages to group per summary.
    summary_chunk_size: usize,
    /// Generated summaries for older message chunks.
    summaries: Vec<Summary>,
    /// Index up to which messages have been summarized.
    summarized_up_to: usize,
}

impl HistoryCompressor {
    /// Creates a new compressor.
    ///
    /// - `recent_window`: number of most recent messages to keep as-is
    /// - `summary_chunk_size`: how many messages per summary block
    pub fn new(recent_window: usize, summary_chunk_size: usize) -> Self {
        Self {
            all_messages: Vec::new(),
            recent_window,
            summary_chunk_size,
            summaries: Vec::new(),
            summarized_up_to: 0,
        }
    }

    /// Adds a message to the history.
    pub fn add_message(&mut self, message: Message) {
        self.all_messages.push(message);
    }

    /// Compresses older messages using the provided summarizer function.
    /// Returns the number of new summaries created.
    pub fn compress(&mut self, summarizer: &dyn Fn(&[Message]) -> String) -> usize {
        let total = self.all_messages.len();
        if total <= self.recent_window {
            return 0;
        }

        let eligible = total - self.recent_window;
        let mut new_summaries = 0;

        while self.summarized_up_to + self.summary_chunk_size <= eligible {
            let chunk_start = self.summarized_up_to;
            let chunk_end = chunk_start + self.summary_chunk_size;
            let chunk = &self.all_messages[chunk_start..chunk_end];

            let summary_text = summarizer(chunk);
            self.summaries.push(Summary {
                original_count: self.summary_chunk_size,
                text: summary_text,
            });

            self.summarized_up_to = chunk_end;
            new_summaries += 1;
        }

        new_summaries
    }

    /// Builds the compressed context: summaries + recent messages.
    pub fn build_compressed_context(&self) -> Vec<Message> {
        let mut context = Vec::new();

        // Add summaries as system messages
        for summary in &self.summaries {
            context.push(Message::system(&summary.to_string()));
        }

        // Add unsummarized-but-not-recent messages (partial chunk not yet summarized)
        let recent_start = self.all_messages.len().saturating_sub(self.recent_window);
        if self.summarized_up_to < recent_start {
            for msg in &self.all_messages[self.summarized_up_to..recent_start] {
                context.push(msg.clone());
            }
        }

        // Add recent messages
        for msg in &self.all_messages[recent_start..] {
            context.push(msg.clone());
        }

        context
    }

    /// Builds the full uncompressed context (all messages).
    pub fn build_full_context(&self) -> Vec<Message> {
        self.all_messages.clone()
    }

    /// Estimates tokens for a set of messages.
    fn estimate_tokens(messages: &[Message]) -> usize {
        messages.iter().map(|m| m.content.len() / 4 + 1).sum()
    }

    /// Returns compression statistics.
    pub fn stats(&self) -> CompressionStats {
        let compressed = self.build_compressed_context();
        let uncompressed_tokens = Self::estimate_tokens(&self.all_messages);
        let compressed_tokens = Self::estimate_tokens(&compressed);

        let savings = if uncompressed_tokens > 0 {
            (1.0 - compressed_tokens as f64 / uncompressed_tokens as f64) * 100.0
        } else {
            0.0
        };

        let recent_start = self.all_messages.len().saturating_sub(self.recent_window);

        CompressionStats {
            total_messages: self.all_messages.len(),
            recent_messages_kept: self.all_messages.len() - recent_start,
            messages_summarized: self.summarized_up_to,
            summaries_count: self.summaries.len(),
            uncompressed_tokens,
            compressed_tokens,
            savings_percent: savings,
        }
    }

    /// Returns the number of messages stored.
    pub fn message_count(&self) -> usize {
        self.all_messages.len()
    }

    /// Returns the number of summaries generated.
    pub fn summary_count(&self) -> usize {
        self.summaries.len()
    }

    /// Returns a reference to all summaries.
    pub fn summaries(&self) -> &[Summary] {
        &self.summaries
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_messages(n: usize) -> Vec<Message> {
        (0..n)
            .map(|i| {
                if i % 2 == 0 {
                    Message::user(&format!("User question number {}. It has some detail.", i))
                } else {
                    Message::assistant(&format!(
                        "Assistant answer number {}. Here is a detailed explanation.",
                        i
                    ))
                }
            })
            .collect()
    }

    #[test]
    fn test_no_compression_when_within_window() {
        let mut compressor = HistoryCompressor::new(10, 5);
        for msg in make_messages(5) {
            compressor.add_message(msg);
        }
        let created = compressor.compress(&default_summarizer);
        assert_eq!(created, 0);
        assert_eq!(compressor.summary_count(), 0);
    }

    #[test]
    fn test_single_chunk_compression() {
        let mut compressor = HistoryCompressor::new(4, 4);
        for msg in make_messages(10) {
            compressor.add_message(msg);
        }

        let created = compressor.compress(&default_summarizer);
        assert_eq!(created, 1); // (10 - 4) = 6 eligible, 6 / 4 = 1 full chunk
        assert_eq!(compressor.summary_count(), 1);

        let context = compressor.build_compressed_context();
        // 1 summary + 2 unsummarized middle messages + 4 recent = 7
        assert_eq!(context.len(), 7);
        assert_eq!(context[0].role, "system"); // summary
    }

    #[test]
    fn test_multiple_chunks_compression() {
        let mut compressor = HistoryCompressor::new(4, 4);
        for msg in make_messages(16) {
            compressor.add_message(msg);
        }

        let created = compressor.compress(&default_summarizer);
        assert_eq!(created, 3); // (16 - 4) = 12 eligible, 12 / 4 = 3 chunks
        assert_eq!(compressor.summary_count(), 3);

        let context = compressor.build_compressed_context();
        // 3 summaries + 4 recent = 7
        assert_eq!(context.len(), 7);
    }

    #[test]
    fn test_incremental_compression() {
        let mut compressor = HistoryCompressor::new(4, 4);

        // Add 10 messages
        for msg in make_messages(10) {
            compressor.add_message(msg);
        }
        compressor.compress(&default_summarizer);
        assert_eq!(compressor.summary_count(), 1);

        // Add 4 more messages (total 14)
        for msg in make_messages(4) {
            compressor.add_message(msg);
        }
        let created = compressor.compress(&default_summarizer);
        assert_eq!(created, 1); // one more chunk can now be summarized
        assert_eq!(compressor.summary_count(), 2);
    }

    #[test]
    fn test_compressed_saves_tokens() {
        let mut compressor = HistoryCompressor::new(4, 4);
        for msg in make_messages(20) {
            compressor.add_message(msg);
        }
        compressor.compress(&default_summarizer);

        let stats = compressor.stats();
        assert!(
            stats.compressed_tokens < stats.uncompressed_tokens,
            "Compressed ({}) should use fewer tokens than uncompressed ({})",
            stats.compressed_tokens,
            stats.uncompressed_tokens
        );
        assert!(stats.savings_percent > 0.0);
    }

    #[test]
    fn test_stats_report() {
        let mut compressor = HistoryCompressor::new(4, 4);
        for msg in make_messages(12) {
            compressor.add_message(msg);
        }
        compressor.compress(&default_summarizer);

        let stats = compressor.stats();
        assert_eq!(stats.total_messages, 12);
        assert_eq!(stats.recent_messages_kept, 4);
        assert_eq!(stats.messages_summarized, 8);
        assert_eq!(stats.summaries_count, 2);
    }

    #[test]
    fn test_full_context_unchanged() {
        let mut compressor = HistoryCompressor::new(4, 4);
        let messages = make_messages(10);
        for msg in &messages {
            compressor.add_message(msg.clone());
        }
        compressor.compress(&default_summarizer);

        let full = compressor.build_full_context();
        assert_eq!(full.len(), 10);
        assert_eq!(full, messages);
    }

    #[test]
    fn test_custom_summarizer() {
        let mut compressor = HistoryCompressor::new(2, 4);
        for msg in make_messages(8) {
            compressor.add_message(msg);
        }

        let custom = |msgs: &[Message]| -> String {
            format!("Summarized {} messages", msgs.len())
        };

        compressor.compress(&custom);
        let summaries = compressor.summaries();
        assert_eq!(summaries[0].text, "Summarized 4 messages");
    }

    #[test]
    fn test_empty_history() {
        let compressor = HistoryCompressor::new(4, 4);
        let context = compressor.build_compressed_context();
        assert!(context.is_empty());
        let stats = compressor.stats();
        assert_eq!(stats.total_messages, 0);
        assert_eq!(stats.savings_percent, 0.0);
    }

    #[test]
    fn test_recent_messages_are_last_n() {
        let mut compressor = HistoryCompressor::new(3, 3);
        let messages = make_messages(9);
        for msg in &messages {
            compressor.add_message(msg.clone());
        }
        compressor.compress(&default_summarizer);

        let context = compressor.build_compressed_context();
        // Last 3 messages should be the actual last 3 from history
        let recent: Vec<_> = context.iter().filter(|m| m.role != "system").collect();
        assert_eq!(recent.len(), 3);
        assert_eq!(recent[0].content, messages[6].content);
        assert_eq!(recent[1].content, messages[7].content);
        assert_eq!(recent[2].content, messages[8].content);
    }
}
