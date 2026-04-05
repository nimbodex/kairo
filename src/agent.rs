use crate::models::{ApiErrorResponse, ChatCompletion};
use serde_json::json;
use std::path::{Path, PathBuf};

pub struct Agent {
    client: reqwest::Client,
    api_key: String,
    model: String,
    system_prompt: Option<String>,
    history: Vec<serde_json::Value>,
    history_path: Option<PathBuf>,
}

impl Agent {
    pub fn new(api_key: String, model: &str) -> Self {
        Self {
            client: reqwest::Client::new(),
            api_key,
            model: model.to_string(),
            system_prompt: None,
            history: Vec::new(),
            history_path: None,
        }
    }

    pub fn with_system_prompt(mut self, prompt: &str) -> Self {
        self.system_prompt = Some(prompt.to_string());
        self
    }

    pub fn with_persistence(mut self, path: &Path) -> Self {
        if let Ok(data) = std::fs::read_to_string(path) {
            if let Ok(messages) = serde_json::from_str::<Vec<serde_json::Value>>(&data) {
                self.history = messages;
            }
        }
        self.history_path = Some(path.to_path_buf());
        self
    }

    fn save_history(&self) {
        if let Some(path) = &self.history_path {
            if let Ok(json) = serde_json::to_string_pretty(&self.history) {
                let _ = std::fs::write(path, json);
            }
        }
    }

    pub async fn send(&mut self, user_message: &str) -> Result<String, Box<dyn std::error::Error>> {
        self.history.push(json!({
            "role": "user",
            "content": user_message
        }));

        let mut messages = Vec::new();
        if let Some(sys) = &self.system_prompt {
            messages.push(json!({ "role": "system", "content": sys }));
        }
        messages.extend(self.history.clone());

        let body = json!({
            "model": &self.model,
            "messages": messages,
            "reasoning": { "enabled": true }
        });

        let resp = self.client
            .post("https://openrouter.ai/api/v1/chat/completions")
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .await?;

        let status = resp.status();
        let text = resp.text().await?;

        if !status.is_success() {
            // Откатываем добавленное user-сообщение при ошибке API
            self.history.pop();
            let err: ApiErrorResponse = serde_json::from_str(&text)?;
            let raw = err.error.metadata
                .and_then(|m| m.raw)
                .unwrap_or_else(|| err.error.message.clone());
            return Err(format!("[{}] {}", err.error.code, raw).into());
        }

        let completion: ChatCompletion = serde_json::from_str(&text)?;
        let reply = completion.choices[0].message.content.clone();

        self.history.push(json!({
            "role": "assistant",
            "content": &reply
        }));

        self.save_history();

        Ok(reply)
    }

    pub fn history_len(&self) -> usize {
        self.history.len()
    }

    pub fn clear_history(&mut self) {
        self.history.clear();
        self.save_history();
    }
}
