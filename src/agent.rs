use crate::models::{ApiErrorResponse, ChatCompletion};
use serde_json::json;

pub struct Agent {
    client: reqwest::Client,
    api_key: String,
    model: String,
    system_prompt: Option<String>,
    history: Vec<serde_json::Value>,
}

impl Agent {
    pub fn new(api_key: String, model: &str) -> Self {
        Self {
            client: reqwest::Client::new(),
            api_key,
            model: model.to_string(),
            system_prompt: None,
            history: Vec::new(),
        }
    }

    pub fn with_system_prompt(mut self, prompt: &str) -> Self {
        self.system_prompt = Some(prompt.to_string());
        self
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

        Ok(reply)
    }

    pub fn history_len(&self) -> usize {
        self.history.len()
    }

    pub fn clear_history(&mut self) {
        self.history.clear();
    }
}
