mod models;

use serde_json::json;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let api_key = std::env::var("OPENROUTER_API_KEY")?;

    let client = reqwest::Client::new();

    let body = json!({
        "model": "qwen/qwen3.6-plus-preview:free",
        "messages": [
            {
                "role": "user",
                "content": "How many r`s are in the word `strawberry?`"
            }
        ],
        "reasoning": {
            "enabled": true
        }
    });

    let resp = client
        .post("https://openrouter.ai/api/v1/chat/completions")
        .header("Authorization", format!("Bearer {api_key}"))
        .header("Content-Type", "application/json")
        .json(&body)
        .send()
        .await?;

    let status = resp.status();
    let text = resp.text().await?;

    if !status.is_success() {
        let err: models::ApiErrorResponse = serde_json::from_str(&text)?;
        let raw = err.error.metadata
            .and_then(|m| m.raw)
            .unwrap_or_else(|| err.error.message.clone());
        return Err(format!("[{}] {}", err.error.code, raw).into());
    }

    let completion: models::ChatCompletion = serde_json::from_str(&text)?;
    println!("{}", completion.choices[0].message.content);

    Ok(())
}