mod models;

use serde_json::json;

async fn send_request(
    client: &reqwest::Client,
    api_key: &str,
    body: serde_json::Value,
) -> Result<models::ChatCompletion, Box<dyn std::error::Error>> {
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
    Ok(completion)
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let api_key = std::env::var("OPENROUTER_API_KEY")?;
    let client = reqwest::Client::new();

    let question = "How many r`s are in the word `strawberry?`";

    // --- Запрос 1: без ограничений ---
    println!("=== Запрос БЕЗ ограничений ===\n");

    let body_free = json!({
        "model": "qwen/qwen3.6-plus:free",
        "messages": [
            { "role": "user", "content": question }
        ],
        "reasoning": { "enabled": true }
    });

    let result_free = send_request(&client, &api_key, body_free).await?;
    let choice_free = &result_free.choices[0];

    println!("Ответ:\n{}", choice_free.message.content);
    println!("Finish reason: {}", choice_free.finish_reason);
    println!("Токены: prompt={}, completion={}, total={}",
        result_free.usage.prompt_tokens,
        result_free.usage.completion_tokens,
        result_free.usage.total_tokens,
    );

    // --- Запрос 2: с ограничениями ---
    println!("\n=== Запрос С ограничениями ===\n");

    let body_constrained = json!({
        "model": "qwen/qwen3.6-plus:free",
        "messages": [
            {
                "role": "system",
                "content": "Отвечай строго в формате JSON: {\"answer\": <число>, \"explanation\": \"<краткое пояснение>\"}. \
                             Никакого дополнительного текста вне JSON. Ответ должен быть максимально кратким."
            },
            { "role": "user", "content": question }
        ],
        "max_tokens": 100,
        "stop": ["\n\n", "---"],
        "reasoning": { "enabled": true }
    });

    let result_constrained = send_request(&client, &api_key, body_constrained).await?;
    let choice_constrained = &result_constrained.choices[0];

    println!("Ответ:\n{}", choice_constrained.message.content);
    println!("Finish reason: {}", choice_constrained.finish_reason);
    println!("Токены: prompt={}, completion={}, total={}",
        result_constrained.usage.prompt_tokens,
        result_constrained.usage.completion_tokens,
        result_constrained.usage.total_tokens,
    );

    // --- Сравнение ---
    println!("\n=== Сравнение ===\n");
    println!("Без ограничений:  {} токенов completion, finish_reason={}",
        result_free.usage.completion_tokens, choice_free.finish_reason);
    println!("С ограничениями:  {} токенов completion, finish_reason={}",
        result_constrained.usage.completion_tokens, choice_constrained.finish_reason);

    Ok(())
}
