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

fn make_body(messages: Vec<serde_json::Value>) -> serde_json::Value {
    json!({
        "model": "qwen/qwen3.6-plus:free",
        "messages": messages,
        "reasoning": { "enabled": true }
    })
}

fn print_result(label: &str, result: &models::ChatCompletion) {
    let choice = &result.choices[0];
    println!("=== {} ===\n", label);
    println!("Ответ:\n{}\n", choice.message.content);
    println!(
        "Токены: prompt={}, completion={}, total={}\n",
        result.usage.prompt_tokens,
        result.usage.completion_tokens,
        result.usage.total_tokens,
    );
}

const TASK: &str = "\
У фермера есть 3 курицы. Каждая курица несёт 2 яйца в день. \
Фермер собирает яйца каждый день в течение 7 дней. \
Затем он продаёт половину всех яиц, а из оставшихся готовит омлеты, \
на каждый омлет нужно 3 яйца. Сколько омлетов он сможет приготовить?";

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let api_key = std::env::var("OPENROUTER_API_KEY")?;
    let client = reqwest::Client::new();

    // --- Способ 1: прямой ответ ---
    let body1 = make_body(vec![
        json!({ "role": "user", "content": TASK }),
    ]);
    let r1 = send_request(&client, &api_key, body1).await?;
    print_result("Способ 1: Прямой ответ", &r1);

    // --- Способ 2: пошаговое рассуждение ---
    let body2 = make_body(vec![
        json!({
            "role": "system",
            "content": "Решай задачу пошагово. Каждый шаг нумеруй. \
                         В конце дай итоговый ответ в формате: ОТВЕТ: <число>"
        }),
        json!({ "role": "user", "content": TASK }),
    ]);
    let r2 = send_request(&client, &api_key, body2).await?;
    print_result("Способ 2: Пошаговое рассуждение (Chain-of-Thought)", &r2);

    // --- Способ 3: сначала составить промпт, потом решить ---
    // Шаг 3а: просим модель составить промпт
    let body3a = make_body(vec![
        json!({
            "role": "system",
            "content": "Ты — эксперт по составлению промптов. \
                         Пользователь даст тебе задачу. Твоя цель — составить \
                         идеальный промпт для решения этой задачи другой LLM. \
                         Верни ТОЛЬКО текст промпта, без пояснений."
        }),
        json!({ "role": "user", "content": TASK }),
    ]);
    let r3a = send_request(&client, &api_key, body3a).await?;
    let generated_prompt = &r3a.choices[0].message.content;

    println!("=== Способ 3: Мета-промпт (генерация + решение) ===\n");
    println!("Сгенерированный промпт:\n{}\n", generated_prompt);
    println!("---\n");

    // Шаг 3б: решаем задачу сгенерированным промптом
    let body3b = make_body(vec![
        json!({ "role": "user", "content": generated_prompt }),
    ]);
    let r3b = send_request(&client, &api_key, body3b).await?;
    let choice3b = &r3b.choices[0];
    println!("Ответ по сгенерированному промпту:\n{}\n", choice3b.message.content);
    println!(
        "Токены (генерация промпта): prompt={}, completion={}",
        r3a.usage.prompt_tokens, r3a.usage.completion_tokens,
    );
    println!(
        "Токены (решение): prompt={}, completion={}\n",
        r3b.usage.prompt_tokens, r3b.usage.completion_tokens,
    );

    // --- Способ 4: группа экспертов ---
    let body4 = make_body(vec![
        json!({
            "role": "system",
            "content": "Ты симулируешь обсуждение задачи тремя экспертами.\n\n\
                         АНАЛИТИК — разбирает условие задачи, выделяет данные и что нужно найти.\n\
                         ИНЖЕНЕР — выполняет вычисления шаг за шагом.\n\
                         КРИТИК — проверяет решение, ищет ошибки, подтверждает или исправляет.\n\n\
                         Формат ответа:\n\
                         [АНАЛИТИК]: ...\n\
                         [ИНЖЕНЕР]: ...\n\
                         [КРИТИК]: ...\n\
                         ИТОГОВЫЙ ОТВЕТ: <число>"
        }),
        json!({ "role": "user", "content": TASK }),
    ]);
    let r4 = send_request(&client, &api_key, body4).await?;
    print_result("Способ 4: Группа экспертов", &r4);

    // --- Сравнительная таблица ---
    println!("=== Сравнительная таблица ===\n");
    println!(
        "{:<45} {:>12} {:>12}",
        "Способ", "Completion", "Total"
    );
    println!("{}", "-".repeat(69));

    let rows: &[(&str, &models::Usage)] = &[
        ("1. Прямой ответ", &r1.usage),
        ("2. Пошаговое рассуждение", &r2.usage),
        ("3. Мета-промпт (генерация)", &r3a.usage),
        ("3. Мета-промпт (решение)", &r3b.usage),
        ("4. Группа экспертов", &r4.usage),
    ];
    for (name, usage) in rows {
        println!(
            "{:<45} {:>12} {:>12}",
            name, usage.completion_tokens, usage.total_tokens
        );
    }

    println!("\nПравильный ответ: 7 омлетов");
    println!("(3 курицы * 2 яйца * 7 дней = 42; 42 / 2 = 21; 21 / 3 = 7)\n");

    Ok(())
}
