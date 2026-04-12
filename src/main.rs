mod agent;
mod context_strategies;
mod history_compression;
mod invariants;
mod memory_model;
mod models;
mod personalization;
mod state_transitions;
mod task_state_machine;
mod token_tracking;

use serde_json::json;
use std::time::Instant;

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

async fn compare_temperatures(
    client: &reqwest::Client,
    api_key: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n{}", "=".repeat(69));
    println!("   СРАВНЕНИЕ ТЕМПЕРАТУР (temperature = 0 / 0.7 / 1.2)");
    println!("{}\n", "=".repeat(69));

    let prompt = "Придумай название для стартапа, который помогает людям \
                  находить попутчиков для путешествий. Дай ровно 3 варианта \
                  с кратким пояснением (1 предложение) для каждого.";

    let temperatures = [0.0, 0.7, 1.2];
    let labels = [
        "temperature=0 (детерминированный)",
        "temperature=0.7 (сбалансированный)",
        "temperature=1.2 (креативный)",
    ];

    let mut results = Vec::new();

    for (i, &temp) in temperatures.iter().enumerate() {
        let body = json!({
            "model": "qwen/qwen3.6-plus:free",
            "messages": [
                { "role": "user", "content": prompt }
            ],
            "temperature": temp,
            "reasoning": { "enabled": true }
        });

        let result = send_request(client, api_key, body).await?;
        println!("=== {} ===\n", labels[i]);
        println!("Ответ:\n{}\n", result.choices[0].message.content);
        println!(
            "Токены: prompt={}, completion={}, total={}\n",
            result.usage.prompt_tokens,
            result.usage.completion_tokens,
            result.usage.total_tokens,
        );
        results.push(result);
    }

    // --- Сравнительная таблица ---
    println!("=== Сравнение температур: итоги ===\n");
    println!(
        "{:<40} {:>12} {:>12}",
        "Настройка", "Completion", "Total"
    );
    println!("{}", "-".repeat(64));
    for (i, result) in results.iter().enumerate() {
        println!(
            "{:<40} {:>12} {:>12}",
            labels[i],
            result.usage.completion_tokens,
            result.usage.total_tokens,
        );
    }

    println!("\n=== Выводы ===\n");
    println!("temperature=0   — детерминированный: одинаковый результат при повторных запросах.");
    println!("                  Лучше для: фактические вопросы, математика, классификация, код.\n");
    println!("temperature=0.7 — сбалансированный: умеренная вариативность при сохранении качества.");
    println!("                  Лучше для: копирайтинг, диалоги, генерация идей с контролем.\n");
    println!("temperature=1.2 — креативный: максимальное разнообразие, возможны неожиданные ответы.");
    println!("                  Лучше для: мозговой штурм, художественные тексты, нестандартные идеи.\n");

    Ok(())
}

struct ModelResult {
    label: &'static str,
    model_id: &'static str,
    answer: String,
    elapsed_ms: u128,
    prompt_tokens: u32,
    completion_tokens: u32,
    total_tokens: u32,
    cost: f64,
}

async fn compare_models(
    client: &reqwest::Client,
    api_key: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n{}", "=".repeat(69));
    println!("   СРАВНЕНИЕ МОДЕЛЕЙ (слабая / средняя / сильная)");
    println!("{}\n", "=".repeat(69));

    let prompt = "Объясни, почему небо голубое. Дай краткий, но научно точный ответ (3-5 предложений).";

    let models: &[(&str, &str)] = &[
        ("Слабая:  Qwen3 0.6B",    "qwen/qwen3-0.6b:free"),
        ("Средняя: Qwen3 4B",      "qwen/qwen3-4b:free"),
        ("Сильная: Qwen3 235B A22B","qwen/qwen3-235b-a22b:free"),
    ];

    let mut results = Vec::new();

    for &(label, model_id) in models {
        println!("--- {} ({}) ---\n", label, model_id);

        let body = json!({
            "model": model_id,
            "messages": [
                { "role": "user", "content": prompt }
            ],
            "reasoning": { "enabled": true }
        });

        let start = Instant::now();
        let completion = send_request(client, api_key, body).await?;
        let elapsed = start.elapsed().as_millis();

        let answer = &completion.choices[0].message.content;
        println!("Ответ:\n{}\n", answer);
        println!("Время: {} мс", elapsed);
        println!("Токены: prompt={}, completion={}, total={}",
            completion.usage.prompt_tokens,
            completion.usage.completion_tokens,
            completion.usage.total_tokens,
        );
        println!("Стоимость: ${:.6}\n", completion.usage.cost);

        results.push(ModelResult {
            label,
            model_id,
            answer: answer.clone(),
            elapsed_ms: elapsed,
            prompt_tokens: completion.usage.prompt_tokens,
            completion_tokens: completion.usage.completion_tokens,
            total_tokens: completion.usage.total_tokens,
            cost: completion.usage.cost,
        });
    }

    // --- Сравнительная таблица ---
    println!("=== Сравнительная таблица моделей ===\n");
    println!(
        "{:<28} {:>10} {:>10} {:>10} {:>12}",
        "Модель", "Время(мс)", "Compl.tok", "Total tok", "Стоимость"
    );
    println!("{}", "-".repeat(70));
    for r in &results {
        println!(
            "{:<28} {:>10} {:>10} {:>10} {:>12.6}",
            r.label, r.elapsed_ms, r.completion_tokens, r.total_tokens, r.cost
        );
    }

    println!("\n=== Выводы ===\n");
    println!("Слабая модель (0.6B)  — быстрая и дешёвая, но ответы поверхностные,");
    println!("                        могут содержать неточности. Подходит для простых задач.\n");
    println!("Средняя модель (4B)   — баланс скорости и качества. Справляется с большинством");
    println!("                        повседневных задач: суммаризация, перевод, Q&A.\n");
    println!("Сильная модель (235B) — наиболее точные и полные ответы, но медленнее и дороже.");
    println!("                        Для сложных задач: анализ, рассуждения, генерация кода.\n");

    println!("Ссылки на модели:");
    for r in &results {
        let slug = r.model_id.trim_end_matches(":free");
        println!("  {} -> https://openrouter.ai/{}", r.label, slug);
    }
    println!();

    Ok(())
}

async fn run_chat(api_key: String) -> Result<(), Box<dyn std::error::Error>> {
    let history_path = std::path::Path::new("chat_history.json");

    let mut agent = agent::Agent::new(api_key, "qwen/qwen3.6-plus:free")
        .with_system_prompt("Ты — полезный ассистент. Отвечай кратко и по делу.")
        .with_persistence(history_path);

    println!("Kairo Agent (Qwen3.6 Plus)");
    if agent.history_len() > 0 {
        println!("(восстановлено {} сообщений из истории)", agent.history_len());
    }
    println!("Команды: 'выход' — выйти, '/clear' — очистить историю\n");

    let stdin = std::io::stdin();
    loop {
        print!("Вы> ");
        use std::io::Write;
        std::io::stdout().flush()?;

        let mut input = String::new();
        stdin.read_line(&mut input)?;
        let input = input.trim();

        if input.is_empty() {
            continue;
        }
        if input == "выход" || input == "exit" || input == "quit" {
            println!("До свидания!");
            break;
        }
        if input == "/clear" {
            agent.clear_history();
            println!("(история очищена)\n");
            continue;
        }

        match agent.send(input).await {
            Ok(reply) => {
                println!("\nАгент> {}\n", reply);
                println!("(сообщений в истории: {})\n", agent.history_len());
            }
            Err(e) => eprintln!("\nОшибка: {}\n", e),
        }
    }

    Ok(())
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let api_key = std::env::var("OPENROUTER_API_KEY")?;

    if std::env::args().any(|a| a == "--chat") {
        return run_chat(api_key).await;
    }

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

    // --- Сравнение температур ---
    compare_temperatures(&client, &api_key).await?;

    // --- Сравнение моделей ---
    compare_models(&client, &api_key).await?;

    Ok(())
}
