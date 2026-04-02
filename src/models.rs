use serde::Deserialize;

#[derive(Debug, Deserialize)]
pub struct ApiErrorResponse {
    pub error: ApiError,
    pub user_id: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct ApiError {
    pub message: String,
    pub code: u16,
    pub metadata: Option<ApiErrorMetadata>,
}

#[derive(Debug, Deserialize)]
pub struct ApiErrorMetadata {
    pub raw: Option<String>,
    pub provider_name: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct ChatCompletion {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub provider: String,
    pub system_fingerprint: Option<String>,
    pub choices: Vec<Choice>,
    pub usage: Usage,
}

#[derive(Debug, Deserialize)]
pub struct Choice {
    pub index: u32,
    pub finish_reason: String,
    pub native_finish_reason: String,
    pub message: Message,
}

#[derive(Debug, Deserialize)]
pub struct Message {
    pub role: String,
    pub content: String,
    pub refusal: Option<String>,
    pub reasoning: Option<String>,
    #[serde(default)]
    pub reasoning_details: Vec<ReasoningDetail>,
}

#[derive(Debug, Deserialize)]
pub struct ReasoningDetail {
    #[serde(rename = "type")]
    pub kind: String,
    pub text: String,
    pub format: String,
    pub index: u32,
}

#[derive(Debug, Deserialize)]
pub struct Usage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
    pub cost: f64,
    pub is_byok: bool,
    pub prompt_tokens_details: PromptTokensDetails,
    pub cost_details: CostDetails,
    pub completion_tokens_details: CompletionTokensDetails,
}

#[derive(Debug, Deserialize)]
pub struct PromptTokensDetails {
    pub cached_tokens: u32,
    pub cache_write_tokens: u32,
    pub audio_tokens: u32,
    pub video_tokens: u32,
}

#[derive(Debug, Deserialize)]
pub struct CostDetails {
    pub upstream_inference_cost: f64,
    pub upstream_inference_prompt_cost: f64,
    pub upstream_inference_completions_cost: f64,
}

#[derive(Debug, Deserialize)]
pub struct CompletionTokensDetails {
    pub reasoning_tokens: u32,
    pub image_tokens: u32,
    pub audio_tokens: u32,
}
