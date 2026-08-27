//! Immutable configuration required for OpenAI request lowering.

#[derive(Clone, Debug, Default, PartialEq)]
pub struct SamplingDefaults {
    pub temperature: Option<f64>,
    pub top_p: Option<f64>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct RendererConfig {
    pub served_model_name: String,
    pub tool_call_parser: Option<String>,
    pub default_sampling_params: SamplingDefaults,
}
