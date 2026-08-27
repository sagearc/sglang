//! Transport-neutral request-lowering failures.

use thiserror::Error;

#[derive(Debug, Clone, Error)]
pub enum RendererError {
    #[error("{0}")]
    Validation(String),
}

impl From<String> for RendererError {
    fn from(message: String) -> Self {
        Self::Validation(message)
    }
}

impl From<&str> for RendererError {
    fn from(message: &str) -> Self {
        Self::Validation(message.to_owned())
    }
}
