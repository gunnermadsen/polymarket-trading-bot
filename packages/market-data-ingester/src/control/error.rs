use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use serde::Serialize;

use crate::persistence::ProfileWriteError;

#[derive(Debug)]
pub(super) struct ApiError {
    status: StatusCode,
    code: &'static str,
    message: String,
}

impl ApiError {
    pub(super) fn new(status: StatusCode, code: &'static str, message: impl Into<String>) -> Self {
        Self {
            status,
            code,
            message: message.into(),
        }
    }

    pub(super) fn internal(error: impl std::fmt::Display) -> Self {
        tracing::error!(error = %error, "market-data ingester API request failed");
        Self::new(
            StatusCode::INTERNAL_SERVER_ERROR,
            "internal_error",
            "the request could not be completed",
        )
    }
}

impl From<ProfileWriteError> for ApiError {
    fn from(error: ProfileWriteError) -> Self {
        match error {
            ProfileWriteError::NotFound(_) => Self::new(
                StatusCode::NOT_FOUND,
                "profile_not_found",
                error.to_string(),
            ),
            ProfileWriteError::GenerationConflict { .. } => Self::new(
                StatusCode::CONFLICT,
                "generation_conflict",
                error.to_string(),
            ),
            other => Self::internal(other),
        }
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        (
            self.status,
            Json(ApiErrorBody {
                code: self.code,
                message: self.message,
            }),
        )
            .into_response()
    }
}

#[derive(Serialize)]
struct ApiErrorBody {
    code: &'static str,
    message: String,
}
