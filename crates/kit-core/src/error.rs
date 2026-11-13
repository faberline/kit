//! Error types for kit

use thiserror::Error;

/// Result type alias for kit operations
pub type Result<T> = std::result::Result<T, KitError>;

/// Unified error type for all kit operations
#[derive(Error, Debug, Clone)]
pub enum KitError {
    #[error("MongoDB error: {0}")]
    MongoDB(String),

    #[error("Database error: {0}")]
    Database(String),

    #[error("Serialization error: {0}")]
    Serialization(String),

    #[error("Deserialization error: {0}")]
    Deserialization(String),

    #[error("Connection error: {0}")]
    Connection(String),

    #[error("Query error: {0}")]
    Query(String),

    #[error("Validation error: {0}")]
    Validation(String),

    #[error("Internal error: {0}")]
    Internal(String),

    // PostgreSQL-specific error types for better error handling
    /// Unique constraint violation (SQLSTATE 23505)
    #[error("Conflict: {0}")]
    Conflict(String),

    /// Foreign key constraint violation (SQLSTATE 23503, 23502)
    #[error("Foreign key constraint violation: {0}")]
    ForeignKey(String),

    /// Deadlock detected (SQLSTATE 40P01) - retryable
    #[error("Deadlock detected: {0}")]
    Deadlock(String),

    /// Connection timeout - retryable
    #[error("Timeout: {0}")]
    Timeout(String),

    /// Transient error that may succeed on retry
    #[error("Transient error: {0}")]
    Transient(String),
}

impl KitError {
    /// Returns true if this error is potentially retryable
    pub fn is_retryable(&self) -> bool {
        matches!(
            self,
            KitError::Deadlock(_) | KitError::Timeout(_) | KitError::Transient(_)
        )
    }

    /// Returns true if this is a constraint violation error
    pub fn is_constraint_violation(&self) -> bool {
        matches!(self, KitError::Conflict(_) | KitError::ForeignKey(_))
    }
}

impl From<serde_json::Error> for KitError {
    fn from(err: serde_json::Error) -> Self {
        KitError::Serialization(err.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display_mongodb() {
        let err = KitError::MongoDB("connection refused".to_string());
        assert_eq!(err.to_string(), "MongoDB error: connection refused");
    }

    #[test]
    fn test_error_display_database() {
        let err = KitError::Database("invalid query".to_string());
        assert_eq!(err.to_string(), "Database error: invalid query");
    }

    #[test]
    fn test_error_display_serialization() {
        let err = KitError::Serialization("invalid JSON".to_string());
        assert_eq!(err.to_string(), "Serialization error: invalid JSON");
    }

    #[test]
    fn test_error_display_deserialization() {
        let err = KitError::Deserialization("missing field".to_string());
        assert_eq!(err.to_string(), "Deserialization error: missing field");
    }

    #[test]
    fn test_error_display_connection() {
        let err = KitError::Connection("timeout".to_string());
        assert_eq!(err.to_string(), "Connection error: timeout");
    }

    #[test]
    fn test_error_display_query() {
        let err = KitError::Query("invalid operator".to_string());
        assert_eq!(err.to_string(), "Query error: invalid operator");
    }

    #[test]
    fn test_error_display_validation() {
        let err = KitError::Validation("field required".to_string());
        assert_eq!(err.to_string(), "Validation error: field required");
    }

    #[test]
    fn test_error_display_internal() {
        let err = KitError::Internal("unexpected state".to_string());
        assert_eq!(err.to_string(), "Internal error: unexpected state");
    }

    #[test]
    fn test_from_serde_json_error() {
        let json_err = serde_json::from_str::<String>("invalid").unwrap_err();
        let err: KitError = json_err.into();
        assert!(matches!(err, KitError::Serialization(_)));
    }

    #[test]
    #[allow(clippy::unnecessary_literal_unwrap)] // Testing Result type alias
    fn test_result_type_ok() {
        let result: Result<i32> = Ok(42);
        assert_eq!(result.unwrap(), 42);
    }

    #[test]
    fn test_result_type_err() {
        let result: Result<i32> = Err(KitError::Query("failed".to_string()));
        assert!(result.is_err());
    }

    #[test]
    fn test_error_display_conflict() {
        let err = KitError::Conflict("duplicate key value".to_string());
        assert_eq!(err.to_string(), "Conflict: duplicate key value");
    }

    #[test]
    fn test_error_display_foreign_key() {
        let err = KitError::ForeignKey("violates foreign key constraint".to_string());
        assert_eq!(
            err.to_string(),
            "Foreign key constraint violation: violates foreign key constraint"
        );
    }

    #[test]
    fn test_error_display_deadlock() {
        let err = KitError::Deadlock("deadlock detected".to_string());
        assert_eq!(err.to_string(), "Deadlock detected: deadlock detected");
    }

    #[test]
    fn test_error_display_timeout() {
        let err = KitError::Timeout("connection timed out".to_string());
        assert_eq!(err.to_string(), "Timeout: connection timed out");
    }

    #[test]
    fn test_error_display_transient() {
        let err = KitError::Transient("serialization failure".to_string());
        assert_eq!(err.to_string(), "Transient error: serialization failure");
    }

    #[test]
    fn test_is_retryable() {
        assert!(KitError::Deadlock("test".to_string()).is_retryable());
        assert!(KitError::Timeout("test".to_string()).is_retryable());
        assert!(KitError::Transient("test".to_string()).is_retryable());
        assert!(!KitError::Conflict("test".to_string()).is_retryable());
        assert!(!KitError::ForeignKey("test".to_string()).is_retryable());
        assert!(!KitError::Query("test".to_string()).is_retryable());
    }

    #[test]
    fn test_is_constraint_violation() {
        assert!(KitError::Conflict("test".to_string()).is_constraint_violation());
        assert!(KitError::ForeignKey("test".to_string()).is_constraint_violation());
        assert!(!KitError::Deadlock("test".to_string()).is_constraint_violation());
        assert!(!KitError::Query("test".to_string()).is_constraint_violation());
    }
}
