pub mod data;
pub mod extractor;
pub mod features;
pub mod token;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("I/O: {0}")]
    Io(String),
}
