/// Load the GitHub token from a `.token` file in the current directory.
/// Returns `None` if the file does not exist or is empty.
pub fn load() -> Option<String> {
    std::fs::read_to_string(".token")
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
}
