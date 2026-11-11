use crate::features::ReleaseFeatures;
use serde_json;
use std::path::Path;

/// SemVer bump types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ReleaseType {
    Major,
    Minor,
    Patch,
}

impl std::fmt::Display for ReleaseType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ReleaseType::Major => write!(f, "major"),
            ReleaseType::Minor => write!(f, "minor"),
            ReleaseType::Patch => write!(f, "patch"),
        }
    }
}

/// One labelled release observation.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ReleaseRecord {
    pub package: String,
    pub version: String,
    pub release_type: ReleaseType,
    #[serde(flatten)]
    pub features: ReleaseFeatures,
}

// CSV I/O 
//
// The csv crate does not support #[serde(flatten)] directly. 
// So we need to use serde_json for this
// ReleaseRecord serializes to a flat JSON object whose keys become CSV
// columns

/// Serialize a ReleaseRecord to a flat JSON object via serde_json.
/// `#[serde(flatten)]` is handled by serde_json natively.
fn to_json_map(
    r: &ReleaseRecord,
) -> Result<serde_json::Map<String, serde_json::Value>, crate::Error> {
    match serde_json::to_value(r).map_err(|e| crate::Error::Io(e.to_string()))? {
        serde_json::Value::Object(map) => Ok(map),
        _ => Err(crate::Error::Io("expected JSON object".into())),
    }
}

/// Write records to a CSV file.
pub fn write_csv<P: AsRef<Path>>(path: P, records: &[ReleaseRecord]) -> Result<(), crate::Error> {
    if records.is_empty() {
        return Ok(());
    }

    let mut wtr = csv::WriterBuilder::new()
        .from_path(path)
        .map_err(|e| crate::Error::Io(e.to_string()))?;

    // Derive the header from the first record. `serde_json::Map` is a
    // `BTreeMap` here (the `preserve_order` feature is not enabled), so the
    // keys come out sorted ALPHABETICALLY by field name — that is the CSV
    // column order, independent of struct-field declaration order. Every row
    // below is emitted against this same key ordering, so rows stay aligned.
    let first_map = to_json_map(&records[0])?;
    let headers: Vec<String> = first_map.keys().cloned().collect();
    wtr.write_record(&headers)
        .map_err(|e| crate::Error::Io(e.to_string()))?;

    for r in records {
        let map = to_json_map(r)?;
        let row: Vec<String> = headers
            .iter()
            .map(|k| match map.get(k) {
                Some(serde_json::Value::String(s)) => s.clone(),
                Some(v) => v.to_string(),
                None => String::new(),
            })
            .collect();
        wtr.write_record(&row)
            .map_err(|e| crate::Error::Io(e.to_string()))?;
    }
    wtr.flush().map_err(|e| crate::Error::Io(e.to_string()))?;
    Ok(())
}

