use std::{fs, path::Path};

use anyhow::Result;
use serde::de::DeserializeOwned;

pub fn load<T: DeserializeOwned>(path: &Path) -> Result<T> {
    let raw_json = fs::read_to_string(&path)?;
    Ok(serde_json::from_str(&raw_json)?)
}
