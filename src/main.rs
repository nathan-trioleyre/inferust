use std::path::Path;

use anyhow::Result;

use crate::model::config::Config;

mod json_helper;
mod model;

fn main() -> Result<()> {
    let model_config_path = Path::new("data/SmolLM2-135M/config.json");
    let model_config = json_helper::load::<Config>(model_config_path)?;

    println!("{:?}", model_config);

    Ok(())
}
