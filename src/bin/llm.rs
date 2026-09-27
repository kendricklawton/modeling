use anyhow::{Error as E, Result};
use candle_core::{DType, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::generation::LogitsProcessor;
use candle_transformers::models::llama::{Cache, Llama, LlamaConfig};
use hf_hub::{Repo, RepoType, api::sync::Api};
use std::io::Write;
use tokenizers::Tokenizer;

// Import our beautiful new library!
use modeling::utils::device::get_optimal_device;

fn main() -> Result<()> {
    let device = get_optimal_device()?;

    println!("Downloading/loading model weights and tokenizer...");
    let api = Api::new()?;
    let repo = api.repo(Repo::with_revision(
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0".to_string(),
        RepoType::Model,
        "main".to_string(),
    ));

    let tokenizer_filename = repo.get("tokenizer.json")?;
    let config_filename = repo.get("config.json")?;
    let weights_filename = repo.get("model.safetensors")?;

    println!("Loading config and tokenizer...");
    let config_parsed: LlamaConfig =
        serde_json::from_reader(std::fs::File::open(config_filename)?)?;
    let config = config_parsed.into_config(false);
    let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;

    println!("Loading weights into model...");
    let vb =
        unsafe { VarBuilder::from_mmaped_safetensors(&[weights_filename], DType::F32, &device)? };
    let model = Llama::load(vb, &config)?;

    let mut cache = Cache::new(true, DType::F32, &config, &device)?;

    let prompt = "<|system|>\nYou are a helpful assistant.\n<|user|>\nWhat are the benefits of using Rust for machine learning?\n<|assistant|>\n";
    println!("Prompt:\n{}", prompt);
    println!("--- Generating Response ---");

    let tokens = tokenizer.encode(prompt, true).map_err(E::msg)?;
    let mut tokens = tokens.get_ids().to_vec();

    let mut logits_processor = LogitsProcessor::new(299792458, None, None);
    let mut index_pos = 0;

    for index in 0..200 {
        let context_size = if index > 0 { 1 } else { tokens.len() };
        let input = Tensor::new(&tokens[tokens.len() - context_size..], &device)?.unsqueeze(0)?;

        let logits = model.forward(&input, index_pos, &mut cache)?;

        let logits = logits.squeeze(0)?;
        let logits = logits.get(logits.dim(0)? - 1)?;

        let next_token = logits_processor.sample(&logits)?;
        tokens.push(next_token);
        index_pos += context_size;

        if let Some(text) = tokenizer.id_to_token(next_token) {
            let text = text.replace(' ', " "); // Handle SPM token spaces
            print!("{}", text);
            std::io::stdout().flush()?;
        }

        if next_token == 2 || next_token == tokenizer.token_to_id("</s>").unwrap_or(2) {
            break;
        }
    }
    println!("\n---------------------------");

    Ok(())
}
