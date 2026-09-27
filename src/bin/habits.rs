use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use candle_nn::{AdamW, Optimizer, VarBuilder, VarMap, loss};

// Import from our own library!
use modeling::data::dummy::get_habit_data;
use modeling::models::mlp::HabitPredictor;

fn main() -> Result<()> {
    // Note: Tiny models train fast enough on CPU, but you would normally use `modeling::utils::device::get_optimal_device()?`
    let device = Device::Cpu;

    // 1. Data Ingestion
    let (inputs, targets) = get_habit_data(&device)?;

    // 2. Model Initialization
    let varmap = VarMap::new();
    let vs = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let model = HabitPredictor::new(vs)?;

    // 3. Training Loop Setup
    let mut optimizer = AdamW::new_lr(varmap.all_vars(), 0.05)?;

    println!("--- Beginning Training ---");
    for epoch in 1..=200 {
        let predictions = model.forward(&inputs)?;
        let loss = loss::mse(&predictions, &targets)?;

        optimizer.backward_step(&loss)?;

        if epoch % 40 == 0 {
            println!(
                "Epoch {:3} | Error (Loss): {:.4}",
                epoch,
                loss.to_scalar::<f32>()?
            );
        }
    }
    println!("--------------------------\nTraining Complete!\n");

    // 4. Inference / Testing
    let new_sleep = 7.5;
    let new_coffee = 1.0;
    let new_exercise = 20.0;

    let new_day = Tensor::new(&[[new_sleep, new_coffee, new_exercise]], &device)?;
    let predicted_score = model.forward(&new_day)?.to_vec2::<f32>()?[0][0];

    println!("Testing a hypothetical day:");
    println!("  Sleep:    {} hours", new_sleep);
    println!("  Coffee:   {} cups", new_coffee);
    println!("  Exercise: {} mins", new_exercise);
    println!("  >>> Predicted Productivity: {:.2} / 10", predicted_score);

    Ok(())
}
