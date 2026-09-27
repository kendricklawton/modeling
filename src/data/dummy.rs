use anyhow::Result;
use candle_core::{Device, Tensor};

/// Returns (Inputs, Targets) mapping habits to a simulated productivity score
pub fn get_habit_data(device: &Device) -> Result<(Tensor, Tensor)> {
    // Each row is a different day: [Hours of Sleep, Cups of Coffee, Mins of Exercise]
    let inputs = Tensor::new(
        &[
            [8.0f32, 1.0, 30.0], // Day 1
            [5.0, 3.0, 0.0],     // Day 2
            [7.0, 2.0, 45.0],    // Day 3
            [6.0, 2.0, 15.0],    // Day 4
            [9.0, 0.0, 60.0],    // Day 5
        ],
        device,
    )?;

    // [Productivity Score out of 10]
    let targets = Tensor::new(&[[8.5f32], [4.0], [8.0], [6.0], [9.5]], device)?;

    Ok((inputs, targets))
}
