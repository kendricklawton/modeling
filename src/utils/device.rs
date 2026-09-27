use anyhow::Result;
use candle_core::Device;

/// Helper to automatically select the best hardware acceleration available
pub fn get_optimal_device() -> Result<Device> {
    if candle_core::utils::metal_is_available() {
        println!("Using Metal GPU acceleration!");
        Device::new_metal(0).map_err(Into::into)
    } else if candle_core::utils::cuda_is_available() {
        println!("Using CUDA GPU acceleration!");
        Device::new_cuda(0).map_err(Into::into)
    } else {
        println!("Using CPU.");
        Ok(Device::Cpu)
    }
}
