use mamba_transformer_rust::MATModel; // Assuming your MATModel is in the mat module

#[test]
fn test_full_mat_model() {
    // Device setup
    let device = Device::Cpu;
    
    // Initialize model
    let input_dim = 10;
    let hidden_dim = 64;
    let mut model = MATModel::new(input_dim, hidden_dim, &device).unwrap();
    
    // Create some test input data
    let seq_len = 50;
    let batch_size = 32;
    let x = Tensor::randn(&[seq_len, batch_size, input_dim], &device).unwrap();
    
    // Forward pass
    let output = model.forward(&x).unwrap();
    
    // Check output shape
    assert_eq!(output.shape(), x.shape(), "Output shape should match input shape.");
}
