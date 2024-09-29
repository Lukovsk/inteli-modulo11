pub struct Network {
    layers: Vec<Layer>,
    input_shape: Shape,
    output_shape: Shape,
    learning_rate: f32,
}

impl Network {
    pub fn new(input_shape: Shape, output_shape: Shape, learning_rate: f32) -> Network {
        Network {
            layers: Vec::new(),
            input_shape,
            output_shape,
            learning_rate,
        }
    }
    pub fn add_layer(&mut self, layer: Layer) {
        self.layers.push(layer);
    }

    pub fn forward_pass(&self, input_tensor: &Tensor) -> Tensor {}

    pub fn backward_pass(&mut self, input_tensor: &Tensor, output_gradients: &Tensor) {}
}
