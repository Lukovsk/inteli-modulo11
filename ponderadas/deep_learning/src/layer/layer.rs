use crate::Tensor;

pub struct Layer {
    tensors: Vec<Tensor>,
}

impl Layer {
    pub fn new(tensors: Vec<Tensor>) -> Layer {
        Layer { tensors }
    }

    pub fn forward_pass(&self) -> Vec<Tensor> {}

    pub fn backward_pass(&mut self, gradients: Vec<Tensor>) {}

    pub fn update_parameters(&mut self, learning_rate: f32) {}

    pub fn add_tensor(&mut self, tensor: Tensor) {
        self.tensors.push(tensor);
    }

    pub fn get_tensors(&self) -> &[Tensor] {
        &self.tensors
    }
}
