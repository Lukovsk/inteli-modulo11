use crate::Tensor;

use super::Layer;

pub struct DenseLayer {
    weights: Tensor,
    bias: Option<Tensor>,
}

impl DenseLayer {
    pub fn new(weights: Tensor, bias: Option<Tensor>) -> DenseLayer {
        DenseLayer { weights, bias }
    }
}

impl Layer for DenseLayer {
    fn foward(&self, input: &mut Tensor) -> Tensor {
        let mut flat_input = input.flatten_to_column();
        let mut result = flat_input.product(self.weights.clone());

        if let Some(bias_tensor) = &self.bias {
            result = result.apply(|x| x + bias_tensor.matrix[0][0]);
        }

        result
    }
}
