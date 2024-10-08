use crate::Tensor;

use super::Layer;

pub struct ConvLayer {
    kernel: Tensor,
}

impl ConvLayer {
    pub fn new(kernel: Tensor) -> ConvLayer {
        ConvLayer { kernel }
    }
}

impl Layer for ConvLayer {
    fn foward(&self, input: &mut Tensor) -> Tensor {
        input.convolution(&self.kernel.matrix)
    }
}
