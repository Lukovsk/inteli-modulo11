use crate::Tensor;

pub trait Layer {
    fn foward(&self, input: &mut Tensor) -> Tensor;
}
