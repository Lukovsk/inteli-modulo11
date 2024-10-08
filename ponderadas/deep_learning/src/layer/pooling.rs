use crate::{Shape, Tensor};

use super::Layer;

pub enum PoolingType {
    Max,
    Average,
}

pub struct PoolingLayer {
    pool_shape: Shape,
    pool_type: PoolingType,
}

impl PoolingLayer {
    pub fn new(pool_shape: Shape, pool_type: PoolingType) -> PoolingLayer {
        PoolingLayer {
            pool_shape,
            pool_type,
        }
    }
}

impl Layer for PoolingLayer {
    fn foward(&self, input: &mut Tensor) -> Tensor {
        match self.pool_type {
            PoolingType::Max => input.max_pooling(self.pool_shape.clone()),
            PoolingType::Average => input.average_pooling(self.pool_shape.clone()),
        }
    }
}
