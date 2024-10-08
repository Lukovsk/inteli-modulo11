use crate::{layer::Layer, Tensor};

pub struct Network {
    layers: Vec<Box<dyn Layer>>,
}

impl Network {
    pub fn new() -> Network {
        Network { layers: Vec::new() }
    }

    pub fn add_layer(&mut self, layer: Box<dyn Layer>) {
        self.layers.push(layer);
    }

    pub fn forward(&mut self, input: &mut Tensor) -> Tensor {
        let mut output: Tensor = input.clone();
        for layer in &self.layers {
            output = layer.foward(&mut output);
        }

        output
    }
}
