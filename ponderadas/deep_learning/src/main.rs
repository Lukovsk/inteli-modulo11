use deep_learning::{ConvLayer, DenseLayer, Network, PoolingLayer, PoolingType, Shape, Tensor};

fn main() {
    let mut input_tensor: Tensor = Tensor::new(
        6,
        6,
        vec![
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        ],
    );

    let kernel: Tensor = Tensor::new(
        3,
        3,
        vec![
            vec![1.0, 0.0, -1.0],
            vec![1.0, 0.0, -1.0],
            vec![1.0, 0.0, -1.0],
        ],
    );

    let mut network: Network = Network::new();
    network.add_layer(Box::new(ConvLayer::new(kernel)));
    network.add_layer(Box::new(PoolingLayer::new(
        Shape { rows: 2, cols: 2 },
        PoolingType::Average,
    )));
    network.add_layer(Box::new(DenseLayer::new(
        Tensor::new(1, 9, vec![vec![1.0; 9]; 1]),
        None,
    )));

    let output: Tensor = network.forward(&mut input_tensor);

    println!("{:?}", output.matrix)
}

// model.add_layer(Box::new(ConvLayer::new(kernel)));
// model.add_layer(Box::new(PoolLayer::new(Shape { rows: 2, cols: 2 }, PoolingType::Max)));
// model.add_layer(Box::new(DenseLayer::new(Tensor::new(9, 1, vec![vec![1.0]; 9]), None)));
