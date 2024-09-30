use crate::{Shape, Tensor};

#[test]
fn test_tensor_convolutional() {
    let mut tensor = Tensor::new(
        6,
        6,
        vec![
            vec![20.0, 24.0, 11.0, 12.0, 16.0, 19.0],
            vec![19.0, 17.0, 20.0, 23.0, 15.0, 9.0],
            vec![21.0, 40.0, 25.0, 13.0, 14.0, 8.0],
            vec![9.0, 18.0, 8.0, 6.0, 11.0, 22.0],
            vec![31.0, 3.0, 7.0, 9.0, 17.0, 23.0],
            vec![20.0, 12.0, 3.0, 11.0, 19.0, 30.0],
        ],
    );

    let kernel = vec![
        vec![1.0, 0.0, -1.0],
        vec![2.0, 0.0, -2.0],
        vec![1.0, 0.0, -1.0],
    ];

    let new_tensor = tensor.convolution(&kernel);

    let expected_result = vec![
        vec![3.0, 27.0, 16.0, 26.0],
        vec![-8.0, 60.0, 24.0, 8.0],
        vec![22.0, 45.0, -5.0, -41.0],
        vec![66.0, 1.0, -39.0, -63.0],
    ];

    println!("Results for test_tensor_convolutional:");
    for row in &new_tensor.matrix {
        println!("{:?}", row);
    }
    println!("\n");

    assert_eq!(new_tensor.matrix, expected_result);
}

#[test]
fn test_tensor_average_pooling() {
    let mut tensor = Tensor::new(
        6,
        6,
        vec![
            vec![20.0, 24.0, 11.0, 12.0, 16.0, 19.0],
            vec![19.0, 17.0, 20.0, 23.0, 15.0, 9.0],
            vec![21.0, 40.0, 25.0, 13.0, 14.0, 8.0],
            vec![9.0, 18.0, 8.0, 6.0, 11.0, 22.0],
            vec![31.0, 3.0, 7.0, 9.0, 17.0, 23.0],
            vec![20.0, 12.0, 3.0, 11.0, 19.0, 30.0],
        ],
    );

    let expected_result = vec![
        vec![21.88888888888889, 14.333333333333334],
        vec![12.333333333333334, 16.444444444444443],
    ];

    let new_tensor = tensor.average_pooling(Shape { rows: 3, cols: 3 });

    println!("Results for test_tensor_average_pooling:");
    for row in &new_tensor.matrix {
        println!("{:?}", row);
    }
    println!("\n");

    assert_eq!(new_tensor.matrix, expected_result);
}
#[test]
fn test_tensor_average_pooling_2() {
    let mut tensor = Tensor::new(
        2,
        6,
        vec![
            vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ],
    );

    let new_tensor = tensor.average_pooling(Shape { rows: 2, cols: 2 });

    let expected_result = vec![vec![0.0, 0.0, 0.0]];

    assert_eq!(new_tensor.matrix, expected_result);
}

#[test]
fn test_tensor_max_pooling() {
    let mut tensor = Tensor::new(
        6,
        6,
        vec![
            vec![20.0, 24.0, 11.0, 12.0, 16.0, 19.0],
            vec![19.0, 17.0, 20.0, 23.0, 15.0, 9.0],
            vec![21.0, 40.0, 25.0, 13.0, 14.0, 8.0],
            vec![9.0, 18.0, 8.0, 6.0, 11.0, 22.0],
            vec![31.0, 3.0, 7.0, 9.0, 17.0, 23.0],
            vec![20.0, 12.0, 3.0, 11.0, 19.0, 30.0],
        ],
    );

    let expected_result = vec![vec![40.0, 23.0], vec![31.0, 30.0]];

    let new_tensor = tensor.max_pooling(Shape { rows: 3, cols: 3 });

    println!("Results for test_tensor_max_pooling:");
    for row in &new_tensor.matrix {
        println!("{:?}", row);
    }
    println!("\n");

    assert_eq!(new_tensor.matrix, expected_result);
}

#[test]
fn test_tensor_max_pooling_2() {
    let mut tensor = Tensor::new(
        2,
        6,
        vec![
            vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ],
    );

    let new_tensor = tensor.max_pooling(Shape { rows: 2, cols: 2 });

    let expected_result = vec![vec![0.0, 0.0, 0.0]];

    assert_eq!(new_tensor.matrix, expected_result);
}

#[test]
fn test_flattened_tensor_product() {
    let mut tensor = Tensor::new(
        3,
        4,
        vec![
            vec![20.0, 24.0, 11.0, 11.0],
            vec![19.0, 17.0, 20.0, 11.0],
            vec![21.0, 40.0, 25.0, 11.0],
        ],
    );

    let mut tensor_b = Tensor::new(
        4,
        3,
        vec![
            vec![20.0, 24.0, 11.0],
            vec![19.0, 17.0, 11.0],
            vec![21.0, 40.0, 11.0],
            vec![21.0, 40.0, 11.0],
        ],
    );

    let mut flattened_tensor = tensor.flatten_to_column();
    let flattened_tensor_b = tensor_b.flatten();

    let result = vec![
        vec![
            400.0, 480.0, 220.0, 380.0, 340.0, 220.0, 420.0, 800.0, 220.0, 420.0, 800.0, 220.0,
        ],
        vec![
            380.0, 456.0, 209.0, 361.0, 323.0, 209.0, 399.0, 760.0, 209.0, 399.0, 760.0, 209.0,
        ],
        vec![
            420.0, 504.0, 231.0, 399.0, 357.0, 231.0, 441.0, 840.0, 231.0, 441.0, 840.0, 231.0,
        ],
        vec![
            480.0, 576.0, 264.0, 456.0, 408.0, 264.0, 504.0, 960.0, 264.0, 504.0, 960.0, 264.0,
        ],
        vec![
            340.0, 408.0, 187.0, 323.0, 289.0, 187.0, 357.0, 680.0, 187.0, 357.0, 680.0, 187.0,
        ],
        vec![
            800.0, 960.0, 440.0, 760.0, 680.0, 440.0, 840.0, 1600.0, 440.0, 840.0, 1600.0, 440.0,
        ],
        vec![
            220.0, 264.0, 121.0, 209.0, 187.0, 121.0, 231.0, 440.0, 121.0, 231.0, 440.0, 121.0,
        ],
        vec![
            400.0, 480.0, 220.0, 380.0, 340.0, 220.0, 420.0, 800.0, 220.0, 420.0, 800.0, 220.0,
        ],
        vec![
            500.0, 600.0, 275.0, 475.0, 425.0, 275.0, 525.0, 1000.0, 275.0, 525.0, 1000.0, 275.0,
        ],
        vec![
            220.0, 264.0, 121.0, 209.0, 187.0, 121.0, 231.0, 440.0, 121.0, 231.0, 440.0, 121.0,
        ],
        vec![
            220.0, 264.0, 121.0, 209.0, 187.0, 121.0, 231.0, 440.0, 121.0, 231.0, 440.0, 121.0,
        ],
        vec![
            220.0, 264.0, 121.0, 209.0, 187.0, 121.0, 231.0, 440.0, 121.0, 231.0, 440.0, 121.0,
        ],
    ];

    let new_tensor = flattened_tensor.product(flattened_tensor_b);

    println!("Results for test_tensor_cross_product:");
    for i in 0..(new_tensor.matrix.len()) {
        println!("{:?}", new_tensor.matrix[i]);
    }
    println!("\n");

    assert_eq!(new_tensor.matrix, result);
}

#[test]
fn test_tensor_apply_function() {
    let mut tensor = Tensor::new(
        6,
        6,
        vec![
            vec![20.0, 24.0, 11.0, 12.0, 16.0, 19.0],
            vec![19.0, 17.0, 20.0, 23.0, 15.0, 9.0],
            vec![21.0, 40.0, 25.0, 13.0, 14.0, 8.0],
            vec![9.0, 18.0, 8.0, 6.0, 11.0, 22.0],
            vec![31.0, 3.0, 7.0, 9.0, 17.0, 23.0],
            vec![20.0, 12.0, 3.0, 11.0, 19.0, 30.0],
        ],
    );

    let new_tensor = tensor.apply(|val| -> f32 { val * val * val * val });

    let result = vec![
        vec![160000.0, 331776.0, 14641.0, 20736.0, 65536.0, 130321.0],
        vec![130321.0, 83521.0, 160000.0, 279841.0, 50625.0, 6561.0],
        vec![194481.0, 2560000.0, 390625.0, 28561.0, 38416.0, 4096.0],
        vec![6561.0, 104976.0, 4096.0, 1296.0, 14641.0, 234256.0],
        vec![923521.0, 81.0, 2401.0, 6561.0, 83521.0, 279841.0],
        vec![160000.0, 20736.0, 81.0, 14641.0, 130321.0, 810000.0],
    ];

    println!();
    for i in 0..(new_tensor.matrix.len()) {
        println!("{:?}", new_tensor.matrix[i]);
    }

    assert_eq!(new_tensor.matrix, result)
}
