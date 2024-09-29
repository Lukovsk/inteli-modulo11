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
fn test_tensor_cross_product() {
    let mut tensor = Tensor::new(
        3,
        3,
        vec![
            vec![20.0, 24.0, 11.0],
            vec![19.0, 17.0, 20.0],
            vec![21.0, 40.0, 25.0],
        ],
    );

    let tensor_b = Tensor::new(
        3,
        2,
        vec![vec![20.0, 24.0], vec![19.0, 17.0], vec![21.0, 40.0]],
    );

    let result = vec![vec![1087., 1328.], vec![1123., 1545.], vec![1705., 2184.]];

    let new_tensor = tensor.cross_product(tensor_b);

    println!("Results for test_tensor_cross_product:");
    for i in 0..(new_tensor.matrix.len()) {
        println!("{:?}", new_tensor.matrix[i]);
    }
    println!("\n");

    assert_eq!(new_tensor.matrix, result);
}

#[test]
fn test_flattened_tensor_cross_product() {
    let mut tensor = Tensor::new(
        3,
        3,
        vec![
            vec![20.0, 24.0, 11.0],
            vec![19.0, 17.0, 20.0],
            vec![21.0, 40.0, 25.0],
        ],
    );

    let mut tensor_b = Tensor::new(
        3,
        2,
        vec![vec![20.0, 24.0], vec![19.0, 17.0], vec![21.0, 40.0]],
    );

    let mut flattened_tensor = tensor.flatten();
    let flattened_tensor_b = tensor_b.flatten();

    let result = vec![
        vec![1087.0, 1328.0],
        vec![1123.0, 1545.0],
        vec![1705.0, 2184.0],
    ];

    let new_tensor = flattened_tensor.cross_product(flattened_tensor_b);

    println!("Results for test_tensor_cross_product:");
    for i in 0..(new_tensor.matrix.len()) {
        println!("{:?}", new_tensor.matrix[i]);
    }
    println!("\n");

    assert_eq!(new_tensor.matrix, result);
}
