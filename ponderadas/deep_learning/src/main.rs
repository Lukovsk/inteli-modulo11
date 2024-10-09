use core::f32;
use deep_learning::{ConvLayer, Network, PoolingLayer, PoolingType, Shape, Tensor};
use image::{GrayImage, ImageFormat, Luma, Pixel};
use std::{fs::File, path::Path};

fn main() {
    // let mut input_tensor: Tensor = Tensor::new(
    //     6,
    //     6,
    //     vec![
    //         vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    //         vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    //         vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    //         vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    //         vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    //         vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    //     ],
    // );

    let mut input_tensor = image_to_tensor("src/imagem.png");

    let kernel: Tensor = Tensor::new(
        3,
        3,
        vec![
            vec![127.0, -123.0, 519.0],
            vec![1.0, 0.0, -1.0],
            vec![1.0, 0.0, -1.0],
        ],
    );

    let mut network: Network = Network::new();
    network.add_layer(Box::new(PoolingLayer::new(
        Shape { rows: 3, cols: 7 },
        PoolingType::Average,
    )));
    network.add_layer(Box::new(ConvLayer::new(kernel)));
    network.add_layer(Box::new(PoolingLayer::new(
        Shape { rows: 2, cols: 2 },
        PoolingType::Average,
    )));
    // network.add_layer(Box::new(DenseLayer::new(
    //     Tensor::new(1, 9, vec![vec![1.0; 9]; 1]),
    //     None,
    // )));

    let output: Tensor = network.forward(&mut input_tensor);

    let normalized_output = normalize_tensor(&output);

    println!("normalized output rows: {:?}", normalized_output.shape.rows);
    println!("normalized output cols: {:?}", normalized_output.shape.cols);

    tensor_to_image(&normalized_output, "output/imagem2.png");

    // println!("{:?}", output.matrix)
}

fn image_to_tensor(path: &str) -> Tensor {
    let img = image::open(&Path::new(path)).unwrap().to_luma8();

    let (width, height) = img.dimensions();

    println!("image width: {}", width as usize);
    println!("image height: {}", height as usize);

    let mut matrix = vec![vec![0.0; width as usize]; height as usize];

    for (x, y, pixel) in img.enumerate_pixels() {
        let pixel_value = pixel.channels()[0];
        matrix[y as usize][x as usize] = pixel_value as f32 / 255.0;
    }

    Tensor::new(height as usize, width as usize, matrix)
}

fn normalize_tensor(tensor: &Tensor) -> Tensor {
    let max_value = tensor
        .matrix
        .iter()
        .flatten()
        .cloned()
        .fold(f32::NEG_INFINITY, f32::max);
    let min_value = tensor
        .matrix
        .iter()
        .flatten()
        .cloned()
        .fold(f32::INFINITY, f32::min);

    let range = max_value - min_value;

    let normalized_matrix: Vec<Vec<f32>> = tensor
        .matrix
        .iter()
        .map(|row| {
            row.iter()
                .map(|&val| ((val - min_value) / range * 255.0).min(255.0).max(0.0))
                .collect()
        })
        .collect();

    Tensor::new(tensor.shape.rows, tensor.shape.cols, normalized_matrix)
}

fn tensor_to_image(tensor: &Tensor, output_path: &str) {
    let rows = tensor.shape.rows;
    let cols = tensor.shape.cols;

    let mut img: GrayImage = GrayImage::new(cols as u32, rows as u32);

    for row in 0..rows {
        for col in 0..cols {
            let pixel_value = tensor.matrix[row][col] as u8;
            img.put_pixel(col as u32, row as u32, Luma([pixel_value]));
        }
    }

    let path = Path::new(output_path);
    let mut file = File::create(path).expect("Falha ao criar o arquivo");
    img.write_to(&mut file, ImageFormat::Png)
        .expect("Falha ao salvar a imagem");
}
