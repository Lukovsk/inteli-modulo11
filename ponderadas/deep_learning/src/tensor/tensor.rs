use std::f32::INFINITY;

use super::Shape;

pub struct Tensor {
    pub matrix: Vec<Vec<f32>>,
    shape: Shape,
}

impl Tensor {
    pub fn new(rows: usize, cols: usize, matrix: Vec<Vec<f32>>) -> Tensor {
        // Verifica se a matriz tem as dimensões corretas
        if matrix.len() != rows || matrix[0].len() != cols {
            panic!(
                "Invalid matrix dimensions: expected {}x{}, got {}x{}",
                rows,
                cols,
                matrix.len(),
                matrix[0].len()
            );
        }

        Tensor {
            matrix: matrix,
            shape: Shape { rows, cols },
        }
    }

    pub fn convolution(&mut self, kernel: &Vec<Vec<f32>>) -> Tensor {
        // Verificação das dimensões do kernel
        if self.shape.rows < kernel.len() || self.shape.cols < kernel[0].len() {
            panic!(
                "Invalid matrix dimensions for convolution: matrix is {}x{}, kernel is {}x{}",
                self.shape.rows,
                self.shape.cols,
                kernel.len(),
                kernel[0].len()
            );
        }

        // Dimensões do novo tensor após a convolução
        let rows = self.shape.rows - kernel.len() + 1;
        let cols = self.shape.cols - kernel[0].len() + 1;

        let mut new_tensor = Tensor::new(rows, cols, vec![vec![0.0; cols]; rows]);

        for row in 0..rows {
            for col in 0..cols {
                let mut sum: f32 = 0.0;
                for i in 0..kernel.len() {
                    for j in 0..kernel[0].len() {
                        sum += self.matrix[row + i][col + j] * kernel[i][j];
                    }
                }
                new_tensor.matrix[row][col] = sum;
            }
        }

        new_tensor
    }

    pub fn average_pooling(&mut self, shape: Shape) -> Tensor {
        // Verifica se o pooling pode ser aplicado
        if self.shape.rows % shape.rows != 0 || self.shape.cols % shape.cols != 0 {
            panic!("Invalid pooling size");
        }

        let rows = self.shape.rows / shape.rows;
        let cols = self.shape.cols / shape.cols;

        let mut new_tensor = Tensor::new(rows, cols, vec![vec![0.0; cols]; rows]);

        for row in 0..rows {
            for col in 0..cols {
                let mut sum: f32 = 0.0;
                for i in 0..shape.rows {
                    for j in 0..shape.cols {
                        sum += self.matrix[row * shape.rows + i][col * shape.cols + j];
                    }
                }
                new_tensor.matrix[row][col] = sum / (shape.get_size()) as f32;
            }
        }

        new_tensor
    }

    pub fn max_pooling(&mut self, shape: Shape) -> Tensor {
        // Verifica se o pooling pode ser aplicado
        if self.shape.rows % shape.rows != 0 || self.shape.cols % shape.cols != 0 {
            panic!("Invalid pooling size");
        }

        let rows = self.shape.rows / shape.rows;
        let cols = self.shape.cols / shape.cols;

        let mut new_tensor = Tensor::new(rows, cols, vec![vec![0.0; cols]; rows]);

        for row in 0..rows {
            for col in 0..cols {
                let mut max = -INFINITY;
                for i in 0..shape.rows {
                    for j in 0..shape.cols {
                        let value = self.matrix[row * shape.rows + i][col * shape.cols + j];
                        if value > max {
                            max = value;
                        }
                    }
                }
                new_tensor.matrix[row][col] = max;
            }
        }

        new_tensor
    }

    pub fn flatten(&mut self) -> Tensor {
        let mut flatten_vector = Tensor::new(
            1,
            self.shape.get_size(),
            vec![vec![0.0; self.shape.get_size()]; 1],
        );
        let mut counter = 0;
        for row in 0..self.shape.rows {
            for col in 0..self.shape.cols {
                flatten_vector.matrix[0][counter] = self.matrix[row][col];
                counter += 1;
            }
        }
        flatten_vector
    }

    pub fn dot_product(&mut self, b: Tensor) -> f32 {
        if self.shape.cols != b.shape.rows {
            panic!(
                "Invalid matrix dimensions for dot product: {}x{} and {}x{}",
                self.shape.rows, self.shape.cols, b.shape.rows, b.shape.cols
            );
        }

        let mut sum: f32 = 0.0;
        for i in 0..self.shape.cols {
            for j in 0..self.shape.rows {
                sum += self.matrix[j][i] * b.matrix[j][i];
            }
        }

        sum
    }

    pub fn cross_product(&mut self, b: Tensor) -> Tensor {
        b
    }

    pub fn product(&mut self, b: Tensor) -> Tensor {
        if self.shape.cols != b.shape.rows {
            panic!(
                "Invalid matrix dimensions for cross product: {}x{} and {}x{}",
                self.shape.rows, self.shape.cols, b.shape.rows, b.shape.cols
            );
        }

        let mut new_tensor = Tensor::new(
            self.shape.rows,
            b.shape.cols,
            vec![vec![0.0; b.shape.cols]; self.shape.rows],
        );

        for row in 0..self.shape.rows {
            for col in 0..b.shape.cols {
                let mut sum = 0.0;
                for k in 0..self.shape.cols {
                    sum += self.matrix[row][k] * b.matrix[k][col];
                }
                new_tensor.matrix[row][col] = sum;
            }
        }

        new_tensor
    }
}

// let mut sum: f32 = 0.0;
//                 for i in 0..kernel.len() {
//                     for j in 0..kernel[0].len() {
//                         sum += self.matrix[row + i][col + j] * kernel[i][j];
//                     }
//                 }
//                 new_tensor.matrix[row][col] = sum;
