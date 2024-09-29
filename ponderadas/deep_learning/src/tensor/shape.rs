pub struct Shape {
    pub rows: usize,
    pub cols: usize,
}

impl Shape {
    pub fn new(rows: usize, cols: usize) -> Shape {
        Shape { rows, cols }
    }

    pub fn get_size(&self) -> usize {
        self.rows * self.cols
    }
}
