/// A point in 2D space.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct XYPoint<T> {
    pub x: T,
    pub y: T,
}

impl<T> XYPoint<T> {
    pub fn new(x: T, y: T) -> Self {
        Self { x, y }
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct XYTPoint<T> {
    pub x: T,
    pub y: T,
    pub t: T,
}

impl<T> XYTPoint<T> {
    pub fn new(x: T, y: T, t: T) -> Self {
        Self { x, y, t }
    }
}
