use thunderdome::{Index};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct MaterialId(pub(crate) Index);

// impl std::ops::Index<MaterialId> for
