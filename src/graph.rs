use ash::vk;
use rustc_hash::FxHashMap;
use thunderdome::{Arena, Index};

use anyhow::{anyhow, bail, Result};

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ScalarType {
    U8,
    U16,
    U32,
    I8,
    I16,
    I32,
    F32,
    IVec4,
    UVec4,
    Vec4,
    Dims,
    Color,
}

#[derive(Clone, Copy, PartialEq)]
pub enum Scalar {
    U8(u8),
    U16(u16),
    U32(u32),
    I8(i8),
    I16(i16),
    I32(i32),
    F32(f32),
    IVec4([i32; 4]),
    UVec4([u32; 4]),
    Vec4([f32; 4]),
    Dims { width: u32, height: u32 },
    Color { rgba: [u8; 4] },
}

pub struct GraphInput {
    name: String,
    input_type: ScalarType,
    vertex_index: Index,
}

pub struct GraphFn {
    name: String,
    resource_type: (),
    vertex_index: Index,
}

pub enum BindingDesc {
    Image {
        binding: u32,
        read: bool,
        write: bool,
    },
    Buffer {
        binding: u32,
        read: bool,
        write: bool,
    },
    // SampledImage { binding: u32, sampler: () },
}

pub struct LayoutDesc {
    bindings: Vec<BindingDesc>,
}

pub enum ShaderSlot {
    Image {
        binding: u32,
        read: bool,
        write: bool,
        image_ix: Option<Index>,
    },
    Buffer {
        binding: u32,
        read: bool,
        write: bool,
        image_ix: Option<Index>,
    },
    // Scalar {
    //     bytes: Box<[u8]>,
    // },
}

pub enum CmdVertex {
    CopyImage {
        src_img_ix: Index,
        dst_img_ix: Index,
    },
    ComputeShader {
        pipeline_ix: Index,
        bindings: Vec<ShaderSlot>,
        push_constant_size: u32,
        // push_constant_range: (u32, u32),
        // push_constants: Vec<()>,
    },
}

/*
pub enum GraphInput {
    WindowDims { width: u32, height: u32 },
    MousePos { x: f32, y: f32 },
    PushConstant { bytes: Vec<u8>, size: u32 },
}

pub enum VertexKind {
    GraphInput,
    Commands,
    Batch,
}
*/

/*
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct VxId(Index);

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct CmdId(Index);

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct BatchId(Index);

pub enum VertexData {
    GraphInput(GraphInput),
    // should these be in separate arenas? probably
    Commands { commands: Vec<CmdId> },
    Batches { batches: Vec<BatchId> },
    Image,
}
*/

/*
pub struct RenderGraph {
    vertex_data: Arena<VertexData>,

    images: FxHashMap<VxId, Index>,
    command_data: Arena<CmdVertex>,
}
*/

// pub enum VertexKind {
//     GraphInput { some_scalar_desc: () },
//     CommandBuffer { commands: Vec<CmdSeq> },
// }

/*
pub struct VertexType {
    name: String,

    inputs: Vec<SlotType>,
    input_names: Vec<String>,
}
*/

// pub struct VertexType {
// }

// pub struct RenderGraph {
// }
