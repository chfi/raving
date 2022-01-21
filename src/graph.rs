use ash::vk;
use rustc_hash::{FxHashMap, FxHashSet};
use thunderdome::{Arena, Index};

use anyhow::{anyhow, bail, Result};

use crate::vk::SemaphoreIx;

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Vx(Index);

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct GraphInIx(Index);

pub enum VxKind {
    GraphInput,    // gets scalars
    GraphConstant, // produces scalars
    Image,         // any kind, including swapchain
    Compute,       // might become "Pipeline" later
    Command,       // copy images, etc.
    Semaphore,     // kind of an odd one
}

pub enum Edge {
    PipelineBarrier {
        src: Vx,
        dst: Vx,
        layout: Option<(vk::ImageLayout, vk::ImageLayout)>,
    },
    Semaphore {
        wait: Vec<Vx>,
        signal: Vec<Vx>,
        // wait: FxHashSet<SemaphoreIx>,
        // signal: FxHashSet<SemaphoreIx>,
    },
    Identity,
}

pub struct VertexInfo {
    pub name: String,
    pub kind: VxKind,

    pub graph_inputs: Vec<GraphInIx>,
    pub vertex_inputs: Vec<Vx>,
}

pub struct GraphDsl {
    pub graph_inputs: Arena<(String, ScalarType)>,
    pub vertices: Arena<VertexInfo>,

    pub orders: Vec<Vec<Vx>>,

    pub batches: Vec<usize>,
    // edges: Arena<EdgeInfo>,
}

impl GraphDsl {
    pub fn add_vertex(&mut self, name: &str, kind: VxKind) -> Vx {
        let info = VertexInfo {
            name: name.to_string(),
            kind,
            graph_inputs: Vec::new(),
            vertex_inputs: Vec::new(),
        };

        let vx = self.vertices.insert(info);
        Vx(vx)
    }

    pub fn add_graph_input(&mut self, name: &str, ty: ScalarType) -> GraphInIx {
        let gx = self.graph_inputs.insert((name.to_string(), ty));
        GraphInIx(gx)
    }

    // pub fn vx(&self, v: Vx) -> &VertexInfo {
    //     &self.vertices[v
    // }
}

impl std::ops::Index<Vx> for GraphDsl {
    type Output = VertexInfo;

    fn index(&self, vx: Vx) -> &VertexInfo {
        &self.vertices[vx.0]
    }
}

impl std::ops::IndexMut<Vx> for GraphDsl {
    fn index_mut(&mut self, vx: Vx) -> &mut VertexInfo {
        &mut self.vertices[vx.0]
    }
}

pub fn test_graph() -> GraphDsl {
    let mut g = GraphDsl {
        graph_inputs: Arena::new(),
        vertices: Arena::new(),

        orders: Vec::new(),
        batches: Vec::new(),
    };

    let window_size = g.add_graph_input("window_size", ScalarType::Dims);
    let color = g.add_graph_input("color", ScalarType::Color);

    let comp_image = g.add_vertex("comp_image", VxKind::Image);
    g[comp_image].graph_inputs.push(window_size);

    let edge_dims_img = Edge::Identity;

    let compute = g.add_vertex("compute", VxKind::Compute);
    g[compute].graph_inputs.extend([color, window_size]);
    g[compute].vertex_inputs.push(comp_image);

    let edge_color_comp = Edge::Identity;
    let edge_dims_comp = Edge::Identity;

    let edge_img_comp = Edge::PipelineBarrier {
        src: comp_image,
        dst: compute,
        layout: Some((
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            vk::ImageLayout::GENERAL,
        )),
    };

    let swapchain_available =
        g.add_vertex("swapchain_available", VxKind::Semaphore);
    let swapchain_image = g.add_vertex("swapchain_image", VxKind::Image);

    let mut edge_swapchain_avail_img = Edge::Semaphore {
        wait: vec![swapchain_available],
        signal: Vec::new(),
    };

    let copy = g.add_vertex("copy_image", VxKind::Command);
    g[copy].vertex_inputs.extend([comp_image, swapchain_image]);

    let edge_swap_img_copy = Edge::PipelineBarrier {
        src: comp_image,
        dst: swapchain_image,
        layout: Some((
            vk::ImageLayout::GENERAL,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
        )),
    };

    // one alternative
    let order_0 = [compute];
    let order_1 = [swapchain_available, copy];

    // another could be
    // let order_0 = [swapchain_available, compute, copy];

    // these are batch 0 and 1 respectively
    g.orders.push(Vec::from(order_0));
    g.orders.push(Vec::from(order_1));

    g.batches.extend([0, 1]);

    g
}

// pub enum VxInputSlot {
//     Scalar {
//         name: Option<String>,
//         ty: ScalarType,
//     },
//     Image {
//         name: String,
//         binding: u32,
//         read: bool,
//         write: bool,
//         ty: vk::DescriptorType,
//     },
//     Buffer {
//         name: String,
//         binding: u32,
//         read: bool,
//         write: bool,
//         ty: vk::DescriptorType,
//     },
// }

// pub struct VxImage {
//     name: String,
//     format: vk::Format,
//     initial_layout: vk::ImageLayout,
//     inputs: VxInputSlot,
// }

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

/*
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
*/

// pub enum BindingDesc {
//     Image {
//         binding: u32,
//         read: bool,
//         write: bool,
//     },
//     Buffer {
//         binding: u32,
//         read: bool,
//         write: bool,
//     },
//     // SampledImage { binding: u32, sampler: () },
// }

// pub struct LayoutDesc {
//     bindings: Vec<BindingDesc>,
// }

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
