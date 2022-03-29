use bytemuck::{Pod, Zeroable};
use zerocopy::{AsBytes, FromBytes};

use ash::vk;

#[derive(Clone, Copy, Zeroable, Pod, AsBytes, FromBytes)]
#[repr(C)]
pub struct Vx2D {
    pub position: [f32; 2],
}

impl Vx2D {
    pub fn get_binding_desc() -> vk::VertexInputBindingDescription {
        vk::VertexInputBindingDescription::builder()
            .binding(0)
            .stride(std::mem::size_of::<Self>() as u32)
            .input_rate(vk::VertexInputRate::VERTEX)
            .build()
    }

    pub fn get_attribute_descs() -> [vk::VertexInputAttributeDescription; 1] {
        let pos_desc = vk::VertexInputAttributeDescription::builder()
            .binding(0)
            .location(0)
            .format(vk::Format::R32G32_SFLOAT)
            .offset(0)
            .build();

        [pos_desc]
    }
}

#[derive(Clone, Copy, Zeroable, Pod, AsBytes, FromBytes)]
#[repr(C)]
pub struct Vx3D {
    pub position: [f32; 3],
}

impl Vx3D {
    fn get_binding_desc() -> vk::VertexInputBindingDescription {
        vk::VertexInputBindingDescription::builder()
            .binding(0)
            .stride(std::mem::size_of::<Self>() as u32)
            .input_rate(vk::VertexInputRate::VERTEX)
            .build()
    }

    fn get_attribute_descs() -> [vk::VertexInputAttributeDescription; 1] {
        let pos_desc = vk::VertexInputAttributeDescription::builder()
            .binding(0)
            .location(0)
            .format(vk::Format::R32G32B32_SFLOAT)
            .offset(0)
            .build();

        [pos_desc]
    }
}
