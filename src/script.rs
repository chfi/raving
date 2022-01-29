use crossbeam::atomic::AtomicCell;
use parking_lot::{Mutex, RwLock};

use lazy_static::lazy_static;

use rhai::plugin::*;

#[export_module]
pub mod vk {

    use ash::vk;

    #[rhai_mod(name = "ImageLayout")]
    pub mod image_layout {
        macro_rules! layout {
            ($l:ident) => {
                pub const $l: vk::ImageLayout = vk::ImageLayout::$l;
            };
        }

        layout!(UNDEFINED);
        layout!(PREINITIALIZED);
        layout!(GENERAL);
        layout!(TRANSFER_SRC_OPTIMAL);
        layout!(TRANSFER_DST_OPTIMAL);
        layout!(SHADER_READ_ONLY_OPTIMAL);
        layout!(COLOR_ATTACHMENT_OPTIMAL);
        layout!(PRESENT_SRC_KHR);
    }

    #[rhai_mod(name = "DescriptorType")]
    pub mod descriptor_type {
        macro_rules! ty {
            ($t:ident) => {
                pub const $t: vk::DescriptorType = vk::DescriptorType::$t;
            };
        }

        ty!(STORAGE_IMAGE);
        ty!(COMBINED_IMAGE_SAMPLER);
        ty!(SAMPLER);
        ty!(STORAGE_BUFFER);
        ty!(STORAGE_BUFFER_DYNAMIC);
        ty!(STORAGE_TEXEL_BUFFER);
        ty!(UNIFORM_BUFFER);
        ty!(UNIFORM_TEXEL_BUFFER);
        ty!(UNIFORM_BUFFER_DYNAMIC);
        ty!(INPUT_ATTACHMENT);
    }

    #[rhai_mod(name = "ShaderStageFlags")]
    pub mod shader_stage_flags {
        macro_rules! flag {
            ($f:ident) => {
                pub const $f: vk::ShaderStageFlags = vk::ShaderStageFlags::$f;
            };
        }

        flag!(COMPUTE);
        flag!(VERTEX);
        flag!(TESSELLATION_CONTROL);
        flag!(TESSELLATION_EVALUATION);
        flag!(GEOMETRY);
        flag!(FRAGMENT);
        flag!(ALL_GRAPHICS);
        flag!(ALL);
    }

    #[rhai_mod(name = "PipelineStageFlags")]
    pub mod pipeline_stage_flags {
        macro_rules! flag {
            ($f:ident) => {
                pub const $f: vk::PipelineStageFlags =
                    vk::PipelineStageFlags::$f;
            };
        }

        flag!(TOP_OF_PIPE);
        flag!(DRAW_INDIRECT);
        flag!(VERTEX_INPUT);
        flag!(VERTEX_SHADER);
        flag!(TESSELLATION_CONTROL_SHADER);
        flag!(TESSELLATION_EVALUATION_SHADER);
        flag!(GEOMETRY_SHADER);
        flag!(FRAGMENT_SHADER);
        flag!(EARLY_FRAGMENT_TESTS);
        flag!(LATE_FRAGMENT_TESTS);
        flag!(COLOR_ATTACHMENT_OUTPUT);
        flag!(COMPUTE_SHADER);
        flag!(TRANSFER);
        flag!(BOTTOM_OF_PIPE);
        flag!(HOST);
        flag!(ALL_GRAPHICS);
        flag!(ALL_COMMANDS);
    }

    #[rhai_mod(name = "AccessFlags")]
    pub mod access_flags {
        macro_rules! flag {
            ($f:ident) => {
                pub const $f: vk::AccessFlags = vk::AccessFlags::$f;
            };
        }

        flag!(INDIRECT_COMMAND_READ);
        flag!(INDEX_READ);
        flag!(VERTEX_ATTRIBUTE_READ);
        flag!(UNIFORM_READ);
        flag!(INPUT_ATTACHMENT_READ);
        flag!(SHADER_READ);
        flag!(SHADER_WRITE);
        flag!(COLOR_ATTACHMENT_READ);
        flag!(COLOR_ATTACHMENT_WRITE);
        flag!(DEPTH_STENCIL_ATTACHMENT_READ);
        flag!(DEPTH_STENCIL_ATTACHMENT_WRITE);
        flag!(TRANSFER_READ);
        flag!(TRANSFER_WRITE);
        flag!(HOST_READ);
        flag!(HOST_WRITE);
        flag!(MEMORY_READ);
        flag!(MEMORY_WRITE);
    }
}
