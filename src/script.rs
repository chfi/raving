use crossbeam::atomic::AtomicCell;
use parking_lot::{Mutex, RwLock};

use lazy_static::lazy_static;

use crate::vk::resource::index::*;

use rhai::plugin::*;

pub mod console;

#[export_module]
pub mod vk {

    use ash::vk;

    pub mod pipeline {
        use crate::vk::descriptor::BindingDesc;

        pub struct ComputePipeline {
            name: String,
            src_path: String,

            pipeline: PipelineIx,

            bindings: Vec<BindingDesc>,
            pc_size: u64,
        }
    }

    pub mod binding {
        use crate::vk::descriptor::BindingDesc;

        macro_rules! mk_binding {
            ($name:ident, $v:ident) => {
                pub fn $name(binding: i64) -> BindingDesc {
                    BindingDesc::$v {
                        binding: binding as u32,
                    }
                }
            };
        }

        mk_binding!(storage_image, StorageImage);
        mk_binding!(sampled_image, SampledImage);
        mk_binding!(storage_buffer, StorageBuffer);
        mk_binding!(uniform_buffer, UniformBuffer);
    }

    pub mod cmd {

        //
    }

    #[rhai_mod(name = "BufferUsageFlags")]
    pub mod buffer_usage {
        macro_rules! flag {
            ($f:ident) => {
                pub const $f: vk::BufferUsageFlags = vk::BufferUsageFlags::$f;
            };
        }

        flag!(TRANSFER_SRC);
        flag!(TRANSFER_DST);
        flag!(UNIFORM_TEXEL_BUFFER);
        flag!(STORAGE_TEXEL_BUFFER);
        flag!(UNIFORM_BUFFER);
        flag!(STORAGE_BUFFER);
        flag!(INDEX_BUFFER);
        flag!(VERTEX_BUFFER);
        flag!(INDIRECT_BUFFER);

        #[rhai_fn(name = "|")]
        pub fn or(
            f: &mut vk::BufferUsageFlags,
            g: vk::BufferUsageFlags,
        ) -> vk::BufferUsageFlags {
            *f | g
        }

        #[rhai_fn(name = "&")]
        pub fn and(
            f: &mut vk::BufferUsageFlags,
            g: vk::BufferUsageFlags,
        ) -> vk::BufferUsageFlags {
            *f & g
        }
    }

    #[rhai_mod(name = "ImageUsageFlags")]
    pub mod image_usage {
        macro_rules! flag {
            ($f:ident) => {
                pub const $f: vk::ImageUsageFlags = vk::ImageUsageFlags::$f;
            };
        }

        flag!(TRANSFER_SRC);
        flag!(TRANSFER_DST);
        flag!(SAMPLED);
        flag!(STORAGE);
        flag!(COLOR_ATTACHMENT);
        flag!(DEPTH_STENCIL_ATTACHMENT);
        flag!(TRANSIENT_ATTACHMENT);
        flag!(INPUT_ATTACHMENT);

        #[rhai_fn(name = "|")]
        pub fn or(
            f: &mut vk::ImageUsageFlags,
            g: vk::ImageUsageFlags,
        ) -> vk::ImageUsageFlags {
            *f | g
        }

        #[rhai_fn(name = "&")]
        pub fn and(
            f: &mut vk::ImageUsageFlags,
            g: vk::ImageUsageFlags,
        ) -> vk::ImageUsageFlags {
            *f & g
        }
    }

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

    #[rhai_mod(name = "Format")]
    pub mod format {
        macro_rules! fmt {
            ($f:ident) => {
                pub const $f: vk::Format = vk::Format::$f;
            };
        }

        fmt!(UNDEFINED);
        fmt!(R4G4_UNORM_PACK8);
        fmt!(R4G4B4A4_UNORM_PACK16);
        fmt!(B4G4R4A4_UNORM_PACK16);
        fmt!(R5G6B5_UNORM_PACK16);
        fmt!(B5G6R5_UNORM_PACK16);
        fmt!(R5G5B5A1_UNORM_PACK16);
        fmt!(B5G5R5A1_UNORM_PACK16);
        fmt!(A1R5G5B5_UNORM_PACK16);
        fmt!(R8_UNORM);
        fmt!(R8_SNORM);
        fmt!(R8_USCALED);
        fmt!(R8_SSCALED);
        fmt!(R8_UINT);
        fmt!(R8_SINT);
        fmt!(R8_SRGB);
        fmt!(R8G8_UNORM);
        fmt!(R8G8_SNORM);
        fmt!(R8G8_USCALED);
        fmt!(R8G8_SSCALED);
        fmt!(R8G8_UINT);
        fmt!(R8G8_SINT);
        fmt!(R8G8_SRGB);
        fmt!(R8G8B8_UNORM);
        fmt!(R8G8B8_SNORM);
        fmt!(R8G8B8_USCALED);
        fmt!(R8G8B8_SSCALED);
        fmt!(R8G8B8_UINT);
        fmt!(R8G8B8_SINT);
        fmt!(R8G8B8_SRGB);
        fmt!(B8G8R8_UNORM);
        fmt!(B8G8R8_SNORM);
        fmt!(B8G8R8_USCALED);
        fmt!(B8G8R8_SSCALED);
        fmt!(B8G8R8_UINT);
        fmt!(B8G8R8_SINT);
        fmt!(B8G8R8_SRGB);
        fmt!(R8G8B8A8_UNORM);
        fmt!(R8G8B8A8_SNORM);
        fmt!(R8G8B8A8_USCALED);
        fmt!(R8G8B8A8_SSCALED);
        fmt!(R8G8B8A8_UINT);
        fmt!(R8G8B8A8_SINT);
        fmt!(R8G8B8A8_SRGB);
        fmt!(B8G8R8A8_UNORM);
        fmt!(B8G8R8A8_SNORM);
        fmt!(B8G8R8A8_USCALED);
        fmt!(B8G8R8A8_SSCALED);
        fmt!(B8G8R8A8_UINT);
        fmt!(B8G8R8A8_SINT);
        fmt!(B8G8R8A8_SRGB);
        fmt!(A8B8G8R8_UNORM_PACK32);
        fmt!(A8B8G8R8_SNORM_PACK32);
        fmt!(A8B8G8R8_USCALED_PACK32);
        fmt!(A8B8G8R8_SSCALED_PACK32);
        fmt!(A8B8G8R8_UINT_PACK32);
        fmt!(A8B8G8R8_SINT_PACK32);
        fmt!(A8B8G8R8_SRGB_PACK32);
        fmt!(A2R10G10B10_UNORM_PACK32);
        fmt!(A2R10G10B10_SNORM_PACK32);
        fmt!(A2R10G10B10_USCALED_PACK32);
        fmt!(A2R10G10B10_SSCALED_PACK32);
        fmt!(A2R10G10B10_UINT_PACK32);
        fmt!(A2R10G10B10_SINT_PACK32);
        fmt!(A2B10G10R10_UNORM_PACK32);
        fmt!(A2B10G10R10_SNORM_PACK32);
        fmt!(A2B10G10R10_USCALED_PACK32);
        fmt!(A2B10G10R10_SSCALED_PACK32);
        fmt!(A2B10G10R10_UINT_PACK32);
        fmt!(A2B10G10R10_SINT_PACK32);
        fmt!(R16_UNORM);
        fmt!(R16_SNORM);
        fmt!(R16_USCALED);
        fmt!(R16_SSCALED);
        fmt!(R16_UINT);
        fmt!(R16_SINT);
        fmt!(R16_SFLOAT);
        fmt!(R16G16_UNORM);
        fmt!(R16G16_SNORM);
        fmt!(R16G16_USCALED);
        fmt!(R16G16_SSCALED);
        fmt!(R16G16_UINT);
        fmt!(R16G16_SINT);
        fmt!(R16G16_SFLOAT);
        fmt!(R16G16B16_UNORM);
        fmt!(R16G16B16_SNORM);
        fmt!(R16G16B16_USCALED);
        fmt!(R16G16B16_SSCALED);
        fmt!(R16G16B16_UINT);
        fmt!(R16G16B16_SINT);
        fmt!(R16G16B16_SFLOAT);
        fmt!(R16G16B16A16_UNORM);
        fmt!(R16G16B16A16_SNORM);
        fmt!(R16G16B16A16_USCALED);
        fmt!(R16G16B16A16_SSCALED);
        fmt!(R16G16B16A16_UINT);
        fmt!(R16G16B16A16_SINT);
        fmt!(R16G16B16A16_SFLOAT);
        fmt!(R32_UINT);
        fmt!(R32_SINT);
        fmt!(R32_SFLOAT);
        fmt!(R32G32_UINT);
        fmt!(R32G32_SINT);
        fmt!(R32G32_SFLOAT);
        fmt!(R32G32B32_UINT);
        fmt!(R32G32B32_SINT);
        fmt!(R32G32B32_SFLOAT);
        fmt!(R32G32B32A32_UINT);
        fmt!(R32G32B32A32_SINT);
        fmt!(R32G32B32A32_SFLOAT);
        fmt!(R64_UINT);
        fmt!(R64_SINT);
        fmt!(R64_SFLOAT);
        fmt!(R64G64_UINT);
        fmt!(R64G64_SINT);
        fmt!(R64G64_SFLOAT);
        fmt!(R64G64B64_UINT);
        fmt!(R64G64B64_SINT);
        fmt!(R64G64B64_SFLOAT);
        fmt!(R64G64B64A64_UINT);
        fmt!(R64G64B64A64_SINT);
        fmt!(R64G64B64A64_SFLOAT);
        fmt!(B10G11R11_UFLOAT_PACK32);
        fmt!(E5B9G9R9_UFLOAT_PACK32);
        fmt!(D16_UNORM);
        fmt!(X8_D24_UNORM_PACK32);
        fmt!(D32_SFLOAT);
        fmt!(S8_UINT);
        fmt!(D16_UNORM_S8_UINT);
        fmt!(D24_UNORM_S8_UINT);
        fmt!(D32_SFLOAT_S8_UINT);
        fmt!(BC1_RGB_UNORM_BLOCK);
        fmt!(BC1_RGB_SRGB_BLOCK);
        fmt!(BC1_RGBA_UNORM_BLOCK);
        fmt!(BC1_RGBA_SRGB_BLOCK);
        fmt!(BC2_UNORM_BLOCK);
        fmt!(BC2_SRGB_BLOCK);
        fmt!(BC3_UNORM_BLOCK);
        fmt!(BC3_SRGB_BLOCK);
        fmt!(BC4_UNORM_BLOCK);
        fmt!(BC4_SNORM_BLOCK);
        fmt!(BC5_UNORM_BLOCK);
        fmt!(BC5_SNORM_BLOCK);
        fmt!(BC6H_UFLOAT_BLOCK);
        fmt!(BC6H_SFLOAT_BLOCK);
        fmt!(BC7_UNORM_BLOCK);
        fmt!(BC7_SRGB_BLOCK);
        fmt!(ETC2_R8G8B8_UNORM_BLOCK);
        fmt!(ETC2_R8G8B8_SRGB_BLOCK);
        fmt!(ETC2_R8G8B8A1_UNORM_BLOCK);
        fmt!(ETC2_R8G8B8A1_SRGB_BLOCK);
        fmt!(ETC2_R8G8B8A8_UNORM_BLOCK);
        fmt!(ETC2_R8G8B8A8_SRGB_BLOCK);
        fmt!(EAC_R11_UNORM_BLOCK);
        fmt!(EAC_R11_SNORM_BLOCK);
        fmt!(EAC_R11G11_UNORM_BLOCK);
        fmt!(EAC_R11G11_SNORM_BLOCK);
        fmt!(ASTC_4X4_UNORM_BLOCK);
        fmt!(ASTC_4X4_SRGB_BLOCK);
        fmt!(ASTC_5X4_UNORM_BLOCK);
        fmt!(ASTC_5X4_SRGB_BLOCK);
        fmt!(ASTC_5X5_UNORM_BLOCK);
        fmt!(ASTC_5X5_SRGB_BLOCK);
        fmt!(ASTC_6X5_UNORM_BLOCK);
        fmt!(ASTC_6X5_SRGB_BLOCK);
        fmt!(ASTC_6X6_UNORM_BLOCK);
        fmt!(ASTC_6X6_SRGB_BLOCK);
        fmt!(ASTC_8X5_UNORM_BLOCK);
        fmt!(ASTC_8X5_SRGB_BLOCK);
        fmt!(ASTC_8X6_UNORM_BLOCK);
        fmt!(ASTC_8X6_SRGB_BLOCK);
        fmt!(ASTC_8X8_UNORM_BLOCK);
        fmt!(ASTC_8X8_SRGB_BLOCK);
        fmt!(ASTC_10X5_UNORM_BLOCK);
        fmt!(ASTC_10X5_SRGB_BLOCK);
        fmt!(ASTC_10X6_UNORM_BLOCK);
        fmt!(ASTC_10X6_SRGB_BLOCK);
        fmt!(ASTC_10X8_UNORM_BLOCK);
        fmt!(ASTC_10X8_SRGB_BLOCK);
        fmt!(ASTC_10X10_UNORM_BLOCK);
        fmt!(ASTC_10X10_SRGB_BLOCK);
        fmt!(ASTC_12X10_UNORM_BLOCK);
        fmt!(ASTC_12X10_SRGB_BLOCK);
        fmt!(ASTC_12X12_UNORM_BLOCK);
        fmt!(ASTC_12X12_SRGB_BLOCK);
    }
}
