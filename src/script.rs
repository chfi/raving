use crossbeam::atomic::AtomicCell;
use parking_lot::{Mutex, RwLock};

use lazy_static::lazy_static;

use crate::vk::resource::index::*;

use rhai::plugin::*;

pub mod console;

#[export_module]
pub mod vk {

    use ash::vk;

    pub const HELLO: i64 = 42;

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

        pub fn storage_image(binding: i64) -> BindingDesc {
            BindingDesc::StorageImage {
                binding: binding as u32,
            }
        }

        pub fn sampled_image(binding: i64) -> BindingDesc {
            BindingDesc::SampledImage {
                binding: binding as u32,
            }
        }

        pub fn storage_buffer(binding: i64) -> BindingDesc {
            BindingDesc::StorageBuffer {
                binding: binding as u32,
            }
        }

        pub fn uniform_buffer(binding: i64) -> BindingDesc {
            BindingDesc::UniformBuffer {
                binding: binding as u32,
            }
        }
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

        #[rhai_fn(name = "|", global)]
        pub fn or(
            f: vk::BufferUsageFlags,
            g: vk::BufferUsageFlags,
        ) -> vk::BufferUsageFlags {
            f | g
        }

        #[rhai_fn(name = "&", global)]
        pub fn and(
            f: vk::BufferUsageFlags,
            g: vk::BufferUsageFlags,
        ) -> vk::BufferUsageFlags {
            f & g
        }
    }

    #[rhai_mod(name = "ImageUsageFlags")]
    pub mod image_usage {
        use ash::vk::ImageUsageFlags as Flags;

        pub const TRANSFER_SRC: Flags = Flags::TRANSFER_SRC;
        pub const TRANSFER_DST: Flags = Flags::TRANSFER_DST;
        pub const SAMPLED: Flags = Flags::SAMPLED;
        pub const STORAGE: Flags = Flags::STORAGE;
        pub const COLOR_ATTACHMENT: Flags = Flags::COLOR_ATTACHMENT;
        pub const DEPTH_STENCIL_ATTACHMENT: Flags =
            Flags::DEPTH_STENCIL_ATTACHMENT;
        pub const TRANSIENT_ATTACHMENT: Flags = Flags::TRANSIENT_ATTACHMENT;
        pub const INPUT_ATTACHMENT: Flags = Flags::INPUT_ATTACHMENT;

        #[rhai_fn(name = "|", global)]
        pub fn or(f: Flags, g: Flags) -> vk::ImageUsageFlags {
            f | g
        }

        #[rhai_fn(name = "&", global)]
        pub fn and(f: Flags, g: Flags) -> vk::ImageUsageFlags {
            f & g
        }
    }

    #[rhai_mod(name = "ImageLayout")]
    pub mod image_layout {
        use ash::vk::ImageLayout as Layout;

        pub const UNDEFINED: Layout = Layout::UNDEFINED;
        pub const PREINITIALIZED: Layout = Layout::PREINITIALIZED;
        pub const GENERAL: Layout = Layout::GENERAL;
        pub const TRANSFER_SRC_OPTIMAL: Layout = Layout::TRANSFER_SRC_OPTIMAL;
        pub const TRANSFER_DST_OPTIMAL: Layout = Layout::TRANSFER_DST_OPTIMAL;
        pub const SHADER_READ_ONLY_OPTIMAL: Layout =
            Layout::SHADER_READ_ONLY_OPTIMAL;
        pub const COLOR_ATTACHMENT_OPTIMAL: Layout =
            Layout::COLOR_ATTACHMENT_OPTIMAL;
        pub const PRESENT_SRC_KHR: Layout = Layout::PRESENT_SRC_KHR;
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
        use ash::vk::PipelineStageFlags as Flags;

        pub const TOP_OF_PIPE: Flags = Flags::TOP_OF_PIPE;
        pub const DRAW_INDIRECT: Flags = Flags::DRAW_INDIRECT;
        pub const VERTEX_INPUT: Flags = Flags::VERTEX_INPUT;
        pub const VERTEX_SHADER: Flags = Flags::VERTEX_SHADER;
        pub const TESSELLATION_CONTROL_SHADER: Flags =
            Flags::TESSELLATION_CONTROL_SHADER;
        pub const TESSELLATION_EVALUATION_SHADER: Flags =
            Flags::TESSELLATION_EVALUATION_SHADER;
        pub const GEOMETRY_SHADER: Flags = Flags::GEOMETRY_SHADER;
        pub const FRAGMENT_SHADER: Flags = Flags::FRAGMENT_SHADER;
        pub const EARLY_FRAGMENT_TESTS: Flags = Flags::EARLY_FRAGMENT_TESTS;
        pub const LATE_FRAGMENT_TESTS: Flags = Flags::LATE_FRAGMENT_TESTS;
        pub const COLOR_ATTACHMENT_OUTPUT: Flags =
            Flags::COLOR_ATTACHMENT_OUTPUT;
        pub const COMPUTE_SHADER: Flags = Flags::COMPUTE_SHADER;
        pub const TRANSFER: Flags = Flags::TRANSFER;
        pub const BOTTOM_OF_PIPE: Flags = Flags::BOTTOM_OF_PIPE;
        pub const HOST: Flags = Flags::HOST;
        pub const ALL_GRAPHICS: Flags = Flags::ALL_GRAPHICS;
        pub const ALL_COMMANDS: Flags = Flags::ALL_COMMANDS;
    }

    #[rhai_mod(name = "AccessFlags")]
    pub mod access_flags {
        use ash::vk::AccessFlags as Flags;

        pub fn empty() -> Flags {
            Flags::empty()
        }

        pub const INDIRECT_COMMAND_READ: Flags = Flags::INDIRECT_COMMAND_READ;
        pub const INDEX_READ: Flags = Flags::INDEX_READ;
        pub const VERTEX_ATTRIBUTE_READ: Flags = Flags::VERTEX_ATTRIBUTE_READ;
        pub const UNIFORM_READ: Flags = Flags::UNIFORM_READ;
        pub const INPUT_ATTACHMENT_READ: Flags = Flags::INPUT_ATTACHMENT_READ;
        pub const SHADER_READ: Flags = Flags::SHADER_READ;
        pub const SHADER_WRITE: Flags = Flags::SHADER_WRITE;
        pub const COLOR_ATTACHMENT_READ: Flags = Flags::COLOR_ATTACHMENT_READ;
        pub const COLOR_ATTACHMENT_WRITE: Flags = Flags::COLOR_ATTACHMENT_WRITE;
        pub const DEPTH_STENCIL_ATTACHMENT_READ: Flags =
            Flags::DEPTH_STENCIL_ATTACHMENT_READ;
        pub const DEPTH_STENCIL_ATTACHMENT_WRITE: Flags =
            Flags::DEPTH_STENCIL_ATTACHMENT_WRITE;
        pub const TRANSFER_READ: Flags = Flags::TRANSFER_READ;
        pub const TRANSFER_WRITE: Flags = Flags::TRANSFER_WRITE;
        pub const HOST_READ: Flags = Flags::HOST_READ;
        pub const HOST_WRITE: Flags = Flags::HOST_WRITE;
        pub const MEMORY_READ: Flags = Flags::MEMORY_READ;
        pub const MEMORY_WRITE: Flags = Flags::MEMORY_WRITE;
    }

    #[rhai_mod(name = "Format")]
    pub mod format {
        pub const UNDEFINED: vk::Format = vk::Format::UNDEFINED;
        pub const R4G4_UNORM_PACK8: vk::Format = vk::Format::R4G4_UNORM_PACK8;
        pub const R4G4B4A4_UNORM_PACK16: vk::Format =
            vk::Format::R4G4B4A4_UNORM_PACK16;
        pub const B4G4R4A4_UNORM_PACK16: vk::Format =
            vk::Format::B4G4R4A4_UNORM_PACK16;
        pub const R5G6B5_UNORM_PACK16: vk::Format =
            vk::Format::R5G6B5_UNORM_PACK16;
        pub const B5G6R5_UNORM_PACK16: vk::Format =
            vk::Format::B5G6R5_UNORM_PACK16;
        pub const R5G5B5A1_UNORM_PACK16: vk::Format =
            vk::Format::R5G5B5A1_UNORM_PACK16;
        pub const B5G5R5A1_UNORM_PACK16: vk::Format =
            vk::Format::B5G5R5A1_UNORM_PACK16;
        pub const A1R5G5B5_UNORM_PACK16: vk::Format =
            vk::Format::A1R5G5B5_UNORM_PACK16;
        pub const R8_UNORM: vk::Format = vk::Format::R8_UNORM;
        pub const R8_SNORM: vk::Format = vk::Format::R8_SNORM;
        pub const R8_USCALED: vk::Format = vk::Format::R8_USCALED;
        pub const R8_SSCALED: vk::Format = vk::Format::R8_SSCALED;
        pub const R8_UINT: vk::Format = vk::Format::R8_UINT;
        pub const R8_SINT: vk::Format = vk::Format::R8_SINT;
        pub const R8_SRGB: vk::Format = vk::Format::R8_SRGB;
        pub const R8G8_UNORM: vk::Format = vk::Format::R8G8_UNORM;
        pub const R8G8_SNORM: vk::Format = vk::Format::R8G8_SNORM;
        pub const R8G8_USCALED: vk::Format = vk::Format::R8G8_USCALED;
        pub const R8G8_SSCALED: vk::Format = vk::Format::R8G8_SSCALED;
        pub const R8G8_UINT: vk::Format = vk::Format::R8G8_UINT;
        pub const R8G8_SINT: vk::Format = vk::Format::R8G8_SINT;
        pub const R8G8_SRGB: vk::Format = vk::Format::R8G8_SRGB;
        pub const R8G8B8_UNORM: vk::Format = vk::Format::R8G8B8_UNORM;
        pub const R8G8B8_SNORM: vk::Format = vk::Format::R8G8B8_SNORM;
        pub const R8G8B8_USCALED: vk::Format = vk::Format::R8G8B8_USCALED;
        pub const R8G8B8_SSCALED: vk::Format = vk::Format::R8G8B8_SSCALED;
        pub const R8G8B8_UINT: vk::Format = vk::Format::R8G8B8_UINT;
        pub const R8G8B8_SINT: vk::Format = vk::Format::R8G8B8_SINT;
        pub const R8G8B8_SRGB: vk::Format = vk::Format::R8G8B8_SRGB;
        pub const B8G8R8_UNORM: vk::Format = vk::Format::B8G8R8_UNORM;
        pub const B8G8R8_SNORM: vk::Format = vk::Format::B8G8R8_SNORM;
        pub const B8G8R8_USCALED: vk::Format = vk::Format::B8G8R8_USCALED;
        pub const B8G8R8_SSCALED: vk::Format = vk::Format::B8G8R8_SSCALED;
        pub const B8G8R8_UINT: vk::Format = vk::Format::B8G8R8_UINT;
        pub const B8G8R8_SINT: vk::Format = vk::Format::B8G8R8_SINT;
        pub const B8G8R8_SRGB: vk::Format = vk::Format::B8G8R8_SRGB;
        pub const R8G8B8A8_UNORM: vk::Format = vk::Format::R8G8B8A8_UNORM;
        pub const R8G8B8A8_SNORM: vk::Format = vk::Format::R8G8B8A8_SNORM;
        pub const R8G8B8A8_USCALED: vk::Format = vk::Format::R8G8B8A8_USCALED;
        pub const R8G8B8A8_SSCALED: vk::Format = vk::Format::R8G8B8A8_SSCALED;
        pub const R8G8B8A8_UINT: vk::Format = vk::Format::R8G8B8A8_UINT;
        pub const R8G8B8A8_SINT: vk::Format = vk::Format::R8G8B8A8_SINT;
        pub const R8G8B8A8_SRGB: vk::Format = vk::Format::R8G8B8A8_SRGB;
        pub const B8G8R8A8_UNORM: vk::Format = vk::Format::B8G8R8A8_UNORM;
        pub const B8G8R8A8_SNORM: vk::Format = vk::Format::B8G8R8A8_SNORM;
        pub const B8G8R8A8_USCALED: vk::Format = vk::Format::B8G8R8A8_USCALED;
        pub const B8G8R8A8_SSCALED: vk::Format = vk::Format::B8G8R8A8_SSCALED;
        pub const B8G8R8A8_UINT: vk::Format = vk::Format::B8G8R8A8_UINT;
        pub const B8G8R8A8_SINT: vk::Format = vk::Format::B8G8R8A8_SINT;
        pub const B8G8R8A8_SRGB: vk::Format = vk::Format::B8G8R8A8_SRGB;
        pub const A8B8G8R8_UNORM_PACK32: vk::Format =
            vk::Format::A8B8G8R8_UNORM_PACK32;
        pub const A8B8G8R8_SNORM_PACK32: vk::Format =
            vk::Format::A8B8G8R8_SNORM_PACK32;
        pub const A8B8G8R8_USCALED_PACK32: vk::Format =
            vk::Format::A8B8G8R8_USCALED_PACK32;
        pub const A8B8G8R8_SSCALED_PACK32: vk::Format =
            vk::Format::A8B8G8R8_SSCALED_PACK32;
        pub const A8B8G8R8_UINT_PACK32: vk::Format =
            vk::Format::A8B8G8R8_UINT_PACK32;
        pub const A8B8G8R8_SINT_PACK32: vk::Format =
            vk::Format::A8B8G8R8_SINT_PACK32;
        pub const A8B8G8R8_SRGB_PACK32: vk::Format =
            vk::Format::A8B8G8R8_SRGB_PACK32;
        pub const A2R10G10B10_UNORM_PACK32: vk::Format =
            vk::Format::A2R10G10B10_UNORM_PACK32;
        pub const A2R10G10B10_SNORM_PACK32: vk::Format =
            vk::Format::A2R10G10B10_SNORM_PACK32;
        pub const A2R10G10B10_USCALED_PACK32: vk::Format =
            vk::Format::A2R10G10B10_USCALED_PACK32;
        pub const A2R10G10B10_SSCALED_PACK32: vk::Format =
            vk::Format::A2R10G10B10_SSCALED_PACK32;
        pub const A2R10G10B10_UINT_PACK32: vk::Format =
            vk::Format::A2R10G10B10_UINT_PACK32;
        pub const A2R10G10B10_SINT_PACK32: vk::Format =
            vk::Format::A2R10G10B10_SINT_PACK32;
        pub const A2B10G10R10_UNORM_PACK32: vk::Format =
            vk::Format::A2B10G10R10_UNORM_PACK32;
        pub const A2B10G10R10_SNORM_PACK32: vk::Format =
            vk::Format::A2B10G10R10_SNORM_PACK32;
        pub const A2B10G10R10_USCALED_PACK32: vk::Format =
            vk::Format::A2B10G10R10_USCALED_PACK32;
        pub const A2B10G10R10_SSCALED_PACK32: vk::Format =
            vk::Format::A2B10G10R10_SSCALED_PACK32;
        pub const A2B10G10R10_UINT_PACK32: vk::Format =
            vk::Format::A2B10G10R10_UINT_PACK32;
        pub const A2B10G10R10_SINT_PACK32: vk::Format =
            vk::Format::A2B10G10R10_SINT_PACK32;
        pub const R16_UNORM: vk::Format = vk::Format::R16_UNORM;
        pub const R16_SNORM: vk::Format = vk::Format::R16_SNORM;
        pub const R16_USCALED: vk::Format = vk::Format::R16_USCALED;
        pub const R16_SSCALED: vk::Format = vk::Format::R16_SSCALED;
        pub const R16_UINT: vk::Format = vk::Format::R16_UINT;
        pub const R16_SINT: vk::Format = vk::Format::R16_SINT;
        pub const R16_SFLOAT: vk::Format = vk::Format::R16_SFLOAT;
        pub const R16G16_UNORM: vk::Format = vk::Format::R16G16_UNORM;
        pub const R16G16_SNORM: vk::Format = vk::Format::R16G16_SNORM;
        pub const R16G16_USCALED: vk::Format = vk::Format::R16G16_USCALED;
        pub const R16G16_SSCALED: vk::Format = vk::Format::R16G16_SSCALED;
        pub const R16G16_UINT: vk::Format = vk::Format::R16G16_UINT;
        pub const R16G16_SINT: vk::Format = vk::Format::R16G16_SINT;
        pub const R16G16_SFLOAT: vk::Format = vk::Format::R16G16_SFLOAT;
        pub const R16G16B16_UNORM: vk::Format = vk::Format::R16G16B16_UNORM;
        pub const R16G16B16_SNORM: vk::Format = vk::Format::R16G16B16_SNORM;
        pub const R16G16B16_USCALED: vk::Format = vk::Format::R16G16B16_USCALED;
        pub const R16G16B16_SSCALED: vk::Format = vk::Format::R16G16B16_SSCALED;
        pub const R16G16B16_UINT: vk::Format = vk::Format::R16G16B16_UINT;
        pub const R16G16B16_SINT: vk::Format = vk::Format::R16G16B16_SINT;
        pub const R16G16B16_SFLOAT: vk::Format = vk::Format::R16G16B16_SFLOAT;
        pub const R16G16B16A16_UNORM: vk::Format =
            vk::Format::R16G16B16A16_UNORM;
        pub const R16G16B16A16_SNORM: vk::Format =
            vk::Format::R16G16B16A16_SNORM;
        pub const R16G16B16A16_USCALED: vk::Format =
            vk::Format::R16G16B16A16_USCALED;
        pub const R16G16B16A16_SSCALED: vk::Format =
            vk::Format::R16G16B16A16_SSCALED;
        pub const R16G16B16A16_UINT: vk::Format = vk::Format::R16G16B16A16_UINT;
        pub const R16G16B16A16_SINT: vk::Format = vk::Format::R16G16B16A16_SINT;
        pub const R16G16B16A16_SFLOAT: vk::Format =
            vk::Format::R16G16B16A16_SFLOAT;
        pub const R32_UINT: vk::Format = vk::Format::R32_UINT;
        pub const R32_SINT: vk::Format = vk::Format::R32_SINT;
        pub const R32_SFLOAT: vk::Format = vk::Format::R32_SFLOAT;
        pub const R32G32_UINT: vk::Format = vk::Format::R32G32_UINT;
        pub const R32G32_SINT: vk::Format = vk::Format::R32G32_SINT;
        pub const R32G32_SFLOAT: vk::Format = vk::Format::R32G32_SFLOAT;
        pub const R32G32B32_UINT: vk::Format = vk::Format::R32G32B32_UINT;
        pub const R32G32B32_SINT: vk::Format = vk::Format::R32G32B32_SINT;
        pub const R32G32B32_SFLOAT: vk::Format = vk::Format::R32G32B32_SFLOAT;
        pub const R32G32B32A32_UINT: vk::Format = vk::Format::R32G32B32A32_UINT;
        pub const R32G32B32A32_SINT: vk::Format = vk::Format::R32G32B32A32_SINT;
        pub const R32G32B32A32_SFLOAT: vk::Format =
            vk::Format::R32G32B32A32_SFLOAT;
        pub const R64_UINT: vk::Format = vk::Format::R64_UINT;
        pub const R64_SINT: vk::Format = vk::Format::R64_SINT;
        pub const R64_SFLOAT: vk::Format = vk::Format::R64_SFLOAT;
        pub const R64G64_UINT: vk::Format = vk::Format::R64G64_UINT;
        pub const R64G64_SINT: vk::Format = vk::Format::R64G64_SINT;
        pub const R64G64_SFLOAT: vk::Format = vk::Format::R64G64_SFLOAT;
        pub const R64G64B64_UINT: vk::Format = vk::Format::R64G64B64_UINT;
        pub const R64G64B64_SINT: vk::Format = vk::Format::R64G64B64_SINT;
        pub const R64G64B64_SFLOAT: vk::Format = vk::Format::R64G64B64_SFLOAT;
        pub const R64G64B64A64_UINT: vk::Format = vk::Format::R64G64B64A64_UINT;
        pub const R64G64B64A64_SINT: vk::Format = vk::Format::R64G64B64A64_SINT;
        pub const R64G64B64A64_SFLOAT: vk::Format =
            vk::Format::R64G64B64A64_SFLOAT;
        pub const B10G11R11_UFLOAT_PACK32: vk::Format =
            vk::Format::B10G11R11_UFLOAT_PACK32;
        pub const E5B9G9R9_UFLOAT_PACK32: vk::Format =
            vk::Format::E5B9G9R9_UFLOAT_PACK32;
        pub const D16_UNORM: vk::Format = vk::Format::D16_UNORM;
        pub const X8_D24_UNORM_PACK32: vk::Format =
            vk::Format::X8_D24_UNORM_PACK32;
        pub const D32_SFLOAT: vk::Format = vk::Format::D32_SFLOAT;
        pub const S8_UINT: vk::Format = vk::Format::S8_UINT;
        pub const D16_UNORM_S8_UINT: vk::Format = vk::Format::D16_UNORM_S8_UINT;
        pub const D24_UNORM_S8_UINT: vk::Format = vk::Format::D24_UNORM_S8_UINT;
        pub const D32_SFLOAT_S8_UINT: vk::Format =
            vk::Format::D32_SFLOAT_S8_UINT;
        pub const BC1_RGB_UNORM_BLOCK: vk::Format =
            vk::Format::BC1_RGB_UNORM_BLOCK;
        pub const BC1_RGB_SRGB_BLOCK: vk::Format =
            vk::Format::BC1_RGB_SRGB_BLOCK;
        pub const BC1_RGBA_UNORM_BLOCK: vk::Format =
            vk::Format::BC1_RGBA_UNORM_BLOCK;
        pub const BC1_RGBA_SRGB_BLOCK: vk::Format =
            vk::Format::BC1_RGBA_SRGB_BLOCK;
        pub const BC2_UNORM_BLOCK: vk::Format = vk::Format::BC2_UNORM_BLOCK;
        pub const BC2_SRGB_BLOCK: vk::Format = vk::Format::BC2_SRGB_BLOCK;
        pub const BC3_UNORM_BLOCK: vk::Format = vk::Format::BC3_UNORM_BLOCK;
        pub const BC3_SRGB_BLOCK: vk::Format = vk::Format::BC3_SRGB_BLOCK;
        pub const BC4_UNORM_BLOCK: vk::Format = vk::Format::BC4_UNORM_BLOCK;
        pub const BC4_SNORM_BLOCK: vk::Format = vk::Format::BC4_SNORM_BLOCK;
        pub const BC5_UNORM_BLOCK: vk::Format = vk::Format::BC5_UNORM_BLOCK;
        pub const BC5_SNORM_BLOCK: vk::Format = vk::Format::BC5_SNORM_BLOCK;
        pub const BC6H_UFLOAT_BLOCK: vk::Format = vk::Format::BC6H_UFLOAT_BLOCK;
        pub const BC6H_SFLOAT_BLOCK: vk::Format = vk::Format::BC6H_SFLOAT_BLOCK;
        pub const BC7_UNORM_BLOCK: vk::Format = vk::Format::BC7_UNORM_BLOCK;
        pub const BC7_SRGB_BLOCK: vk::Format = vk::Format::BC7_SRGB_BLOCK;
        pub const ETC2_R8G8B8_UNORM_BLOCK: vk::Format =
            vk::Format::ETC2_R8G8B8_UNORM_BLOCK;
        pub const ETC2_R8G8B8_SRGB_BLOCK: vk::Format =
            vk::Format::ETC2_R8G8B8_SRGB_BLOCK;
        pub const ETC2_R8G8B8A1_UNORM_BLOCK: vk::Format =
            vk::Format::ETC2_R8G8B8A1_UNORM_BLOCK;
        pub const ETC2_R8G8B8A1_SRGB_BLOCK: vk::Format =
            vk::Format::ETC2_R8G8B8A1_SRGB_BLOCK;
        pub const ETC2_R8G8B8A8_UNORM_BLOCK: vk::Format =
            vk::Format::ETC2_R8G8B8A8_UNORM_BLOCK;
        pub const ETC2_R8G8B8A8_SRGB_BLOCK: vk::Format =
            vk::Format::ETC2_R8G8B8A8_SRGB_BLOCK;
        pub const EAC_R11_UNORM_BLOCK: vk::Format =
            vk::Format::EAC_R11_UNORM_BLOCK;
        pub const EAC_R11_SNORM_BLOCK: vk::Format =
            vk::Format::EAC_R11_SNORM_BLOCK;
        pub const EAC_R11G11_UNORM_BLOCK: vk::Format =
            vk::Format::EAC_R11G11_UNORM_BLOCK;
        pub const EAC_R11G11_SNORM_BLOCK: vk::Format =
            vk::Format::EAC_R11G11_SNORM_BLOCK;
        pub const ASTC_4X4_UNORM_BLOCK: vk::Format =
            vk::Format::ASTC_4X4_UNORM_BLOCK;
        pub const ASTC_4X4_SRGB_BLOCK: vk::Format =
            vk::Format::ASTC_4X4_SRGB_BLOCK;
        pub const ASTC_5X4_UNORM_BLOCK: vk::Format =
            vk::Format::ASTC_5X4_UNORM_BLOCK;
        pub const ASTC_5X4_SRGB_BLOCK: vk::Format =
            vk::Format::ASTC_5X4_SRGB_BLOCK;
        pub const ASTC_5X5_UNORM_BLOCK: vk::Format =
            vk::Format::ASTC_5X5_UNORM_BLOCK;
        pub const ASTC_5X5_SRGB_BLOCK: vk::Format =
            vk::Format::ASTC_5X5_SRGB_BLOCK;
        pub const ASTC_6X5_UNORM_BLOCK: vk::Format =
            vk::Format::ASTC_6X5_UNORM_BLOCK;
        pub const ASTC_6X5_SRGB_BLOCK: vk::Format =
            vk::Format::ASTC_6X5_SRGB_BLOCK;
        pub const ASTC_6X6_UNORM_BLOCK: vk::Format =
            vk::Format::ASTC_6X6_UNORM_BLOCK;
        pub const ASTC_6X6_SRGB_BLOCK: vk::Format =
            vk::Format::ASTC_6X6_SRGB_BLOCK;
        pub const ASTC_8X5_UNORM_BLOCK: vk::Format =
            vk::Format::ASTC_8X5_UNORM_BLOCK;
        pub const ASTC_8X5_SRGB_BLOCK: vk::Format =
            vk::Format::ASTC_8X5_SRGB_BLOCK;
        pub const ASTC_8X6_UNORM_BLOCK: vk::Format =
            vk::Format::ASTC_8X6_UNORM_BLOCK;
        pub const ASTC_8X6_SRGB_BLOCK: vk::Format =
            vk::Format::ASTC_8X6_SRGB_BLOCK;
        pub const ASTC_8X8_UNORM_BLOCK: vk::Format =
            vk::Format::ASTC_8X8_UNORM_BLOCK;
        pub const ASTC_8X8_SRGB_BLOCK: vk::Format =
            vk::Format::ASTC_8X8_SRGB_BLOCK;
        pub const ASTC_10X5_UNORM_BLOCK: vk::Format =
            vk::Format::ASTC_10X5_UNORM_BLOCK;
        pub const ASTC_10X5_SRGB_BLOCK: vk::Format =
            vk::Format::ASTC_10X5_SRGB_BLOCK;
        pub const ASTC_10X6_UNORM_BLOCK: vk::Format =
            vk::Format::ASTC_10X6_UNORM_BLOCK;
        pub const ASTC_10X6_SRGB_BLOCK: vk::Format =
            vk::Format::ASTC_10X6_SRGB_BLOCK;
        pub const ASTC_10X8_UNORM_BLOCK: vk::Format =
            vk::Format::ASTC_10X8_UNORM_BLOCK;
        pub const ASTC_10X8_SRGB_BLOCK: vk::Format =
            vk::Format::ASTC_10X8_SRGB_BLOCK;
        pub const ASTC_10X10_UNORM_BLOCK: vk::Format =
            vk::Format::ASTC_10X10_UNORM_BLOCK;
        pub const ASTC_10X10_SRGB_BLOCK: vk::Format =
            vk::Format::ASTC_10X10_SRGB_BLOCK;
        pub const ASTC_12X10_UNORM_BLOCK: vk::Format =
            vk::Format::ASTC_12X10_UNORM_BLOCK;
        pub const ASTC_12X10_SRGB_BLOCK: vk::Format =
            vk::Format::ASTC_12X10_SRGB_BLOCK;
        pub const ASTC_12X12_UNORM_BLOCK: vk::Format =
            vk::Format::ASTC_12X12_UNORM_BLOCK;
        pub const ASTC_12X12_SRGB_BLOCK: vk::Format =
            vk::Format::ASTC_12X12_SRGB_BLOCK;
    }
}
