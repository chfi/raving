use crossbeam::atomic::AtomicCell;
use parking_lot::{Mutex, RwLock};

use lazy_static::lazy_static;

use crate::vk::resource::index::*;

use rhai::plugin::*;

pub mod console;

#[export_module]
pub mod vk {

    use ash::vk;

    #[rhai_mod(name = "SamplerCreateInfo")]
    pub mod sampler_create_info {

        #[rhai_fn(skip)]
        pub(crate) fn build(
            mut map: rhai::Map,
        ) -> anyhow::Result<vk::SamplerCreateInfo> {
            let mut info = vk::SamplerCreateInfo::default();

            macro_rules! take_cast_set {
                ($field:ident, $ty:ty) => {
                    if let Some($field) = map
                        .remove(stringify!(field))
                        .and_then(|$field| $field.try_cast::<$ty>())
                    {
                        info.$field = $field;
                    }
                };
            }

            take_cast_set!(flags, vk::SamplerCreateFlags);
            take_cast_set!(mag_filter, vk::Filter);
            take_cast_set!(min_filter, vk::Filter);
            take_cast_set!(mipmap_mode, vk::SamplerMipmapMode);
            take_cast_set!(address_mode_u, vk::SamplerAddressMode);
            take_cast_set!(address_mode_v, vk::SamplerAddressMode);
            take_cast_set!(address_mode_w, vk::SamplerAddressMode);
            take_cast_set!(mip_lod_bias, f32);
            take_cast_set!(anisotropy_enable, u32);
            take_cast_set!(max_anisotropy, f32);
            take_cast_set!(compare_enable, u32);
            take_cast_set!(compare_op, vk::CompareOp);
            take_cast_set!(min_lod, f32);
            take_cast_set!(max_lod, f32);
            take_cast_set!(border_color, vk::BorderColor);
            take_cast_set!(unnormalized_coordinates, u32);

            Ok(info)
        }
    }

    #[rhai_mod(name = "CompareOp")]
    pub mod compare_op {
        pub const NEVER: ash::vk::CompareOp = ash::vk::CompareOp::NEVER;
        pub const LESS: ash::vk::CompareOp = ash::vk::CompareOp::LESS;
        pub const EQUAL: ash::vk::CompareOp = ash::vk::CompareOp::EQUAL;
        pub const LESS_OR_EQUAL: ash::vk::CompareOp =
            ash::vk::CompareOp::LESS_OR_EQUAL;
        pub const GREATER: ash::vk::CompareOp = ash::vk::CompareOp::GREATER;
        pub const NOT_EQUAL: ash::vk::CompareOp = ash::vk::CompareOp::NOT_EQUAL;
        pub const GREATER_OR_EQUAL: ash::vk::CompareOp =
            ash::vk::CompareOp::GREATER_OR_EQUAL;
        pub const ALWAYS: ash::vk::CompareOp = ash::vk::CompareOp::ALWAYS;
    }

    #[rhai_mod(name = "Filter")]
    pub mod filter {
        pub const NEAREST: ash::vk::Filter = ash::vk::Filter::NEAREST;
        pub const LINEAR: ash::vk::Filter = ash::vk::Filter::LINEAR;
    }

    #[rhai_mod(name = "SamplerMipmapMode")]
    pub mod sampler_mipmap_mode {
        pub const NEAREST: ash::vk::SamplerMipmapMode =
            ash::vk::SamplerMipmapMode::NEAREST;
        pub const LINEAR: ash::vk::SamplerMipmapMode =
            ash::vk::SamplerMipmapMode::LINEAR;
    }

    #[rhai_mod(name = "SamplerAddressMode")]
    pub mod sampler_address_mode {
        pub const REPEAT: ash::vk::SamplerAddressMode =
            ash::vk::SamplerAddressMode::REPEAT;
        pub const MIRRORED_REPEAT: ash::vk::SamplerAddressMode =
            ash::vk::SamplerAddressMode::MIRRORED_REPEAT;
        pub const CLAMP_TO_EDGE: ash::vk::SamplerAddressMode =
            ash::vk::SamplerAddressMode::CLAMP_TO_EDGE;
        pub const CLAMP_TO_BORDER: ash::vk::SamplerAddressMode =
            ash::vk::SamplerAddressMode::CLAMP_TO_BORDER;
    }

    #[rhai_mod(name = "BorderColor")]
    pub mod border_color {
        pub const FLOAT_TRANSPARENT_BLACK: ash::vk::BorderColor =
            ash::vk::BorderColor::FLOAT_TRANSPARENT_BLACK;
        pub const INT_TRANSPARENT_BLACK: ash::vk::BorderColor =
            ash::vk::BorderColor::INT_TRANSPARENT_BLACK;
        pub const FLOAT_OPAQUE_BLACK: ash::vk::BorderColor =
            ash::vk::BorderColor::FLOAT_OPAQUE_BLACK;
        pub const INT_OPAQUE_BLACK: ash::vk::BorderColor =
            ash::vk::BorderColor::INT_OPAQUE_BLACK;
        pub const FLOAT_OPAQUE_WHITE: ash::vk::BorderColor =
            ash::vk::BorderColor::FLOAT_OPAQUE_WHITE;
        pub const INT_OPAQUE_WHITE: ash::vk::BorderColor =
            ash::vk::BorderColor::INT_OPAQUE_WHITE;
    }

    #[rhai_mod(name = "BufferUsageFlags")]
    pub mod buffer_usage {

        pub const TRANSFER_SRC: ash::vk::BufferUsageFlags =
            ash::vk::BufferUsageFlags::TRANSFER_SRC;
        pub const TRANSFER_DST: ash::vk::BufferUsageFlags =
            ash::vk::BufferUsageFlags::TRANSFER_DST;
        pub const UNIFORM_TEXEL_BUFFER: ash::vk::BufferUsageFlags =
            ash::vk::BufferUsageFlags::UNIFORM_TEXEL_BUFFER;
        pub const STORAGE_TEXEL_BUFFER: ash::vk::BufferUsageFlags =
            ash::vk::BufferUsageFlags::STORAGE_TEXEL_BUFFER;
        pub const UNIFORM_BUFFER: ash::vk::BufferUsageFlags =
            ash::vk::BufferUsageFlags::UNIFORM_BUFFER;
        pub const STORAGE_BUFFER: ash::vk::BufferUsageFlags =
            ash::vk::BufferUsageFlags::STORAGE_BUFFER;
        pub const INDEX_BUFFER: ash::vk::BufferUsageFlags =
            ash::vk::BufferUsageFlags::INDEX_BUFFER;
        pub const VERTEX_BUFFER: ash::vk::BufferUsageFlags =
            ash::vk::BufferUsageFlags::VERTEX_BUFFER;
        pub const INDIRECT_BUFFER: ash::vk::BufferUsageFlags =
            ash::vk::BufferUsageFlags::INDIRECT_BUFFER;

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

    #[rhai_mod(name = "MemoryLocation")]
    pub mod memory_location {
        use gpu_allocator::MemoryLocation as Loc;

        pub const GPU_ONLY: Loc = Loc::GpuOnly;
        pub const UNKNOWN: Loc = Loc::Unknown;
        pub const CPU_TO_GPU: Loc = Loc::CpuToGpu;
        pub const GPU_TO_CPU: Loc = Loc::CpuToGpu;
    }

    #[rhai_mod(name = "ShaderStageFlags")]
    pub mod shader_stage_flags {

        pub const COMPUTE: ash::vk::ShaderStageFlags =
            ash::vk::ShaderStageFlags::COMPUTE;
        pub const VERTEX: ash::vk::ShaderStageFlags =
            ash::vk::ShaderStageFlags::VERTEX;
        pub const TESSELLATION_CONTROL: ash::vk::ShaderStageFlags =
            ash::vk::ShaderStageFlags::TESSELLATION_CONTROL;
        pub const TESSELLATION_EVALUATION: ash::vk::ShaderStageFlags =
            ash::vk::ShaderStageFlags::TESSELLATION_EVALUATION;
        pub const GEOMETRY: ash::vk::ShaderStageFlags =
            ash::vk::ShaderStageFlags::GEOMETRY;
        pub const FRAGMENT: ash::vk::ShaderStageFlags =
            ash::vk::ShaderStageFlags::FRAGMENT;
        pub const ALL_GRAPHICS: ash::vk::ShaderStageFlags =
            ash::vk::ShaderStageFlags::ALL_GRAPHICS;
        pub const ALL: ash::vk::ShaderStageFlags =
            ash::vk::ShaderStageFlags::ALL;

        #[rhai_fn(name = "|", global)]
        pub fn or(
            f: vk::ShaderStageFlags,
            g: vk::ShaderStageFlags,
        ) -> vk::ShaderStageFlags {
            f | g
        }

        #[rhai_fn(name = "&", global)]
        pub fn and(
            f: vk::ShaderStageFlags,
            g: vk::ShaderStageFlags,
        ) -> vk::ShaderStageFlags {
            f & g
        }
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

        #[rhai_fn(name = "|", global)]
        pub fn or(
            f: vk::PipelineStageFlags,
            g: vk::PipelineStageFlags,
        ) -> vk::PipelineStageFlags {
            f | g
        }

        #[rhai_fn(name = "&", global)]
        pub fn and(
            f: vk::PipelineStageFlags,
            g: vk::PipelineStageFlags,
        ) -> vk::PipelineStageFlags {
            f & g
        }
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

        #[rhai_fn(name = "|", global)]
        pub fn or(f: vk::AccessFlags, g: vk::AccessFlags) -> vk::AccessFlags {
            f | g
        }

        #[rhai_fn(name = "&", global)]
        pub fn and(f: vk::AccessFlags, g: vk::AccessFlags) -> vk::AccessFlags {
            f & g
        }
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
