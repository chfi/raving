use ash::{vk, Device};

use gpu_allocator::vulkan::{Allocation, AllocationCreateDesc, Allocator};

#[allow(unused_imports)]
use anyhow::{anyhow, bail, Result};
use thunderdome::{Arena, Index};

use super::{
    context::VkContext,
    descriptor::{
        DescriptorAllocator, DescriptorBuilder, DescriptorLayoutCache,
    },
};

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PipelineIx(Index);

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct DescSetIx(Index);

// TODO make private
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ImageIx(pub(super) Index);

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ImageViewIx(Index);

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct BufferIx(Index);

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SemaphoreIx(Index);

pub struct GpuResources {
    descriptor_allocator: DescriptorAllocator,
    layout_cache: DescriptorLayoutCache,
    pub descriptor_sets: Arena<vk::DescriptorSet>,

    pipelines: Arena<(vk::Pipeline, vk::PipelineLayout)>,

    pub(super) images: Arena<ImageRes>,

    image_views: Arena<(vk::ImageView, ImageIx)>,
    semaphores: Arena<vk::Semaphore>,
    pub(super) semaphores_old: Vec<vk::Semaphore>,
}

impl GpuResources {
    pub fn dispatch_compute(
        &self,
        device: &Device,
        cmd: vk::CommandBuffer,
        pipeline_ix: PipelineIx,
        image_ix: ImageIx,
        desc_set_ix: DescSetIx,
        width: u32,
        height: u32,
        color: [f32; 4],
    ) -> Result<()> {
        let (pipeline, pipeline_layout) = *self
            .pipelines
            .get(pipeline_ix.0)
            .ok_or(anyhow!("tried to dispatch with nonexistent pipeline"))?;

        let image = self.images.get(image_ix.0).ok_or(anyhow!(
            "tried to use nonexistent image in compute dispatch"
        ))?;

        let desc_set = *self
            .descriptor_sets
            .get(desc_set_ix.0)
            .ok_or(anyhow!("descriptor set missing for compute dispatch"))?;

        // transition image TRANSFER_SRC_OPTIMAL -> GENERAL

        use vk::AccessFlags as Access;
        use vk::PipelineStageFlags as Stage;

        let memory_barriers = [];
        let buffer_barriers = [];

        let from_transfer_barrier = vk::ImageMemoryBarrier::builder()
            .src_access_mask(Access::TRANSFER_READ)
            .dst_access_mask(Access::SHADER_WRITE)
            .old_layout(vk::ImageLayout::UNDEFINED)
            .new_layout(vk::ImageLayout::GENERAL)
            .image(image.image)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            })
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .build();

        let src_stage_mask = Stage::TRANSFER;
        let dst_stage_mask = Stage::COMPUTE_SHADER;

        let image_barriers = [from_transfer_barrier];

        unsafe {
            device.cmd_pipeline_barrier(
                cmd,
                src_stage_mask,
                dst_stage_mask,
                vk::DependencyFlags::BY_REGION,
                &memory_barriers,
                &buffer_barriers,
                &image_barriers,
            );
        };

        // dispatch

        unsafe {
            let bind_point = vk::PipelineBindPoint::COMPUTE;
            device.cmd_bind_pipeline(cmd, bind_point, pipeline);

            let desc_sets = [desc_set];
            let null = [];

            device.cmd_bind_descriptor_sets(
                cmd,
                bind_point,
                pipeline_layout,
                0,
                &desc_sets,
                &null,
            );

            // let float_consts = [1f32, 0f32, 0f32, 1f32];

            let push_constants = [
                width as u32,
                height as u32,
                // 0u32,
                // 0u32,
            ];

            let mut bytes: Vec<u8> = Vec::with_capacity(24);
            bytes.extend_from_slice(bytemuck::cast_slice(&color));
            bytes.extend_from_slice(bytemuck::cast_slice(&push_constants));

            device.cmd_push_constants(
                cmd,
                pipeline_layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                &bytes,
            );
        };

        let x_size = 16;
        let y_size = 16;

        let x_groups = (width / x_size) + width % x_size;
        let y_groups = (height / y_size) + height % y_size;

        unsafe {
            device.cmd_dispatch(cmd, x_groups, y_groups, 1);
        };

        // transition image GENERAL -> TRANSFER_SRC_OPTIMAL
        let from_general_barrier = vk::ImageMemoryBarrier::builder()
            .src_access_mask(Access::SHADER_WRITE)
            .dst_access_mask(Access::TRANSFER_READ)
            .old_layout(vk::ImageLayout::GENERAL)
            .new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
            .image(image.image)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            })
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .build();

        let dst_stage_mask = Stage::TRANSFER;
        let src_stage_mask = Stage::COMPUTE_SHADER;

        let image_barriers = [from_general_barrier];

        unsafe {
            device.cmd_pipeline_barrier(
                cmd,
                src_stage_mask,
                dst_stage_mask,
                vk::DependencyFlags::BY_REGION,
                &memory_barriers,
                &buffer_barriers,
                &image_barriers,
            );
        };

        Ok(())
    }

    pub fn new(context: &VkContext) -> Result<Self> {
        let descriptor_allocator = DescriptorAllocator::new(context)?;
        let layout_cache =
            DescriptorLayoutCache::new(context.device().to_owned());

        let result = Self {
            descriptor_allocator,
            layout_cache,
            descriptor_sets: Arena::new(),

            pipelines: Arena::new(),

            images: Arena::new(),
            image_views: Arena::new(),
            semaphores: Arena::new(),

            semaphores_old: Vec::new(),
        };

        Ok(result)
    }

    pub fn allocate_image(
        &mut self,
        ctx: &VkContext,
        allocator: &mut Allocator,
        width: u32,
        height: u32,
        format: vk::Format,
        usage: vk::ImageUsageFlags,
    ) -> Result<ImageIx> {
        let image = ImageRes::allocate_2d(
            allocator, ctx, width, height, format, usage,
        )?;
        let ix = self.images.insert(image);
        Ok(ImageIx(ix))
    }

    pub fn allocate_semaphore(
        &mut self,
        ctx: &VkContext,
    ) -> Result<SemaphoreIx> {
        let semaphore_info = vk::SemaphoreCreateInfo::builder().build();
        let semaphore =
            unsafe { ctx.device().create_semaphore(&semaphore_info, None) }?;
        let ix = self.semaphores.insert(semaphore);

        Ok(SemaphoreIx(ix))
    }

    pub fn create_image_view_for_image(
        &mut self,
        ctx: &VkContext,
        image_ix: ImageIx,
    ) -> Result<ImageViewIx> {
        let img = self.images.get(image_ix.0).ok_or(anyhow!(
            "tried to create image view for nonexistent image"
        ))?;

        let view = img.create_image_view(ctx)?;
        let ix = self.image_views.insert((view, image_ix));

        Ok(ImageViewIx(ix))
    }

    pub fn create_compute_desc_set(
        &mut self,
        view_ix: ImageViewIx,
    ) -> Result<DescSetIx> {
        let (view, _image_ix) =
            *self.image_views.get(view_ix.0).ok_or(anyhow!(
                "tried to create descriptor set using nonexistent image view"
            ))?;

        let mut builder = DescriptorBuilder::begin(
            &mut self.layout_cache,
            &mut self.descriptor_allocator,
        );

        let img_info = vk::DescriptorImageInfo::builder()
            .image_layout(vk::ImageLayout::GENERAL)
            .image_view(view)
            .build();

        let image_info = [img_info];
        builder.bind_image(
            0,
            &image_info,
            vk::DescriptorType::STORAGE_IMAGE,
            vk::ShaderStageFlags::COMPUTE,
        );

        let set = builder.build()?;

        let ix = self.descriptor_sets.insert(set);

        Ok(DescSetIx(ix))
    }

    fn storage_image_layout_info() -> vk::DescriptorSetLayoutCreateInfo {
        let output_image_binding = vk::DescriptorSetLayoutBinding::builder()
            .binding(0)
            .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .build();

        let bindings = [output_image_binding];
        let layout_info = vk::DescriptorSetLayoutCreateInfo::builder()
            .bindings(&bindings)
            .build();

        layout_info
    }

    // TODO this is completely temporary and will be handled in a
    // generic way that uses reflection to find the pipeline layout
    pub fn load_compute_shader(
        &mut self,
        context: &VkContext,
        shader: &[u8],
    ) -> Result<PipelineIx> {
        let comp_src = {
            let mut cursor = std::io::Cursor::new(shader);
            ash::util::read_spv(&mut cursor)
        }?;

        let create_info = vk::ShaderModuleCreateInfo::builder()
            .code(&comp_src)
            .build();

        let shader_module = unsafe {
            context.device().create_shader_module(&create_info, None)
        }?;

        let pipeline_layout = {
            let pc_size = std::mem::size_of::<[i32; 2]>()
                + std::mem::size_of::<[f32; 4]>();

            let pc_range = vk::PushConstantRange::builder()
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
                .offset(0)
                .size(pc_size as u32)
                .build();

            let pc_ranges = [pc_range];

            let layout_info = Self::storage_image_layout_info();

            let layout =
                self.layout_cache.get_descriptor_layout(&layout_info)?;
            let layouts = [layout];

            let layout_info = vk::PipelineLayoutCreateInfo::builder()
                .set_layouts(&layouts)
                .push_constant_ranges(&pc_ranges)
                .build();

            unsafe {
                context.device().create_pipeline_layout(&layout_info, None)
            }
        }?;

        let entry_point = std::ffi::CString::new("main").unwrap();

        let comp_state_info = vk::PipelineShaderStageCreateInfo::builder()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(shader_module)
            .name(&entry_point)
            .build();

        let pipeline_info = vk::ComputePipelineCreateInfo::builder()
            .layout(pipeline_layout)
            .stage(comp_state_info)
            .build();

        let pipeline_infos = [pipeline_info];

        let result = unsafe {
            context.device().create_compute_pipelines(
                vk::PipelineCache::null(),
                &pipeline_infos,
                None,
            )
        };

        let pipelines = match result {
            Ok(pipelines) => pipelines,
            Err((pipelines, err)) => {
                log::warn!("{:?}", err);
                pipelines
            }
        };

        let pipeline = pipelines[0];

        let ix = self.pipelines.insert((pipeline, pipeline_layout));

        unsafe {
            context.device().destroy_shader_module(shader_module, None);
        }

        Ok(PipelineIx(ix))
    }
}

#[allow(dead_code)]
pub struct ImageRes {
    pub(super) image: vk::Image,
    pub(super) format: vk::Format,

    alloc: Allocation,
    pub(super) layout: vk::ImageLayout,
    img_type: vk::ImageType,

    pub(super) extent: vk::Extent3D,
}

impl ImageRes {
    pub fn create_image_view(&self, ctx: &VkContext) -> Result<vk::ImageView> {
        let create_info = vk::ImageViewCreateInfo::builder()
            .image(self.image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(self.format)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            })
            .build();

        let view =
            unsafe { ctx.device().create_image_view(&create_info, None) }?;

        Ok(view)
    }

    pub fn allocate_2d(
        allocator: &mut Allocator,
        ctx: &VkContext,
        width: u32,
        height: u32,
        format: vk::Format,
        usage: vk::ImageUsageFlags,
    ) -> Result<Self> {
        let extent = vk::Extent3D {
            width,
            height,
            depth: 1,
        };

        let tiling = vk::ImageTiling::OPTIMAL;
        let sample_count = vk::SampleCountFlags::TYPE_1;
        let flags = vk::ImageCreateFlags::empty();

        let img_type = vk::ImageType::TYPE_2D;
        let image_info = vk::ImageCreateInfo::builder()
            .image_type(img_type)
            .extent(extent)
            .mip_levels(1)
            .array_layers(1)
            .format(format)
            .tiling(tiling)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .samples(sample_count)
            .flags(flags)
            .build();

        let device = ctx.device();

        let (image, requirements) = unsafe {
            let image = device.create_image(&image_info, None)?;
            let reqs = device.get_image_memory_requirements(image);

            (image, reqs)
        };

        let alloc_desc = AllocationCreateDesc {
            name: "tmp",
            requirements,
            location: gpu_allocator::MemoryLocation::GpuOnly,
            linear: false,
        };

        let alloc = allocator.allocate(&alloc_desc)?;

        unsafe {
            device.bind_image_memory(image, alloc.memory(), alloc.offset())
        }?;

        Ok(Self {
            image,
            format,
            alloc,
            layout: vk::ImageLayout::UNDEFINED,
            img_type,
            extent,
        })
    }
}
