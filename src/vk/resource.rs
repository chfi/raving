use ash::{vk, Device};

use gpu_allocator::{
    vulkan::{Allocation, AllocationCreateDesc, Allocator},
    MemoryLocation,
};

#[allow(unused_imports)]
use anyhow::{anyhow, bail, Result};
use thunderdome::Arena;

use super::{
    context::VkContext,
    descriptor::{
        BindingDesc, BindingInput, DescriptorAllocator, DescriptorBuilder,
        DescriptorLayoutCache,
    },
};

pub mod graph;
pub mod index;

use graph::*;
pub use index::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum PipelineType {
    Compute,
}

pub struct GpuResources {
    descriptor_allocator: DescriptorAllocator,
    layout_cache: DescriptorLayoutCache,
    descriptor_sets: Arena<vk::DescriptorSet>,

    pipelines: Arena<(vk::Pipeline, vk::PipelineLayout)>,

    buffers: Arena<BufferRes>,

    images: Arena<ImageRes>,
    image_views: Arena<(vk::ImageView, ImageIx)>,
    samplers: Arena<vk::Sampler>,

    semaphores: Arena<vk::Semaphore>,
    fences: Arena<vk::Fence>,
}

impl GpuResources {
    pub fn new(context: &VkContext) -> Result<Self> {
        let descriptor_allocator = DescriptorAllocator::new(context)?;
        let layout_cache =
            DescriptorLayoutCache::new(context.device().to_owned());

        let result = Self {
            descriptor_allocator,
            layout_cache,
            descriptor_sets: Arena::new(),

            pipelines: Arena::new(),

            buffers: Arena::new(),

            images: Arena::new(),
            image_views: Arena::new(),
            samplers: Arena::new(),

            semaphores: Arena::new(),
            fences: Arena::new(),
        };

        Ok(result)
    }

    pub fn dispatch_compute(
        &self,
        cmd: vk::CommandBuffer,
        device: &Device,
        pipeline_ix: PipelineIx,
        image_ix: ImageIx,
        desc_set_ix: DescSetIx,
        width: u32,
        height: u32,
        color: [f32; 4],
    ) -> Result<()> {
        let (pipeline, pipeline_layout) = self[pipeline_ix];

        // transition image TRANSFER_SRC_OPTIMAL -> GENERAL
        self.transition_image(
            cmd,
            device,
            image_ix,
            vk::AccessFlags::TRANSFER_READ,
            vk::PipelineStageFlags::TRANSFER,
            vk::AccessFlags::SHADER_WRITE,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::GENERAL,
        );

        // dispatch

        let desc_set = self[desc_set_ix];

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

            let push_constants = [width as u32, height as u32];

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
        self.transition_image(
            cmd,
            device,
            image_ix,
            vk::AccessFlags::SHADER_WRITE,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::AccessFlags::TRANSFER_READ,
            vk::PipelineStageFlags::TRANSFER,
            vk::ImageLayout::GENERAL,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
        );

        Ok(())
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
        let name = None;

        let image = ImageRes::allocate_2d(
            ctx, allocator, width, height, format, usage, name,
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

    pub fn allocate_fence(&mut self, ctx: &VkContext) -> Result<FenceIx> {
        let fence_info = vk::FenceCreateInfo::builder().build();
        let fence = unsafe { ctx.device().create_fence(&fence_info, None) }?;
        let ix = self.fences.insert(fence);
        Ok(FenceIx(ix))
    }

    pub fn allocate_desc_set(
        &mut self,
        bind_descs: &[BindingDesc],
        bind_inputs: &[BindingInput],
        stage_flags: vk::ShaderStageFlags,
    ) -> Result<DescSetIx> {
        if bind_descs.is_empty() || bind_descs.len() != bind_inputs.len() {
            bail!(
                "Binding descriptions did not match inputs in length: {} vs {}",
                bind_descs.len(),
                bind_inputs.len()
            );
        }

        let mut builder = DescriptorBuilder::begin(
            &mut self.layout_cache,
            &mut self.descriptor_allocator,
        );

        use BindingDesc as Desc;
        use BindingInput as In;

        use parking_lot::Mutex;

        let img_infos = Mutex::new(Vec::new());

        for (desc, input) in bind_descs.iter().zip(bind_inputs) {
            match *desc {
                Desc::StorageImage { binding } => {
                    if input.binding() != binding {
                        bail!("Binding descriptions and input order do not match: {} vs {}",
                              binding,
                              input.binding());
                    }

                    if let In::ImageView { view, .. } = input {
                        let (view, _image) = self.image_views[view.0];

                        let img_info = vk::DescriptorImageInfo::builder()
                            .image_layout(vk::ImageLayout::GENERAL)
                            .image_view(view)
                            .build();

                        let image_info = vec![img_info];

                        let (image_info, len) = {
                            let mut infos = img_infos.lock();
                            let ix = infos.len();

                            let len = image_info.len();

                            infos.push(image_info);

                            let lol = infos[ix].as_ptr();
                            (lol, len)
                        };

                        unsafe {
                            let lmao: &[vk::DescriptorImageInfo] =
                                std::slice::from_raw_parts(image_info, len);

                            builder.bind_image(
                                binding,
                                lmao,
                                vk::DescriptorType::STORAGE_IMAGE,
                                stage_flags,
                            );
                        }
                    } else {
                        bail!(
                            "Incompatible binding: {:?} vs {:?}",
                            desc,
                            input
                        );
                    }
                }
                Desc::SampledImage { binding } => {
                    if let In::ImageView { view, .. } = input {
                        let (view, _image) = self.image_views[view.0];

                        let img_info = vk::DescriptorImageInfo::builder()
                            .image_layout(
                                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                            )
                            .image_view(view)
                            .build();

                        let image_info = vec![img_info];

                        let (image_info, len) = {
                            let mut infos = img_infos.lock();
                            let ix = infos.len();

                            let len = image_info.len();

                            infos.push(image_info);

                            let lol = infos[ix].as_ptr();
                            (lol, len)
                        };

                        unsafe {
                            let lmao: &[vk::DescriptorImageInfo] =
                                std::slice::from_raw_parts(image_info, len);

                            builder.bind_image(
                                binding,
                                lmao,
                                vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                                stage_flags,
                            );
                        }
                    } else {
                        bail!(
                            "Incompatible binding: {:?} vs {:?}",
                            desc,
                            input
                        );
                    }
                }
            }
        }

        let set = builder.build()?;

        let ix = self.descriptor_sets.insert(set);

        Ok(DescSetIx(ix))
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

    // TODO this is completely temporary and will be handled in a
    // generic way that uses reflection to find the pipeline layout
    pub fn load_compute_shader(
        &mut self,
        context: &VkContext,
        bindings: &[BindingDesc],
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

            let bindings = BindingDesc::layout_bindings(
                bindings,
                vk::ShaderStageFlags::COMPUTE,
            )?;

            let layout_info = vk::DescriptorSetLayoutCreateInfo::builder()
                .bindings(&bindings)
                .build();

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

    pub fn transition_image(
        &self,
        cmd: vk::CommandBuffer,
        device: &Device,
        image: ImageIx,
        src_access_mask: vk::AccessFlags,
        src_stage_mask: vk::PipelineStageFlags,
        dst_access_mask: vk::AccessFlags,
        dst_stage_mask: vk::PipelineStageFlags,
        old_layout: vk::ImageLayout,
        new_layout: vk::ImageLayout,
    ) {
        let image = &self[image];

        let image_barrier = vk::ImageMemoryBarrier::builder()
            .src_access_mask(src_access_mask)
            .dst_access_mask(dst_access_mask)
            .old_layout(old_layout)
            .new_layout(new_layout)
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

        let memory_barriers = [];
        let buffer_barriers = [];
        let image_barriers = [image_barrier];

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
    }
}

#[allow(dead_code)]
pub struct BufferRes {
    name: Option<String>,
    pub(super) buffer: vk::Buffer,

    elem_size: usize,
    len: usize,

    location: gpu_allocator::MemoryLocation,

    alloc: Allocation,
}

impl BufferRes {
    pub fn gpu_readable(&self) -> bool {
        use gpu_allocator::MemoryLocation::*;
        match self.location {
            Unknown => false,
            GpuOnly => true,
            CpuToGpu => true,
            GpuToCpu => false,
        }
    }
    pub fn gpu_writable(&self) -> bool {
        use gpu_allocator::MemoryLocation::*;
        match self.location {
            Unknown => false,
            GpuOnly => true,
            CpuToGpu => false,
            GpuToCpu => true,
        }
    }

    pub fn host_readable(&self) -> bool {
        use gpu_allocator::MemoryLocation::*;
        match self.location {
            Unknown => false,
            GpuOnly => false,
            CpuToGpu => true,
            GpuToCpu => true,
        }
    }

    pub fn host_writable(&self) -> bool {
        use gpu_allocator::MemoryLocation::*;
        match self.location {
            Unknown => false,
            GpuOnly => false,
            CpuToGpu => true,
            GpuToCpu => false,
        }
    }

    // pub fn upload_to_self<T: Copy + Eq>(
    pub fn upload_to_self_bytes(
        &mut self,
        ctx: &VkContext,
        allocator: &mut Allocator,
        src: &[u8],
    ) -> Result<()> {
        assert!(!self.host_writable());

        let staging_usage = vk::BufferUsageFlags::TRANSFER_SRC;
        let location = MemoryLocation::CpuToGpu;

        let mut staging = Self::allocate_for_type::<u8>(
            ctx,
            allocator,
            location,
            staging_usage,
            src.len(),
            Some("tmp staging buffer"),
        )?;

        if let Some(stg) = staging.alloc.mapped_slice_mut() {
            log::warn!("in mapped slice!");
            stg.clone_from_slice(src);
        } else {
            bail!("couldn't map staging buffer memory");
        }

        todo!();

        Ok(())
    }

    pub fn cleanup(
        self,
        ctx: &VkContext,
        allocator: &mut Allocator,
    ) -> Result<()> {
        unsafe {
            ctx.device().destroy_buffer(self.buffer, None);
        }

        allocator.free(self.alloc)?;
        Ok(())
    }

    pub fn allocate(
        ctx: &VkContext,
        allocator: &mut Allocator,
        location: gpu_allocator::MemoryLocation,
        usage: vk::BufferUsageFlags,
        elem_size: usize,
        len: usize,
        name: Option<&str>,
    ) -> Result<Self> {
        let size = elem_size * len;

        let buf_info = vk::BufferCreateInfo::builder()
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .usage(usage)
            .size(size as _)
            .build();

        let device = ctx.device();

        let (buffer, reqs) = unsafe {
            let buffer = device.create_buffer(&buf_info, None)?;
            let reqs = device.get_buffer_memory_requirements(buffer);
            (buffer, reqs)
        };

        let alloc_desc = AllocationCreateDesc {
            name: name.unwrap_or("tmp"),
            requirements: reqs,
            location,
            linear: true,
        };

        let alloc = allocator.allocate(&alloc_desc)?;

        unsafe {
            device.bind_buffer_memory(buffer, alloc.memory(), alloc.offset())
        }?;

        Ok(Self {
            name: name.map(|n| n.to_string()),
            buffer,

            elem_size,
            len,

            location,

            alloc,
        })
    }

    pub fn allocate_for_type<T: Copy>(
        ctx: &VkContext,
        allocator: &mut Allocator,
        location: gpu_allocator::MemoryLocation,
        usage: vk::BufferUsageFlags,
        len: usize,
        name: Option<&str>,
    ) -> Result<Self> {
        let elem_size = std::mem::size_of::<T>();
        Self::allocate(ctx, allocator, location, usage, elem_size, len, name)
    }
}

#[allow(dead_code)]
pub struct ImageRes {
    name: Option<String>,
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
        ctx: &VkContext,
        allocator: &mut Allocator,
        width: u32,
        height: u32,
        format: vk::Format,
        usage: vk::ImageUsageFlags,
        name: Option<&str>,
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
            name: name.unwrap_or("tmp"),
            requirements,
            location: gpu_allocator::MemoryLocation::GpuOnly,
            linear: false,
        };

        let alloc = allocator.allocate(&alloc_desc)?;

        unsafe {
            device.bind_image_memory(image, alloc.memory(), alloc.offset())
        }?;

        Ok(Self {
            name: name.map(|n| n.to_string()),
            image,
            format,
            alloc,
            layout: vk::ImageLayout::UNDEFINED,
            img_type,
            extent,
        })
    }
}
