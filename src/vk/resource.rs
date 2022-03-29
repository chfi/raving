use std::{
    collections::{BTreeMap, HashMap},
    path::PathBuf,
};

use ash::{vk, Device};

use gpu_allocator::{
    vulkan::{Allocation, AllocationCreateDesc, Allocator},
    MemoryLocation,
};

#[allow(unused_imports)]
use anyhow::{anyhow, bail, Result};
use rspirv_reflect::DescriptorInfo;
use thunderdome::Arena;

use crate::{
    vk::descriptor::{
        DescriptorAllocator, DescriptorBuilder, DescriptorLayoutCache,
        DescriptorLayoutInfo, DescriptorUpdateBuilder,
    },
    vk::{VkContext, VkEngine},
};

pub mod buffer;
pub mod img;
pub mod index;
pub mod vertex;

pub use buffer::*;
pub use img::*;
pub use index::*;
pub use vertex::*;

pub struct GpuResources {
    descriptor_allocator: DescriptorAllocator,
    layout_cache: DescriptorLayoutCache,
    pub(super) descriptor_sets: Arena<vk::DescriptorSet>,

    // compute_pipelines: Arena<ComputePipeline>,
    pipelines: Arena<(vk::Pipeline, vk::PipelineLayout)>,

    shaders: Arena<ShaderInfo>,
    shader_file_cache: HashMap<PathBuf, ShaderIx>,

    pub(super) render_passes: Arena<vk::RenderPass>,

    pub(super) buffers: Arena<BufferRes>,

    pub(super) images: Arena<ImageRes>,
    // image_views: Arena<(vk::ImageView, ImageIx)>,
    pub(super) image_views: Arena<vk::ImageView>,
    samplers: Arena<vk::Sampler>,

    semaphores: Arena<vk::Semaphore>,
    pub(super) fences: Arena<vk::Fence>,
}

#[derive(Clone)]
pub struct ShaderInfo {
    // name: String,
    spirv: Vec<u32>,

    set_infos: BTreeMap<u32, BTreeMap<u32, DescriptorInfo>>,
    push_constant_range: Option<(u32, u32)>,

    stage: vk::ShaderStageFlags,
}

impl ShaderInfo {
    pub fn set_layout_info(&self, set_ix: u32) -> Result<DescriptorLayoutInfo> {
        let set_info = self
            .set_infos
            .get(&set_ix)
            .ok_or(anyhow!("Shader lacks set index {}", set_ix))?;

        let mut bindings = set_info
            .iter()
            .map(|(binding, info)| {
                let count = match info.binding_count {
                    rspirv_reflect::BindingCount::One => 1,
                    rspirv_reflect::BindingCount::StaticSized(n) => n as u32,
                    rspirv_reflect::BindingCount::Unbounded => {
                        log::warn!("unbounded binding count; set manually");
                        0
                    }
                };

                let ash_ty = vk::DescriptorType::from_raw(info.ty.0 as i32);

                let binding = vk::DescriptorSetLayoutBinding::builder()
                    .binding(*binding)
                    .descriptor_count(count)
                    .descriptor_type(ash_ty)
                    .stage_flags(self.stage)
                    .build();

                binding
            })
            .collect::<Vec<_>>();

        bindings.sort_by_key(|b| b.binding);

        Ok(DescriptorLayoutInfo { bindings })
    }
}

#[derive(Clone)]
pub struct ComputePipeline {
    pub pipeline: vk::Pipeline,
    pub layout: vk::PipelineLayout,
    pub shader: ShaderIx,
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

            shaders: Arena::new(),
            shader_file_cache: HashMap::default(),

            render_passes: Arena::new(),

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

    //// Buffer methods

    pub fn allocate_buffer(
        &mut self,
        ctx: &VkContext,
        allocator: &mut Allocator,
        location: gpu_allocator::MemoryLocation,
        elem_size: usize,
        len: usize,
        usage: vk::BufferUsageFlags,
        name: Option<&str>,
    ) -> Result<BufferRes> {
        let buffer = BufferRes::allocate(
            ctx, allocator, location, usage, elem_size, len, name,
        )?;

        if let Some(name) = name {
            VkEngine::set_debug_object_name(ctx, buffer.buffer, name)?;
        }

        Ok(buffer)
    }

    pub fn insert_buffer(&mut self, buffer: BufferRes) -> BufferIx {
        let ix = self.buffers.insert(buffer);
        BufferIx(ix)
    }

    #[must_use = "If a BufferRes is returned, it must be freed manually or inserted into another index, otherwise it will leak"]
    pub fn insert_buffer_at(
        &mut self,
        ix: BufferIx,
        buffer: BufferRes,
    ) -> Option<BufferRes> {
        self.buffers.insert_at(ix.0, buffer)
    }

    #[must_use = "If a BufferRes is returned, it must be freed manually or inserted into another index, otherwise it will leak"]
    pub fn take_buffer(&mut self, ix: BufferIx) -> Option<BufferRes> {
        self.buffers.remove(ix.0)
    }

    pub fn free_buffer(
        &mut self,
        ctx: &VkContext,
        alloc: &mut Allocator,
        buffer: BufferRes,
    ) -> Result<()> {
        buffer.cleanup(ctx, alloc)?;
        Ok(())
    }

    pub fn destroy_buffer(
        &mut self,
        ctx: &VkContext,
        alloc: &mut Allocator,
        ix: BufferIx,
    ) -> Result<()> {
        if let Some(buffer) = self.take_buffer(ix) {
            self.free_buffer(ctx, alloc, buffer)?;
        }
        Ok(())
    }

    //// Image methods

    /*
    pub fn insert_image_at(
        &mut self,
        ctx: &VkContext,
        alloc: &mut Allocator,
        image: ImageRes,
        ix: ImageIx,
    ) -> Result<()> {
        if let Some(_old_res) = self.images.insert_at(ix.0, image) {
            self.free_image(ctx, alloc, ix)?;
        }
        Ok(())
    }
    */

    pub fn allocate_image(
        &mut self,
        ctx: &VkContext,
        allocator: &mut Allocator,
        width: u32,
        height: u32,
        format: vk::Format,
        usage: vk::ImageUsageFlags,
        name: Option<&str>,
    ) -> Result<ImageRes> {
        let image = ImageRes::allocate_2d(
            ctx, allocator, width, height, format, usage, name,
        )?;

        if let Some(name) = name {
            VkEngine::set_debug_object_name(ctx, image.image, name)?;
        }

        Ok(image)
    }

    pub fn insert_image(&mut self, image: ImageRes) -> ImageIx {
        let ix = self.images.insert(image);
        ImageIx(ix)
    }

    #[must_use = "If an ImageRes is returned, it must be freed manually or inserted into another index, otherwise it will leak"]
    pub fn insert_image_at(
        &mut self,
        ix: ImageIx,
        image: ImageRes,
    ) -> Option<ImageRes> {
        self.images.insert_at(ix.0, image)
    }

    #[must_use = "If an ImageRes is returned, it must be freed manually or inserted into another index, otherwise it will leak"]
    pub fn take_image(&mut self, ix: ImageIx) -> Option<ImageRes> {
        self.images.remove(ix.0)
    }

    pub fn free_image(
        &mut self,
        ctx: &VkContext,
        alloc: &mut Allocator,
        image: ImageRes,
    ) -> Result<()> {
        image.cleanup(ctx, alloc)?;
        Ok(())
    }

    pub fn destroy_image(
        &mut self,
        ctx: &VkContext,
        alloc: &mut Allocator,
        ix: ImageIx,
    ) -> Result<()> {
        if let Some(image) = self.take_image(ix) {
            self.free_image(ctx, alloc, image)?;
        }
        Ok(())
    }

    //// Image view methods

    pub fn new_image_view(
        &self,
        ctx: &VkContext,
        image: &ImageRes,
    ) -> Result<vk::ImageView> {
        let view = image.create_image_view(ctx)?;
        Ok(view)
    }

    pub fn insert_image_view(
        &mut self,
        image_view: vk::ImageView,
    ) -> ImageViewIx {
        let ix = self.image_views.insert(image_view);
        ImageViewIx(ix)
    }

    #[must_use = "If a vk::ImageView is returned, it must be freed manually or inserted into another index, otherwise it will leak"]
    pub fn insert_image_view_at(
        &mut self,
        ix: ImageViewIx,
        image_view: vk::ImageView,
    ) -> Option<vk::ImageView> {
        self.image_views.insert_at(ix.0, image_view)
    }

    #[must_use = "If a vk::ImageView is returned, it must be freed manually or inserted into another index, otherwise it will leak"]
    pub fn take_image_view(
        &mut self,
        ix: ImageViewIx,
    ) -> Option<vk::ImageView> {
        self.image_views.remove(ix.0)
    }

    #[must_use = "If a vk::ImageView is returned, it must be freed manually or inserted into another index, otherwise it will leak"]
    pub fn recreate_image_view(
        &mut self,
        ctx: &VkContext,
        image_ix: ImageIx,
        image_view_ix: ImageViewIx,
    ) -> Result<Option<vk::ImageView>> {
        let image = &self[image_ix];
        let new_view = self.new_image_view(ctx, image)?;
        let old_view = self.image_views.insert_at(image_view_ix.0, new_view);
        Ok(old_view)
    }

    //// Sampler methods

    pub fn insert_sampler(
        &mut self,
        context: &VkContext,
        sampler_info: vk::SamplerCreateInfo,
    ) -> Result<SamplerIx> {
        let sampler =
            unsafe { context.device().create_sampler(&sampler_info, None) }?;
        let ix = self.samplers.insert(sampler);
        Ok(SamplerIx(ix))
    }

    #[must_use = "If a vk::Sampler is returned, it must be freed manually or inserted into another index, otherwise it will leak"]
    pub fn take_sampler(&mut self, ix: SamplerIx) -> Option<vk::Sampler> {
        self.samplers.remove(ix.0)
    }

    //// Render pass methods

    /*
    pub fn create_line_render_pass(
        &self,
        ctx: &VkContext,
        color_attch_desc: vk::AttachmentDescription,
        // initial_layout: vk::ImageLayout,
        // final_layout: vk::ImageLayout,
    ) -> Result<vk::RenderPass> {
        let color_attch_desc = vk::AttachmentDescription::builder()
            .format(swapchain_props.format.format)
            .samples(msaa_samples)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::DONT_CARE)
            .initial_layout(initial_layout)
            .final_layout(final_layout)
            .build();
    }
    */

    //// Shader methods

    pub fn load_shader(
        &mut self,
        shader_path: &str,
        stage: vk::ShaderStageFlags,
    ) -> Result<ShaderInfo> {
        let spirv = {
            let mut file = std::fs::File::open(shader_path)?;
            ash::util::read_spv(&mut file)?
        };

        let (sets, pcs) = rspirv_reflect::Reflection::new_from_spirv(
            bytemuck::cast_slice(&spirv),
        )
        .and_then(|i| {
            let sets = i.get_descriptor_sets()?;
            let pcs = i.get_push_constant_range()?;
            Ok((sets, pcs))
        })
        .map_err(|e| anyhow!("{:?}", e))?;

        let push_constant_range = pcs.map(|pc| (pc.offset, pc.size));

        let shader = ShaderInfo {
            spirv,
            set_infos: sets,
            push_constant_range,
            stage,
        };

        Ok(shader)
    }

    pub fn insert_shader(&mut self, shader: ShaderInfo) -> ShaderIx {
        let ix = self.shaders.insert(shader);
        ShaderIx(ix)
    }

    //// Pipeline methods

    pub fn create_graphics_pipeline_tmp(
        &mut self,
        context: &VkContext,
        vert_shader_ix: ShaderIx,
        frag_shader_ix: ShaderIx,
        render_pass: vk::RenderPass,
    ) -> Result<PipelineIx> {
        /*
                if (*&self[vert_shader_ix].stage & vk::ShaderStageFlags::VERTEX).is_empty()
                    ||
        (*&self[frag_shader_ix].stage & vk::ShaderStageFlags::FRAGMENT).is_empty()

                {
                    bail!("Tried to create a compute pipeline from shader {:?} which has stage flags {:?}", shader_ix, &self[shader_ix].stage);
                }
                */

        let vert = self[vert_shader_ix].clone();
        let frag = self[frag_shader_ix].clone();

        let vert_create_info = vk::ShaderModuleCreateInfo::builder()
            .code(&vert.spirv)
            .build();

        let vert_shader_module = unsafe {
            context
                .device()
                .create_shader_module(&vert_create_info, None)
        }?;

        let frag_create_info = vk::ShaderModuleCreateInfo::builder()
            .code(&frag.spirv)
            .build();

        let frag_shader_module = unsafe {
            context
                .device()
                .create_shader_module(&frag_create_info, None)
        }?;

        let pipeline_layout = {
            let vert_pc_range =
                if let Some((offset, size)) = vert.push_constant_range {
                    vk::PushConstantRange::builder()
                        .stage_flags(vk::ShaderStageFlags::VERTEX)
                        .offset(offset)
                        .size(size)
                        .build()
                } else {
                    // TODO not sure if this is correct!
                    vk::PushConstantRange::builder()
                        .stage_flags(vk::ShaderStageFlags::VERTEX)
                        .offset(0)
                        .size(0)
                        .build()
                };

            let frag_pc_range =
                if let Some((offset, size)) = frag.push_constant_range {
                    vk::PushConstantRange::builder()
                        .stage_flags(vk::ShaderStageFlags::FRAGMENT)
                        .offset(offset)
                        .size(size)
                        .build()
                } else {
                    // TODO not sure if this is correct!
                    vk::PushConstantRange::builder()
                        .stage_flags(vk::ShaderStageFlags::FRAGMENT)
                        .offset(0)
                        .size(0)
                        .build()
                };

            let pc_ranges = [vert_pc_range, frag_pc_range];

            let mut layouts = Vec::new();

            layouts.extend(vert.set_infos.keys().filter_map(|&set_ix| {
                let info = vert.set_layout_info(set_ix).ok()?;
                let layout =
                    self.layout_cache.get_descriptor_layout_new(&info).ok()?;
                Some(layout)
            }));

            layouts.extend(frag.set_infos.keys().filter_map(|&set_ix| {
                let info = frag.set_layout_info(set_ix).ok()?;
                let layout =
                    self.layout_cache.get_descriptor_layout_new(&info).ok()?;
                Some(layout)
            }));

            let layout_info = vk::PipelineLayoutCreateInfo::builder()
                .set_layouts(&layouts)
                .push_constant_ranges(&pc_ranges)
                .build();

            unsafe {
                context.device().create_pipeline_layout(&layout_info, None)
            }
        }?;

        let entry_point = std::ffi::CString::new("main").unwrap();

        let vert_state_info = vk::PipelineShaderStageCreateInfo::builder()
            .stage(vk::ShaderStageFlags::VERTEX)
            .module(vert_shader_module)
            .name(&entry_point)
            .build();

        let frag_state_info = vk::PipelineShaderStageCreateInfo::builder()
            .stage(vk::ShaderStageFlags::FRAGMENT)
            .module(frag_shader_module)
            .name(&entry_point)
            .build();

        let stages = [vert_state_info, frag_state_info];

        let pipeline_info = vk::GraphicsPipelineCreateInfo::builder()
            .layout(pipeline_layout)
            .stages(&stages)
            .render_pass(render_pass)
            .build();

        /*
        let mut pipeline_info = vk::GraphicsPipelineCreateInfo::builder()
            .stages(&shader_stages_create_infos)
            .vertex_input_state(&vert_input_info)
            .input_assembly_state(&input_assembly_info)
            .viewport_state(&viewport_info)
            .dynamic_state(&dynamic_state_info)
            .rasterization_state(&rasterizer_info)
            .multisample_state(&multisampling_info)
            .color_blend_state(&color_blending_info)
            .layout(layout)
            .render_pass(render_pass)
            .subpass(0);
            */

        let pipeline_infos = [pipeline_info];

        let result = unsafe {
            context.device().create_graphics_pipelines(
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
            context
                .device()
                .destroy_shader_module(vert_shader_module, None);
            context
                .device()
                .destroy_shader_module(frag_shader_module, None);
        }

        Ok(PipelineIx(ix))
    }

    pub fn create_compute_pipeline(
        &mut self,
        context: &VkContext,
        shader_ix: ShaderIx,
    ) -> Result<PipelineIx> {
        if (*&self[shader_ix].stage & vk::ShaderStageFlags::COMPUTE).is_empty()
        {
            bail!("Tried to create a compute pipeline from shader {:?} which has stage flags {:?}", shader_ix, &self[shader_ix].stage);
        }

        let shader = self[shader_ix].clone();

        let create_info = vk::ShaderModuleCreateInfo::builder()
            .code(&shader.spirv)
            .build();

        let shader_module = unsafe {
            context.device().create_shader_module(&create_info, None)
        }?;

        let pipeline_layout = {
            let pc_range =
                if let Some((offset, size)) = shader.push_constant_range {
                    vk::PushConstantRange::builder()
                        .stage_flags(vk::ShaderStageFlags::COMPUTE)
                        .offset(offset)
                        .size(size)
                        .build()
                } else {
                    // TODO not sure if this is correct!
                    vk::PushConstantRange::builder()
                        .stage_flags(vk::ShaderStageFlags::COMPUTE)
                        .offset(0)
                        .size(0)
                        .build()
                };

            let pc_ranges = [pc_range];

            let layouts = shader
                .set_infos
                .keys()
                .filter_map(|&set_ix| {
                    let info = shader.set_layout_info(set_ix).ok()?;
                    let layout = self
                        .layout_cache
                        .get_descriptor_layout_new(&info)
                        .ok()?;
                    Some(layout)
                })
                .collect::<Vec<_>>();

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

    //// Descriptor set methods

    pub fn allocate_desc_set_raw<F>(
        &mut self,
        layout_info: &DescriptorLayoutInfo,
        set_info: &BTreeMap<u32, DescriptorInfo>,
        write_builder: F,
    ) -> Result<vk::DescriptorSet>
    where
        F: FnOnce(&Self, &mut DescriptorUpdateBuilder) -> Result<()>,
    {
        let layout =
            self.layout_cache.get_descriptor_layout_new(&layout_info)?;

        let desc_set = self.descriptor_allocator.allocate(layout)?;

        let mut builder = DescriptorUpdateBuilder::new(set_info);

        write_builder(self, &mut builder)?;

        builder.apply(
            &mut self.layout_cache,
            &mut self.descriptor_allocator,
            desc_set,
        );

        Ok(desc_set)
    }

    pub fn allocate_desc_set<F>(
        &mut self,
        shader_ix: ShaderIx,
        set: u32,
        write_builder: F,
    ) -> Result<vk::DescriptorSet>
    where
        F: FnOnce(&Self, &mut DescriptorUpdateBuilder) -> Result<()>,
    {
        let layout_info = self[shader_ix].set_layout_info(set)?;

        let set_info = self[shader_ix].set_infos.get(&set).ok_or(anyhow!(
            "Tried to allocate descriptor set {} for incompatible shader",
            set
        ))?;

        let set_info = set_info.clone();

        self.allocate_desc_set_raw(&layout_info, &set_info, write_builder)
    }

    pub fn insert_desc_set(
        &mut self,
        desc_set: vk::DescriptorSet,
    ) -> DescSetIx {
        let ix = self.descriptor_sets.insert(desc_set);
        DescSetIx(ix)
    }

    pub fn insert_desc_set_at(
        &mut self,
        ix: DescSetIx,
        desc_set: vk::DescriptorSet,
    ) -> Option<vk::DescriptorSet> {
        self.descriptor_sets.insert_at(ix.0, desc_set)
    }

    //// Semaphores and fences

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

    pub fn destroy_fence(
        &mut self,
        device: &Device,
        fence: FenceIx,
    ) -> Option<()> {
        let f = self.fences.remove(fence.0)?;

        unsafe {
            device.destroy_fence(f, None);
        };

        Some(())
    }

    // pub fn create_image_view_for_image(
    //     &mut self,
    //     ctx: &VkContext,
    //     image_ix: ImageIx,
    // ) -> Result<ImageViewIx> {
    //     let img = self.images.get(image_ix.0).ok_or(anyhow!(
    //         "tried to create image view for nonexistent image"
    //     ))?;

    //     let view = img.create_image_view(ctx)?;
    //     let ix = self.image_views.insert(view);

    //     Ok(ImageViewIx(ix))
    // }

    //// Command methods

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

        VkEngine::transition_image(
            cmd,
            device,
            image.image,
            src_access_mask,
            src_stage_mask,
            dst_access_mask,
            dst_stage_mask,
            old_layout,
            new_layout,
        );
    }

    //// Cleanup

    pub fn cleanup(
        &mut self,
        ctx: &VkContext,
        alloc: &mut Allocator,
    ) -> Result<()> {
        let device = ctx.device();

        self.descriptor_allocator.cleanup()?;

        self.layout_cache.cleanup()?;

        for (_ix, &sampler) in self.samplers.iter() {
            unsafe {
                device.destroy_sampler(sampler, None);
            }
        }

        for (_ix, &view) in self.image_views.iter() {
            unsafe {
                device.destroy_image_view(view, None);
            }
        }

        for (_, image) in self.images.drain() {
            image.cleanup(ctx, alloc)?;
        }

        for (_, buffer) in self.buffers.drain() {
            buffer.cleanup(ctx, alloc)?;
        }

        for (_ix, &semaphore) in self.semaphores.iter() {
            unsafe {
                device.destroy_semaphore(semaphore, None);
            }
        }

        for (_ix, &fence) in self.fences.iter() {
            unsafe {
                device.destroy_fence(fence, None);
            }
        }

        for (_ix, (pipeline, layout)) in self.pipelines.iter() {
            unsafe {
                device.destroy_pipeline_layout(*layout, None);
                device.destroy_pipeline(*pipeline, None);
            }
        }

        Ok(())
    }
}
