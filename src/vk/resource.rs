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

use super::{
    context::VkContext,
    descriptor::{
        BindingDesc, DescriptorAllocator, DescriptorBuilder,
        DescriptorLayoutCache, DescriptorLayoutInfo, DescriptorUpdateBuilder,
    },
    VkEngine,
};

pub mod index;

pub use index::*;

pub struct GpuResources {
    descriptor_allocator: DescriptorAllocator,
    layout_cache: DescriptorLayoutCache,
    descriptor_sets: Arena<vk::DescriptorSet>,

    // compute_pipelines: Arena<ComputePipeline>,
    pipelines: Arena<(vk::Pipeline, vk::PipelineLayout)>,

    shaders: Arena<ShaderInfo>,
    shader_file_cache: HashMap<PathBuf, ShaderIx>,

    buffers: Arena<BufferRes>,

    images: Arena<ImageRes>,
    image_views: Arena<(vk::ImageView, ImageIx)>,
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

    pub fn insert_buffer(&mut self, buffer: BufferRes) -> BufferIx {
        let ix = self.buffers.insert(buffer);
        BufferIx(ix)
    }

    pub fn allocate_buffer(
        &mut self,
        ctx: &VkContext,
        allocator: &mut Allocator,
        location: gpu_allocator::MemoryLocation,
        elem_size: usize,
        len: usize,
        usage: vk::BufferUsageFlags,
        name: Option<&str>,
    ) -> Result<BufferIx> {
        let buffer = BufferRes::allocate(
            ctx, allocator, location, usage, elem_size, len, name,
        )?;

        if let Some(name) = name {
            VkEngine::set_debug_object_name(ctx, buffer.buffer, name)?;
        }

        let ix = self.buffers.insert(buffer);

        Ok(BufferIx(ix))
    }

    pub fn allocate_image(
        &mut self,
        ctx: &VkContext,
        allocator: &mut Allocator,
        width: u32,
        height: u32,
        format: vk::Format,
        usage: vk::ImageUsageFlags,
        name: Option<&str>,
    ) -> Result<ImageIx> {
        let image = ImageRes::allocate_2d(
            ctx, allocator, width, height, format, usage, name,
        )?;

        if let Some(name) = name {
            VkEngine::set_debug_object_name(ctx, image.image, name)?;
        }

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

    pub fn allocate_desc_set<F>(
        &mut self,
        shader_ix: ShaderIx,
        set: u32,
        write_builder: F,
    ) -> Result<DescSetIx>
    where
        F: FnOnce(&mut DescriptorUpdateBuilder) -> Result<()>,
    {
        let layout_info = self[shader_ix].set_layout_info(set)?;

        let layout =
            self.layout_cache.get_descriptor_layout_new(&layout_info)?;

        let desc_set = self.descriptor_allocator.allocate(layout)?;

        let set_info = self[shader_ix].set_infos.get(&set).ok_or(anyhow!(
            "Tried to allocate descriptor set {} for incompatible shader",
            set
        ))?;

        let mut builder = DescriptorUpdateBuilder::new(set_info);

        write_builder(&mut builder);

        builder.apply(
            &mut self.layout_cache,
            &mut self.descriptor_allocator,
            desc_set,
        );

        let ix = self.descriptor_sets.insert(desc_set);

        Ok(DescSetIx(ix))
    }

    pub fn allocate_desc_set_dyn(
        &mut self,
        stage_flags: vk::ShaderStageFlags,
        descriptors: &BTreeMap<u32, rspirv_reflect::DescriptorInfo>,
        inputs: &[rhai::Map],
    ) -> Result<DescSetIx> {
        //
        use std::any::TypeId;

        use crate::script::console::frame::{BindableVar, Resolvable};

        #[rustfmt::skip]
        macro_rules! get_and_cast {
            ($map:ident, $field:literal, $type:ty) => {
                {
                    let val = $map.get($field).unwrap();

                    if val.type_id() == TypeId::of::<BindableVar>() {
                        let var = val.clone_cast::<BindableVar>();
                        let bufdyn = var.value.take();
                        var.value.store(bufdyn.clone());
                        let res =
                            bufdyn.unwrap().cast::<$type>();
                        res
                    } else {
                        let res = val.clone_cast::<Resolvable<$type>>();
                        res.get_unwrap()
                    }
                }
            };
        }

        let mut builder = DescriptorBuilder::begin();

        for input in inputs {
            let binding = input
                .get("binding")
                .ok_or(anyhow!("input record is missing `binding` field"))?;

            let binding = binding.as_int().map_err(|e| anyhow!("{}", e))?;
            let binding = binding as u32;

            let info = descriptors
                .get(&binding)
                .ok_or(anyhow!("descriptors missing binding: {}", binding))?;

            let ash_ty = vk::DescriptorType::from_raw(info.ty.0 as i32);

            use rspirv_reflect::DescriptorType as Ty;

            if matches!(
                info.ty,
                Ty::STORAGE_BUFFER
                    | Ty::UNIFORM_BUFFER
                    | Ty::STORAGE_BUFFER_DYNAMIC
                    | Ty::UNIFORM_BUFFER_DYNAMIC
            ) {
                let buf_ix = get_and_cast!(input, "buffer", BufferIx);

                let buffer = &self.buffers[buf_ix.0];
                let buf_info = vk::DescriptorBufferInfo::builder()
                    .buffer(buffer.buffer)
                    .offset(0)
                    .range(vk::WHOLE_SIZE)
                    .build();

                let buffer_info = [buf_info];
                builder.bind_buffer(binding, &buffer_info, ash_ty, stage_flags);
            }

            if matches!(
                info.ty,
                Ty::STORAGE_IMAGE
                    | Ty::SAMPLED_IMAGE
                    | Ty::COMBINED_IMAGE_SAMPLER
            ) {
                let layout = if info.ty == Ty::STORAGE_IMAGE {
                    vk::ImageLayout::GENERAL
                } else {
                    vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL
                };

                let view_ix = get_and_cast!(input, "image_view", ImageViewIx);
                let (view, _image_ix) = self.image_views[view_ix.0];

                let img_info = vk::DescriptorImageInfo::builder()
                    .image_layout(layout)
                    .image_view(view)
                    .build();

                builder.bind_image(binding, &[img_info], ash_ty, stage_flags);
            }

            if matches!(info.ty, Ty::SAMPLER | Ty::COMBINED_IMAGE_SAMPLER) {
                todo!();
            }

            if matches!(
                info.ty,
                Ty::UNIFORM_TEXEL_BUFFER | Ty::STORAGE_TEXEL_BUFFER
            ) {
                log::error!("Texel buffers not yet supported");
            }
        }

        let set = builder
            .build(&mut self.layout_cache, &mut self.descriptor_allocator)?;

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

    pub fn load_shader(
        &mut self,
        shader_path: &str,
        stage: vk::ShaderStageFlags,
    ) -> Result<ShaderIx> {
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

        let ix = self.shaders.insert(shader);

        Ok(ShaderIx(ix))
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

    pub fn load_compute_shader_runtime(
        &mut self,
        context: &VkContext,
        shader_path: &str,
        bindings: &[BindingDesc],
        push_constants_len: usize,
    ) -> Result<PipelineIx> {
        let comp_src = {
            let mut file = std::fs::File::open(shader_path)?;
            ash::util::read_spv(&mut file)
        }?;

        let (desc_sets, push_const) = {
            let result = rspirv_reflect::Reflection::new_from_spirv(
                bytemuck::cast_slice(&comp_src),
            )
            .and_then(|i| {
                let sets = i.get_descriptor_sets()?;
                let pcs = i.get_push_constant_range()?;
                Ok((sets, pcs))
            });

            match result {
                Ok((sets, pcs)) => {
                    log::warn!("sets: {:?}", sets);
                    if let Some(pcs) = &pcs {
                        log::warn!(
                            "push constants: ({}, {})",
                            pcs.offset,
                            pcs.size
                        );
                    }
                    (sets, pcs)
                }
                Err(err) => {
                    anyhow::bail!("SPIR-V reflection error: {:?}", err);
                }
            }
        };

        // .unwrap();
        // dbg!(info.get_descriptor_sets().unwrap());

        let create_info = vk::ShaderModuleCreateInfo::builder()
            .code(&comp_src)
            .build();

        let shader_module = unsafe {
            context.device().create_shader_module(&create_info, None)
        }?;

        let pipeline_layout = {
            let pc_range = vk::PushConstantRange::builder()
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
                .offset(0)
                .size(push_constants_len as u32)
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

    pub fn free_buffer(
        &mut self,
        ctx: &VkContext,
        alloc: &mut Allocator,
        ix: BufferIx,
    ) -> Result<()> {
        let buf_res = self
            .buffers
            .remove(ix.0)
            .ok_or(anyhow!("Tried to free buffer that did not exist"))?;

        buf_res.cleanup(ctx, alloc)?;

        Ok(())
    }

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

        for (_ix, &(view, _)) in self.image_views.iter() {
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

#[allow(dead_code)]
pub struct BufferRes {
    name: Option<String>,
    pub buffer: vk::Buffer,

    pub elem_size: usize,
    pub len: usize,

    pub location: gpu_allocator::MemoryLocation,

    pub alloc: Allocation,
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

    pub fn size_bytes(&self) -> usize {
        self.len * self.elem_size
    }

    pub fn upload_to_self_bytes(
        &mut self,
        ctx: &VkContext,
        allocator: &mut Allocator,
        src: &[u8],
        cmd: vk::CommandBuffer,
    ) -> Result<Self> {
        // assert!(!self.host_writable());
        let device = ctx.device();

        let staging_usage = vk::BufferUsageFlags::TRANSFER_SRC;
        let location = MemoryLocation::CpuToGpu;

        log::warn!("src: {:?}", src);

        let len = src.len().min(self.size_bytes());

        let mut staging = Self::allocate_for_type::<u8>(
            ctx,
            allocator,
            location,
            staging_usage,
            len,
            Some("tmp staging buffer"),
        )?;

        if let Some(stg) = staging.alloc.mapped_slice_mut() {
            log::warn!("src.len() {}", src.len());
            log::warn!("stg.len() {}", stg.len());
            stg[..src.len()].clone_from_slice(src);
        } else {
            bail!("couldn't map staging buffer memory");
        }

        VkEngine::copy_buffer(
            device,
            cmd,
            staging.buffer,
            self.buffer,
            len,
            None,
            None,
        );

        Ok(staging)
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
    pub image: vk::Image,
    pub format: vk::Format,

    alloc: Allocation,
    pub layout: vk::ImageLayout,
    img_type: vk::ImageType,

    pub extent: vk::Extent3D,
}

impl ImageRes {
    pub fn fill_from_pixels(
        &mut self,
        device: &Device,
        ctx: &VkContext,
        allocator: &mut Allocator,
        pixel_bytes: impl IntoIterator<Item = u8>,
        elem_size: usize,
        layout: vk::ImageLayout,
        cmd: vk::CommandBuffer,
    ) -> Result<BufferRes> {
        let staging_usage = vk::BufferUsageFlags::TRANSFER_SRC;
        let location = MemoryLocation::CpuToGpu;

        let len = self.extent.width * self.extent.height * elem_size as u32;

        let mut staging = BufferRes::allocate_for_type::<u8>(
            ctx,
            allocator,
            location,
            staging_usage,
            len as usize,
            Some("tmp staging buffer"),
        )?;

        let bytes = pixel_bytes.into_iter().collect::<Vec<_>>();

        if let Some(stg) = staging.alloc.mapped_slice_mut() {
            log::warn!("in mapped slice!");
            stg.clone_from_slice(&bytes);
        } else {
            bail!("couldn't map staging buffer memory");
        }

        VkEngine::copy_buffer_to_image(
            device,
            cmd,
            staging.buffer,
            self.image,
            layout,
            self.extent,
            None,
        );

        Ok(staging)
    }

    pub fn cleanup(
        self,
        ctx: &VkContext,
        allocator: &mut Allocator,
    ) -> Result<()> {
        unsafe {
            ctx.device().destroy_image(self.image, None);
        }

        allocator.free(self.alloc)?;
        Ok(())
    }

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
