use ash::{
    extensions::khr::{Surface, Swapchain},
    vk, Device, Entry,
};

use crossbeam::atomic::AtomicCell;
use gpu_allocator::vulkan::Allocator;
use rspirv_reflect::DescriptorInfo;
use rustc_hash::FxHashMap;
use winit::window::Window;

use anyhow::{anyhow, bail, Result};

use std::{
    collections::{BTreeMap, HashMap},
    sync::Arc,
};

use thunderdome::{Arena, Index};

pub mod context;
pub mod debug;
pub mod descriptor;
pub mod init;
pub mod resource;
pub mod util;

pub use resource::*;

use context::Queues;

pub use context::VkContext;

use crate::{compositor::Compositor, vk::descriptor::DescriptorLayoutInfo};

pub const FRAME_OVERLAP: usize = 2;

pub struct VkEngine {
    pub allocator: Allocator,
    pub resources: GpuResources,
    pub context: VkContext,

    pub queues: Queues,

    pub swapchain: Swapchain,
    pub swapchain_khr: vk::SwapchainKHR,
    pub swapchain_props: SwapchainProperties,

    pub swapchain_images: Vec<vk::Image>,
    #[allow(dead_code)]
    pub swapchain_image_views: Vec<vk::ImageView>,

    // pub swapchain_texture_sets: Vec<DescSetIx>,
    pub swapchain_storage_desc_sets: Vec<DescSetIx>,

    command_pool: vk::CommandPool,

    frame_number: usize,
}

#[derive(Default, Clone)]
pub struct BatchInput {
    pub swapchain_image: Option<vk::Image>,
    pub storage_set: Option<DescSetIx>,
}

pub type WinSizeResourcesFn = Arc<
    dyn Fn(
            u32,
            u32,
            &VkContext,
            &mut GpuResources,
            &mut Allocator,
        ) -> Result<WinSizeResourcesBuilder>
        + Send
        + Sync
        + 'static,
>;

pub struct FrameResources {
    semaphores: Vec<SemaphoreIx>,
    // semaphore_map: FxHashMap<(usize, usize), SemaphoreIx>,
    fence: FenceIx,
    command_buffers: Vec<vk::CommandBuffer>,

    executing: AtomicCell<bool>,
    // framebuffers: Vec<vk::Framebuffer>,
    // window_size_resources_fn: Option<WinSizeResourcesFn>,
    // window_size_indices: WinSizeIndices,
    // window_size_desc_sets: HashMap<String, vk::DescriptorSet>,
}

// #[derive(Clone, Copy, PartialEq)]
// pub enum WinSizeDescSetInfo {
//     StorageImage { format: vk::Format },
//     SampledImage { format: vk::Format },
// }

// impl WinSizeDescSetInfo {

// }

// #[derive(Debug, Default, Clone)]
// pub struct WinSizeIndices {
//     pub images: HashMap<String, ImageIx>,
//     pub image_views: HashMap<String, ImageViewIx>,
//     pub desc_sets: HashMap<String, DescSetIx>,
//     pub framebuffers: HashMap<String, FramebufferIx>,
// }

/*
#[derive(Default)]
pub struct WinSizeResources {
    indices: WinSizeIndices,
}

impl WinSizeResources {

}
*/

impl FrameResources {
    pub fn new(
        ctx: &VkContext,
        res: &mut GpuResources,
        queue_ix: u32,
        semaphore_count: usize,
        cmd_buf_count: usize,
    ) -> Result<Self> {
        let semaphores = (0..semaphore_count)
            .filter_map(|_| res.allocate_semaphore(ctx).ok())
            .collect();

        let fence = res.allocate_fence(ctx)?;

        let create_flags = vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER;

        let command_pool_info = vk::CommandPoolCreateInfo::builder()
            .queue_family_index(queue_ix)
            .flags(create_flags)
            .build();

        let command_pool = unsafe {
            ctx.device().create_command_pool(&command_pool_info, None)
        }?;

        let command_buffers = {
            let alloc_info = vk::CommandBufferAllocateInfo::builder()
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_pool(command_pool)
                .command_buffer_count(cmd_buf_count as u32)
                .build();

            let bufs =
                unsafe { ctx.device().allocate_command_buffers(&alloc_info) }?;

            bufs
        };

        Ok(Self {
            semaphores,
            fence,
            command_buffers,

            executing: false.into(),
            // window_size_resources_fn: None,
            // window_size_desc_sets: HashMap::default(),
            // window_size_indices: WinSizeIndices::default(),
        })
    }

    /*
    pub fn set_window_size_resources<F>(&mut self, f: F) -> Result<()>
    where
        F: Fn(
                u32,
                u32,
                &VkContext,
                &mut GpuResources,
                &mut Allocator,
            ) -> Result<WinSizeResourcesBuilder>
            + Send
            + Sync
            + 'static,
    {
        self.window_size_resources = Some(Arc::new(f) as _);
        todo!();
        Ok(())
    }
    */
}

impl VkEngine {
    pub fn new(window: &Window) -> Result<Self> {
        let entry = Entry::linked();

        log::debug!("Created Vulkan entry");
        let instance = init::create_instance(&entry, window)?;
        log::debug!("Created Vulkan instance");

        let surface = Surface::new(&entry, &instance);
        let surface_khr = unsafe {
            ash_window::create_surface(&entry, &instance, window, None)
        }?;
        log::debug!("Created window surface");

        let debug_utils = debug::setup_debug_utils(&entry, &instance);

        let (physical_device, graphics_ix) = init::choose_physical_device(
            &instance,
            &surface,
            surface_khr,
            None,
        )?;

        let (device, graphics_queue) = init::create_logical_device(
            &instance,
            physical_device,
            graphics_ix,
        )?;

        let allocator_create = gpu_allocator::vulkan::AllocatorCreateDesc {
            instance: instance.clone(),
            device: device.clone(),
            physical_device,
            debug_settings: Default::default(),
            buffer_device_address: false,
        };

        let mut allocator =
            gpu_allocator::vulkan::Allocator::new(&allocator_create)?;

        let vk_context = VkContext::new(
            entry,
            instance,
            debug_utils,
            surface,
            surface_khr,
            physical_device,
            device,
        )?;

        let winit::dpi::PhysicalSize { width, height } = window.inner_size();

        let (swapchain, swapchain_khr, swapchain_props, images) =
            init::create_swapchain_and_images(
                &vk_context,
                graphics_ix,
                [width, height],
            )?;

        let swapchain_image_views = init::create_swapchain_image_views(
            vk_context.device(),
            &images,
            swapchain_props,
        )?;

        let msaa_samples = vk_context.get_max_usable_sample_count();

        let queues = Queues::init(graphics_queue, graphics_ix)?;

        let frame_number = 0;

        let mut resources = GpuResources::new(&vk_context)?;

        let swapchain_storage_desc_sets = {
            let layout = {
                let mut info = DescriptorLayoutInfo::default();

                let binding = vk::DescriptorSetLayoutBinding::builder()
                    .binding(0)
                    .descriptor_count(1)
                    .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                    .stage_flags(
                        vk::ShaderStageFlags::COMPUTE
                            | vk::ShaderStageFlags::VERTEX
                            | vk::ShaderStageFlags::FRAGMENT,
                    )
                    .build();

                info.bindings.push(binding);
                info
            };

            let set_info = {
                let info = DescriptorInfo {
                    ty: rspirv_reflect::DescriptorType::STORAGE_IMAGE,
                    binding_count: rspirv_reflect::BindingCount::One,
                    name: "out_image".to_string(),
                };

                Some((0u32, info)).into_iter().collect::<BTreeMap<_, _>>()
            };

            let mut views = Vec::new();
            for view in swapchain_image_views.iter() {
                let desc_set = resources.allocate_desc_set_raw(
                    &layout,
                    &set_info,
                    |res, builder| {
                        let info = ash::vk::DescriptorImageInfo::builder()
                            .image_layout(vk::ImageLayout::GENERAL)
                            .image_view(*view)
                            .build();

                        builder.bind_image(0, &[info]);

                        Ok(())
                    },
                )?;
                let set_ix = resources.insert_desc_set(desc_set);
                views.push(set_ix);
            }
            views
        };

        let device = vk_context.device();

        let create_flags = vk::CommandPoolCreateFlags::empty();
        let command_pool_info = vk::CommandPoolCreateInfo::builder()
            .queue_family_index(graphics_ix)
            .flags(create_flags)
            .build();

        let command_pool =
            unsafe { device.create_command_pool(&command_pool_info, None) }?;

        let engine = VkEngine {
            allocator,
            resources,

            context: vk_context,
            queues,
            swapchain,
            swapchain_khr,
            swapchain_props,
            swapchain_images: images,
            swapchain_image_views,

            swapchain_storage_desc_sets,

            command_pool,

            frame_number,
        };

        Ok(engine)
    }

    pub fn cleanup_swapchain(&mut self) {
        let device = self.device();
        unsafe {
            self.swapchain_image_views
                .iter()
                .for_each(|v| device.destroy_image_view(*v, None));

            self.swapchain.destroy_swapchain(self.swapchain_khr, None);
        }
    }

    pub fn wait_gpu_idle(&self) -> Result<()> {
        unsafe { self.device().device_wait_idle() }?;
        Ok(())
    }

    pub fn swapchain_dimensions(&self) -> [u32; 2] {
        [
            self.swapchain_props.extent.width,
            self.swapchain_props.extent.height,
        ]
    }

    pub fn recreate_swapchain(
        &mut self,
        dimensions: Option<[u32; 2]>,
    ) -> Result<()> {
        // TODO should only wait on the present queue, when i've added
        // multiple queue family support
        self.wait_gpu_idle()?;

        self.cleanup_swapchain();

        let dimensions = dimensions.unwrap_or([
            self.swapchain_props.extent.width,
            self.swapchain_props.extent.height,
        ]);

        let graphics_ix = self.queues.thread.queue_family_index;

        let (swapchain, swapchain_khr, swapchain_props, images) =
            init::create_swapchain_and_images(
                self.ctx(),
                graphics_ix,
                dimensions,
            )?;

        let swapchain_image_views = init::create_swapchain_image_views(
            self.device(),
            &images,
            swapchain_props,
        )?;

        self.swapchain = swapchain;
        self.swapchain_khr = swapchain_khr;
        self.swapchain_props = swapchain_props;
        self.swapchain_images = images;

        // storage image desc sets
        {
            let layout = {
                let mut info = DescriptorLayoutInfo::default();

                let binding = vk::DescriptorSetLayoutBinding::builder()
                    .binding(0)
                    .descriptor_count(1)
                    .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                    .stage_flags(
                        vk::ShaderStageFlags::COMPUTE
                            | vk::ShaderStageFlags::VERTEX
                            | vk::ShaderStageFlags::FRAGMENT,
                    )
                    .build();

                info.bindings.push(binding);
                info
            };

            let set_info = {
                let info = DescriptorInfo {
                    ty: rspirv_reflect::DescriptorType::STORAGE_IMAGE,
                    binding_count: rspirv_reflect::BindingCount::One,
                    name: "out_image".to_string(),
                };

                Some((0u32, info)).into_iter().collect::<BTreeMap<_, _>>()
            };

            let mut views = Vec::new();
            for (ix, view) in self
                .swapchain_storage_desc_sets
                .iter()
                .zip(swapchain_image_views.iter())
            {
                let desc_set = self.resources.allocate_desc_set_raw(
                    &layout,
                    &set_info,
                    |res, builder| {
                        let info = ash::vk::DescriptorImageInfo::builder()
                            .image_layout(vk::ImageLayout::GENERAL)
                            .image_view(*view)
                            .build();

                        builder.bind_image(0, &[info]);

                        Ok(())
                    },
                )?;
                self.resources.insert_desc_set_at(*ix, desc_set);
                views.push(desc_set);
            }
            views
        };

        self.swapchain_image_views = swapchain_image_views;

        // (later) create render passes (render passes that depend on
        // swapchain extent)

        // recreate framebuffers?

        // signal that resources should be recreated?

        Ok(())
    }

    pub fn allocate_command_buffer(&mut self) -> Result<vk::CommandBuffer> {
        let ctx = &self.context;

        let command_buffer = {
            let alloc_info = vk::CommandBufferAllocateInfo::builder()
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_pool(self.command_pool)
                .command_buffer_count(1)
                .build();

            let bufs =
                unsafe { ctx.device().allocate_command_buffers(&alloc_info) }?;

            bufs[0]
        };

        Ok(command_buffer)
    }

    pub fn free_command_buffer(&mut self, cmd: vk::CommandBuffer) {
        unsafe {
            self.context
                .device()
                .free_command_buffers(self.command_pool, &[cmd]);
        };
    }

    pub fn submit_queue_semaphore(
        &mut self,
        cmd: vk::CommandBuffer,
        semaphore: SemaphoreIx,
        wait_dst_stage_mask: vk::PipelineStageFlags,
    ) -> Result<FenceIx> {
        let ctx = &self.context;
        let device = ctx.device();
        let fence = self.resources.allocate_fence(ctx)?;

        let cmds = [cmd];

        let semaphore = self.resources[semaphore];

        let wait_semaphores = [semaphore];
        let wait_mask = [wait_dst_stage_mask];

        let batch = vk::SubmitInfo::builder()
            .command_buffers(&cmds)
            .wait_semaphores(&wait_semaphores)
            .wait_dst_stage_mask(&wait_mask)
            .signal_semaphores(&[])
            .build();

        let submit_infos = [batch];

        let queue = self.queues.thread.queue;

        unsafe {
            device.queue_submit(queue, &submit_infos, self.resources[fence])?;
        }

        Ok(fence)
    }

    pub fn submit_queue_fn<F, T>(&mut self, f: F) -> Result<T>
    where
        F: FnOnce(
            &VkContext,
            &mut GpuResources,
            &mut Allocator,
            vk::CommandBuffer,
        ) -> Result<T>,
    {
        let cmd = self.allocate_command_buffer()?;
        let cmd_begin_info = vk::CommandBufferBeginInfo::builder()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

        unsafe {
            self.context
                .device()
                .begin_command_buffer(cmd, &cmd_begin_info)?;
        }

        let result =
            f(&self.context, &mut self.resources, &mut self.allocator, cmd);

        unsafe { self.context.device().end_command_buffer(cmd) }?;

        if result.is_err() {
            self.free_command_buffer(cmd);
        }

        let result = result?;

        let ctx = &self.context;
        let device = ctx.device();
        let fence = self.resources.allocate_fence(ctx)?;

        let cmds = [cmd];

        let batch = vk::SubmitInfo::builder()
            .command_buffers(&cmds)
            .wait_semaphores(&[])
            .wait_dst_stage_mask(&[])
            .signal_semaphores(&[])
            .build();

        let submit_infos = [batch];

        let queue = self.queues.thread.queue;

        unsafe {
            device.queue_submit(queue, &submit_infos, self.resources[fence])?;
        }

        self.block_on_fence(fence)?;

        self.free_command_buffer(cmd);

        Ok(result)
    }

    pub fn submit_queue(&mut self, cmd: vk::CommandBuffer) -> Result<FenceIx> {
        let ctx = &self.context;
        let device = ctx.device();
        let fence = self.resources.allocate_fence(ctx)?;

        let cmds = [cmd];

        let batch = vk::SubmitInfo::builder()
            .command_buffers(&cmds)
            .wait_semaphores(&[])
            .wait_dst_stage_mask(&[])
            .signal_semaphores(&[])
            .build();

        let submit_infos = [batch];

        let queue = self.queues.thread.queue;

        unsafe {
            device.queue_submit(queue, &submit_infos, self.resources[fence])?;
        }

        Ok(fence)
    }

    pub fn with_resources<T, F>(&self, f: F) -> Result<T>
    where
        F: FnOnce(&VkContext, &GpuResources) -> Result<T>,
    {
        f(&self.context, &self.resources)
    }

    pub fn with_allocators_ref<T, F>(&self, f: F) -> Result<T>
    where
        F: FnOnce(&VkContext, &GpuResources, &Allocator) -> Result<T>,
    {
        f(&self.context, &self.resources, &self.allocator)
    }

    pub fn with_allocators<T, F>(&mut self, f: F) -> Result<T>
    where
        F: FnOnce(&VkContext, &mut GpuResources, &mut Allocator) -> Result<T>,
    {
        f(&self.context, &mut self.resources, &mut self.allocator)
    }

    pub fn ctx(&self) -> &VkContext {
        &self.context
    }

    pub fn device(&self) -> &Device {
        self.context.device()
    }

    pub fn current_frame_number(&self) -> usize {
        self.frame_number
    }

    pub fn dispatch_compute(
        resources: &GpuResources,
        device: &Device,
        cmd: vk::CommandBuffer,
        pipeline_ix: PipelineIx,
        desc_set_indices: &[DescSetIx],
        push_constants: &[u8],
        groups: (u32, u32, u32),
    ) -> vk::CommandBuffer {
        let (pipeline, pipeline_layout) =
            resources[pipeline_ix].pipeline_and_layout();

        let desc_sets = desc_set_indices
            .iter()
            .map(|&ix| resources[ix])
            .collect::<Vec<_>>();

        unsafe {
            let bind_point = vk::PipelineBindPoint::COMPUTE;
            device.cmd_bind_pipeline(cmd, bind_point, pipeline);

            let null = [];

            device.cmd_bind_descriptor_sets(
                cmd,
                bind_point,
                pipeline_layout,
                0,
                &desc_sets,
                &null,
            );

            if !push_constants.is_empty() {
                device.cmd_push_constants(
                    cmd,
                    pipeline_layout,
                    vk::ShaderStageFlags::COMPUTE,
                    0,
                    push_constants,
                );
            }
        };

        unsafe { device.cmd_dispatch(cmd, groups.0, groups.1, groups.2) };

        cmd
    }

    pub fn copy_buffer(
        device: &Device,
        cmd: vk::CommandBuffer,
        src: vk::Buffer,
        dst: vk::Buffer,
        len: usize,
        src_offset: Option<u64>,
        dst_offset: Option<u64>,
    ) {
        let region = vk::BufferCopy {
            src_offset: src_offset.unwrap_or_default(),
            dst_offset: dst_offset.unwrap_or_default(),
            size: len as u64,
        };
        let regions = [region];

        unsafe { device.cmd_copy_buffer(cmd, src, dst, &regions) };
    }

    pub fn copy_buffer_to_image(
        device: &Device,
        cmd: vk::CommandBuffer,
        src: vk::Buffer,
        dst: vk::Image,
        dst_layout: vk::ImageLayout,
        extent: vk::Extent3D,
        src_offset: Option<u64>,
    ) {
        let region = vk::BufferImageCopy::builder()
            .buffer_offset(src_offset.unwrap_or_default())
            .buffer_row_length(0)
            .buffer_image_height(0)
            .image_subresource(vk::ImageSubresourceLayers {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                mip_level: 0,
                base_array_layer: 0,
                layer_count: 1,
            })
            .image_offset(vk::Offset3D { x: 0, y: 0, z: 0 })
            .image_extent(extent)
            .build();

        let regions = [region];

        unsafe {
            device.cmd_copy_buffer_to_image(cmd, src, dst, dst_layout, &regions)
        }
    }

    pub fn copy_image_to_buffer(
        device: &Device,
        cmd: vk::CommandBuffer,
        src: vk::Image,
        dst: vk::Buffer,
        extent: vk::Extent3D,
        dst_offset: Option<u64>,
    ) {
        let region = vk::BufferImageCopy::builder()
            .buffer_offset(dst_offset.unwrap_or_default())
            .buffer_row_length(0)
            .buffer_image_height(0)
            .image_subresource(vk::ImageSubresourceLayers {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                mip_level: 0,
                base_array_layer: 0,
                layer_count: 1,
            })
            .image_offset(vk::Offset3D { x: 0, y: 0, z: 0 })
            .image_extent(vk::Extent3D {
                width: extent.width,
                height: extent.height,
                depth: 1,
            })
            .build();

        let regions = [region];

        unsafe {
            device.cmd_copy_image_to_buffer(
                cmd,
                src,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                dst,
                &regions,
            )
        }
    }

    pub fn copy_image(
        device: &Device,
        cmd: vk::CommandBuffer,
        src: vk::Image,
        dst: vk::Image,
        extent: vk::Extent3D,
        src_layout: vk::ImageLayout,
        dst_layout: vk::ImageLayout,
    ) {
        let src_subres = vk::ImageSubresourceLayers {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            mip_level: 0,
            base_array_layer: 0,
            layer_count: 1,
        };

        let region = vk::ImageCopy::builder()
            .src_offset(vk::Offset3D { x: 0, y: 0, z: 0 })
            .dst_offset(vk::Offset3D { x: 0, y: 0, z: 0 })
            .src_subresource(src_subres)
            .dst_subresource(src_subres)
            .extent(extent)
            .build();

        let regions = [region];

        unsafe {
            device
                .cmd_copy_image(cmd, src, src_layout, dst, dst_layout, &regions)
        };
    }

    pub fn transition_image(
        cmd: vk::CommandBuffer,
        device: &Device,
        image: vk::Image,
        src_access_mask: vk::AccessFlags,
        src_stage_mask: vk::PipelineStageFlags,
        dst_access_mask: vk::AccessFlags,
        dst_stage_mask: vk::PipelineStageFlags,
        old_layout: vk::ImageLayout,
        new_layout: vk::ImageLayout,
    ) {
        let image_barrier = vk::ImageMemoryBarrier::builder()
            .src_access_mask(src_access_mask)
            .dst_access_mask(dst_access_mask)
            .old_layout(old_layout)
            .new_layout(new_layout)
            .image(image)
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

    pub fn submit_batches_fence_alt(
        &mut self,
        batches: &[&dyn Fn(
            &VkContext,
            &mut GpuResources,
            &mut Allocator,
            vk::CommandBuffer,
        ) -> anyhow::Result<()>],
    ) -> Result<FenceIx> {
        let cmd = self.allocate_command_buffer()?;

        let cmd_begin_info = vk::CommandBufferBeginInfo::builder()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

        unsafe {
            self.context
                .device()
                .begin_command_buffer(cmd, &cmd_begin_info)?;
        }

        let ctx = &self.context;
        let dev = ctx.device();

        let res = &mut self.resources;
        let alloc = &mut self.allocator;

        for batch in batches.iter() {
            batch(ctx, res, alloc, cmd)?;
        }

        unsafe { dev.end_command_buffer(cmd) }?;

        let fence_ix = self.submit_queue(cmd)?;

        Ok(fence_ix)
    }

    pub fn submit_batches_fence(
        &mut self,
        batches: &[Arc<
            dyn Fn(
                    &VkContext,
                    &mut GpuResources,
                    &mut Allocator,
                    vk::CommandBuffer,
                ) -> anyhow::Result<()>
                + Send
                + Sync,
        >],
    ) -> Result<FenceIx> {
        let cmd = self.allocate_command_buffer()?;

        let cmd_begin_info = vk::CommandBufferBeginInfo::builder()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

        unsafe {
            self.context
                .device()
                .begin_command_buffer(cmd, &cmd_begin_info)?;
        }

        let ctx = &self.context;
        let dev = ctx.device();

        let res = &mut self.resources;
        let alloc = &mut self.allocator;

        for batch in batches.iter() {
            batch(ctx, res, alloc, cmd)?;
        }

        unsafe { dev.end_command_buffer(cmd) }?;

        let fence_ix = self.submit_queue(cmd)?;

        Ok(fence_ix)
    }

    // TODO this should allow for custom timeouts & not always block
    // and remove
    pub fn block_on_fence(&mut self, ix: FenceIx) -> Result<()> {
        let fence =
            self.resources.fences.remove(ix.0).ok_or(anyhow!(
                "tried to block on fence that does not exist"
            ))?;

        let fences = [fence];

        let dev = self.context.device();

        unsafe {
            dev.wait_for_fences(&fences, true, 10_000_000_000)?;
            dev.reset_fences(&fences)?;
        };

        Ok(())
    }

    pub fn draw_compositor(
        &mut self,
        compositor: &Compositor,
        clear_color: [f32; 3],
        width: u32,
        height: u32,
    ) -> Result<(ImageIx, ImageViewIx)> {
        let render_pass = compositor.clear_pass;

        let format = vk::Format::R8G8B8A8_UNORM;

        let usage = vk::ImageUsageFlags::COLOR_ATTACHMENT
            | vk::ImageUsageFlags::SAMPLED
            | vk::ImageUsageFlags::TRANSFER_SRC;

        // allocate image
        let image = self.resources.allocate_image(
            &self.context,
            &mut self.allocator,
            width,
            height,
            format,
            usage,
            Some(&format!("Compositor Image: ({}, {})", width, height)),
        )?;

        let image_view =
            self.resources.new_image_view(&self.context, &image)?;

        // create framebuffer
        let attchs = [image_view];
        let framebuffer = self.resources.create_framebuffer(
            &self.context,
            render_pass,
            &attchs,
            width,
            height,
        )?;

        VkEngine::set_debug_object_name(
            &self.context,
            framebuffer,
            &format!("Compositor Framebuffer: ({}, {})", width, height),
        )?;

        // let fb_ix = self.resources.insert_framebuffer(framebuffer);
        let cmd = self.allocate_command_buffer()?;

        unsafe {
            let cmd_begin_info = vk::CommandBufferBeginInfo::builder()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

            self.context
                .device()
                .begin_command_buffer(cmd, &cmd_begin_info)?;
        }

        // draw compositor image
        let extent = vk::Extent2D { width, height };
        compositor.draw_impl(
            framebuffer,
            extent,
            Some(clear_color),
            self.context.device(),
            &self.resources,
            cmd,
        )?;

        unsafe {
            self.context.device().end_command_buffer(cmd)?;
        }

        let fence = self.submit_queue(cmd)?;

        // await results
        self.block_on_fence(fence)?;

        // free resources
        self.free_command_buffer(cmd);
        unsafe {
            self.context.device().destroy_framebuffer(framebuffer, None);
        }

        let image = self.resources.insert_image(image);
        let image_view = self.resources.insert_image_view(image_view);

        Ok((image, image_view))
    }

    // this one blocks as it waits on the blitting fence
    pub fn display_image(
        &mut self,
        image: ImageIx,
        src_layout: vk::ImageLayout,
    ) -> Result<bool> {
        let s_img_avail = self.resources.allocate_semaphore(&self.context)?;

        let img_available_semaphore = self.resources[s_img_avail];

        let swapchain_img_ix = unsafe {
            let result = self.swapchain.acquire_next_image(
                self.swapchain_khr,
                std::u64::MAX,
                img_available_semaphore,
                vk::Fence::null(),
            );

            match result {
                Ok((img_index, _)) => img_index,
                Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                    log::warn!("Swapchain out of date");
                    self.resources
                        .destroy_semaphore(&self.context, s_img_avail);
                    return Ok(false);
                }
                Err(error) => bail!(
                    "Error while acquiring next swapchain image: {}",
                    error
                ),
            }
        };

        let cmd = self.allocate_command_buffer()?;

        let [width, height] = self.swapchain_dimensions();

        let dst_extent = vk::Extent2D { width, height };

        let swapchain_img = self.swapchain_images[swapchain_img_ix as usize];

        let src_img = &self.resources[image];
        let img_extent = src_img.extent;

        let device = &self.context.device();
        unsafe {
            let cmd_begin_info = vk::CommandBufferBeginInfo::builder()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

            device.begin_command_buffer(cmd, &cmd_begin_info)?;

            Self::transition_image(
                cmd,
                self.context.device(),
                swapchain_img,
                vk::AccessFlags::NONE,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::AccessFlags::TRANSFER_WRITE,
                vk::PipelineStageFlags::TRANSFER,
                vk::ImageLayout::UNDEFINED,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            );

            let src_subres = vk::ImageSubresourceLayers::builder()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .base_array_layer(0)
                .mip_level(0)
                .layer_count(1)
                .build();

            let dst_subres = src_subres.clone();

            let src_0 = vk::Offset3D::builder().x(0).y(0).z(0).build();

            let src_1 = vk::Offset3D::builder()
                .x(img_extent.width as i32)
                .y(img_extent.height as i32)
                .z(1)
                .build();

            let dst_1 = vk::Offset3D::builder()
                .x(dst_extent.width as i32)
                .y(dst_extent.height as i32)
                .z(1)
                .build();

            let blit = vk::ImageBlit::builder()
                .src_subresource(src_subres)
                .src_offsets([src_0, src_1])
                .dst_subresource(dst_subres)
                .dst_offsets([src_0, dst_1])
                .build();

            let regions = [blit];
            let filter = vk::Filter::LINEAR;
            device.cmd_blit_image(
                cmd,
                src_img.image,
                src_layout,
                swapchain_img,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &regions,
                filter,
            );

            Self::transition_image(
                cmd,
                self.context.device(),
                swapchain_img,
                vk::AccessFlags::TRANSFER_WRITE,
                vk::PipelineStageFlags::TRANSFER,
                vk::AccessFlags::NONE,
                vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                vk::ImageLayout::PRESENT_SRC_KHR,
            );

            device.end_command_buffer(cmd)
        }?;

        let fence = self.submit_queue_semaphore(
            cmd,
            s_img_avail,
            vk::PipelineStageFlags::ALL_GRAPHICS,
        )?;

        self.block_on_fence(fence)?;

        let present_wait = [];
        let swapchains = [self.swapchain_khr];
        let img_indices = [swapchain_img_ix];

        let present_info = vk::PresentInfoKHR::builder()
            .wait_semaphores(&present_wait)
            .swapchains(&swapchains)
            .image_indices(&img_indices)
            .build();

        let queue = self.queues.thread.queue;
        let result =
            unsafe { self.swapchain.queue_present(queue, &present_info) };

        self.wait_gpu_idle()?;
        self.resources.destroy_semaphore(&self.context, s_img_avail);

        self.free_command_buffer(cmd);

        Ok(true)
    }

    pub fn draw_from_batches<'a>(
        &mut self,
        frame: &mut FrameResources,
        batches: &[&Box<
            dyn Fn(&Device, &GpuResources, &BatchInput, vk::CommandBuffer) + 'a,
        >],
        batch_dependencies: &[Option<Vec<(usize, vk::PipelineStageFlags)>>],
        acquire_image_batch: usize,
    ) -> Result<bool> {
        let ctx = &self.context;
        let device = self.context.device();

        // let frame_n = self.frame_number;
        // dbg!(self.frame_number);
        // let f_ix = frame_n % FRAME_OVERLAP;

        if frame.executing.load() {
            let fences = [self.resources[frame.fence]];

            unsafe {
                device.wait_for_fences(&fences, true, 1_000_000_000)?;
                device.reset_fences(&fences)?;
            };
            frame.executing.store(false);
        }

        // TODO hackyyyyyyy
        let present_ix = frame.semaphores.len() - 1;
        let img_available = self.resources[frame.semaphores[present_ix]];

        let swapchain_img_ix = unsafe {
            let result = self.swapchain.acquire_next_image(
                self.swapchain_khr,
                std::u64::MAX,
                img_available,
                vk::Fence::null(),
            );

            match result {
                Ok((img_index, _)) => img_index,
                Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                    log::warn!("Swapchain out of date");
                    return Ok(false);
                }
                Err(error) => bail!(
                    "Error while acquiring next swapchain image: {}",
                    error
                ),
            }
        };

        let swapchain_img = self.swapchain_images[swapchain_img_ix as usize];
        let storage_set =
            self.swapchain_storage_desc_sets[swapchain_img_ix as usize];

        for (ix, &cmd) in frame.command_buffers.iter().enumerate() {
            if batches.get(ix).is_none() {
                break;
            }

            let cmd_begin_info = vk::CommandBufferBeginInfo::builder()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

            unsafe {
                device.reset_command_buffer(
                    cmd,
                    vk::CommandBufferResetFlags::empty(),
                )?;

                device.begin_command_buffer(cmd, &cmd_begin_info)?;
            }

            let batch_input = BatchInput {
                storage_set: Some(storage_set),
                swapchain_image: (ix == acquire_image_batch)
                    .then(|| swapchain_img),
            };

            (&batches[ix])(device, &self.resources, &batch_input, cmd);

            unsafe { device.end_command_buffer(cmd) }?;
        }

        let mut batch_dep_info: Vec<(Vec<usize>, Vec<vk::PipelineStageFlags>)> =
            Vec::new();

        let mut batch_rev_deps: Vec<Vec<usize>> =
            vec![Vec::new(); frame.command_buffers.len()];

        let mut semaphore_map: FxHashMap<(usize, usize), SemaphoreIx> =
            FxHashMap::default();
        let mut sem_ix = 0usize;

        let mut get_semaphore = {
            |a: usize, b: usize| {
                if let Some(s) = semaphore_map.get(&(a, b)) {
                    *s
                } else {
                    let s = frame.semaphores[sem_ix];
                    semaphore_map.insert((a, b), s);
                    sem_ix += 1;
                    s
                }
            }
        };

        for (ix, deps) in batch_dependencies.iter().enumerate() {
            let mut wait = Vec::new();
            let mut wait_mask = Vec::new();

            if let Some(deps) = deps {
                for (dep_ix, mask) in deps {
                    wait.push(*dep_ix);
                    wait_mask.push(*mask);

                    let _ = get_semaphore(*dep_ix, ix);

                    let rev = &mut batch_rev_deps[*dep_ix];
                    rev.push(ix);
                }
            }

            batch_dep_info.push((wait, wait_mask));
        }

        let mut batch_data = Vec::new();

        let mut last_semaphore = None;

        for (ix, ((deps_ix, stages), revs_ix)) in
            std::iter::zip(batch_dep_info, batch_rev_deps).enumerate()
        {
            let mut wait_semaphores = deps_ix
                .into_iter()
                .map(|prev| {
                    let s = get_semaphore(prev, ix);
                    self.resources[s]
                })
                .collect::<Vec<_>>();
            let mut wait_mask =
                stages.into_iter().map(|s| s).collect::<Vec<_>>();

            if ix == acquire_image_batch {
                wait_semaphores.push(img_available);
                wait_mask.push(vk::PipelineStageFlags::BOTTOM_OF_PIPE);
            }

            let mut signal_semaphores = revs_ix
                .into_iter()
                .map(|next| {
                    let s = get_semaphore(ix, next);
                    self.resources[s]
                })
                .collect::<Vec<_>>();

            if ix == batches.len() - 1 {
                let s = get_semaphore(ix, std::usize::MAX);
                signal_semaphores.push(self.resources[s]);
                last_semaphore = Some(s);
            }

            let cmd_bufs = [frame.command_buffers[ix]];

            batch_data.push((
                cmd_bufs,
                wait_semaphores,
                wait_mask,
                signal_semaphores,
            ));
        }

        let queue = self.queues.thread.queue;

        let submit_infos = batch_data
            .iter()
            .map(|(cmd_bufs, wait, wmask, signal)| {
                let submit_info = vk::SubmitInfo::builder()
                    .command_buffers(cmd_bufs)
                    .wait_semaphores(wait)
                    .wait_dst_stage_mask(wmask)
                    .signal_semaphores(signal)
                    .build();

                submit_info
            })
            .collect::<Vec<_>>();

        unsafe {
            device.queue_submit(
                queue,
                &submit_infos,
                self.resources[frame.fence],
            )?;
        }

        frame.executing.store(true);

        let copy_done_semaphore = self.resources[last_semaphore.unwrap()];

        let present_wait = [copy_done_semaphore];
        let swapchains = [self.swapchain_khr];
        let img_indices = [swapchain_img_ix];

        let present_info = vk::PresentInfoKHR::builder()
            .wait_semaphores(&present_wait)
            .swapchains(&swapchains)
            .image_indices(&img_indices)
            .build();

        let result =
            unsafe { self.swapchain.queue_present(queue, &present_info) };

        match result {
            Ok(true) | Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                log::warn!("swapchain surface suboptimal");
                return Ok(false);
            }
            Err(error) => panic!("Failed to present queue: {}", error),
            _ => {}
        }

        unsafe {
            device.queue_wait_idle(queue)?;
        };

        self.frame_number += 1;

        Ok(true)
    }

    pub fn set_debug_object_name<T: ash::vk::Handle>(
        ctx: &VkContext,
        object: T,
        name: &str,
    ) -> Result<()> {
        use std::ffi::CString;

        if let Some(utils) = ctx.debug_utils() {
            let name = CString::new(name)?;

            let debug_name_info = vk::DebugUtilsObjectNameInfoEXT::builder()
                .object_type(T::TYPE)
                .object_handle(object.as_raw())
                .object_name(&name)
                .build();

            unsafe {
                utils.debug_utils_set_object_name(
                    ctx.device().handle(),
                    &debug_name_info,
                )?;
            }
        }

        Ok(())
    }

    /// Given a desired uniform buffer size in bytes, return the
    /// smallest size that matches the Vulkan implementations'
    /// `minUniformBufferOffsetAlignment` limit.
    ///
    /// Any buffer that is used as a uniform buffer must have a size
    /// that's divisible with that device limit.
    pub fn aligned_ubo_size(ctx: &VkContext, ubo_size: usize) -> usize {
        let dev_align =
            ctx.phys_device_props()
                .limits
                .min_uniform_buffer_offset_alignment as usize;

        if ubo_size == 0 {
            return dev_align;
        }

        if ubo_size % dev_align == 0 {
            return ubo_size;
        }

        let blocks = ubo_size / dev_align;

        (1 + blocks) * dev_align
    }
}

#[derive(Clone, Copy, Debug)]
pub struct SwapchainProperties {
    pub extent: vk::Extent2D,
    pub present_mode: vk::PresentModeKHR,
    pub format: vk::SurfaceFormatKHR,
}

struct SwapchainSupportDetails {
    capabilities: vk::SurfaceCapabilitiesKHR,
    formats: Vec<vk::SurfaceFormatKHR>,
    present_modes: Vec<vk::PresentModeKHR>,
}

impl SwapchainSupportDetails {
    fn new(
        device: vk::PhysicalDevice,
        surface: &Surface,
        surface_khr: vk::SurfaceKHR,
    ) -> Result<Self> {
        unsafe {
            let capabilities = surface
                .get_physical_device_surface_capabilities(
                    device,
                    surface_khr,
                )?;

            let formats = surface
                .get_physical_device_surface_formats(device, surface_khr)?;

            let present_modes = surface
                .get_physical_device_surface_present_modes(
                    device,
                    surface_khr,
                )?;

            Ok(Self {
                capabilities,
                formats,
                present_modes,
            })
        }
    }

    fn get_ideal_swapchain_properties(
        &self,
        preferred_dimensions: [u32; 2],
    ) -> SwapchainProperties {
        let format = Self::choose_swapchain_surface_format(&self.formats);
        let present_mode =
            Self::choose_swapchain_surface_present_mode(&self.present_modes);
        let extent = Self::choose_swapchain_extent(
            self.capabilities,
            preferred_dimensions,
        );
        SwapchainProperties {
            format,
            present_mode,
            extent,
        }
    }

    /// Choose the swapchain surface format.
    ///
    /// Will choose B8G8R8A8_UNORM/SRGB_NONLINEAR if possible or
    /// the first available otherwise.
    fn choose_swapchain_surface_format(
        available_formats: &[vk::SurfaceFormatKHR],
    ) -> vk::SurfaceFormatKHR {
        if available_formats.len() == 1
            && available_formats[0].format == vk::Format::UNDEFINED
        {
            return vk::SurfaceFormatKHR {
                format: vk::Format::B8G8R8A8_UNORM,
                color_space: vk::ColorSpaceKHR::SRGB_NONLINEAR,
            };
        }

        *available_formats
            .iter()
            .find(|format| {
                format.format == vk::Format::B8G8R8A8_UNORM
                    && format.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
            })
            .unwrap_or(&available_formats[0])
    }

    /// Choose the swapchain present mode.
    ///
    /// Will favor MAILBOX if present otherwise FIFO.
    /// If none is present it will fallback to IMMEDIATE.
    fn choose_swapchain_surface_present_mode(
        available_present_modes: &[vk::PresentModeKHR],
    ) -> vk::PresentModeKHR {
        let checkit = |v| available_present_modes.contains(&v).then(|| v);

        checkit(vk::PresentModeKHR::FIFO)
            .or(checkit(vk::PresentModeKHR::MAILBOX))
            .unwrap_or(vk::PresentModeKHR::IMMEDIATE)
    }

    /// Choose the swapchain extent.
    ///
    /// If a current extent is defined it will be returned.
    /// Otherwise the surface extent clamped between the min
    /// and max image extent will be returned.
    fn choose_swapchain_extent(
        capabilities: vk::SurfaceCapabilitiesKHR,
        preferred_dimensions: [u32; 2],
    ) -> vk::Extent2D {
        if capabilities.current_extent.width != std::u32::MAX {
            return capabilities.current_extent;
        }

        let min = capabilities.min_image_extent;
        let max = capabilities.max_image_extent;
        let width = preferred_dimensions[0].min(max.width).max(min.width);
        let height = preferred_dimensions[1].min(max.height).max(min.height);
        vk::Extent2D { width, height }
    }
}
