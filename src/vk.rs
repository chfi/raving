use ash::{
    extensions::khr::{Surface, Swapchain},
    vk, Device, Entry,
};

use crossbeam::atomic::AtomicCell;
use gpu_allocator::vulkan::Allocator;
use rustc_hash::FxHashMap;
use winit::window::Window;

use anyhow::{anyhow, bail, Result};

use std::sync::Arc;

use thunderdome::{Arena, Index};

pub mod context;
pub mod debug;
pub mod descriptor;
pub mod init;
pub mod resource;
pub mod util;

pub use resource::*;

use context::{Queues, VkContext};

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

    command_pool: vk::CommandPool,

    frame_number: usize,
}

#[derive(Default, Clone)]
pub struct BatchInput {
    pub swapchain_image: Option<vk::Image>,
}

pub struct FrameResources {
    semaphores: Vec<SemaphoreIx>,
    // semaphore_map: FxHashMap<(usize, usize), SemaphoreIx>,
    fence: FenceIx,
    command_buffers: Vec<vk::CommandBuffer>,

    executing: AtomicCell<bool>,
}

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
        })
    }
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

        let width = 800u32;
        let height = 600u32;

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

        let resources = GpuResources::new(&vk_context)?;

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

            command_pool,

            frame_number,
        };

        Ok(engine)
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
        desc_set_ix: DescSetIx,
        push_constants: &[u8],
        groups: (u32, u32, u32),
    ) -> vk::CommandBuffer {
        let (pipeline, pipeline_layout) = resources[pipeline_ix];

        let desc_set = resources[desc_set_ix];

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

    pub fn draw_from_batches(
        &mut self,
        frame: &mut FrameResources,
        batches: &[&Box<
            dyn Fn(&Device, &GpuResources, &BatchInput, vk::CommandBuffer),
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
                    log::warn!("error out of date");
                    return Ok(false);
                }
                Err(error) => bail!(
                    "Error while acquiring next swapchain image: {}",
                    error
                ),
            }
        };

        let swapchain_img = self.swapchain_images[swapchain_img_ix as usize];

        for (ix, &cmd) in frame.command_buffers.iter().enumerate() {
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
            batch_dep_info.into_iter().zip(batch_rev_deps).enumerate()
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
