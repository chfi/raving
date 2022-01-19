use ash::{
    extensions::khr::{Surface, Swapchain},
    vk, Device, Entry,
};

use gpu_allocator::vulkan::Allocator;
use winit::window::Window;

use anyhow::{anyhow, bail, Result};

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

    queues: Queues,

    pub swapchain: Swapchain,
    pub swapchain_khr: vk::SwapchainKHR,
    pub swapchain_props: SwapchainProperties,

    swapchain_images: Vec<vk::Image>,
    #[allow(dead_code)]
    swapchain_image_views: Vec<vk::ImageView>,

    frames: [FrameData; FRAME_OVERLAP],
    frame_number: usize,
}

impl VkEngine {
    pub fn new(window: &Window) -> Result<Self> {
        let entry = Entry::linked();

        log::debug!("Created Vulkan entry");
        let instance = init::create_instance(&entry, window)?;
        log::debug!("Created Vulkan instance");

        let surface = Surface::new(&entry, &instance);
        let surface_khr = unsafe { ash_window::create_surface(&entry, &instance, window, None) }?;
        log::debug!("Created window surface");

        let debug_utils = debug::setup_debug_utils(&entry, &instance);

        let (physical_device, graphics_ix) =
            init::choose_physical_device(&instance, &surface, surface_khr, None)?;

        let (device, graphics_queue) =
            init::create_logical_device(&instance, physical_device, graphics_ix)?;

        let allocator_create = gpu_allocator::vulkan::AllocatorCreateDesc {
            instance: instance.clone(),
            device: device.clone(),
            physical_device,
            debug_settings: Default::default(),
            buffer_device_address: false,
        };

        let allocator = gpu_allocator::vulkan::Allocator::new(&allocator_create)?;

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
            init::create_swapchain_and_images(&vk_context, graphics_ix, [width, height])?;
        let swapchain_image_views =
            init::create_swapchain_image_views(vk_context.device(), &images, swapchain_props)?;

        let msaa_samples = vk_context.get_max_usable_sample_count();

        let queues = Queues::init(graphics_queue, graphics_ix)?;

        let create_flags = vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER;

        let frames = [
            FrameData::new(&vk_context, graphics_ix, create_flags)?,
            FrameData::new(&vk_context, graphics_ix, create_flags)?,
        ];

        let frame_number = 0;

        let mut resources = GpuResources::new(&vk_context)?;

        let device = vk_context.device();

        let semaphore_info = vk::SemaphoreCreateInfo::builder().build();
        let semaphore = unsafe { device.create_semaphore(&semaphore_info, None) }?;
        resources.semaphores.push(semaphore);
        let semaphore = unsafe { device.create_semaphore(&semaphore_info, None) }?;
        resources.semaphores.push(semaphore);

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

            frames,
            frame_number,
        };

        Ok(engine)
    }

    pub fn ctx(&self) -> &VkContext {
        &self.context
    }

    pub fn device(&self) -> &Device {
        self.context.device()
    }

    pub fn current_frame(&self) -> &FrameData {
        &self.frames[self.frame_number % FRAME_OVERLAP]
    }

    pub fn draw_from_compute(
        &mut self,
        pipeline_ix: Index,
        image_ix: Index,
        desc_set_ix: Index,
        width: u32,
        height: u32,
    ) -> Result<bool> {
        let ctx = &self.context;
        let device = self.context.device();

        let frame_n = self.frame_number;
        let f_ix = frame_n % FRAME_OVERLAP;

        let frame = &self.frames[f_ix];

        if frame_n != f_ix {
            let fences = [frame.render_fence];

            unsafe {
                device.wait_for_fences(&fences, true, 1_000_000_000)?;
                device.reset_fences(&fences)?;
            };
        }

        let img_available = frame.present_semaphore;

        let swapchain_img_ix = unsafe {
            let result = self.swapchain.acquire_next_image(
                self.swapchain_khr,
                std::u64::MAX,
                img_available,
                vk::Fence::null(),
            );

            match result {
                Ok((img_index, _)) => img_index,
                Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => return Ok(true),
                Err(error) => bail!("Error while acquiring next swapchain image: {}", error),
            }
        };

        unsafe {
            device.reset_command_buffer(
                frame.main_command_buffer,
                vk::CommandBufferResetFlags::empty(),
            )?;
            device.reset_command_buffer(
                frame.copy_command_buffer,
                vk::CommandBufferResetFlags::empty(),
            )?;
        };

        let swapchain_img = self.swapchain_images[swapchain_img_ix as usize];

        {
            let cmd = frame.main_command_buffer;

            let cmd_begin_info = vk::CommandBufferBeginInfo::builder()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

            unsafe { device.begin_command_buffer(cmd, &cmd_begin_info) }?;

            self.resources.dispatch_compute(
                device,
                cmd,
                pipeline_ix,
                image_ix,
                desc_set_ix,
                width,
                height,
            )?;

            unsafe { device.end_command_buffer(cmd) }?;
        };

        {
            let cmd = frame.copy_command_buffer;

            let cmd_begin_info = vk::CommandBufferBeginInfo::builder()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

            unsafe { device.begin_command_buffer(cmd, &cmd_begin_info) }?;

            let src_img = self.resources.images.get(image_ix).unwrap();
            let dst_img = swapchain_img;

            // transition swapchain image UNDEFINED -> GENERAL

            use vk::AccessFlags as Access;
            use vk::PipelineStageFlags as Stage;

            let memory_barriers = [];
            let buffer_barriers = [];

            let from_undefined_barrier = vk::ImageMemoryBarrier::builder()
                .src_access_mask(Access::NONE_KHR)
                .dst_access_mask(Access::NONE_KHR)
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .image(dst_img)
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

            let src_stage_mask = Stage::TOP_OF_PIPE;
            let dst_stage_mask = Stage::TRANSFER;

            let image_barriers = [from_undefined_barrier];

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

            let src_layout = vk::ImageLayout::TRANSFER_SRC_OPTIMAL;
            let dst_layout = vk::ImageLayout::TRANSFER_DST_OPTIMAL;

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
                .extent(src_img.extent)
                .build();

            let regions = [region];

            unsafe {
                device.cmd_copy_image(
                    cmd,
                    src_img.image,
                    src_layout,
                    dst_img,
                    dst_layout,
                    &regions,
                )
            };

            // transition swapchain image GENERAL -> PRESENT

            let memory_barriers = [];
            let buffer_barriers = [];

            let from_transfer_barrier = vk::ImageMemoryBarrier::builder()
                .src_access_mask(Access::TRANSFER_WRITE)
                .dst_access_mask(Access::MEMORY_READ)
                .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .new_layout(vk::ImageLayout::PRESENT_SRC_KHR)
                .image(dst_img)
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
            let dst_stage_mask = Stage::TOP_OF_PIPE;

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

            unsafe { device.end_command_buffer(cmd) }?;
        };

        let main_bufs = [frame.main_command_buffer];

        let main_wait = [];
        let main_signal = [frame.render_semaphore];
        let main_wait_stages = [];

        let main_batch = vk::SubmitInfo::builder()
            .wait_semaphores(&main_wait)
            .wait_dst_stage_mask(&main_wait_stages)
            .signal_semaphores(&main_signal)
            .command_buffers(&main_bufs)
            .build();

        let copy_semaphore = self.resources.semaphores[f_ix];

        let copy_bufs = [frame.copy_command_buffer];

        let copy_wait = [frame.render_semaphore, img_available];
        let copy_signal = [copy_semaphore];
        let copy_wait_stages = [
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::PipelineStageFlags::TRANSFER,
        ];

        let copy_batch = vk::SubmitInfo::builder()
            .wait_semaphores(&copy_wait)
            .wait_dst_stage_mask(&copy_wait_stages)
            .signal_semaphores(&copy_signal)
            .command_buffers(&copy_bufs)
            .build();

        let queue = self.queues.thread.queue;

        unsafe {
            let batches = [main_batch, copy_batch];

            device.queue_submit(queue, &batches, frame.render_fence)?;
        }

        let swapchains = [self.swapchain_khr];
        let img_indices = [swapchain_img_ix];

        let present_info = vk::PresentInfoKHR::builder()
            .wait_semaphores(&copy_signal)
            .swapchains(&swapchains)
            .image_indices(&img_indices)
            .build();

        let result = unsafe { self.swapchain.queue_present(queue, &present_info) };

        match result {
            Ok(true) | Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                return Ok(true);
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

    pub fn allocate_image(
        &mut self,
        width: u32,
        height: u32,
        format: vk::Format,
        usage: vk::ImageUsageFlags,
    ) -> Result<Index> {
        self.resources.allocate_image(
            &mut self.allocator,
            &self.context,
            width,
            height,
            format,
            usage,
        )
    }
}

pub struct FrameData {
    present_semaphore: vk::Semaphore,
    render_semaphore: vk::Semaphore,

    render_fence: vk::Fence,

    command_pool: vk::CommandPool,
    main_command_buffer: vk::CommandBuffer,
    copy_command_buffer: vk::CommandBuffer,
}

impl FrameData {
    pub fn new(
        ctx: &VkContext,
        queue_ix: u32,
        create_flags: vk::CommandPoolCreateFlags,
    ) -> Result<Self> {
        let dev = ctx.device();

        let semaphore_info = vk::SemaphoreCreateInfo::builder().build();
        let present_semaphore = unsafe { dev.create_semaphore(&semaphore_info, None) }?;
        let render_semaphore = unsafe { dev.create_semaphore(&semaphore_info, None) }?;

        let fence_info = vk::FenceCreateInfo::builder().build();
        let render_fence = unsafe { dev.create_fence(&fence_info, None) }?;

        let command_pool_info = vk::CommandPoolCreateInfo::builder()
            .queue_family_index(queue_ix)
            .flags(create_flags)
            .build();

        let command_pool = unsafe { dev.create_command_pool(&command_pool_info, None) }?;

        let (main, copy) = {
            let alloc_info = vk::CommandBufferAllocateInfo::builder()
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_pool(command_pool)
                .command_buffer_count(2)
                // .command_buffer_count(2 * FRAME_OVERLAP as u32)
                .build();

            let bufs = unsafe { dev.allocate_command_buffers(&alloc_info) }?;
            (bufs[0], bufs[1])
        };
        let main_command_buffer = main;
        let copy_command_buffer = copy;

        Ok(Self {
            present_semaphore,
            render_semaphore,

            render_fence,
            command_pool,
            main_command_buffer,
            copy_command_buffer,
        })
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
            let capabilities =
                surface.get_physical_device_surface_capabilities(device, surface_khr)?;

            let formats = surface.get_physical_device_surface_formats(device, surface_khr)?;

            let present_modes =
                surface.get_physical_device_surface_present_modes(device, surface_khr)?;

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
        let present_mode = Self::choose_swapchain_surface_present_mode(&self.present_modes);
        let extent = Self::choose_swapchain_extent(self.capabilities, preferred_dimensions);
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
        if available_formats.len() == 1 && available_formats[0].format == vk::Format::UNDEFINED {
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
