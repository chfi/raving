use ash::{
    extensions::khr::{Surface, Swapchain},
    // version::DeviceV1_0,
    vk::{self},
    Device,
    Entry,
};

use gpu_allocator::vulkan::Allocator;
use winit::window::Window;

use anyhow::Result;

use self::context::{Queues, VkContext, VkQueueThread};

pub mod context;
pub mod debug;
pub mod init;
pub mod resource;
pub mod util;

pub mod compute;
pub mod graph;

pub const FRAME_OVERLAP: usize = 2;

pub struct GpuResources {
    // TODO replace this with a system that allocates new descriptor
    // pools as needed
    descriptor_pool: vk::DescriptorPool,
    pub descriptor_sets: Vec<vk::DescriptorSet>,

    pipelines: Vec<(vk::Pipeline, vk::PipelineLayout)>,

    storage_image_layout: vk::DescriptorSetLayout,
}

impl GpuResources {
    pub fn new(context: &VkContext) -> Result<Self> {
        // TODO replace this with a system that allocates new descriptor
        // pools as needed
        let descriptor_pool = {
            let descriptor_count = 1024;
            let max_sets = descriptor_count * 4;

            let pool_size = |ty: vk::DescriptorType| vk::DescriptorPoolSize {
                ty,
                descriptor_count,
            };

            let sampled_image_size = pool_size(vk::DescriptorType::COMBINED_IMAGE_SAMPLER);
            let storage_buffer_size = pool_size(vk::DescriptorType::STORAGE_BUFFER);
            let storage_image_size = pool_size(vk::DescriptorType::STORAGE_IMAGE);

            let pool_sizes = [sampled_image_size, storage_buffer_size, storage_image_size];

            let pool_info = vk::DescriptorPoolCreateInfo::builder()
                .pool_sizes(&pool_sizes)
                .max_sets(max_sets)
                .build();

            unsafe { context.device().create_descriptor_pool(&pool_info, None) }
        }?;

        let storage_image_layout = {
            use vk::ShaderStageFlags as Stages;

            let output_image_binding = vk::DescriptorSetLayoutBinding::builder()
                .binding(0)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .descriptor_count(1)
                .stage_flags(Stages::COMPUTE)
                .build();

            let bindings = [output_image_binding];

            let layout_info = vk::DescriptorSetLayoutCreateInfo::builder()
                .bindings(&bindings)
                .build();

            let layout = unsafe {
                context
                    .device()
                    .create_descriptor_set_layout(&layout_info, None)
            }?;

            layout
        };

        let result = Self {
            descriptor_pool,
            descriptor_sets: Vec::new(),

            pipelines: Vec::new(),

            storage_image_layout,
        };

        Ok(result)
    }

    // TODO this is completely temporary
    pub fn load_compute_shader(&mut self, context: &VkContext, shader: &[u8]) -> Result<usize> {
        let comp_src = {
            let mut cursor = std::io::Cursor::new(shader);
            ash::util::read_spv(&mut cursor)
        }?;

        let create_info = vk::ShaderModuleCreateInfo::builder()
            .code(&comp_src)
            .build();

        let shader_module = unsafe { context.device().create_shader_module(&create_info, None) }?;

        let pipeline_layout = {
            let pc_size = std::mem::size_of::<[f32; 4]>() + std::mem::size_of::<[i32; 2]>();

            let pc_range = vk::PushConstantRange::builder()
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
                .offset(0)
                .size(pc_size as u32)
                .build();

            let pc_ranges = [pc_range];

            let layouts = [self.storage_image_layout];

            let layout_info = vk::PipelineLayoutCreateInfo::builder()
                .set_layouts(&layouts)
                .push_constant_ranges(&pc_ranges)
                .build();

            unsafe { context.device().create_pipeline_layout(&layout_info, None) }
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

        let i = self.pipelines.len();

        self.pipelines.push((pipeline, pipeline_layout));

        unsafe {
            context.device().destroy_shader_module(shader_module, None);
        }

        Ok(i)
    }

    pub fn storage_image_layout(&self) -> vk::DescriptorSetLayout {
        self.storage_image_layout
    }

    // returns the index of the descriptor set
    pub fn allocate_storage_image_set(&mut self, context: &VkContext) -> Result<usize> {
        let i = self.descriptor_sets.len();

        let layouts = [self.storage_image_layout];

        let alloc_info = vk::DescriptorSetAllocateInfo::builder()
            .descriptor_pool(self.descriptor_pool)
            .set_layouts(&layouts)
            .build();

        let sets = unsafe { context.device().allocate_descriptor_sets(&alloc_info) }?;

        self.descriptor_sets.push(sets[0]);

        Ok(i)
    }
}

pub struct VkEngine {
    allocator: Allocator,
    context: VkContext,

    queues: Queues,

    pub swapchain: Swapchain,
    pub swapchain_khr: vk::SwapchainKHR,
    pub swapchain_props: SwapchainProperties,

    swapchain_images: Vec<vk::Image>,
    swapchain_image_views: Vec<vk::ImageView>,
    // swapchain_framebuffers: Vec<vk::Framebuffer>,
    frames: [FrameData; FRAME_OVERLAP],
    frame_number: usize,
}

impl VkEngine {
    pub fn new(window: &Window) -> Result<Self> {
        let entry = Entry::linked();

        // let instance_exts = init::instance_extensions(&entry)?;

        log::debug!("Created Vulkan entry");
        let instance = init::create_instance(&entry, window)?;
        log::debug!("Created Vulkan instance");

        let surface = Surface::new(&entry, &instance);
        let surface_khr = unsafe { ash_window::create_surface(&entry, &instance, window, None) }?;
        log::debug!("Created window surface");

        let debug_utils = debug::setup_debug_utils(&entry, &instance);

        // let (physical_device, graphics_ix, present_ix, compute_ix) = init::choose_physical_device(
        let (physical_device, graphics_ix) = init::choose_physical_device(
            &instance,
            &surface,
            surface_khr,
            None,
            // args.force_graphics_device.as_deref(),
        )?;

        // let (device, graphics_queue, present_queue, _compute_queue) = init::create_logical_device(
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
        // let allocator = vk_mem::Allocator::new(&allocator_create_info)?;

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
                // present_ix,
                [width, height],
            )?;
        let swapchain_image_views =
            init::create_swapchain_image_views(vk_context.device(), &images, swapchain_props)?;

        let msaa_samples = vk_context.get_max_usable_sample_count();

        let queues = Queues::init(graphics_queue, graphics_ix)?;

        let create_flags = vk::CommandPoolCreateFlags::empty();

        let frames = [
            FrameData::new(&vk_context, graphics_ix, create_flags)?,
            FrameData::new(&vk_context, graphics_ix, create_flags)?,
        ];

        let frame_number = 0;

        let engine = VkEngine {
            allocator,
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

    pub fn current_frame(&self) -> &FrameData {
        &self.frames[self.frame_number % FRAME_OVERLAP]
    }

    pub fn draw_next_frame(&mut self) -> Result<bool> {
        let ctx = &self.context;
        let device = ctx.device();

        let f_ix = self.frame_number % FRAME_OVERLAP;

        // wait for previous frame (no such thing if this is frame zero)
        /*
        if self.frame_number != 0 {
            let prev_ix = (self.frame_number - 1) % FRAME_OVERLAP;
            let prev_frame = &self.frames[prev_ix];

            let fences = [prev_frame.render_fence];

            unsafe {
        // timeout of 1 second
                device.wait_for_fences(fences, wait_all, timeout)
            }
        }
        */

        let present_semaphore = self.frames[f_ix].present_semaphore;

        let img_result = unsafe {
            self.swapchain.acquire_next_image(
                self.swapchain_khr,
                1_000_000_000,
                present_semaphore,
                vk::Fence::null(),
            )
        };

        let swap_img_index = match img_result {
            Ok((img_index)) => img_index,
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => return Ok(false),
            Err(error) => anyhow::bail!("Error while acquiring next image: {}", error),
        };

        unsafe {
            device.reset_command_buffer(
                self.frames[f_ix].main_command_buffer,
                vk::CommandBufferResetFlags::empty(),
            )
        }?;

        let cmd = self.frames[f_ix].main_command_buffer;

        let cmd_begin_info = vk::CommandBufferBeginInfo::builder()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

        unsafe { device.begin_command_buffer(cmd, &cmd_begin_info) }?;

        self.frame_number += 1;

        Ok(true)
    }
}

pub struct FrameData {
    present_semaphore: vk::Semaphore,
    render_semaphore: vk::Semaphore,

    render_fence: vk::Fence,

    command_pool: vk::CommandPool,
    main_command_buffer: vk::CommandBuffer,
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

        let cmd_buf = {
            let alloc_info = vk::CommandBufferAllocateInfo::builder()
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_pool(command_pool)
                .command_buffer_count(FRAME_OVERLAP as u32)
                .build();

            let bufs = unsafe { dev.allocate_command_buffers(&alloc_info) }?;
            bufs[0]
        };
        let main_command_buffer = cmd_buf;

        Ok(Self {
            present_semaphore,
            render_semaphore,

            render_fence,
            command_pool,
            main_command_buffer,
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
