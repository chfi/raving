use ash::{
    extensions::khr::{Surface, Swapchain},
    // version::DeviceV1_0,
    vk,
    Device,
    Entry,
};

use anyhow::Result;
use gpu_allocator::vulkan::{Allocation, AllocationCreateDesc};

use super::{context::VkContext, VkEngine};

pub struct ImageRes {
    image: vk::Image,
    format: vk::Format,
    alloc: Allocation,
    layout: vk::ImageLayout,
    img_type: vk::ImageType,

    extent: vk::Extent3D,
    // width: u32,
    // height: u32,
}

impl ImageRes {
    pub fn allocate_2d(
        engine: &mut VkEngine,
        // ctx: &VkContext,
        width: u32,
        height: u32,
        format: vk::Format,
        initial_layout: vk::ImageLayout,
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
            .extent(extent)
            .mip_levels(1)
            .array_layers(1)
            .format(format)
            .tiling(tiling)
            .initial_layout(initial_layout)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .samples(sample_count)
            .flags(flags)
            .build();

        let device = engine.context.device();

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

        let alloc = engine.allocator.allocate(&alloc_desc)?;

        unsafe { device.bind_image_memory(image, alloc.memory(), alloc.offset()) }?;

        Ok(Self {
            image,
            format,
            alloc,
            layout: initial_layout,
            img_type,
            extent,
        })
    }
}
