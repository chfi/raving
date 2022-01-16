use ash::{
    extensions::khr::{Surface, Swapchain},
    // version::DeviceV1_0,
    vk,
    Device,
    Entry,
};

use anyhow::Result;
use gpu_allocator::vulkan::{Allocation, AllocationCreateDesc, Allocator};

use super::{context::VkContext, VkEngine};

pub struct ImageRes {
    pub(super) image: vk::Image,
    pub(super) format: vk::Format,

    alloc: Allocation,
    pub(super) layout: vk::ImageLayout,
    img_type: vk::ImageType,

    pub(super) extent: vk::Extent3D,
    // width: u32,
    // height: u32,
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

        let view = unsafe { ctx.device().create_image_view(&create_info, None) }?;

        Ok(view)
    }

    pub fn allocate_2d(
        allocator: &mut Allocator,
        ctx: &VkContext,
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
