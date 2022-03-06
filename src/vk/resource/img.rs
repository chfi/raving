use ash::{vk, Device};

use gpu_allocator::{
    vulkan::{Allocation, AllocationCreateDesc, Allocator},
    MemoryLocation,
};

#[allow(unused_imports)]
use anyhow::{anyhow, bail, Result};

use crate::vk::{VkContext, VkEngine};

use super::BufferRes;

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
