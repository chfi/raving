use crate::vk::{GpuResources, VkEngine};

use super::resource::index::*;
use ash::{vk, Device};

#[allow(unused_imports)]
use anyhow::{anyhow, bail, Result};

pub fn copy_batch(
    src: ImageIx,
    dst_img: vk::Image,
    device: &Device,
    resources: &GpuResources,
    cmd: vk::CommandBuffer,
) {
    let img = &resources[src];

    VkEngine::transition_image(
        cmd,
        &device,
        dst_img,
        vk::AccessFlags::NONE_KHR,
        vk::PipelineStageFlags::TOP_OF_PIPE,
        vk::AccessFlags::NONE_KHR,
        vk::PipelineStageFlags::TRANSFER,
        vk::ImageLayout::UNDEFINED,
        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
    );

    VkEngine::copy_image(
        &device,
        cmd,
        img.image,
        dst_img,
        img.extent,
        vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
    );

    VkEngine::transition_image(
        cmd,
        &device,
        dst_img,
        vk::AccessFlags::TRANSFER_WRITE,
        vk::PipelineStageFlags::TRANSFER,
        vk::AccessFlags::MEMORY_READ,
        vk::PipelineStageFlags::HOST,
        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        vk::ImageLayout::PRESENT_SRC_KHR,
    );
}
