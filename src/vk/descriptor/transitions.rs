use std::collections::{HashMap, VecDeque};

use ash::{
    extensions::khr::{Surface, Swapchain},
    // version::DeviceV1_0,
    vk::{self},
    Device,
    Entry,
};

use gpu_allocator::vulkan::Allocator;
use rustc_hash::FxHashMap;
use winit::window::Window;

use anyhow::{anyhow, bail, Result};

use thunderdome::{Arena, Index};

use super::*;

pub fn compute_shader_img_read() -> (vk::AccessFlags, vk::PipelineStageFlags) {
    (
        vk::AccessFlags::SHADER_READ,
        vk::PipelineStageFlags::COMPUTE_SHADER,
    )
}

pub fn compute_shader_img_write() -> (vk::AccessFlags, vk::PipelineStageFlags) {
    (
        vk::AccessFlags::SHADER_WRITE,
        vk::PipelineStageFlags::COMPUTE_SHADER,
    )
}

pub fn copy_img_read() -> (vk::AccessFlags, vk::PipelineStageFlags) {
    (
        vk::AccessFlags::TRANSFER_READ,
        vk::PipelineStageFlags::TRANSFER,
    )
}

pub fn copy_img_write() -> (vk::AccessFlags, vk::PipelineStageFlags) {
    (
        vk::AccessFlags::TRANSFER_WRITE,
        vk::PipelineStageFlags::TRANSFER,
    )
}
