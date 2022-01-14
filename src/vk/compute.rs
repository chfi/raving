use ash::{
    extensions::{
        ext::DebugUtils,
        khr::{Surface, Swapchain},
    },
    // version::{DeviceV1_0, EntryV1_0, InstanceV1_0},
    vk::SurfaceKHR,
};
use ash::{vk, Device, Entry, Instance};

use winit::window::Window;

use std::ffi::{CStr, CString};

use anyhow::Result;

use super::{context::*, debug::*, SwapchainProperties, SwapchainSupportDetails};

pub struct PipelineId(usize);
pub struct DescSetLayoutId(usize);

pub struct ComputeOp {
    pipeline: PipelineId,
    desc_set_layout: DescSetLayoutId,
    // pipeline
    // pipeline: vk::Pipeline,
    // pipeline_layout:
}
