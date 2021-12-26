
use ash::{
    extensions::khr::{Surface, Swapchain},
    // version::DeviceV1_0,
    vk, Device, Entry,
};

use std::{sync::Arc, path::PathBuf};

use anyhow::Result;


pub struct RenderGraph {
    //
}

pub struct FullscreenEffect<Input = ()> {
    // shader_path: Arc<PathBuf>,
    pipeline: vk::Pipeline,
    pipeline_layout: vk::PipelineLayout,

    attch_format: vk::Format,
    attch_initial_layout: vk::ImageLayout,
    attch_final_layout: vk::ImageLayout,

    input: Input,
}

/*
pub enum TaskType {
    CopyBufferToBuffer { src: (), tgt: (), size: usize },
    CopyBufferToImage { src: (), tgt: (), dims: usize },
    CopyImageToBuffer { src: (), tgt: (), dims: usize },
    CopyImageToImage { src: (), tgt: (), dims: usize },


}
*/