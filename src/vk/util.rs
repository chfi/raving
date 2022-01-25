use crate::graph::GraphDsl;
use crate::vk::descriptor::{BindingDesc, BindingInput};
use crate::vk::{
    BatchInput, DescSetIx, FrameResources, GpuResources, ImageIx, PipelineIx,
    VkEngine,
};
use ash::{vk, Device};

use gpu_allocator::vulkan::Allocator;
use png::Decoder;

use anyhow::Result;

use super::context::VkContext;
use super::BufferIx;

#[derive(Clone, Copy)]
pub struct TextRenderer {
    pub pipeline: PipelineIx,
    pub set: DescSetIx,

    pub font_image: ImageIx,
    pub text_buffer: BufferIx,
    // text_len: usize,
}

impl TextRenderer {
    // font has to be 8x8 monospace, in a png, for now
    pub fn new(
        ctx: &VkContext,
        res: &mut GpuResources,
        allocator: &mut Allocator,
        font_img_path: &str,
    ) -> Result<Self> {
        // dst dims, start pos
        let pc_size = std::mem::size_of::<[i32; 4]>();

        let bindings = [
            BindingDesc::StorageImage { binding: 0 },
            BindingDesc::UniformBuffer { binding: 1 },
            BindingDesc::StorageImage { binding: 2 },
        ];

        todo!();
    }
}

#[derive(Clone, Copy)]
pub struct ExampleState {
    pub fill_pipeline: PipelineIx,
    pub fill_set: DescSetIx,
    pub fill_image: ImageIx,

    pub flip_pipeline: PipelineIx,
    pub flip_set: DescSetIx,
    pub flip_image: ImageIx,

    pub text_pipeline: PipelineIx,
    pub text_set: DescSetIx,
    pub text_image: ImageIx,
}

pub fn text_batch(
    state: ExampleState,
    device: &Device,
    resources: &GpuResources,
    input: &BatchInput,
    cmd: vk::CommandBuffer,
    // pos: (usize, usize),
) {
    let text_src = &resources[state.text_image];
    let dst = &resources[state.fill_image];

    let width = dst.extent.width;
    let height = dst.extent.height;

    /*
    VkEngine::transition_image(
        cmd,
        &device,
        src.image,
        vk::AccessFlags::SHADER_WRITE,
        vk::PipelineStageFlags::COMPUTE_SHADER,
        vk::AccessFlags::SHADER_READ,
        vk::PipelineStageFlags::COMPUTE_SHADER,
        vk::ImageLayout::GENERAL,
        vk::ImageLayout::GENERAL,
    );
    */

    VkEngine::transition_image(
        cmd,
        &device,
        dst.image,
        vk::AccessFlags::SHADER_WRITE,
        vk::PipelineStageFlags::COMPUTE_SHADER,
        vk::AccessFlags::SHADER_WRITE,
        vk::PipelineStageFlags::COMPUTE_SHADER,
        vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
        // vk::ImageLayout::GENERAL,
        vk::ImageLayout::GENERAL,
    );

    let push_constants = [width as u32, height as u32, 0, 0];

    let mut bytes: Vec<u8> = Vec::with_capacity(16);
    bytes.extend_from_slice(bytemuck::cast_slice(&push_constants));

    let x_size = 16;
    let y_size = 16;

    let x_groups = (width / x_size) + width % x_size;
    let y_groups = (height / y_size) + height % y_size;

    let groups = (x_groups, y_groups, 1);

    VkEngine::dispatch_compute(
        resources,
        &device,
        cmd,
        state.text_pipeline,
        state.text_set,
        bytes.as_slice(),
        groups,
    );

    VkEngine::transition_image(
        cmd,
        &device,
        dst.image,
        vk::AccessFlags::SHADER_WRITE,
        vk::PipelineStageFlags::COMPUTE_SHADER,
        vk::AccessFlags::TRANSFER_READ,
        vk::PipelineStageFlags::TRANSFER,
        // vk::AccessFlags::SHADER_WRITE,
        // vk::PipelineStageFlags::COMPUTE_SHADER,
        vk::ImageLayout::GENERAL,
        vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
    );
}

pub fn flip_batch(
    state: ExampleState,
    device: &Device,
    resources: &GpuResources,
    input: &BatchInput,
    cmd: vk::CommandBuffer,
) {
    let src = &resources[state.fill_image];
    let dst = &resources[state.flip_image];

    let width = src.extent.width;
    let height = src.extent.height;

    VkEngine::transition_image(
        cmd,
        &device,
        src.image,
        vk::AccessFlags::SHADER_WRITE,
        vk::PipelineStageFlags::COMPUTE_SHADER,
        vk::AccessFlags::SHADER_READ,
        vk::PipelineStageFlags::COMPUTE_SHADER,
        vk::ImageLayout::GENERAL,
        vk::ImageLayout::GENERAL,
    );

    VkEngine::transition_image(
        cmd,
        &device,
        dst.image,
        vk::AccessFlags::empty(),
        vk::PipelineStageFlags::TOP_OF_PIPE,
        vk::AccessFlags::SHADER_WRITE,
        vk::PipelineStageFlags::COMPUTE_SHADER,
        vk::ImageLayout::UNDEFINED,
        vk::ImageLayout::GENERAL,
    );

    let push_constants = [width as u32, height as u32];

    let mut bytes: Vec<u8> = Vec::with_capacity(8);
    bytes.extend_from_slice(bytemuck::cast_slice(&push_constants));

    let x_size = 16;
    let y_size = 16;

    let x_groups = (width / x_size) + width % x_size;
    let y_groups = (height / y_size) + height % y_size;

    let groups = (x_groups, y_groups, 1);

    VkEngine::dispatch_compute(
        resources,
        &device,
        cmd,
        state.flip_pipeline,
        state.flip_set,
        bytes.as_slice(),
        groups,
    );

    VkEngine::transition_image(
        cmd,
        &device,
        dst.image,
        vk::AccessFlags::SHADER_WRITE,
        vk::PipelineStageFlags::COMPUTE_SHADER,
        vk::AccessFlags::TRANSFER_READ,
        vk::PipelineStageFlags::TRANSFER,
        vk::ImageLayout::GENERAL,
        vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
    );
}

pub fn compute_batch(
    state: ExampleState,
    device: &Device,
    resources: &GpuResources,
    input: &BatchInput,
    cmd: vk::CommandBuffer,
) {
    let image = &resources[state.fill_image];

    let width = image.extent.width;
    let height = image.extent.height;

    VkEngine::transition_image(
        cmd,
        &device,
        image.image,
        vk::AccessFlags::empty(),
        vk::PipelineStageFlags::TOP_OF_PIPE,
        vk::AccessFlags::SHADER_WRITE,
        vk::PipelineStageFlags::COMPUTE_SHADER,
        vk::ImageLayout::UNDEFINED,
        vk::ImageLayout::GENERAL,
    );

    let push_constants = [width as u32, height as u32];

    // let color = [1f32, 0.0, 0.0, 1.0];
    // let mut bytes: Vec<u8> = Vec::with_capacity(24);
    // bytes.extend_from_slice(bytemuck::cast_slice(&color));

    let mut bytes: Vec<u8> = Vec::with_capacity(8);
    bytes.extend_from_slice(bytemuck::cast_slice(&push_constants));

    let x_size = 16;
    let y_size = 16;

    let x_groups = (width / x_size) + width % x_size;
    let y_groups = (height / y_size) + height % y_size;

    let groups = (x_groups, y_groups, 1);

    VkEngine::dispatch_compute(
        resources,
        &device,
        cmd,
        state.fill_pipeline,
        state.fill_set,
        bytes.as_slice(),
        groups,
    );

    VkEngine::transition_image(
        cmd,
        &device,
        image.image,
        vk::AccessFlags::SHADER_WRITE,
        vk::PipelineStageFlags::COMPUTE_SHADER,
        vk::AccessFlags::TRANSFER_READ,
        vk::PipelineStageFlags::TRANSFER,
        vk::ImageLayout::GENERAL,
        vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
    );
}

pub fn copy_batch(
    state: ExampleState,
    device: &Device,
    resources: &GpuResources,
    input: &BatchInput,
    cmd: vk::CommandBuffer,
) {
    let image = &resources[state.fill_image];

    let src_img = image.image;

    let dst_img = input.swapchain_image.unwrap();

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
        src_img,
        dst_img,
        image.extent,
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
