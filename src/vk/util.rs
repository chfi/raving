use crate::graph::GraphDsl;
use crate::vk::descriptor::{BindingDesc, BindingInput};
use crate::vk::{BatchInput, FrameResources, GpuResources, VkEngine};

use super::resource::index::*;
use ash::{vk, Device};

use gpu_allocator::vulkan::Allocator;
// use png::Decoder;

use anyhow::Result;

use super::context::VkContext;

#[derive(Clone, Copy)]
pub struct TextRenderer {
    pub pipeline: PipelineIx,
    pub set: DescSetIx,

    pub font_image: ImageIx,
    pub font_view: ImageViewIx,
    pub text_buffer: BufferIx,
    // text_len: usize,
}

impl TextRenderer {
    // font has to be 8x8 monospace, in a png, for now
    pub fn new(
        engine: &mut VkEngine,
        // ctx: &VkContext,
        // res: &mut GpuResources,
        // allocator: &mut Allocator,
        font_img_path: &str,
        out_image: ImageViewIx,
    ) -> Result<Self> {
        // dst dims, start pos
        let pc_size = std::mem::size_of::<[i32; 4]>();

        let bindings = [
            BindingDesc::StorageImage { binding: 0 },
            BindingDesc::UniformBuffer { binding: 1 },
            BindingDesc::StorageImage { binding: 2 },
        ];

        let result = engine.with_allocators(|ctx, res, alloc| {
            let pipeline = res.load_compute_shader_runtime(
                ctx,
                "shaders/text.comp.spv",
                &bindings,
                pc_size,
            )?;

            let font_image = res.allocate_image(
                ctx,
                alloc,
                1024,
                8,
                vk::Format::R8G8B8A8_UNORM,
                vk::ImageUsageFlags::STORAGE
                    | vk::ImageUsageFlags::TRANSFER_DST,
                Some("text:font_image"),
            )?;
            let font_view = res.create_image_view_for_image(ctx, font_image)?;

            let usage = vk::BufferUsageFlags::TRANSFER_DST
                | vk::BufferUsageFlags::UNIFORM_BUFFER;

            let text_buffer = res.allocate_buffer(
                ctx,
                alloc,
                1,
                256,
                usage,
                Some("text:text_data_buffer"),
            )?;

            let inputs = [
                BindingInput::ImageView {
                    binding: 0,
                    view: font_view,
                },
                BindingInput::Buffer {
                    binding: 1,
                    buffer: text_buffer,
                },
                BindingInput::ImageView {
                    binding: 2,
                    view: out_image,
                },
            ];

            let set = res.allocate_desc_set(
                &bindings,
                &inputs,
                vk::ShaderStageFlags::COMPUTE,
            )?;

            Ok(Self {
                pipeline,
                set,
                font_image,
                font_view,
                text_buffer,
            })
        })?;

        {
            let res = &engine.resources;
            engine.set_debug_object_name(
                res[result.pipeline].0,
                "text:pipeline",
            )?;
            engine.set_debug_object_name(
                res[result.set],
                "text:descriptor_set",
            )?;
            engine.set_debug_object_name(
                res[result.font_image].image,
                "text:font_image",
            )?;
            engine.set_debug_object_name(
                res[result.text_buffer].buffer,
                "text:text_data_buffer",
            )?;
        }

        {
            let cmd = engine.allocate_command_buffer()?;

            let cmd_begin_info = vk::CommandBufferBeginInfo::builder()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

            unsafe {
                engine
                    .context
                    .device()
                    .begin_command_buffer(cmd, &cmd_begin_info)?;
            }

            let context = &engine.context;
            let res = &mut engine.resources;
            let alloc = &mut engine.allocator;

            let font_vk = &mut res[result.font_image];

            //////////////

            VkEngine::transition_image(
                cmd,
                context.device(),
                font_vk.image,
                vk::AccessFlags::empty(),
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::AccessFlags::TRANSFER_WRITE,
                vk::PipelineStageFlags::TRANSFER,
                vk::ImageLayout::UNDEFINED,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            );

            use image::io::Reader as ImageReader;

            let font = ImageReader::open(font_img_path)?.decode()?;
            let font_rgba8 = font.as_rgba8().unwrap();

            let pixel_bytes =
                font_rgba8.enumerate_pixels().flat_map(|(_, _, col)| {
                    let [r, g, b, a] = col.0;
                    [r, g, b, a].into_iter()
                });

            let staging = font_vk.fill_from_pixels(
                context.device(),
                context,
                alloc,
                pixel_bytes,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                cmd,
            )?;

            VkEngine::transition_image(
                cmd,
                context.device(),
                font_vk.image,
                vk::AccessFlags::TRANSFER_WRITE,
                vk::PipelineStageFlags::TRANSFER,
                vk::AccessFlags::MEMORY_READ,
                vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            );

            //////////////

            unsafe { engine.context.device().end_command_buffer(cmd) }?;

            let fence_ix = engine.submit_queue(cmd)?;
            let fence = engine.resources[fence_ix];

            let fences = [fence];
            unsafe {
                engine.context.device().wait_for_fences(
                    &fences,
                    true,
                    10_000_000_000,
                )?;
                engine.context.device().reset_fences(&fences)?;
                engine.context.device().destroy_fence(fence, None);
            };

            staging.cleanup(&engine.context, &mut engine.allocator)?;

            engine.free_command_buffer(cmd);
        }

        Ok(result)
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
