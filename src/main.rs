use engine::graph::GraphDsl;
use engine::vk::{
    BatchInput, DescSetIx, FrameResources, GpuResources, ImageIx, PipelineIx,
    VkEngine,
};

use ash::{vk, Device};

use engine::vk::descriptor::{BindingDesc, BindingInput};
use flexi_logger::{Duplicate, FileSpec, Logger};
use winit::event::{Event, WindowEvent};
// use winit::platform::unix::*;
use winit::{event_loop::EventLoop, window::WindowBuilder};

use anyhow::Result;

#[derive(Clone, Copy)]
struct ExampleState {
    fill_pipeline: PipelineIx,
    fill_set: DescSetIx,
    fill_image: ImageIx,

    flip_pipeline: PipelineIx,
    flip_set: DescSetIx,
    flip_image: ImageIx,

    text_pipeline: PipelineIx,
    text_set: DescSetIx,
    text_image: ImageIx,
}

fn flip_batch(
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

fn compute_batch(
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

    // VkEngine::transition_image(
    //     cmd,
    //     &device,
    //     image.image,
    //     vk::AccessFlags::SHADER_WRITE,
    //     vk::PipelineStageFlags::COMPUTE_SHADER,
    //     vk::AccessFlags::TRANSFER_READ,
    //     vk::PipelineStageFlags::TRANSFER,
    //     vk::ImageLayout::GENERAL,
    //     vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
    // );
}

fn copy_batch(
    state: ExampleState,
    device: &Device,
    resources: &GpuResources,
    input: &BatchInput,
    cmd: vk::CommandBuffer,
) {
    let image = &resources[state.flip_image];

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

fn main() -> Result<()> {
    // let args: Args = argh::from_env();
    // let _logger = set_up_logger(&args).unwrap();

    let spec = "debug";
    let _logger = Logger::try_with_env_or_str(spec)?
        .log_to_file(FileSpec::default())
        .duplicate_to_stderr(Duplicate::Debug)
        .start()?;

    let event_loop = EventLoop::new();

    let width = 800;
    let height = 600;

    let window = WindowBuilder::new()
        .with_title("engine")
        .with_inner_size(winit::dpi::PhysicalSize::new(width, height))
        .build(&event_loop)?;

    let mut engine = VkEngine::new(&window)?;

    let example_state = engine.with_allocators(|ctx, res, alloc| {
        let fill_bindings = [BindingDesc::StorageImage { binding: 0 }];

        // let fill_pc_size =
        //     std::mem::size_of::<[i32; 2]>() + std::mem::size_of::<[f32; 4]>();
        let fill_pc_size = std::mem::size_of::<[i32; 2]>();

        let fill_pipeline = res.load_compute_shader_runtime(
            ctx,
            "shaders/trig_color.comp.spv",
            &fill_bindings,
            fill_pc_size,
        )?;

        let flip_bindings = [
            BindingDesc::StorageImage { binding: 0 },
            BindingDesc::StorageImage { binding: 1 },
        ];
        let flip_pc_size = std::mem::size_of::<[i32; 2]>();

        let flip_pipeline = res.load_compute_shader_runtime(
            ctx,
            "shaders/flip.comp.spv",
            &flip_bindings,
            flip_pc_size,
        )?;

        let text_bindings = [
            BindingDesc::StorageImage { binding: 0 },
            BindingDesc::StorageImage { binding: 1 },
        ];
        let text_pc_size = std::mem::size_of::<[i32; 4]>();

        let text_pipeline = res.load_compute_shader_runtime(
            ctx,
            "shaders/text.comp.spv",
            &text_bindings,
            text_pc_size,
        )?;

        let fill_image = res.allocate_image(
            ctx,
            alloc,
            width,
            height,
            vk::Format::R8G8B8A8_UNORM,
            vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::TRANSFER_SRC,
        )?;

        let flip_image = res.allocate_image(
            ctx,
            alloc,
            width,
            height,
            vk::Format::R8G8B8A8_UNORM,
            vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::TRANSFER_SRC,
        )?;

        let text_image = res.allocate_image(
            ctx,
            alloc,
            1024,
            8,
            vk::Format::R8G8B8A8_UNORM,
            vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::TRANSFER_DST,
        )?;

        let fill_view = res.create_image_view_for_image(ctx, fill_image)?;
        let flip_view = res.create_image_view_for_image(ctx, flip_image)?;
        let text_view = res.create_image_view_for_image(ctx, text_image)?;

        let fill_inputs = [BindingInput::ImageView {
            binding: 0,
            view: fill_view,
        }];

        let fill_set = res.allocate_desc_set(
            &fill_bindings,
            &fill_inputs,
            vk::ShaderStageFlags::COMPUTE,
        )?;

        let flip_inputs = [
            BindingInput::ImageView {
                binding: 0,
                view: fill_view,
            },
            BindingInput::ImageView {
                binding: 1,
                view: flip_view,
            },
        ];
        let flip_set = res.allocate_desc_set(
            &flip_bindings,
            &flip_inputs,
            vk::ShaderStageFlags::COMPUTE,
        )?;

        let text_inputs = [
            BindingInput::ImageView {
                binding: 0,
                view: fill_view,
            },
            BindingInput::ImageView {
                binding: 1,
                view: flip_view,
            },
        ];
        let text_set = res.allocate_desc_set(
            &flip_bindings,
            &flip_inputs,
            vk::ShaderStageFlags::COMPUTE,
        )?;

        Ok(ExampleState {
            fill_pipeline,
            fill_set,
            fill_image,

            flip_pipeline,
            flip_set,
            flip_image,

            text_pipeline,
            text_set,
            text_image,
        })
    })?;

    let mut text_buffer = engine.with_allocators(|ctx, res, alloc| {
        let elem_size = std::mem::size_of::<u32>();
        let len = 1024 * 8;
        let format = vk::Format::R8G8B8A8_UNORM;
        let usage = vk::BufferUsageFlags::TRANSFER_SRC
            | vk::BufferUsageFlags::TRANSFER_DST;

        let buf =
            res.allocate_buffer(ctx, alloc, elem_size, len, format, usage)?;

        Ok(buf)
    })?;

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

        let mut bytes = Vec::with_capacity(1024 * 8 * 4);

        for ix in 0..(1024 * 8) {
            let col = ix % 1024;
            let row = ix / 1024;

            let v = ((col % 32) * 4) as u8;

            bytes.push(v);
            bytes.push(v);
            bytes.push(v);
            bytes.push(255);
        }

        let context = &engine.context;
        let res = &mut engine.resources;
        let alloc = &mut engine.allocator;

        let buffer = &mut res[text_buffer];

        buffer.upload_to_self_bytes(
            context.device(),
            context,
            alloc,
            bytes.as_slice(),
            cmd,
        )?;

        let text_src = &res[text_buffer];
        let text_img = &res[example_state.text_image];

        VkEngine::transition_image(
            cmd,
            context.device(),
            text_img.image,
            vk::AccessFlags::empty(),
            vk::PipelineStageFlags::TOP_OF_PIPE,
            vk::AccessFlags::TRANSFER_WRITE,
            vk::PipelineStageFlags::TRANSFER,
            vk::ImageLayout::GENERAL,
            vk::ImageLayout::GENERAL,
        );

        VkEngine::copy_buffer_to_image(
            context.device(),
            cmd,
            text_src.buffer,
            text_img.image,
            vk::ImageLayout::GENERAL,
            vk::Extent3D {
                width: 1024,
                height: 8,
                depth: 1,
            },
            None,
        );

        /*
        let src_r = &engine.resources[text_buffer];
        let dst_r = &engine.resources[example_state.text_image];

        let extent = vk::Extent3D {
            width: 1024,
            height: 8,
            depth: 1,
        };

        VkEngine::copy_buffer_to_image(
            engine.context.device(),
            cmd,
            src_r.buffer,
            dst_r.image,
            extent,
            None,
        );
        */

        unsafe { engine.context.device().end_command_buffer(cmd) }?;

        let fence_ix = engine.submit_queue(cmd)?;
        let fence = engine.resources[fence_ix];

        let fences = [fence];
        unsafe {
            engine.context.device().wait_for_fences(
                &fences,
                true,
                1_000_000_000,
            )?;
            engine.context.device().reset_fences(&fences)?;
        };

        engine.free_command_buffer(cmd);
    }

    /*
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

        let src_r = &engine.resources[text_buffer];
        let dst_r = &engine.resources[example_state.text_image];

        let extent = vk::Extent3D {
            width: 1024,
            height: 8,
            depth: 1,
        };

        VkEngine::copy_buffer_to_image(
            engine.context.device(),
            cmd,
            src_r.buffer,
            dst_r.image,
            extent,
            None,
        );

        unsafe { engine.context.device().end_command_buffer(cmd) }?;

        let fence_ix = engine.submit_queue(cmd)?;
        let fence = engine.resources[fence_ix];

        let fences = [fence];
        unsafe {
            engine.context.device().wait_for_fences(
                &fences,
                true,
                1_000_000_000,
            )?;
            engine.context.device().reset_fences(&fences)?;
        };

        engine.free_command_buffer(cmd);
    }
    */

    /*
    let (pipeline, image, desc_set) =
        engine.with_allocators(|ctx, res, alloc| {
            let bindings = [BindingDesc::StorageImage { binding: 0 }];

            // let pc_size_1 = std::mem::size_of::<[i32; 2]>()
            //     + std::mem::size_of::<[f32; 4]>();
            let pc_size_1 = std::mem::size_of::<[i32; 2]>();

            // let pipeline = res.load_compute_shader_runtime(
            //     ctx,
            //     "shaders/fill_color.comp.spv",
            //     &bindings,
            //     pc_size_1,
            // )?;
            let pipeline = res.load_compute_shader_runtime(
                ctx,
                "shaders/trig_color.comp.spv",
                &bindings,
                pc_size_1,
            )?;

            let image = res.allocate_image(
                ctx,
                alloc,
                width,
                height,
                // right now this image is copied to the swapchain, which on
                // my system uses BGRA rather than RGBA, so this is just a
                // temporary fix
                // vk::Format::B8G8R8A8_UNORM,
                vk::Format::R8G8B8A8_UNORM,
                vk::ImageUsageFlags::STORAGE
                    | vk::ImageUsageFlags::TRANSFER_SRC,
            )?;

            let view = res.create_image_view_for_image(ctx, image)?;

            let bind_inputs = [BindingInput::ImageView { binding: 0, view }];

            let set = res.allocate_desc_set(
                &bindings,
                &bind_inputs,
                vk::ShaderStageFlags::COMPUTE,
            )?;

            Ok((pipeline, image, set))
        })?;
    */

    let mut frames = {
        let queue_ix = engine.queues.thread.queue_family_index;

        let semaphore_count = 32;
        let cmd_buf_count = 3;

        let mut new_frame = || {
            engine
                .with_allocators(|ctx, res, _alloc| {
                    FrameResources::new(
                        ctx,
                        res,
                        queue_ix,
                        semaphore_count,
                        cmd_buf_count,
                    )
                })
                .unwrap()
        };
        [new_frame(), new_frame()]
    };

    let main_batch = Box::new(
        move |dev: &Device,
              res: &GpuResources,
              input: &BatchInput,
              cmd: vk::CommandBuffer| {
            compute_batch(example_state, dev, res, input, cmd)
        },
    ) as Box<_>;

    let copy_batch = Box::new(
        move |dev: &Device,
              res: &GpuResources,
              input: &BatchInput,
              cmd: vk::CommandBuffer| {
            copy_batch(example_state, dev, res, input, cmd)
        },
    ) as Box<_>;

    let flip_batch = Box::new(
        move |dev: &Device,
              res: &GpuResources,
              input: &BatchInput,
              cmd: vk::CommandBuffer| {
            flip_batch(example_state, dev, res, input, cmd)
        },
    ) as Box<_>;

    let batches = [main_batch, flip_batch, copy_batch];

    let deps = vec![
        None,
        Some(vec![(0, vk::PipelineStageFlags::COMPUTE_SHADER)]),
        Some(vec![(1, vk::PipelineStageFlags::COMPUTE_SHADER)]),
    ];

    std::thread::sleep(std::time::Duration::from_millis(100));

    let start = std::time::Instant::now();

    event_loop.run(move |event, _, control_flow| {
        *control_flow = winit::event_loop::ControlFlow::Poll;

        let mut _dirty_swapchain = false;

        match event {
            Event::MainEventsCleared => {
                let t = start.elapsed().as_secs_f32();

                let r = (t.sin() + 1.0) / 2.0;
                let b = (t.cos() + 1.0) / 2.0;

                let color = [r, 1.0, b, 1.0];

                let f_ix = engine.current_frame_number();
                let frame = &mut frames[f_ix % engine::vk::FRAME_OVERLAP];

                let render_success = engine
                    .draw_from_batches(frame, &batches, deps.as_slice(), 2)
                    .unwrap();

                /*
                let render_success = engine
                    .draw_from_compute(
                        pipeline, image, desc_set, width, height, color,
                    )
                    .unwrap();
                */

                if !render_success {
                    _dirty_swapchain = true;
                }
            }
            Event::RedrawEventsCleared => {
                //
            }
            Event::WindowEvent { event, .. } => match event {
                WindowEvent::CloseRequested => {
                    log::trace!("WindowEvent::CloseRequested");
                    *control_flow = winit::event_loop::ControlFlow::Exit;
                }
                WindowEvent::Resized { .. } => {
                    _dirty_swapchain = true;
                }
                _ => (),
            },
            Event::LoopDestroyed => {
                log::trace!("Event::LoopDestroyed");
            }
            _ => (),
        }
    });
}
