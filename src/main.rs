use engine::script::console::frame::{FrameBuilder, Resolvable};
use engine::script::console::BatchBuilder;
use engine::vk::{
    BatchInput, BufferIx, DescSetIx, FrameResources, GpuResources, ImageIx,
    ImageViewIx, PipelineIx, VkEngine,
};

use engine::vk::util::*;

use ash::{vk, Device};

use engine::vk::descriptor::{BindingDesc, BindingInput};
use flexi_logger::{Duplicate, FileSpec, Logger};
use winit::event::{Event, WindowEvent};
// use winit::platform::unix::*;
use winit::{event_loop::EventLoop, window::WindowBuilder};

use std::sync::Arc;

use anyhow::Result;

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

    let (out_image, out_view) = engine.with_allocators(|ctx, res, alloc| {
        let out_image = res.allocate_image(
            ctx,
            alloc,
            width,
            height,
            vk::Format::R8G8B8A8_UNORM,
            vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::TRANSFER_SRC,
            Some("out_image"),
        )?;

        let out_view = res.create_image_view_for_image(ctx, out_image)?;

        Ok((out_image, out_view))
    })?;

    log::warn!("MODULE BUILDER");

    let mut builder = FrameBuilder::from_script("test.rhai")?;

    builder.bind_var("out_image", out_image)?;
    builder.bind_var("out_view", out_view)?;

    engine.with_allocators(|ctx, res, alloc| {
        builder.resolve(ctx, res, alloc)?;
        Ok(())
    })?;
    log::warn!("is resolved: {}", builder.is_resolved());

    let mut rhai_engine = engine::script::console::create_batch_engine();

    let arc_module = Arc::new(builder.module.clone());

    rhai_engine.register_static_module("self", arc_module.clone());

    let init = rhai::Func::<(), BatchBuilder>::create_from_ast(
        rhai_engine,
        builder.ast.clone_functions_only(),
        "init",
    );

    let mut rhai_engine = engine::script::console::create_batch_engine();
    rhai_engine.register_static_module("self", arc_module.clone());

    let draw_background =
        rhai::Func::<(i64, i64, f32), BatchBuilder>::create_from_ast(
            rhai_engine,
            builder.ast.clone_functions_only(),
            "background",
        );

    let mut rhai_engine = engine::script::console::create_batch_engine();
    rhai_engine.register_static_module("self", arc_module);

    let draw_at = rhai::Func::<(i64, i64, f32), BatchBuilder>::create_from_ast(
        rhai_engine,
        builder.ast.clone_functions_only(),
        "draw_at",
    );

    let mut frames = {
        let queue_ix = engine.queues.thread.queue_family_index;

        // hardcoded for now
        let semaphore_count = 3;
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

    let copy_batch = Box::new(
        move |dev: &Device,
              res: &GpuResources,
              input: &BatchInput,
              cmd: vk::CommandBuffer| {
            copy_batch(out_image, input.swapchain_image.unwrap(), dev, res, cmd)
        },
    ) as Box<_>;

    std::thread::sleep(std::time::Duration::from_millis(100));

    let start = std::time::Instant::now();

    {
        let init_builder = init()?;

        let fence =
            engine.submit_batches_fence(init_builder.init_fn.as_slice())?;

        engine.block_on_fence(fence)?;
    }

    event_loop.run(move |event, _, control_flow| {
        *control_flow = winit::event_loop::ControlFlow::Poll;

        let mut _dirty_swapchain = false;

        match event {
            Event::MainEventsCleared => {
                let t = start.elapsed().as_secs_f32();

                let f_ix = engine.current_frame_number();
                let frame = &mut frames[f_ix % engine::vk::FRAME_OVERLAP];

                let bg_batch = draw_background(800, 600, t).unwrap();
                let bg_batch_fn = bg_batch.build();
                let bg_rhai_batch = bg_batch_fn.clone();

                let batch = draw_at(800, 600, t).unwrap();
                let batch_fn = batch.build();
                let rhai_batch = batch_fn.clone();

                let bg_batch = Box::new(
                    move |dev: &Device,
                          res: &GpuResources,
                          _input: &BatchInput,
                          cmd: vk::CommandBuffer| {
                        bg_rhai_batch(dev, res, cmd);
                    },
                ) as Box<_>;

                let text_batch = Box::new(
                    move |dev: &Device,
                          res: &GpuResources,
                          _input: &BatchInput,
                          cmd: vk::CommandBuffer| {
                        rhai_batch(dev, res, cmd);
                    },
                ) as Box<_>;

                let batches = [&bg_batch, &text_batch, &copy_batch];

                let deps = vec![
                    None,
                    Some(vec![(0, vk::PipelineStageFlags::COMPUTE_SHADER)]),
                    Some(vec![(1, vk::PipelineStageFlags::COMPUTE_SHADER)]),
                ];

                let render_success = engine
                    .draw_from_batches(frame, &batches, deps.as_slice(), 2)
                    .unwrap();

                if !render_success {
                    _dirty_swapchain = true;
                }
            }
            Event::RedrawEventsCleared => {
                //
            }
            Event::WindowEvent { event, .. } => match event {
                WindowEvent::CloseRequested => {
                    log::debug!("WindowEvent::CloseRequested");
                    *control_flow = winit::event_loop::ControlFlow::Exit;
                }
                WindowEvent::Resized { .. } => {
                    _dirty_swapchain = true;
                }
                _ => (),
            },
            Event::LoopDestroyed => {
                log::debug!("Event::LoopDestroyed");

                unsafe {
                    let queue = engine.queues.thread.queue;
                    engine.context.device().queue_wait_idle(queue).unwrap();
                };

                let ctx = &engine.context;
                let res = &mut engine.resources;
                let alloc = &mut engine.allocator;

                res.cleanup(ctx, alloc).unwrap();
            }
            _ => (),
        }
    });
}
