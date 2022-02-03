use engine::script::console::frame::FrameBuilder;
use engine::script::console::{BatchBuilder, ModuleBuilder, Resolvable};
use engine::vk::{
    BatchInput, DescSetIx, FrameResources, GpuResources, ImageIx, ImageViewIx,
    PipelineIx, VkEngine,
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

    let example_state = engine.with_allocators(|ctx, res, alloc| {
        let fill_bindings = [BindingDesc::StorageImage { binding: 0 }];
        let flip_bindings = [
            BindingDesc::StorageImage { binding: 0 },
            BindingDesc::StorageImage { binding: 1 },
        ];

        let fill_pc_size = std::mem::size_of::<[i32; 2]>();
        let flip_pc_size = std::mem::size_of::<[i32; 2]>();

        let fill_pipeline = res.load_compute_shader_runtime(
            ctx,
            "shaders/trig_color.comp.spv",
            &fill_bindings,
            fill_pc_size,
        )?;

        let flip_pipeline = res.load_compute_shader_runtime(
            ctx,
            "shaders/flip.comp.spv",
            &flip_bindings,
            flip_pc_size,
        )?;

        let fill_image = res.allocate_image(
            ctx,
            alloc,
            width,
            height,
            vk::Format::R8G8B8A8_UNORM,
            vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::TRANSFER_SRC,
            Some("outer-fill_image"),
        )?;
        let flip_image = res.allocate_image(
            ctx,
            alloc,
            width,
            height,
            vk::Format::R8G8B8A8_UNORM,
            vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::TRANSFER_SRC,
            Some("flip_image"),
        )?;

        let fill_view = res.create_image_view_for_image(ctx, fill_image)?;
        let flip_view = res.create_image_view_for_image(ctx, flip_image)?;

        let fill_inputs = [BindingInput::ImageView {
            binding: 0,
            view: fill_view,
        }];
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

        let fill_set = res.allocate_desc_set(
            &fill_bindings,
            &fill_inputs,
            vk::ShaderStageFlags::COMPUTE,
        )?;
        let flip_set = res.allocate_desc_set(
            &flip_bindings,
            &flip_inputs,
            vk::ShaderStageFlags::COMPUTE,
        )?;

        Ok(ExampleState {
            fill_pipeline,
            fill_set,
            fill_image,
            fill_view,

            flip_pipeline,
            flip_set,
            flip_image,
        })
    })?;

    log::warn!("MODULE BUILDER");

    let mut builder = FrameBuilder::from_script("test.rhai")?;

    // let (mut builder, module) = ModuleBuilder::from_script("test.rhai")?;

    let mut line_renderer = LineRenderer::new(&mut engine)?;

    let lines = ["hello world", "e", "l", "l", "o     world", "???"];

    line_renderer.update_lines(&mut engine.resources, lines)?;

    dbg!();

    log::warn!("is resolved: {}", builder.is_resolved());
    // log::warn!("binding pipeline variable");
    // builder.bind_pipeline_var("pipeline", line_renderer.pipeline);

    builder.bind_var("out_image", example_state.fill_image)?;
    builder.bind_var("out_view", example_state.fill_view)?;

    builder.bind_var("text_buffer", line_renderer.text_buffer)?;
    builder.bind_var("line_buffer", line_renderer.line_buffer)?;

    engine.with_allocators(|ctx, res, alloc| {
        builder.resolve(ctx, res, alloc)?;
        Ok(())
    })?;

    log::warn!("binding descriptor set variables");
    // builder.bind_desc_set_var("bg_desc_set", example_state.fill_set);
    // builder.bind_desc_set_var("line_desc_set", line_renderer.set);
    log::warn!("is resolved: {}", builder.is_resolved());

    let mut rhai_engine = engine::script::console::create_batch_engine();

    let arc_module = Arc::new(builder.module.clone());

    // let arc_module: Arc<rhai::Module> = module.into();
    rhai_engine.register_static_module("self", arc_module.clone());

    let init = rhai::Func::<(), BatchBuilder>::create_from_ast(
        rhai_engine,
        builder.ast.clone_functions_only(),
        "init",
    );

    let mut rhai_engine = engine::script::console::create_batch_engine();
    rhai_engine.register_static_module("self", arc_module.clone());

    let draw_background =
        rhai::Func::<(i64, i64), BatchBuilder>::create_from_ast(
            rhai_engine,
            builder.ast.clone_functions_only(),
            "background",
        );

    let mut rhai_engine = engine::script::console::create_batch_engine();
    rhai_engine.register_static_module("self", arc_module);

    let draw_at =
        rhai::Func::<(i64, i64, i64, i64), BatchBuilder>::create_from_ast(
            rhai_engine,
            builder.ast.clone_functions_only(),
            "draw_at",
        );

    dbg!();

    dbg!();

    {
        let e = example_state;
        let res = &engine.resources;

        engine.set_debug_object_name(res[e.fill_image].image, "fill_image")?;
        engine.set_debug_object_name(res[e.flip_image].image, "flip_image")?;
    }

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

    let copy_batch = Box::new(
        move |dev: &Device,
              res: &GpuResources,
              input: &BatchInput,
              cmd: vk::CommandBuffer| {
            copy_batch(example_state, dev, res, input, cmd)
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

                let r = (t.sin() + 1.0) / 2.0;
                let b = (t.cos() + 1.0) / 2.0;

                let color = [r, 1.0, b, 1.0];

                let f_ix = engine.current_frame_number();
                let frame = &mut frames[f_ix % engine::vk::FRAME_OVERLAP];

                let x = 400.0 + 200.0 * t.sin();
                let y = 300.0 + 160.0 * t.cos();

                let bg_batch = draw_background(800, 600).unwrap();
                let bg_batch_fn = bg_batch.build();
                let bg_rhai_batch = bg_batch_fn.clone();

                let batch = draw_at(x as i64, y as i64, 800, 600).unwrap();
                let batch_fn = batch.build();
                let rhai_batch = batch_fn.clone();

                let bg_batch = Box::new(
                    move |dev: &Device,
                          res: &GpuResources,
                          input: &BatchInput,
                          cmd: vk::CommandBuffer| {
                        bg_rhai_batch(dev, res, cmd);
                    },
                ) as Box<_>;

                let text_batch = Box::new(
                    move |dev: &Device,
                          res: &GpuResources,
                          input: &BatchInput,
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
