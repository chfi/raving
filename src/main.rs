use raving::script::console::frame::FrameBuilder;
use raving::script::console::BatchBuilder;
use raving::vk::descriptor::DescriptorLayoutInfo;
use raving::vk::{
    BatchInput, FrameResources, GpuResources, VkEngine, WinSizeIndices,
    WinSizeResourcesBuilder,
};

use raving::vk::util::*;

use ash::{vk, Device};

use flexi_logger::{Duplicate, FileSpec, Logger};
use rspirv_reflect::DescriptorInfo;
use winit::event::{Event, WindowEvent};
use winit::{event_loop::EventLoop, window::WindowBuilder};

// #[cfg(target_os = "linux")]
// use winit::platform::unix::*;

use std::collections::BTreeMap;
use std::sync::Arc;

use anyhow::{anyhow, bail, Result};

fn main() -> Result<()> {
    let mut args = std::env::args();

    let _ = args.next().unwrap();

    let script_path = args.next().ok_or(anyhow!("Provide a script path"))?;

    // let args: Args = argh::from_env();

    let spec = "debug";
    let _logger = Logger::try_with_env_or_str(spec)?
        .log_to_file(FileSpec::default())
        .duplicate_to_stderr(Duplicate::Debug)
        .start()?;

    /*
    let event_loop: EventLoop<()>;

    #[cfg(target_os = "linux")]
    {
        use winit::platform::unix::EventLoopExtUnix;
        event_loop = EventLoop::new_x11()?;
        // event_loop = if args.force_x11 || !instance_exts.wayland_surface {
        //     if let Ok(ev_loop) = EventLoop::new_x11() {
        //         log::debug!("Using X11 event loop");
        //         ev_loop
        //     } else {
        //         error!(
        //             "Error initializing X11 window, falling back to default"
        //         );
        //         EventLoop::new()
        //     }
        // } else {
        //     log::debug!("Using default event loop");
        //     EventLoop::new()
        // };
    }

    #[cfg(not(target_os = "linux"))]
    {
        log::debug!("Using default event loop");
        let event_loop = EventLoop::new();
    }
    */

    let event_loop = EventLoop::new();

    let width = 800;
    let height = 600;

    let window = WindowBuilder::new()
        .with_title("engine")
        .with_inner_size(winit::dpi::PhysicalSize::new(width, height))
        .build(&event_loop)?;

    let mut engine = VkEngine::new(&window)?;

    let window_storage_set_info = {
        let info = DescriptorInfo {
            ty: rspirv_reflect::DescriptorType::STORAGE_IMAGE,
            binding_count: rspirv_reflect::BindingCount::One,
            name: "out_image".to_string(),
        };

        Some((0u32, info)).into_iter().collect::<BTreeMap<_, _>>()
    };

    let window_storage_image_layout = {
        let mut info = DescriptorLayoutInfo::default();

        let binding = vk::DescriptorSetLayoutBinding::builder()
            .binding(0)
            .descriptor_count(1)
            .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
            .build();

        info.bindings.push(binding);
        info
    };

    let mut win_size_resource_index = WinSizeIndices::default();

    let win_size_res_builder = move |engine: &mut VkEngine,
                                     width: u32,
                                     height: u32|
          -> Result<WinSizeResourcesBuilder> {
        let mut builder = WinSizeResourcesBuilder::default();

        let (img, view, desc_set) =
            engine.with_allocators(|ctx, res, alloc| {
                dbg!();
                let out_image = res.allocate_image(
                    ctx,
                    alloc,
                    width,
                    height,
                    vk::Format::R8G8B8A8_UNORM,
                    vk::ImageUsageFlags::STORAGE
                        | vk::ImageUsageFlags::TRANSFER_SRC,
                    Some("out_image"),
                )?;

                dbg!();
                let out_view = res.new_image_view(ctx, &out_image)?;

                dbg!();
                let out_desc_set = res.allocate_desc_set_raw(
                    &window_storage_image_layout,
                    &window_storage_set_info,
                    |res, builder| {
                        let info = ash::vk::DescriptorImageInfo::builder()
                            .image_layout(vk::ImageLayout::GENERAL)
                            .image_view(out_view)
                            .build();

                        builder.bind_image(0, &[info]);

                        Ok(())
                    },
                )?;
                dbg!();

                Ok((out_image, out_view, out_desc_set))
            })?;

        builder.images.insert("out_image".to_string(), img);
        builder
            .image_views
            .insert("out_image_view".to_string(), view);
        builder
            .desc_sets
            .insert("out_desc_set".to_string(), desc_set);

        //
        Ok(builder)
    };

    {
        let builder = win_size_res_builder(&mut engine, 800, 600)?;
        engine.with_allocators(|ctx, res, alloc| {
            builder.insert(&mut win_size_resource_index, ctx, res, alloc)?;
            Ok(())
        })?;
    }

    let out_image = *win_size_resource_index.images.get("out_image").unwrap();
    let out_view = *win_size_resource_index
        .image_views
        .get("out_image_view")
        .unwrap();
    let out_desc_set = *win_size_resource_index
        .desc_sets
        .get("out_desc_set")
        .unwrap();

    // let (out_image, out_view) = engine.with_allocators(|ctx, res, alloc| {
    //     let out_image = res.allocate_image(
    //         ctx,
    //         alloc,
    //         width,
    //         height,
    //         vk::Format::R8G8B8A8_UNORM,
    //         vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::TRANSFER_SRC,
    //         Some("out_image"),
    //     )?;

    //     let out_view = res.new_image_view(ctx, &out_image)?;

    //     let out_image_ix = res.insert_image(out_image);
    //     let out_view_ix = res.insert_image_view(out_view);

    //     Ok((out_image_ix, out_view_ix))
    // })?;

    log::warn!("MODULE BUILDER");

    let mut builder = FrameBuilder::from_script(&script_path)?;

    // builder.bind_var("out_image",
    // builder.bind_var("out_view", out_view)?;
    builder.bind_var("out_image", out_image)?;
    builder.bind_var("out_view", out_view)?;
    builder.bind_var("out_desc_set", out_desc_set)?;

    engine.with_allocators(|ctx, res, alloc| {
        builder.resolve(ctx, res, alloc)?;
        Ok(())
    })?;
    log::warn!("is resolved: {}", builder.is_resolved());

    let mut rhai_engine = raving::script::console::create_batch_engine();

    let arc_module = Arc::new(builder.module.clone());

    rhai_engine.register_static_module("self", arc_module.clone());

    let init = rhai::Func::<(), BatchBuilder>::create_from_ast(
        rhai_engine,
        builder.ast.clone_functions_only(),
        "init",
    );

    let mut rhai_engine = raving::script::console::create_batch_engine();
    rhai_engine.register_static_module("self", arc_module.clone());

    let draw_background =
        rhai::Func::<(i64, i64), BatchBuilder>::create_from_ast(
            rhai_engine,
            builder.ast.clone_functions_only(),
            "background",
        );

    // let mut rhai_engine = engine::script::console::create_batch_engine();
    // rhai_engine.register_static_module("self", arc_module);

    // let draw_foreground =
    //     rhai::Func::<(i64, i64, f32), BatchBuilder>::create_from_ast(
    //         rhai_engine,
    //         builder.ast.clone_functions_only(),
    //         "foreground",
    //     );

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

        if !init_builder.init_fn.is_empty() {
            log::warn!("submitting init batches");
            let fence =
                engine.submit_batches_fence(init_builder.init_fn.as_slice())?;

            engine.block_on_fence(fence)?;
        }
    }

    let mut recreate_swapchain = false;

    event_loop.run(move |event, _, control_flow| {
        *control_flow = winit::event_loop::ControlFlow::Poll;

        match event {
            Event::MainEventsCleared => {
                let t = start.elapsed().as_secs_f32();

                let f_ix = engine.current_frame_number();
                // dbg!(t);
                // dbg!(f_ix);
                let frame = &mut frames[f_ix % raving::vk::FRAME_OVERLAP];

                let bg_batch = draw_background(800, 600).unwrap();
                let bg_batch_fn = bg_batch.build();
                let bg_rhai_batch = bg_batch_fn.clone();

                // let fg_batch = draw_foreground(800, 600, t).unwrap();
                // let fg_batch_fn = fg_batch.build();
                // let fg_rhai_batch = fg_batch_fn.clone();

                let bg_batch = Box::new(
                    move |dev: &Device,
                          res: &GpuResources,
                          _input: &BatchInput,
                          cmd: vk::CommandBuffer| {
                        bg_rhai_batch(dev, res, cmd);
                    },
                ) as Box<_>;

                // let fg_batch = Box::new(
                //     move |dev: &Device,
                //           res: &GpuResources,
                //           _input: &BatchInput,
                //           cmd: vk::CommandBuffer| {
                //         fg_rhai_batch(dev, res, cmd);
                //     },
                // ) as Box<_>;

                // let batches = [&bg_batch, &fg_batch, &copy_batch];
                let batches = [&bg_batch, &copy_batch];

                let deps = vec![
                    None,
                    Some(vec![(0, vk::PipelineStageFlags::COMPUTE_SHADER)]),
                    // Some(vec![(1, vk::PipelineStageFlags::COMPUTE_SHADER)]),
                ];

                // dbg!();
                let render_success = engine
                    .draw_from_batches(frame, &batches, deps.as_slice(), 1)
                    .unwrap();

                // dbg!();

                if !render_success {
                    dbg!();
                    recreate_swapchain = true;
                }
            }
            Event::RedrawEventsCleared => {
                if recreate_swapchain {
                    recreate_swapchain = false;

                    let size = window.inner_size();

                    if size.width > 0 && size.height > 0 {
                        log::debug!(
                            "Recreating swapchain with window size {:?}",
                            size
                        );
                    }
                }
                //
            }
            Event::WindowEvent { event, .. } => match event {
                WindowEvent::CloseRequested => {
                    log::debug!("WindowEvent::CloseRequested");
                    *control_flow = winit::event_loop::ControlFlow::Exit;
                }
                WindowEvent::Resized { .. } => {
                    recreate_swapchain = true;
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
