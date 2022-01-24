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
    fill_color_pipeline: PipelineIx,
    fill_color_desc: DescSetIx,
    image: ImageIx,
}

fn compute_batch(
    state: ExampleState,
    device: &Device,
    resources: &GpuResources,
    input: &BatchInput,
    cmd: vk::CommandBuffer,
) {
    let image = &resources[state.image];

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

    let color = [1f32, 0.0, 0.0, 1.0];

    let mut bytes: Vec<u8> = Vec::with_capacity(24);
    bytes.extend_from_slice(bytemuck::cast_slice(&color));
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
        state.fill_color_pipeline,
        state.fill_color_desc,
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

fn copy_batch(
    state: ExampleState,
    device: &Device,
    resources: &GpuResources,
    input: &BatchInput,
    cmd: vk::CommandBuffer,
) {
    let image = &resources[state.image];

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

    let (pipeline, image, desc_set) =
        engine.with_allocators(|ctx, res, alloc| {
            let bindings = [BindingDesc::StorageImage { binding: 0 }];

            let pc_size_1 = std::mem::size_of::<[i32; 2]>()
                + std::mem::size_of::<[f32; 4]>();

            let pipeline = res.load_compute_shader_runtime(
                ctx,
                "shaders/fill_color.comp.spv",
                &bindings,
                pc_size_1,
            )?;

            /*
            let pc_size_2 = std::mem::size_of::<[i32; 2]>();

            let flipline = res.load_compute_shader_runtime(
                ctx,
                "shaders/circle_flip.comp.spv",
                &bindings,
                pc_size_2,
            )?;
            */

            let image = res.allocate_image(
                ctx,
                alloc,
                width,
                height,
                // right now this image is copied to the swapchain, which on
                // my system uses BGRA rather than RGBA, so this is just a
                // temporary fix
                vk::Format::B8G8R8A8_UNORM,
                // vk::Format::R8G8B8A8_UNORM,
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

    let ex_state = ExampleState {
        fill_color_pipeline: pipeline,
        fill_color_desc: desc_set,
        image,
    };

    let mut frames = {
        let queue_ix = engine.queues.thread.queue_family_index;

        let semaphore_count = 32;
        let cmd_buf_count = 2;

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
            compute_batch(ex_state, dev, res, input, cmd)
        },
    ) as Box<_>;

    let copy_batch = Box::new(
        move |dev: &Device,
              res: &GpuResources,
              input: &BatchInput,
              cmd: vk::CommandBuffer| {
            copy_batch(ex_state, dev, res, input, cmd)
        },
    ) as Box<_>;

    let batches = [main_batch, copy_batch];

    let deps = vec![
        None,
        // Some(vec![(0, vk::PipelineStageFlags::COMPUTE_SHADER)]),
        Some(vec![(0, vk::PipelineStageFlags::TRANSFER)]),
        // Some(vec![(0, vk::PipelineStageFlags::empty())]),
    ];
    // let deps = vec![(0, 1)];

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
                    .draw_from_batches(frame, &batches, deps.as_slice(), 1)
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
