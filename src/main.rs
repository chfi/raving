use engine::graph::GraphDsl;
use engine::vk::VkEngine;

use ash::vk;

use flexi_logger::{Duplicate, FileSpec, Logger};
use winit::event::{Event, WindowEvent};
// use winit::platform::unix::*;
use winit::{event_loop::EventLoop, window::WindowBuilder};

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

    /*

    */

    let mut dsl = engine::graph::test_graph();

    // graph inputs
    let window_size = [width, height];
    let color = [1.0, 0.0, 0.0, 1.0];

    // image
    let comp_image = engine.allocate_image(
        width,
        height,
        // right now this image is copied to the swapchain, which on
        // my system uses BGRA rather than RGBA, so this is just a
        // temporary fix
        vk::Format::B8G8R8A8_UNORM,
        // vk::Format::R8G8B8A8_UNORM,
        vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::TRANSFER_SRC,
    )?;

    // let swapchain_available =

    /*

    */

    let shader_code = engine::include_shader!("fill_color.comp.spv");

    let pipeline_ix = engine
        .resources
        .load_compute_shader(&engine.context, shader_code)?;

    let image_ix = engine.allocate_image(
        width,
        height,
        // right now this image is copied to the swapchain, which on
        // my system uses BGRA rather than RGBA, so this is just a
        // temporary fix
        vk::Format::B8G8R8A8_UNORM,
        // vk::Format::R8G8B8A8_UNORM,
        vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::TRANSFER_SRC,
    )?;

    let view_ix = engine
        .resources
        .create_image_view_for_image(&engine.context, image_ix)?;

    let desc_set_ix = engine.resources.create_compute_desc_set(view_ix)?;

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

                let render_success = engine
                    .draw_from_compute(pipeline_ix, image_ix, desc_set_ix, width, height, color)
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
