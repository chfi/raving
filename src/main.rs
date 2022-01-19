use engine::vk::VkEngine;

use ash::{
    extensions::khr::{Surface, Swapchain},
    vk::{self},
    Device, Entry,
};

use flexi_logger::{Duplicate, FileSpec, Logger};
use winit::event::{Event, WindowEvent};
use winit::platform::unix::*;
use winit::{
    event_loop::EventLoop,
    window::{Window, WindowBuilder},
};

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

    let shader_code = engine::include_shader!("fill_color.comp.spv");

    let pipeline_ix = engine
        .resources
        .load_compute_shader(&engine.context, shader_code)?;

    let image_ix = engine.allocate_image(
        width,
        height,
        vk::Format::R8G8B8A8_UNORM,
        vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::TRANSFER_SRC,
    )?;

    let view_ix = engine
        .resources
        .create_image_view_for_image(&engine.context, image_ix)?;

    let desc_set_ix = engine.resources.create_compute_desc_set(view_ix)?;

    std::thread::sleep(std::time::Duration::from_millis(100));

    event_loop.run(move |event, _, control_flow| {
        *control_flow = winit::event_loop::ControlFlow::Poll;

        let mut dirty_swapchain = false;

        match event {
            Event::MainEventsCleared => {
                let render_success = engine
                    .draw_from_compute(pipeline_ix, image_ix, desc_set_ix, width, height)
                    .unwrap();
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
                    dirty_swapchain = true;
                }
                _ => (),
            },
            Event::LoopDestroyed => {
                log::trace!("Event::LoopDestroyed");
            }
            _ => (),
        }
    });

    // println!("Hello, world!");

    // Ok(())
}
