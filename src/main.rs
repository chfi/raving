use engine::vk::{self, VkEngine};

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

    // log::debug!("Logger initalized");

    let event_loop = EventLoop::new();

    let width = 800;
    let height = 600;

    let window = WindowBuilder::new()
        .with_title("engine")
        .with_inner_size(winit::dpi::PhysicalSize::new(width, height))
        .build(&event_loop)?;

    dbg!();
    let mut engine = VkEngine::new(&window)?;

    let shader_code = engine::include_shader!("fill_color.comp.spv");

    dbg!();
    let pipeline_ix = engine
        .resources
        .load_compute_shader(&engine.context, shader_code)
        .unwrap();

    dbg!();

    let image_ix = engine
        .resources
        .allocate_image_for_compute(&mut engine.allocator, &engine.context, width, height)
        .unwrap();

    dbg!();
    let view_ix = engine
        .resources
        .create_image_view_for_image(&engine.context, image_ix)
        .unwrap();

    dbg!();
    let desc_set_ix = engine
        .resources
        .create_desc_set_for_image(&engine.context, view_ix)
        .unwrap();

    dbg!();

    std::thread::sleep(std::time::Duration::from_millis(100));

    event_loop.run(move |event, _, control_flow| {
        *control_flow = winit::event_loop::ControlFlow::Poll;

        let mut dirty_swapchain = false;

        match event {
            Event::MainEventsCleared => {
                let render_success = engine
                    .draw_from_compute(pipeline_ix, image_ix, desc_set_ix, width, height)
                    .unwrap();

                //     let screen_dims = app.dims();
                //     let mouse_pos = app.mouse_pos();
                //     main_view.update_view_animation(screen_dims, mouse_pos);

                //     let edge_ubo = app.settings.edge_renderer().load();

                //     for er in edge_renderer.iter_mut() {
                //         er.write_ubo(&edge_ubo).unwrap();
                //     }

                //     let focus = &app.shared_state().gui_focus_state;
                //     if !focus.mouse_over_gui() {
                //         main_view.produce_context(&context_mgr);
                //         // main_view.send_context(context_menu.tx());
                //     }
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
