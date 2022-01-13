use engine::vk::{self, VkEngine};

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

    // log::debug!("Logger initalized");

    let event_loop = EventLoop::new();

    let window = WindowBuilder::new()
        .with_title("Gfaestus")
        .with_inner_size(winit::dpi::PhysicalSize::new(800, 600))
        .build(&event_loop)?;

    let mut engine = VkEngine::new(&window)?;

    event_loop.run(move |event, _, control_flow| {
        *control_flow = winit::event_loop::ControlFlow::Poll;

        // let event = if let Some(ev) = event.to_static() {
        //     ev
        // } else {
        //     return;
        // };

        let mut dirty_swapchain = false;

        match event {
            // Event::MainEventsCleared => {
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
            // }
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

    Ok(())
}
