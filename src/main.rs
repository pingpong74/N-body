use std::time::Instant;

use winit::{
    application::ApplicationHandler,
    event::{DeviceEvent, WindowEvent},
    event_loop::{ActiveEventLoop, EventLoop},
    window::{Window, WindowAttributes, WindowId},
};

mod app;
mod camera;
mod radix_sort;
mod renderer;
mod simulator;
use app::application::Application;

use crate::app::application::Particle;

struct Runner {
    app: Option<Application>,
    window: Option<Window>,
    particles: Option<Vec<Particle>>,
}

impl ApplicationHandler for Runner {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window = event_loop
            .create_window(WindowAttributes::default().with_title("Vulkan App").with_inner_size(winit::dpi::LogicalSize::new(1280.0, 720.0)))
            .expect("Failed to create window");

        window.set_cursor_grab(winit::window::CursorGrabMode::Locked).expect(":(");
        window.set_cursor_visible(false);

        self.app = Some(Application::new(&window, self.particles.take().unwrap()));
        self.window = Some(window);
    }

    fn window_event(&mut self, _: &ActiveEventLoop, _: WindowId, event: WindowEvent) {
        let app = self.app.as_mut().unwrap();
        let window = self.window.as_ref().unwrap();
        let size = window.inner_size();

        app.handle_window_event(&event);

        match event {
            WindowEvent::CloseRequested => std::process::exit(0),

            WindowEvent::RedrawRequested => {
                let instant = Instant::now();
                app.update(size.width, size.height);
                window.request_redraw();
                let duration = instant.elapsed();
                //println!("{}\r", duration.as_millis());
            }

            WindowEvent::Resized(size) => {
                app.resize(size.width, size.height);
            }
            _ => {}
        }
    }

    fn device_event(&mut self, _: &ActiveEventLoop, _: winit::event::DeviceId, event: DeviceEvent) {
        let app = self.app.as_mut().unwrap();
        app.handle_device_event(&event);
    }
}

fn main() {
    sgpu::add_shader_directory("shaders/");

    const N: usize = 32;

    let particles: Vec<Particle> = (0..N)
        .map(|i| {
            let t = i as f32 / N as f32;
            Particle {
                position: [
                    (t * 37.0).sin() * 75.0,
                    (t * 53.0).cos() * 75.0,
                    (t * 71.0).sin() * 75.0,
                ],
                velocity: [
                    (t * 51.0).sin() * 0.0,
                    (t * 25.5).cos() * 0.0,
                    (t * 7.0).sin() * 0.0,
                ],
                mass: 1.0,
                radius: 1.5 + (t * 10.0).sin(),
            }
        })
        .collect();

    let event_loop = EventLoop::new().unwrap();
    let mut runner = Runner {
        app: None,
        window: None,
        particles: Some(particles),
    };

    event_loop.run_app(&mut runner).unwrap();
}
