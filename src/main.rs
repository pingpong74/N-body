use rand::Rng;
use std::{
    f32::consts::TAU,
    time::{Duration, Instant},
};

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
    dt: Duration,
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
                app.update(size.width, size.height, self.dt.as_secs_f32());
                window.request_redraw();
                let duration = instant.elapsed();

                self.dt = duration;
                println!("{}\r", duration.as_millis());
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

    const N: usize = 1 << 18;
    let radius = 2500.0;
    let mut rng = rand::rng();
    let particles: Vec<Particle> = (0..N)
        .map(|_| {
            let r = rng.random_range(0.0..radius);
            let theta = rng.random_range(0.0..360.0f32).to_radians();
            let x = r * theta.cos();
            let z = r * theta.sin();
            let y = rng.random_range(-150.0..150.0);
            let vx = r * theta.sin();
            let vz = -r * theta.cos();

            Particle {
                position: [x, 0.0, z],
                velocity: [0.0; 3],
                mass: 10.0,
                radius: 0.3,
            }
        })
        .collect();

    let event_loop = EventLoop::new().unwrap();
    let mut runner = Runner {
        app: None,
        window: None,
        particles: Some(particles),
        dt: Duration::from_secs_f64(0.016),
    };

    event_loop.run_app(&mut runner).unwrap();
}
