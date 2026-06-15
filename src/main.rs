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

fn gaussian(rng: &mut impl rand::Rng) -> f32 {
    let u1: f32 = rng.random();
    let u2: f32 = rng.random();

    (-2.0 * u1.ln()).sqrt() * (TAU * u2).cos()
}

pub fn create_spiral_galaxy(count: usize, center: [f32; 3], galaxy_velocity: [f32; 3], radius: f32, arms: usize) -> Vec<Particle> {
    let mut rng = rand::rng();

    let mut particles = Vec::with_capacity(count);

    let bulge_fraction = 0.15;
    let bulge_count = (count as f32 * bulge_fraction) as usize;

    // ------------------------
    // Bulge
    // ------------------------
    for _ in 0..bulge_count {
        let r = radius * 0.15 * rng.random::<f32>().powf(2.5);

        let theta = rng.random_range(0.0..TAU);
        let phi = rng.random_range(0.0..std::f32::consts::PI);

        let x = r * phi.sin() * theta.cos();
        let y = r * phi.cos();
        let z = r * phi.sin() * theta.sin();

        let mass = rng.random_range(5.0..20.0);

        particles.push(Particle {
            position: [
                center[0] + x,
                center[1] + y,
                center[2] + z,
            ],
            velocity: galaxy_velocity,
            mass,
            radius: 0.3,
        });
    }

    // ------------------------
    // Disk + Arms
    // ------------------------
    for _ in bulge_count..count {
        // Exponential radial distribution
        let r = radius * rng.random::<f32>().powf(2.0);

        let arm = rng.random_range(0..arms) as f32 * TAU / arms as f32;

        // Logarithmic spiral
        let spiral_theta = arm + 3.5 * (r / radius) * TAU + rng.random_range(-0.25..0.25);

        let x = r * spiral_theta.cos();
        let z = r * spiral_theta.sin();

        // Gaussian thickness
        let scale_height = radius * 0.03;

        let y = gaussian(&mut rng) * scale_height;

        // Heavier stars near center
        let mass = 1.0 + 4.0 * (1.0 - r / radius).powf(2.0);

        // Approximate circular velocity
        let enclosed_mass = count as f32 * (r / radius).powf(1.5);

        let orbital_speed = (enclosed_mass / (r + 1.0)).sqrt();

        let tx = -spiral_theta.sin();
        let tz = spiral_theta.cos();

        particles.push(Particle {
            position: [
                center[0] + x,
                center[1] + y,
                center[2] + z,
            ],
            velocity: [
                galaxy_velocity[0] + tx * orbital_speed + gaussian(&mut rng) * 0.5,
                galaxy_velocity[1] + gaussian(&mut rng) * 0.5,
                galaxy_velocity[2] + tz * orbital_speed + gaussian(&mut rng) * 0.5,
            ],
            mass,
            radius: 0.3,
        });
    }

    particles
}

fn main() {
    sgpu::add_shader_directory("shaders/");

    let mut particles = Vec::new();

    particles.extend(create_spiral_galaxy(1 << 17, [-1500.0, 0.0, 0.0], [20.0, 0.0, 0.0], 1000.0, 4));

    println!("{}", particles.len());

    let event_loop = EventLoop::new().unwrap();
    let mut runner = Runner {
        app: None,
        window: None,
        particles: Some(particles),
        dt: Duration::from_secs_f64(0.016),
    };

    event_loop.run_app(&mut runner).unwrap();
}
