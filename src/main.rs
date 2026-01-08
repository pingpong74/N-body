use std::time::Instant;

use winit::{
    application::ApplicationHandler,
    event::{DeviceEvent, WindowEvent},
    event_loop::{ActiveEventLoop, EventLoop},
    window::{Window, WindowAttributes, WindowId},
};

use glam::*;
use rand::Rng;

mod app;
mod camera;
mod renderer;
mod simulation;
use app::application::Application;

use crate::app::application::Particle;

pub struct SimulationInfo {
    gen_func: fn() -> Vec<Particle>,
    grid_lenght: u32,
    cell_size: u32,
}

const PARTICLE_COUNT: usize = 100000;

fn rotate_y(v: glam::Vec3, angle_rad: f32) -> glam::Vec3 {
    let c = angle_rad.cos();
    let s = angle_rad.sin();
    glam::Vec3::new(v.x * c + v.z * s, v.y, -v.x * s + v.z * c)
}

pub fn create_particles() -> Vec<Particle> {
    let mut rng = rand::rng();
    let mut particles = Vec::with_capacity(PARTICLE_COUNT);

    let center_a = glam::Vec3::new(33000.0, 30000.0, 40000.0);
    let center_b = glam::Vec3::new(40000.0, 30000.0, 40000.0);

    let blob_radius = 1000.0;
    let half = PARTICLE_COUNT / 2;

    // Blob direction
    let base_dir_a = 10.0 * (center_b - center_a).normalize(); // A → B
    let base_dir_b = -base_dir_a; // B → A

    let angle = 90.0_f32.to_radians();

    // Rotate directions by ±30°
    let linear_a = rotate_y(base_dir_a, angle) * 40.0;
    let linear_b = rotate_y(base_dir_b, -angle) * 40.0;

    // --- NEW: angular velocities
    let angular_a = glam::Vec3::new(0.0, 0.2, 0.0); // spin around Y
    let angular_b = glam::Vec3::new(0.0, -0.15, 0.0); // opposite spin

    for i in 0..PARTICLE_COUNT {
        // Random point inside unit sphere
        let offset = loop {
            let p = glam::Vec3::new(
                rng.random_range(-1.0..1.0),
                rng.random_range(-1.0..1.0),
                rng.random_range(-1.0..1.0),
            );
            if p.length_squared() <= 1.0 {
                break p * blob_radius;
            }
        };

        let (center, linear, angular) = if i < half {
            (center_a, linear_a, angular_a)
        } else {
            (center_b, linear_b, angular_b)
        };

        let pos = center + offset;

        // --- NEW: rotational velocity contribution: v_rot = ω × r
        let v_rot = angular.cross(offset);

        // --- final particle velocity
        let vel = linear + v_rot;

        let mass = rng.random_range(0.1..200.0);
        let radius = rng.random_range(1.0..10.0);

        particles.push(Particle {
            mass,
            position: pos.into(),
            velocity: vel.into(),
            radius,
        });
    }

    particles
}

fn create_rect() -> Vec<Particle> {
    let mut particles: Vec<Particle> = Vec::new();
    let mut rng = rand::rng();

    for i in 0..PARTICLE_COUNT {
        let pos = Vec3::new(
            rng.random_range(1000.0..3000.0),
            rng.random_range(1000.0..3000.0),
            rng.random_range(1000.0..3000.0),
        );

        particles.push(Particle {
            mass: rng.random_range(0.1..5.0),
            velocity: Vec3::ZERO.into(),
            position: pos.into(),
            radius: rng.random_range(0.1..5.0),
        });
    }

    particles
}

struct Runner {
    app: Option<Application>,
    window: Option<Window>,
    sim_info: SimulationInfo,
}

impl ApplicationHandler for Runner {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window = event_loop
            .create_window(
                WindowAttributes::default()
                    .with_title("Vulkan App")
                    .with_inner_size(winit::dpi::LogicalSize::new(1280.0, 720.0)),
            )
            .expect("Failed to create window");

        window
            .set_cursor_grab(winit::window::CursorGrabMode::Locked)
            .expect(":(");
        window.set_cursor_visible(false);

        self.app = Some(Application::new(
            &window,
            (self.sim_info.gen_func)(),
            self.sim_info.cell_size,
            self.sim_info.grid_lenght,
        ));
        self.window = Some(window);
    }

    fn window_event(
        &mut self,
        _event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
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

    fn device_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        device_id: winit::event::DeviceId,
        event: DeviceEvent,
    ) {
        let app = self.app.as_mut().unwrap();
        app.handle_device_event(&event);
    }
}

fn main() {
    nexion::add_shader_directory("shaders/");

    let event_loop = EventLoop::new().unwrap();
    let mut runner = Runner {
        app: None,
        window: None,
        sim_info: SimulationInfo {
            gen_func: create_particles,
            grid_lenght: 60000,
            cell_size: 512,
        },
    };

    event_loop.run_app(&mut runner).unwrap();
}
