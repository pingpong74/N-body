use sgpu::*;
use winit::{
    event::{DeviceEvent, WindowEvent},
    window::Window,
};

use crate::{app::input::InputManager, camera::Camera};

#[repr(C)]
#[derive(Clone, Copy)]
pub struct Particle {
    pub velocity: [f32; 3],
    pub mass: f32,
    pub position: [f32; 3],
    pub radius: f32,
}

pub struct Application {
    input_manager: InputManager,
    swapchain: Swapchain,
    camera: Camera,
}

impl Application {
    pub fn new(window: &Window, particles: Vec<Particle>) -> Application {
        let size = window.inner_size();
        sgpu_init(&SgpuInititizationInfo::default_from_window(window));

        let swapchain = create_swapchain(
            window,
            &SwapchainDescription {
                format: sgpu::Format::Rgba16Float,
                frames_in_flight: 2,
                width: size.width,
                height: size.height,
            },
        );

        return Application {
            input_manager: InputManager::new(),
            swapchain: swapchain,
            camera: Camera::new(
                glam::Vec3::new(31201.0, 31996.0, 49475.0),
                size.width as f32 / size.height as f32,
            ),
        };
    }

    pub fn handle_window_event(&mut self, window_event: &WindowEvent) {
        self.input_manager.handle_window_event(&window_event);
    }

    pub fn handle_device_event(&mut self, device_event: &DeviceEvent) {
        self.input_manager.handle_device_event(&device_event);
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        self.camera.resize(width, height);
        self.swapchain.resize(width, height);
    }

    pub fn update(&mut self, width: u32, height: u32) {
        self.camera.process_input(&self.input_manager, 1.0 / 60.0);
        self.input_manager.begin_frame();

        let swapchain_image = self.swapchain.acquire_image();
    }
}
