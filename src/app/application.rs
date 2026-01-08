use nexion::{utils::vulkan_context::VulkanContext, *};
use winit::{
    event::{DeviceEvent, WindowEvent},
    window::Window,
};

use crate::{
    app::input::InputManager, camera::Camera, renderer::Renderer,
    simulation::particle_simulation::ParticleSimulation,
};

const PARTICLES_BINDING: u32 = 4;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Zeroable, bytemuck::Pod)]
pub struct Particle {
    pub mass: f32,
    pub velocity: [f32; 3],
    pub position: [f32; 3],
    pub radius: f32,
}

pub struct Application {
    input_manager: InputManager,
    renderer: Renderer,
    simulation: ParticleSimulation,
    camera: Camera,
    sim_exec_cmd_buffer: ExecutableCommandBuffer,
    fence: Fence,
    compute_finished_semaphore: Semaphore,
    particle_buffer: BufferID,
    vkc: VulkanContext,
}

impl Application {
    pub fn new(
        window: &Window,
        particles: Vec<Particle>,
        cell_size: u32,
        grid_lenght: u32,
    ) -> Application {
        let size = window.inner_size();

        let vkc = VulkanContext::new(
            window,
            &InstanceDescription {
                api_version: nexion::ApiVersion::VkApi1_3,
                enable_validation_layers: true,
            },
            &DeviceDescription {
                use_compute_queue: true,
                use_transfer_queue: true,
                atomic_float_operations: true,
                ..Default::default()
            },
            &SwapchainDescription {
                width: size.width,
                height: size.height,
                image_count: 3,
            },
        );

        let particle_buffer = vkc.create_buffer(&BufferDescription {
            memory_type: MemoryType::DeviceLocal,
            size: particles.len() as u64 * std::mem::size_of::<Particle>() as u64,
            usage: BufferUsage::STORAGE | BufferUsage::TRANSFER_DST,
            create_mapped: false,
        });

        let particle_staging_buffer = vkc.create_buffer(&BufferDescription {
            memory_type: MemoryType::PreferHost,
            size: particles.len() as u64 * std::mem::size_of::<Particle>() as u64,
            usage: BufferUsage::TRANSFER_SRC,
            create_mapped: true,
        });

        vkc.write_data_to_buffer(particle_staging_buffer, particles.as_slice());

        let mut rec = vkc.create_command_recorder(QueueType::Transfer);

        rec.begin_recording(CommandBufferUsage::OneTimeSubmit);
        rec.copy_buffer(&BufferCopyInfo {
            src_buffer: particle_staging_buffer,
            dst_buffer: particle_buffer,
            regions: vec![CopyRegion {
                src_offset: 0,
                dst_offset: 0,
                size: particles.len() as u64 * std::mem::size_of::<Particle>() as u64,
            }],
        });
        let exe = rec.end_recording();

        vkc.submit(&QueueSubmitInfo {
            fence: None,
            command_buffers: &[exe],
            wait_semaphores: &[],
            signal_semaphores: &[],
        });

        vkc.wait_idle();

        vkc.destroy_buffer(particle_staging_buffer);

        vkc.write_buffer(&BufferWriteInfo {
            buffer: particle_buffer,
            offset: 0,
            range: particles.len() as u64 * std::mem::size_of::<Particle>() as u64,
            index: PARTICLES_BINDING,
        });

        println!(
            "Grid Size: {} Cell width: {}",
            grid_lenght.next_power_of_two(),
            cell_size.next_power_of_two()
        );

        println!(
            "Particle buffer of size {}",
            particles.len() as u64 * std::mem::size_of::<Particle>() as u64
        );

        let mut simulation = ParticleSimulation::new(
            vkc.clone(),
            grid_lenght.next_power_of_two(),
            cell_size.next_power_of_two(),
            particle_buffer,
            particles.len() as u32,
        );

        return Application {
            input_manager: InputManager::new(),
            renderer: Renderer::new(vkc.clone(), particle_buffer, particles.len() as u32),
            sim_exec_cmd_buffer: simulation.record(),
            simulation: simulation,
            camera: Camera::new(
                glam::Vec3::new(31201.0, 31996.0, 49475.0),
                size.width as f32 / size.height as f32,
            ),
            fence: vkc.create_fence(true),
            compute_finished_semaphore: vkc.create_binary_semaphore(),
            particle_buffer: particle_buffer,
            vkc: vkc,
        };
    }

    pub fn handle_window_event(&mut self, window_event: &WindowEvent) {
        self.input_manager.handle_window_event(&window_event);
    }

    pub fn handle_device_event(&mut self, device_event: &DeviceEvent) {
        self.input_manager.handle_device_event(&device_event);
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        self.vkc.resize(width, height);
        self.camera.resize(width, height);
    }

    pub fn update(&mut self, width: u32, height: u32) {
        self.camera.process_input(&self.input_manager, 1.0 / 60.0);
        self.input_manager.begin_frame();

        self.vkc.wait_fence(self.fence);
        self.vkc.reset_fence(self.fence);

        let (img, img_view, image_semaphore, present_semaphore) = self.vkc.acquire_image();

        self.vkc.submit(&QueueSubmitInfo {
            fence: None,
            command_buffers: &[self.sim_exec_cmd_buffer],
            wait_semaphores: &[],
            signal_semaphores: &[SemaphoreInfo {
                semaphore: self.compute_finished_semaphore,
                pipeline_stage: PipelineStage::ComputeShader,
                value: None,
            }],
        });

        let render_exec_cmd_buffer =
            self.renderer
                .record(width, height, &self.camera, img, img_view);

        self.vkc.submit(&QueueSubmitInfo {
            fence: Some(self.fence),
            command_buffers: &[render_exec_cmd_buffer],
            wait_semaphores: &[
                SemaphoreInfo {
                    semaphore: image_semaphore,
                    pipeline_stage: PipelineStage::ColorAttachmentOutput,
                    value: None,
                },
                SemaphoreInfo {
                    semaphore: self.compute_finished_semaphore,
                    pipeline_stage: PipelineStage::VertexInput,
                    value: None,
                },
            ],
            signal_semaphores: &[SemaphoreInfo {
                semaphore: present_semaphore,
                pipeline_stage: PipelineStage::BottomOfPipe,
                value: None,
            }],
        });

        self.vkc.present();
    }
}

impl Drop for Application {
    fn drop(&mut self) {
        self.vkc.destroy_fence(self.fence);
        self.vkc.destroy_semaphore(self.compute_finished_semaphore);
        self.vkc.destroy_buffer(self.particle_buffer);
    }
}
