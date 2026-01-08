use std::u64;

use nexion::{utils::vulkan_context::VulkanContext, *};

use crate::simulation::lod_grids::LodPyramid;

const LOD_0_BINDING: u32 = 0;
const LOD_1_BINDING: u32 = 1;
const LOD_2_BINDING: u32 = 2;
const LOD_3_BINDING: u32 = 3;

struct SimulationPipelines {
    scatter: ComputePipeline,
    downsample: ComputePipeline,
    update: ComputePipeline,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Zeroable, bytemuck::Pod)]
struct ScatterParams {
    cell_size: u32,
    grid_res: u32,
    num_of_particles: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Zeroable, bytemuck::Pod)]
struct DownsampleParams {
    src_res: u32,
    src_binding: u32,
    dst_res: u32,
    dst_binding: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Zeroable, bytemuck::Pod)]
struct UpdateParams {
    cell_size: u32,
    lod_0_res: u32,
    dt: f32,
    fall_off: f32,
}

pub struct ParticleSimulation {
    vkc: VulkanContext,
    lods: LodPyramid,
    pipelines: SimulationPipelines,
    recorder: CommandRecorder,
    particle_buffer: BufferID,
    num_of_particles: u32,
}

impl ParticleSimulation {
    pub fn new(
        vkc: VulkanContext,
        grid_size: u32,
        cell_size: u32,
        particle_buffer: BufferID,
        num_of_particles: u32,
    ) -> ParticleSimulation {
        let sim_pipeline = SimulationPipelines {
            scatter: vkc.create_compute_pipeline(&ComputePipelineDescription {
                push_constants: PushConstantsDescription {
                    stage_flags: ShaderStages::ALL,
                    size: std::mem::size_of::<ScatterParams>() as u32,
                    offset: 0,
                },
                shader_path: "shaders/scatter.slang",
            }),
            downsample: vkc.create_compute_pipeline(&ComputePipelineDescription {
                push_constants: PushConstantsDescription {
                    stage_flags: ShaderStages::ALL,
                    size: std::mem::size_of::<DownsampleParams>() as u32,
                    offset: 0,
                },
                shader_path: "shaders/downsample.slang",
            }),
            update: vkc.create_compute_pipeline(&ComputePipelineDescription {
                push_constants: PushConstantsDescription {
                    stage_flags: ShaderStages::ALL,
                    size: std::mem::size_of::<UpdateParams>() as u32,
                    offset: 0,
                },
                shader_path: "shaders/update.slang",
            }),
        };

        let lods = LodPyramid::new(&vkc, grid_size / cell_size as u32, cell_size);

        return ParticleSimulation {
            lods: lods,
            pipelines: sim_pipeline,
            recorder: vkc.create_command_recorder(QueueType::Compute),
            vkc: vkc,
            particle_buffer: particle_buffer,
            num_of_particles: num_of_particles,
        };
    }

    pub fn record(&mut self) -> ExecutableCommandBuffer {
        self.recorder
            .begin_recording(CommandBufferUsage::SimultaneousUse);

        let groups = (self.num_of_particles + 255) / 256;

        self.recorder.fill_buffer(&BufferFillInfo {
            buffer: self.lods.lods[0].buffer,
            offset: 0,
            size: u64::MAX,
            data: 0,
        });

        self.recorder
            .pipeline_barrier(&[Barrier::Buffer(BufferBarrier {
                buffer: self.lods.lods[0].buffer,
                src_stage: PipelineStage::Transfer,
                dst_stage: PipelineStage::ComputeShader,
                src_access: AccessType::TransferWrite,
                dst_access: AccessType::ShaderWrite,
                ..Default::default()
            })]);

        // Lod 0
        self.recorder.bind_pipeline(&self.pipelines.scatter);

        self.recorder.set_push_constants(
            &ScatterParams {
                cell_size: self.lods.lods[0].cell_size,
                grid_res: self.lods.lods[0].resolution,
                num_of_particles: self.num_of_particles,
            },
            &self.pipelines.scatter,
        );

        self.recorder.dispatch(&DispatchInfo {
            group_count_x: groups,
            group_count_y: 1,
            group_count_z: 1,
        });

        self.recorder
            .pipeline_barrier(&[Barrier::Buffer(BufferBarrier {
                buffer: self.lods.lods[0].buffer,
                src_stage: PipelineStage::ComputeShader,
                dst_stage: PipelineStage::ComputeShader,
                src_access: AccessType::ShaderWrite,
                dst_access: AccessType::ShaderRead,
                ..Default::default()
            })]);

        self.recorder.bind_pipeline(&self.pipelines.downsample);

        // Lod 1
        self.recorder.set_push_constants(
            &DownsampleParams {
                src_binding: LOD_0_BINDING,
                dst_binding: LOD_1_BINDING,
                src_res: self.lods.lods[0].resolution,
                dst_res: self.lods.lods[1].resolution,
            },
            &self.pipelines.downsample,
        );

        self.recorder.dispatch(&DispatchInfo {
            group_count_x: (self.lods.lods[1].resolution + 7) / 8,
            group_count_y: (self.lods.lods[1].resolution + 7) / 8,
            group_count_z: (self.lods.lods[1].resolution + 7) / 8,
        });

        self.recorder
            .pipeline_barrier(&[Barrier::Buffer(BufferBarrier {
                buffer: self.lods.lods[1].buffer,
                src_stage: PipelineStage::ComputeShader,
                dst_stage: PipelineStage::ComputeShader,
                src_access: AccessType::ShaderWrite,
                dst_access: AccessType::ShaderRead,
                ..Default::default()
            })]);

        // Lod 2

        self.recorder.set_push_constants(
            &DownsampleParams {
                src_binding: LOD_1_BINDING,
                dst_binding: LOD_2_BINDING,
                src_res: self.lods.lods[1].resolution,
                dst_res: self.lods.lods[2].resolution,
            },
            &self.pipelines.downsample,
        );

        self.recorder.dispatch(&DispatchInfo {
            group_count_x: (self.lods.lods[2].resolution + 7) / 8,
            group_count_y: (self.lods.lods[2].resolution + 7) / 8,
            group_count_z: (self.lods.lods[2].resolution + 7) / 8,
        });

        self.recorder
            .pipeline_barrier(&[Barrier::Buffer(BufferBarrier {
                buffer: self.lods.lods[2].buffer,
                src_stage: PipelineStage::ComputeShader,
                dst_stage: PipelineStage::ComputeShader,
                src_access: AccessType::ShaderWrite,
                dst_access: AccessType::ShaderRead,
                ..Default::default()
            })]);

        // Lod 3
        self.recorder.set_push_constants(
            &DownsampleParams {
                src_binding: LOD_2_BINDING,
                dst_binding: LOD_3_BINDING,
                src_res: self.lods.lods[2].resolution,
                dst_res: self.lods.lods[3].resolution,
            },
            &self.pipelines.downsample,
        );

        self.recorder.dispatch(&DispatchInfo {
            group_count_x: (self.lods.lods[3].resolution + 7) / 8,
            group_count_y: (self.lods.lods[3].resolution + 7) / 8,
            group_count_z: (self.lods.lods[3].resolution + 7) / 8,
        });

        self.recorder
            .pipeline_barrier(&[Barrier::Buffer(BufferBarrier {
                buffer: self.lods.lods[3].buffer,
                src_stage: PipelineStage::ComputeShader,
                dst_stage: PipelineStage::ComputeShader,
                src_access: AccessType::ShaderWrite,
                dst_access: AccessType::ShaderRead,
                ..Default::default()
            })]);

        // Updating the particles
        self.recorder.bind_pipeline(&self.pipelines.update);

        self.recorder.set_push_constants(
            &UpdateParams {
                cell_size: self.lods.lods[0].cell_size,
                dt: 1.0 / 60.0,
                fall_off: 0.05,
                lod_0_res: self.lods.lods[0].resolution,
            },
            &self.pipelines.update,
        );

        self.recorder.dispatch(&DispatchInfo {
            group_count_x: groups,
            group_count_y: 1,
            group_count_z: 1,
        });

        self.recorder
            .pipeline_barrier(&[Barrier::Buffer(BufferBarrier {
                buffer: self.particle_buffer,
                src_stage: PipelineStage::ComputeShader,
                dst_stage: PipelineStage::ComputeShader,
                src_access: AccessType::ShaderWrite,
                dst_access: AccessType::ShaderRead,
                ..Default::default()
            })]);

        return self.recorder.end_recording();
    }
}

impl Drop for ParticleSimulation {
    fn drop(&mut self) {
        for i in 0..4 {
            self.vkc.destroy_buffer(self.lods.lods[i].buffer);
        }
    }
}
