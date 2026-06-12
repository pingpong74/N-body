use std::num;

use sgpu::{AccessType, Buffer, BufferDescription, BufferUsage, CommandBuffer, ComputePipeline, GlobalBarrier, create_buffer, create_compute_pipeline};

use crate::{app::application::Particle, radix_sort::RadixSorter};

struct Node {
    mass: f32,
    com: [f32; 3],
    size: f32,

    start: u32,
    count: u32,
}

pub struct SimulationPipelines {
    morton: ComputePipeline,

    build_tree: ComputePipeline,
    compute_mass: ComputePipeline,

    integrate: ComputePipeline,
}

impl SimulationPipelines {
    fn new() -> SimulationPipelines {
        return SimulationPipelines {
            morton: create_compute_pipeline(include_bytes!("../compiled/morton_code.spv")),
            build_tree: create_compute_pipeline(&[]),
            compute_mass: create_compute_pipeline(&[]),
            integrate: create_compute_pipeline(&[]),
        };
    }
}

pub struct Simulator {
    particle_buffer: Buffer,
    // morton codes and sorted indices
    morton_buffer: Buffer,
    indices: Buffer,

    // final buffer which contains the tree
    tree_buffer: Buffer,

    radix_sort: RadixSorter,
    pipelines: SimulationPipelines,

    num_of_particles: u32,
}

impl Simulator {
    pub fn new(particles: &[Particle]) -> Simulator {
        let particle_size = std::mem::size_of::<Particle>() as u64;
        let num_of_particles = particles.len() as u64;

        let particle_buffer = create_buffer(&sgpu::BufferDescription {
            usage: BufferUsage::STORAGE | BufferUsage::TRANSFER_DST,
            size: particle_size * num_of_particles,
            memory_type: sgpu::MemoryType::DeviceLocal,
        });

        // each morton code is just a u32
        let morton_buffer = create_buffer(&sgpu::BufferDescription {
            usage: BufferUsage::STORAGE | BufferUsage::TRANSFER_DST,
            size: 4 * num_of_particles,
            memory_type: sgpu::MemoryType::DeviceLocal,
        });

        let sorted_indices_buffer = create_buffer(&sgpu::BufferDescription {
            usage: BufferUsage::STORAGE | BufferUsage::TRANSFER_DST,
            size: 4 * num_of_particles,
            memory_type: sgpu::MemoryType::DeviceLocal,
        });

        let tree_buffer = create_buffer(&BufferDescription {
            usage: BufferUsage::STORAGE | BufferUsage::TRANSFER_DST,
            size: 2 * num_of_particles * std::mem::size_of::<Node>() as u64,
            memory_type: sgpu::MemoryType::DeviceLocal,
        });

        return Simulator {
            particle_buffer: particle_buffer,
            morton_buffer: morton_buffer,
            indices: sorted_indices_buffer,
            tree_buffer: tree_buffer,
            radix_sort: RadixSorter::new(num_of_particles as usize),
            pipelines: SimulationPipelines::new(),
            num_of_particles: num_of_particles as u32,
        };
    }

    pub fn record_simulation(&self, cmd: &mut CommandBuffer) {
        let groups = u32::div_ceil(self.num_of_particles, 256);

        cmd.bind_compute_pipeline(&self.pipelines.morton);
        cmd.dispatch(groups, 1, 1);
        cmd.barriers(
            Some(&GlobalBarrier {
                previous_accesses: &[AccessType::ComputeShaderStorageWrite],
                next_accesses: &[AccessType::ComputeShaderStorageWrite],
            }),
            &[],
        );

        self.radix_sort.record(cmd, self.morton_buffer, self.indices);

        cmd.barriers(
            Some(&GlobalBarrier {
                previous_accesses: &[AccessType::ComputeShaderStorageWrite],
                next_accesses: &[AccessType::ComputeShaderStorageRead],
            }),
            &[],
        );

        // build the tree
        cmd.bind_compute_pipeline(&self.pipelines.build_tree);
        cmd.dispatch(groups, 1, 1);

        cmd.barriers(
            Some(&GlobalBarrier {
                previous_accesses: &[AccessType::ComputeShaderStorageWrite],
                next_accesses: &[AccessType::ComputeShaderStorageWrite],
            }),
            &[],
        );

        // compute com for each node
        cmd.bind_compute_pipeline(&self.pipelines.compute_mass);
        cmd.dispatch(groups, 1, 1);

        cmd.barriers(
            Some(&GlobalBarrier {
                previous_accesses: &[AccessType::ComputeShaderStorageWrite],
                next_accesses: &[AccessType::ComputeShaderStorageRead],
            }),
            &[],
        );

        // traverse and ingtegrate
        cmd.bind_compute_pipeline(&self.pipelines.integrate);
        cmd.dispatch(groups, 1, 1);
    }
}
