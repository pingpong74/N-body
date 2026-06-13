use std::num;

use sgpu::*;

use crate::{app::application::Particle, radix_sort::RadixSorter};

const NODE_SIZE: u64 = 64;

#[repr(C)]
#[derive(Clone, Copy)]
struct BoundsPc {
    particle_buffer_id: u32,
    bounds_buffer_id: u32,
    num_of_particles: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct MortonPc {
    particle_buffer_id: u32,
    morton_buffer_id: u32,
    num_of_particles: u32,
    bounds_buffer: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct BuildTreePc {
    morton_buffer_id: u32,
    indices_buffer_id: u32,
    node_buffer_id: u32,
    num_of_particles: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct ComputeMassPc {
    particle_buffer_id: u32,
    indices_buffer_id: u32,
    node_buffer_id: u32,
    counter_buffer_id: u32,
    num_of_particles: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct IntegratePc {
    particle_buffer_id: u32,
    indices_buffer_id: u32,
    node_buffer_id: u32,
    num_of_particles: u32,
    dt: f32,
    theta: f32,
    epsilon: f32,
}

pub struct SimulationPipelines {
    bound: ComputePipeline,
    morton: ComputePipeline,
    build_tree: ComputePipeline,
    compute_mass: ComputePipeline,
    integrate: ComputePipeline,
}

impl SimulationPipelines {
    fn new() -> SimulationPipelines {
        return SimulationPipelines {
            bound: create_compute_pipeline(include_bytes!("../compiled/bounds.spv")),
            morton: create_compute_pipeline(include_bytes!("../compiled/morton_code.spv")),
            build_tree: create_compute_pipeline(include_bytes!("../compiled/tree.spv")),
            compute_mass: create_compute_pipeline(include_bytes!("../compiled/com.spv")),
            integrate: create_compute_pipeline(include_bytes!("../compiled/integrate.spv")),
        };
    }
}

pub struct Simulator {
    read_back: Buffer,
    pub particle_buffer: Buffer,

    // store the bounds calculated on gpu
    bounds_buffer: Buffer,
    // morton codes and sorted indices
    morton_buffer: Buffer,
    indices: Buffer,
    // final buffer which contains the tree
    tree_buffer: Buffer,
    counter: Buffer, // temp buffer for accumalating mass per node

    radix_sort: RadixSorter,
    pipelines: SimulationPipelines,

    num_of_particles: u32,
}

impl Simulator {
    pub fn new(particles: &[Particle]) -> Simulator {
        let particle_size = std::mem::size_of::<Particle>() as u64;
        let num_of_particles = particles.len() as u64;

        let staging = create_buffer(&BufferDescription {
            usage: BufferUsage::TRANSFER_DST | BufferUsage::TRANSFER_SRC,
            size: particle_size * num_of_particles * 4,
            memory_type: MemoryType::PreferHost,
        });

        let particle_buffer = create_buffer(&sgpu::BufferDescription {
            usage: BufferUsage::STORAGE | BufferUsage::TRANSFER_SRC | BufferUsage::TRANSFER_DST,
            size: num_of_particles * particle_size,
            memory_type: sgpu::MemoryType::DeviceLocal,
        });

        {
            let slice = staging.as_mut_slice();

            for i in 0..particles.len() {
                slice[i] = particles[i];
            }

            let mut cmd = record(QueueType::Transfer);
            cmd.copy_buffer(staging, particle_buffer, 0, 0, num_of_particles * particle_size);
            wait(submit(&[cmd]));
        }

        let bounds = create_buffer(&sgpu::BufferDescription {
            usage: BufferUsage::STORAGE | BufferUsage::TRANSFER_DST | BufferUsage::TRANSFER_SRC,
            size: 2 * 4 * 4,
            memory_type: sgpu::MemoryType::DeviceLocal,
        });

        // each morton code is just a u32
        let morton_buffer = create_buffer(&sgpu::BufferDescription {
            usage: BufferUsage::STORAGE | BufferUsage::TRANSFER_DST | BufferUsage::TRANSFER_SRC,
            size: 4 * num_of_particles,
            memory_type: sgpu::MemoryType::DeviceLocal,
        });

        let sorted_indices_buffer = create_buffer(&sgpu::BufferDescription {
            usage: BufferUsage::STORAGE | BufferUsage::TRANSFER_DST | BufferUsage::TRANSFER_SRC,
            size: 4 * num_of_particles,
            memory_type: sgpu::MemoryType::DeviceLocal,
        });

        let tree_buffer = create_buffer(&BufferDescription {
            usage: BufferUsage::STORAGE | BufferUsage::TRANSFER_DST | BufferUsage::TRANSFER_SRC,
            size: 2 * num_of_particles * NODE_SIZE,
            memory_type: sgpu::MemoryType::DeviceLocal,
        });

        let counter_buffer = create_buffer(&BufferDescription {
            usage: BufferUsage::STORAGE | BufferUsage::TRANSFER_DST | BufferUsage::TRANSFER_SRC,
            size: (num_of_particles - 1) * 4,
            memory_type: sgpu::MemoryType::DeviceLocal,
        });

        return Simulator {
            read_back: staging,
            particle_buffer: particle_buffer,
            bounds_buffer: bounds,
            morton_buffer: morton_buffer,
            indices: sorted_indices_buffer,
            tree_buffer: tree_buffer,
            counter: counter_buffer,
            radix_sort: RadixSorter::new(num_of_particles as usize),
            pipelines: SimulationPipelines::new(),
            num_of_particles: num_of_particles as u32,
        };
    }

    pub fn record_simulation(&self, cmd: &mut CommandBuffer) {
        let groups = u32::div_ceil(self.num_of_particles, 256);

        cmd.fill_buffer(&self.bounds_buffer, 0, 4 * 4, 0);
        cmd.fill_buffer(&self.bounds_buffer, 4 * 4, 4 * 4, 0xFFFFFFFF);
        cmd.bind_compute_pipeline(&self.pipelines.bound);
        cmd.push_constants(&BoundsPc {
            particle_buffer_id: self.particle_buffer.descriptor_index(),
            num_of_particles: self.num_of_particles,
            bounds_buffer_id: self.bounds_buffer.descriptor_index(),
        });
        cmd.dispatch(groups, 1, 1);
        cmd.barriers(
            Some(&GlobalBarrier {
                previous_accesses: &[AccessType::ComputeShaderStorageWrite],
                next_accesses: &[AccessType::ComputeShaderStorageWrite],
            }),
            &[],
        );

        cmd.bind_compute_pipeline(&self.pipelines.morton);
        cmd.push_constants(&MortonPc {
            particle_buffer_id: self.particle_buffer.descriptor_index(),
            morton_buffer_id: self.morton_buffer.descriptor_index(),
            num_of_particles: self.num_of_particles,
            bounds_buffer: self.bounds_buffer.descriptor_index(),
        });
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
        cmd.push_constants(&BuildTreePc {
            morton_buffer_id: self.morton_buffer.descriptor_index(),
            indices_buffer_id: self.indices.descriptor_index(),
            node_buffer_id: self.tree_buffer.descriptor_index(),
            num_of_particles: self.num_of_particles,
        });
        cmd.dispatch(u32::div_ceil(self.num_of_particles - 1, 256), 1, 1);
        cmd.barriers(
            Some(&GlobalBarrier {
                previous_accesses: &[AccessType::ComputeShaderStorageWrite],
                next_accesses: &[AccessType::ComputeShaderStorageWrite],
            }),
            &[],
        );

        // compute com for each node
        cmd.fill_buffer(&self.counter, 0, u64::MAX, 0);
        cmd.barriers(
            Some(&GlobalBarrier {
                previous_accesses: &[AccessType::TransferWrite],
                next_accesses: &[AccessType::ComputeShaderStorageWrite],
            }),
            &[],
        );

        cmd.bind_compute_pipeline(&self.pipelines.compute_mass);
        cmd.push_constants(&ComputeMassPc {
            particle_buffer_id: self.particle_buffer.descriptor_index(),
            indices_buffer_id: self.indices.descriptor_index(),
            node_buffer_id: self.tree_buffer.descriptor_index(),
            counter_buffer_id: self.counter.descriptor_index(),
            num_of_particles: self.num_of_particles,
        });
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
        cmd.push_constants(&IntegratePc {
            particle_buffer_id: self.particle_buffer.descriptor_index(),
            indices_buffer_id: self.indices.descriptor_index(),
            node_buffer_id: self.tree_buffer.descriptor_index(),
            num_of_particles: self.num_of_particles,
            dt: 0.016,
            theta: 0.9,
            epsilon: 0.7,
        });
        cmd.dispatch(groups, 1, 1);
    }

    pub fn readback_raw(&self, buffer: &Buffer, byte_size: u64) -> Vec<u8> {
        let mut cmd = record(QueueType::Compute);
        cmd.copy_buffer(*buffer, self.read_back, 0, 0, byte_size);
        wait(submit(&[cmd]));

        let slice = self.read_back.as_slice::<u8>();
        slice[..byte_size as usize].to_vec()
    }

    pub fn readback_as<T: Clone>(&self, buffer: &Buffer, count: usize) -> Vec<T> {
        let byte_size = (count * std::mem::size_of::<T>()) as u64;
        let bytes = self.readback_raw(buffer, byte_size);
        let ptr = bytes.as_ptr() as *const T;
        unsafe { std::slice::from_raw_parts(ptr, count).to_vec() }
    }
}

#[test]
fn test() {
    sgpu_init(&SgpuInititizationInfo::default());

    const N: usize = 16;

    let particles: Vec<Particle> = (0..N)
        .map(|i| {
            let t = i as f32 / N as f32;
            Particle {
                position: [
                    (t * 37.0).sin() * 100.0,
                    (t * 53.0).cos() * 100.0,
                    (t * 71.0).sin() * 100.0,
                ],
                velocity: [0.0; 3],
                mass: 1.0,
                radius: 1.0,
            }
        })
        .collect();

    println!("=== INPUT PARTICLES ===");
    for i in 0..N {
        println!("  [{}] pos=({:.2}, {:.2}, {:.2})", i, particles[i].position[0], particles[i].position[1], particles[i].position[2]);
    }

    let sim = Simulator::new(&particles);

    let mut c = record(sgpu::QueueType::Compute);
    sim.record_simulation(&mut c);
    wait(submit(&[c]));

    // BOUNDS
    let bounds_u = sim.readback_as::<u32>(&sim.bounds_buffer, 8);
    let bounds_f = sim.readback_as::<f32>(&sim.bounds_buffer, 8);
    println!("\n=== BOUNDS RAW (u32 hex / f32) ===");
    for i in 0..8 {
        println!("  [{}] = {:#010x} / {:.4}", i, bounds_u[i], bounds_f[i]);
    }

    // MORTON (before sort — but we only have post-sort morton)
    let morton = sim.readback_as::<u32>(&sim.morton_buffer, N);
    println!("\n=== MORTON CODES (post sort) ===");
    for i in 0..N {
        println!("  [{}] = {:#010x}", i, morton[i]);
    }

    // INDICES
    let indices = sim.readback_as::<u32>(&sim.indices, N);
    println!("\n=== SORTED INDICES ===");
    for i in 0..N {
        println!("  [{}] -> particle {}", i, indices[i]);
    }

    // FULL TREE DUMP
    // Node = float4 aabb_min, float4 aabb_max, float4 com, uint left, uint right, uint parent, uint is_leaf
    // = 16 floats = 16 u32s = 64 bytes
    let num_nodes = 2 * N - 1; // n-1 internal + n leaves
    let tree_f = sim.readback_as::<f32>(&sim.tree_buffer, num_nodes * 16);
    let tree_u = sim.readback_as::<u32>(&sim.tree_buffer, num_nodes * 16);
    println!("\n=== TREE NODES ({} total: {} internal, {} leaves) ===", num_nodes, N - 1, N);
    for node in 0..num_nodes {
        let b = node * 16;
        let aabb_min = (tree_f[b + 0], tree_f[b + 1], tree_f[b + 2]);
        let aabb_max = (tree_f[b + 4], tree_f[b + 5], tree_f[b + 6]);
        let com = (tree_f[b + 8], tree_f[b + 9], tree_f[b + 10]);
        let mass = tree_f[b + 11];
        let left = tree_u[b + 12];
        let right = tree_u[b + 13];
        let parent = tree_u[b + 14];
        let is_leaf = tree_u[b + 15];
        let kind = if node < N - 1 { "INT" } else { "LEAF" };
        println!(
            "  node {:2} [{}] is_leaf={} left={:2} right={:2} parent={:2} mass={:.1} \
                  com=({:.1},{:.1},{:.1}) aabb=[({:.1},{:.1},{:.1})->({:.1},{:.1},{:.1})]",
            node, kind, is_leaf, left, right, parent, mass, com.0, com.1, com.2, aabb_min.0, aabb_min.1, aabb_min.2, aabb_max.0, aabb_max.1, aabb_max.2
        );
    }

    // PARTICLES AFTER
    let out_f = sim.readback_as::<f32>(&sim.particle_buffer, N * 8);
    println!("\n=== PARTICLES AFTER 1 FRAME ===");
    // Particle layout: float4 velocity_mass, float4 position_radius
    for i in 0..N {
        let b = i * 8;
        let vel = (out_f[b + 0], out_f[b + 1], out_f[b + 2]);
        let mass = out_f[b + 3];
        let pos = (out_f[b + 4], out_f[b + 5], out_f[b + 6]);
        println!("  [{}] pos=({:.3},{:.3},{:.3}) vel=({:.4},{:.4},{:.4}) mass={:.1}", i, pos.0, pos.1, pos.2, vel.0, vel.1, vel.2, mass);
    }
}
