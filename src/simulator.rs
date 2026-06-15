use sgpu::*;
use winit::event::MouseButton;

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
struct ParentsPc {
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
    parent_set: ComputePipeline,
    compute_mass: ComputePipeline,
    integrate: ComputePipeline,
}

impl SimulationPipelines {
    fn new() -> SimulationPipelines {
        return SimulationPipelines {
            bound: create_compute_pipeline(include_bytes!("../compiled/bounds.spv")),
            morton: create_compute_pipeline(include_bytes!("../compiled/morton_code.spv")),
            build_tree: create_compute_pipeline(include_bytes!("../compiled/tree.spv")),
            parent_set: create_compute_pipeline(include_bytes!("../compiled/parent.spv")),
            compute_mass: create_compute_pipeline(include_bytes!("../compiled/com.spv")),
            integrate: create_compute_pipeline(include_bytes!("../compiled/integrate.spv")),
        };
    }
}

pub struct Simulator {
    debug: Buffer,
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
            usage: BufferUsage::TRANSFER_DST | BufferUsage::TRANSFER_SRC | BufferUsage::STORAGE,
            size: particle_size * num_of_particles * 4,
            memory_type: MemoryType::PreferHost,
        });

        let debug = create_buffer(&BufferDescription {
            usage: BufferUsage::TRANSFER_DST | BufferUsage::TRANSFER_SRC | BufferUsage::STORAGE,
            size: particle_size * num_of_particles * 4,
            memory_type: MemoryType::DeviceLocal,
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
            debug: debug,
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

        cmd.fill_buffer(&self.tree_buffer, 0, 2 * self.num_of_particles as u64 * NODE_SIZE, 1000000000);
        cmd.barriers(
            Some(&GlobalBarrier {
                previous_accesses: &[AccessType::TransferWrite],
                next_accesses: &[
                    AccessType::ComputeShaderStorageWrite,
                    AccessType::ComputeShaderStorageRead,
                ],
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
                next_accesses: &[
                    AccessType::ComputeShaderStorageWrite,
                    AccessType::ComputeShaderStorageRead,
                ],
            }),
            &[],
        );

        // a pass for setting parents
        cmd.bind_compute_pipeline(&self.pipelines.parent_set);
        cmd.push_constants(&ParentsPc {
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
        cmd.fill_buffer(&self.counter, 0, (self.num_of_particles as u64 - 1) * 4, 0);
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
            theta: 0.8,
            epsilon: 0.3,
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

    pub fn record_and_time_simulation(&self) {
        // Time each pass individually by submitting and waiting separately
        let groups = u32::div_ceil(self.num_of_particles, 256);

        let time_pass = |name: &str, f: &dyn Fn(&mut CommandBuffer)| {
            let mut cmd = record(QueueType::Compute);
            f(&mut cmd);
            let start = std::time::Instant::now();
            wait(submit(&[cmd]));
            println!("{}: {:.2}ms", name, start.elapsed().as_secs_f64() * 1000.0);
        };

        time_pass("bounds", &|cmd| {
            cmd.fill_buffer(&self.bounds_buffer, 0, 4 * 4, 0);
            cmd.fill_buffer(&self.bounds_buffer, 4 * 4, 4 * 4, 0xFFFFFFFF);
            cmd.bind_compute_pipeline(&self.pipelines.bound);
            cmd.push_constants(&BoundsPc {
                particle_buffer_id: self.particle_buffer.descriptor_index(),
                num_of_particles: self.num_of_particles,
                bounds_buffer_id: self.bounds_buffer.descriptor_index(),
            });
            cmd.dispatch(groups, 1, 1);
        });

        time_pass("morton", &|cmd| {
            cmd.bind_compute_pipeline(&self.pipelines.morton);
            cmd.push_constants(&MortonPc {
                particle_buffer_id: self.particle_buffer.descriptor_index(),
                morton_buffer_id: self.morton_buffer.descriptor_index(),
                num_of_particles: self.num_of_particles,
                bounds_buffer: self.bounds_buffer.descriptor_index(),
            });
            cmd.dispatch(groups, 1, 1);
        });

        time_pass("radix_sort", &|cmd| {
            self.radix_sort.record(cmd, self.morton_buffer, self.indices);
        });

        time_pass("build_tree", &|cmd| {
            cmd.fill_buffer(&self.tree_buffer, 0, 2 * self.num_of_particles as u64 * NODE_SIZE, 0);
            cmd.barriers(
                Some(&GlobalBarrier {
                    previous_accesses: &[AccessType::TransferWrite],
                    next_accesses: &[
                        AccessType::ComputeShaderStorageWrite,
                        AccessType::ComputeShaderStorageRead,
                    ],
                }),
                &[],
            );
            cmd.bind_compute_pipeline(&self.pipelines.build_tree);
            cmd.push_constants(&BuildTreePc {
                morton_buffer_id: self.morton_buffer.descriptor_index(),
                indices_buffer_id: self.indices.descriptor_index(),
                node_buffer_id: self.tree_buffer.descriptor_index(),
                num_of_particles: self.num_of_particles,
            });
            cmd.dispatch(u32::div_ceil(self.num_of_particles - 1, 256), 1, 1);
        });

        time_pass("parent_set", &|cmd| {
            cmd.bind_compute_pipeline(&self.pipelines.parent_set);
            cmd.push_constants(&ParentsPc {
                node_buffer_id: self.tree_buffer.descriptor_index(),
                num_of_particles: self.num_of_particles,
            });
            cmd.dispatch(u32::div_ceil(self.num_of_particles - 1, 256), 1, 1);
        });

        time_pass("compute_mass", &|cmd| {
            cmd.fill_buffer(&self.counter, 0, (self.num_of_particles as u64 - 1) * 4, 0);
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
        });

        time_pass("integrate", &|cmd| {
            cmd.bind_compute_pipeline(&self.pipelines.integrate);
            cmd.push_constants(&IntegratePc {
                particle_buffer_id: self.particle_buffer.descriptor_index(),
                indices_buffer_id: self.indices.descriptor_index(),
                node_buffer_id: self.tree_buffer.descriptor_index(),
                num_of_particles: self.num_of_particles,
                dt: 0.016,
                theta: 0.8,
                epsilon: 0.3,
            });
            cmd.dispatch(groups, 1, 1);
        });
    }
}

#[derive(Debug, Clone)]
struct Node {
    // keeping w component empty for safe alignment
    aabb_min: [f32; 4],
    aabb_max: [f32; 4],
    // w component is the mass
    com: [f32; 4],
    left: u32,
    right: u32,
    parent: u32,
    is_leaf: u32,
}

#[test]
fn test() {
    use rand::Rng;

    sgpu_init(&SgpuInititizationInfo::default());

    const N: usize = 1 << 20;

    let radius = 1000.0;
    let mut rng = rand::rng();

    let particles: Vec<Particle> = (0..N)
        .map(|_| {
            // Exponential radial distribution
            let r = rng.random_range(0.0..radius);
            let theta = rng.random_range(0.0..360.0f32).to_radians();

            let x = r * theta.cos();
            let z = r * theta.sin();

            Particle {
                position: [x, 0.0, z],
                velocity: [0.0; 3],
                mass: 1.0,
                radius: 0.3,
            }
        })
        .collect();

    let sim = Simulator::new(&particles);

    for i in 0..10 {
        println!("{}", i);
        sim.record_and_time_simulation();
        println!();
    }

    let dbg = sim.readback_as::<Node>(&sim.tree_buffer, 2 * N - 1);
    validate_and_trace(&dbg, N);
}

fn validate_and_trace(nodes: &[Node], num_particles: usize) {
    let total_nodes = 2 * num_particles - 1;
    let leaf_offset = num_particles - 1;

    // 1. Find the first node with an invalid child pointer
    for i in 0..leaf_offset {
        let node = &nodes[i];
        if node.left >= total_nodes as u32 {
            println!("Invalid left child at internal node {}", i);
            println!("  left = {}, right = {}", node.left, node.right);
            println!("Tracing parent chain from this node up to root:");
            let mut cur = i;
            while cur != 0xFFFFFFFF as usize && cur < total_nodes {
                println!("  node {}", cur);
                let parent = nodes[cur].parent as usize;
                if parent == 0xFFFFFFFF as usize {
                    break;
                }
                cur = parent;
            }
            // Also find which node points to this node as a child
            println!("Finding nodes that have this node as child:");
            for j in 0..leaf_offset {
                if nodes[j].left == i as u32 || nodes[j].right == i as u32 {
                    println!("  node {} has this node as child (left={}, right={})", j, nodes[j].left, nodes[j].right);
                }
            }
            panic!("Tree validation failed at node {}", i);
        }
        if node.right >= total_nodes as u32 {
            println!("Invalid right child at internal node {}", i);
            println!("  left = {}, right = {}", node.left, node.right);
            let mut cur = i;
            while cur != 0xFFFFFFFF as usize && cur < total_nodes {
                println!("  node {}", cur);
                let parent = nodes[cur].parent as usize;
                if parent == 0xFFFFFFFF as usize {
                    break;
                }
                cur = parent;
            }
            panic!("Tree validation failed at node {}", i);
        }
    }

    // 2. Check that every internal node's parent pointer matches
    for i in 0..leaf_offset {
        let left = nodes[i].left as usize;
        let right = nodes[i].right as usize;
        if left < total_nodes && nodes[left].parent != i as u32 {
            println!("Parent mismatch: node {} parent is {} but node {} has left child {}", left, nodes[left].parent, i, left);
            let mut cur = i;
            while cur != 0xFFFFFFFF as usize && cur < total_nodes {
                println!("  node {}", cur);
                let parent = nodes[cur].parent as usize;
                if parent == 0xFFFFFFFF as usize {
                    break;
                }
                cur = parent;
            }
            panic!("Parent validation failed");
        }
        if right < total_nodes && nodes[right].parent != i as u32 {
            println!("Parent mismatch: node {} parent is {} but node {} has right child {}", right, nodes[right].parent, i, right);
            let mut cur = i;
            while cur != 0xFFFFFFFF as usize && cur < total_nodes {
                println!("  node {}", cur);
                let parent = nodes[cur].parent as usize;
                if parent == 0xFFFFFFFF as usize {
                    break;
                }
                cur = parent;
            }
            panic!("Parent validation failed");
        }
    }

    println!("Tree validation passed.");
}
