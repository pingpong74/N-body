use sgpu::{AccessType, Buffer, BufferUsage, CommandBuffer, ComputePipeline, GlobalBarrier, create_buffer, create_compute_pipeline};

#[repr(C)]
#[derive(Clone, Copy)]
struct HistogramPc {
    count: u32,
    shift: u32,
    workgroups: u32,
    blocks_per_workgroup: u32,
    in_buffer_id: u32,
    histogram_buffer_id: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct SortPc {
    count: u32,
    shift: u32,
    workgroups: u32,
    blocks_per_workgroup: u32,
    in_buffer_id: u32,
    out_buffer_id: u32,
    in_index_id: u32,
    out_index_id: u32,
    histogram_buffer_id: u32,
}

pub struct RadixSorter {
    temp_buffer: Buffer,
    temp_index_buffer: Buffer,
    histogram_buffer: Buffer,

    histogram_pipeline: ComputePipeline,
    sort_pipeline: ComputePipeline,

    count: u32,
    blocks_per_wg: u32,
    dispatch_x: u32,
}

impl RadixSorter {
    pub fn new(count: usize) -> RadixSorter {
        let blocks_per_wg: u32 = 32;
        let global_inv = (count as u32 + blocks_per_wg - 1) / blocks_per_wg;
        let groups = (global_inv + 255) / 256;

        let temp = create_buffer(&sgpu::BufferDescription {
            usage: BufferUsage::STORAGE,
            size: count as u64 * 4,
            memory_type: sgpu::MemoryType::DeviceLocal,
        });

        let temp_indices = create_buffer(&sgpu::BufferDescription {
            usage: BufferUsage::STORAGE,
            size: count as u64 * 4,
            memory_type: sgpu::MemoryType::DeviceLocal,
        });

        let histogram_buffer = create_buffer(&sgpu::BufferDescription {
            usage: BufferUsage::STORAGE,
            size: groups as u64 * 256 * 4,
            memory_type: sgpu::MemoryType::DeviceLocal,
        });

        return RadixSorter {
            temp_buffer: temp,
            temp_index_buffer: temp_indices,
            histogram_buffer: histogram_buffer,
            histogram_pipeline: create_compute_pipeline(include_bytes!("../compiled/histogram.spv")),
            sort_pipeline: create_compute_pipeline(include_bytes!("../compiled/sort.spv")),
            count: count as u32,
            blocks_per_wg: blocks_per_wg,
            dispatch_x: groups,
        };
    }

    pub fn record(&self, cmd: &mut CommandBuffer, values: Buffer, indices: Buffer) {
        let val_buffers = [
            values,
            self.temp_buffer,
        ];

        let index_buffers = [
            indices,
            self.temp_index_buffer,
        ];

        for pass in 0..4u32 {
            let shift = pass * 8;

            let src_val = val_buffers[pass as usize % 2];
            let dst_val = val_buffers[(pass as usize + 1) % 2];

            let src_index = index_buffers[pass as usize % 2];
            let dst_index = index_buffers[(pass as usize + 1) % 2];

            cmd.bind_compute_pipeline(&self.histogram_pipeline);
            cmd.push_constants(&HistogramPc {
                count: self.count,
                shift,
                workgroups: self.dispatch_x,
                blocks_per_workgroup: self.blocks_per_wg,
                in_buffer_id: src_val.descriptor_index(),
                histogram_buffer_id: self.histogram_buffer.descriptor_index(),
            });
            cmd.dispatch(self.dispatch_x, 1, 1);

            cmd.barriers(
                Some(&GlobalBarrier {
                    previous_accesses: &[AccessType::ComputeShaderStorageWrite],
                    next_accesses: &[AccessType::ComputeShaderStorageRead],
                }),
                &[],
            );

            cmd.bind_compute_pipeline(&self.sort_pipeline);
            cmd.push_constants(&SortPc {
                count: self.count,
                shift,
                workgroups: self.dispatch_x,
                blocks_per_workgroup: self.blocks_per_wg,
                in_buffer_id: src_val.descriptor_index(),
                out_buffer_id: dst_val.descriptor_index(),
                in_index_id: src_index.descriptor_index(),
                out_index_id: dst_index.descriptor_index(),
                histogram_buffer_id: self.histogram_buffer.descriptor_index(),
            });
            cmd.dispatch(self.dispatch_x, 1, 1);

            if pass < 3 {
                cmd.barriers(
                    Some(&GlobalBarrier {
                        previous_accesses: &[AccessType::ComputeShaderStorageWrite],
                        next_accesses: &[AccessType::ComputeShaderStorageRead],
                    }),
                    &[],
                );
            }
        }
    }
}

/*#[cfg(test)]
mod test {
    use std::time::Instant;

    use sgpu::*;

    use crate::radix_sort::RadixSorter;

    #[test]
    fn test() {
        sgpu_init(&SgpuInititizationInfo {
            enable_validation_layers: false,
            ..Default::default()
        });

        const LEN: usize = 1_000_000;

        let staging_buffer = create_buffer(&BufferDescription {
            usage: BufferUsage::TRANSFER_SRC | BufferUsage::TRANSFER_DST,
            size: LEN as u64 * 4,
            memory_type: MemoryType::PreferHost,
        });

        let buf = create_buffer(&BufferDescription {
            usage: BufferUsage::STORAGE | BufferUsage::TRANSFER_DST | BufferUsage::TRANSFER_SRC,
            size: LEN as u64 * 4,
            memory_type: MemoryType::DeviceLocal,
        });

        let indices = create_buffer(&BufferDescription {
            usage: BufferUsage::STORAGE | BufferUsage::TRANSFER_DST | BufferUsage::TRANSFER_SRC,
            size: LEN as u64 * 4,
            memory_type: MemoryType::DeviceLocal,
        });

        let stating_slice = staging_buffer.as_mut_slice();

        let mut og_array = vec![0u32; LEN];

        for i in 0..LEN {
            og_array[i] = LEN as u32 - i as u32;
            stating_slice[i] = og_array[i];
        }

        let mut cmd = record(QueueType::Transfer);
        cmd.copy_buffer(staging_buffer, buf, 0, 0, LEN as u64 * 4);
        wait(submit(&[cmd]));

        for i in 0..LEN {
            stating_slice[i] = i as u32;
        }

        let mut cmd = record(QueueType::Transfer);
        cmd.copy_buffer(staging_buffer, indices, 0, 0, LEN as u64 * 4);
        wait(submit(&[cmd]));

        let sorter = RadixSorter::new(LEN);

        let cpu_start = Instant::now();
        let mut cmd = record(QueueType::Compute);
        sorter.record(&mut cmd, buf, indices);
        let cpu_end = cpu_start.elapsed();

        let gpu_start = Instant::now();
        wait(submit(&[cmd]));
        let gpu_end = gpu_start.elapsed();

        println!("CPU time: {:?} \nGPU time: {:?}", cpu_end, gpu_end);

        let mut cmd = record(QueueType::Transfer);
        cmd.copy_buffer(buf, staging_buffer, 0, 0, LEN as u64 * 4);
        wait(submit(&[cmd]));

        // already checked that the sort is correct
        /*for i in 0..(LEN - 1) {
            if stating_slice[i] > stating_slice[i + 1] {
                panic!()
            }
        }*/
    }
}*/
