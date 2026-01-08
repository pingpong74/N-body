use nexion::{utils::vulkan_context::VulkanContext, *};

pub const NO_OF_LODS: usize = 4;

pub struct Grid {
    pub buffer: BufferID,
    pub resolution: u32,
    pub cell_size: u32,
}

pub struct LodPyramid {
    pub lods: [Grid; NO_OF_LODS],
}

impl LodPyramid {
    pub fn new(vkc: &VulkanContext, grid_size: u32, cell_size: u32) -> LodPyramid {
        return LodPyramid {
            lods: std::array::from_fn(|i| {
                let res = grid_size >> i;
                let cell_size_i = cell_size << i;

                println!(
                    "Lod {} of side {} cell size {} size of buffer is {}",
                    i as u32,
                    res,
                    cell_size_i,
                    (res as u64 * res as u64 * res as u64) * std::mem::size_of::<f32>() as u64
                );

                let b = vkc.create_buffer(&BufferDescription {
                    usage: BufferUsage::STORAGE | BufferUsage::TRANSFER_DST,
                    size: (res as u64 * res as u64 * res as u64)
                        * std::mem::size_of::<f32>() as u64,
                    memory_type: MemoryType::DeviceLocal,
                    create_mapped: false,
                });

                vkc.write_buffer(&BufferWriteInfo {
                    buffer: b,
                    offset: 0,
                    range: u64::MAX,
                    index: i as u32,
                });

                return Grid {
                    buffer: b,
                    resolution: res,
                    cell_size: cell_size_i,
                };
            }),
        };
    }
}
