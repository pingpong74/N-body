use sgpu::*;
use winit::dpi::PhysicalSize;

use crate::camera::Camera;

#[repr(C)]
#[derive(Clone, Copy)]
struct VertexPushConstants {
    view: [[f32; 4]; 4],
    proj: [[f32; 4]; 4],
    particle_buffer_id: u32,
}

pub struct Renderer {
    pipeline: RasterizationPipeline,
    depth_image: Image,
    size: PhysicalSize<u32>,
    num_of_particles: u32,
}

impl Renderer {
    pub fn new(num_of_particles: u32, size: PhysicalSize<u32>) -> Renderer {
        let raster_pipeline = create_rasterization_pipeline(&RasterizationPipelineDescription {
            vertex_shader: include_bytes!("../compiled/vertex.spv"),
            fragment_shader: include_bytes!("../compiled/fragment.spv"),
            topology: PrimitiveTopology::TriangleList,
            polygon_mode: PolygonMode::Fill,
            outputs: PipelineOutputs {
                color: &[Format::Rgba16Float],
                depth: Some(Format::D32Float),
                stencil: None,
            },
            ..Default::default()
        });

        let depth_image = create_depth_image(size.width, size.height);

        return Renderer {
            pipeline: raster_pipeline,
            depth_image,
            size,
            num_of_particles,
        };
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        self.size.height = height;
        self.size.width = width;

        destroy_image(self.depth_image);

        self.depth_image = create_depth_image(width, height);
    }

    pub fn record_rendering(&self, cmd: &mut CommandBuffer, camera: &Camera, swapchain_img: &AcquiredImage, particle_buffer: &Buffer) {
        cmd.wait_for_swapchain_image(&swapchain_img);

        cmd.image_barrier(&ImageBarrier {
            view: swapchain_img.image().default_view(),
            previous_accesses: &[AccessType::None],
            next_accesses: &[AccessType::ColorAttachmentWrite],
            discard_contents: true,
            ..Default::default()
        });

        cmd.begin_rendering(
            &RenderingBeginInfo {
                render_area: RenderArea {
                    offset: Offset2D { x: 0, y: 0 },
                    extent: Extent2D {
                        width: self.size.width,
                        height: self.size.height,
                    },
                },
                color_attachments: &[
                    RenderingAttachment {
                        image_view: swapchain_img.image().default_view(),
                        clear_value: ClearValue::ColorFloat([0.5, 0.8, 0.4, 1.0]),
                        ..Default::default()
                    },
                ],
                depth_attachment: Some(RenderingAttachment {
                    image_view: self.depth_image.default_view(),
                    clear_value: ClearValue::DepthStencil {
                        depth: 1.0,
                        stencil: 0,
                    },
                    ..Default::default()
                }),
                ..Default::default()
            },
            |r| {
                r.bind_rasterization_pipeline(&self.pipeline);
                r.push_constants(&VertexPushConstants {
                    particle_buffer_id: particle_buffer.descriptor_index(),
                    view: camera.view().to_cols_array_2d(),
                    proj: camera.projection().to_cols_array_2d(),
                });
                r.set_viewport(self.size.width, self.size.height);
                r.set_scissor(self.size.width, self.size.height);
                r.draw(self.num_of_particles * 6, 1, 0, 0);
            },
        );

        cmd.image_barrier(&ImageBarrier {
            view: swapchain_img.image().default_view(),
            previous_accesses: &[AccessType::ColorAttachmentWrite],
            next_accesses: &[AccessType::Present],
            discard_contents: false,
            ..Default::default()
        });
    }
}

fn create_depth_image(width: u32, height: u32) -> Image {
    let img = create_image(&ImageDescription {
        usage: ImageUsage::DEPTH_STENCIL_ATTACHMENT,
        format: Format::D32Float,
        extent: Extent3D {
            width: width,
            height: height,
            depth: 1,
        },
        memory_type: MemoryType::DeviceLocal,
        default_view: ImageViewDescription {
            subresources: ImageSubresources {
                aspect: ImageAspect::DEPTH,
                ..Default::default()
            },
            ..Default::default()
        },

        ..Default::default()
    });

    let mut c = record(QueueType::Graphics);
    c.barriers(
        None,
        &[ImageBarrier {
            view: img.default_view(),
            previous_accesses: &[AccessType::None],
            next_accesses: &[AccessType::DepthStencilAttachmentWrite],
            ..Default::default()
        }],
    );

    wait(submit(&[c]));

    return img;
}
