use nexion::{utils::vulkan_context::VulkanContext, *};

use crate::camera::Camera;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Zeroable, bytemuck::Pod)]
struct DrawParams {
    view_proj: [[f32; 4]; 4],
    cam_pos: [f32; 3],
    focal_lenght: f32,
}

pub struct Renderer {
    vkc: VulkanContext,
    particle_buffer: BufferID,
    num_of_particles: u32,
    pipeline: RasterizationPipeline,
    recorder: CommandRecorder,
}

impl Renderer {
    pub fn new(vkc: VulkanContext, particle_buffer: BufferID, num_of_particles: u32) -> Renderer {
        return Renderer {
            pipeline: vkc.create_rasterization_pipeline(&RasterizationPipelineDescription {
                push_constants: PushConstantsDescription {
                    stage_flags: ShaderStages::ALL,
                    offset: 0,
                    size: std::mem::size_of::<DrawParams>() as u32,
                },
                geometry: GeometryStage::Classic {
                    vertex_input: VertexInputDescription::default(),
                    topology: InputTopology::PointList,
                    vertex_shader: "shaders/vertex.slang",
                },
                fragment_shader_path: "shaders/fragment.slang",
                cull_mode: CullMode::None,
                polygon_mode: PolygonMode::Fill,
                alpha_blend_enable: false,
                outputs: PipelineOutputs {
                    color: &[Format::Rgba16Float],
                    depth: None,
                    stencil: None,
                },
                ..Default::default()
            }),
            particle_buffer: particle_buffer,
            num_of_particles: num_of_particles,
            recorder: vkc.create_command_recorder(QueueType::Graphics),
            vkc: vkc,
        };
    }

    pub fn record(
        &mut self,
        width: u32,
        height: u32,
        camera: &Camera,
        img: ImageID,
        img_view: ImageViewID,
    ) -> ExecutableCommandBuffer {
        self.recorder.reset();

        self.recorder
            .begin_recording(CommandBufferUsage::OneTimeSubmit);

        self.recorder
            .pipeline_barrier(&[Barrier::Image(ImageBarrier {
                image: img,
                old_layout: ImageLayout::Undefined,
                new_layout: ImageLayout::ColorAttachment,
                src_stage: PipelineStage::TopOfPipe,
                dst_stage: PipelineStage::ColorAttachmentOutput,
                src_access: AccessType::None,
                dst_access: AccessType::ColorAttachmentWrite,
                ..Default::default()
            })]);

        self.recorder.set_push_constants(
            &DrawParams {
                view_proj: camera.view_proj().to_cols_array_2d(),
                cam_pos: camera.position.into(),
                focal_lenght: camera.focal_lenght(),
            },
            &self.pipeline,
        );

        self.recorder.begin_rendering(&RenderingBeginInfo {
            render_area: RenderArea {
                extent: Extent2D {
                    width: width,
                    height: height,
                },
                offset: Offset2D { x: 0, y: 0 },
            },
            rendering_flags: RenderingFlags::None,
            view_mask: 0,
            layer_count: 1,
            color_attachments: &[RenderingAttachment {
                image_view: img_view,
                image_layout: ImageLayout::ColorAttachment,
                clear_value: ClearValue::ColorFloat([0.0, 0.0, 0.0, 1.0]),
                ..Default::default()
            }],
            depth_attachment: None,
            stencil_attachment: None,
        });

        self.recorder.bind_pipeline(&self.pipeline);
        self.recorder.set_viewport_and_scissor(width, height);
        self.recorder.draw(self.num_of_particles, 1, 0, 0);

        self.recorder.end_rendering();

        self.recorder
            .pipeline_barrier(&[Barrier::Image(ImageBarrier {
                image: img,
                old_layout: ImageLayout::ColorAttachment,
                new_layout: ImageLayout::PresentSrc,
                src_stage: PipelineStage::ColorAttachmentOutput,
                dst_stage: PipelineStage::BottomOfPipe,
                src_access: AccessType::ColorAttachmentWrite,
                dst_access: AccessType::None,
                ..Default::default()
            })]);

        return self.recorder.end_recording();
    }
}
