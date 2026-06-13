import "../sgpu.slang";
import "../common.slang";

struct PushConstants {
  float4x4 view;
  float4x4 proj;
  uint particle_buffer_id;
};

[vk_push_constant]
PushConstants pc;

static const float2 QUAD[6] = { float2(-1, -1), float2(1, -1), float2(1, 1),
                                float2(-1, -1), float2(1, 1),  float2(-1, 1) };

struct VertexOutput {
  float4 position : SV_Position;
  [vk_location(0)]
  float3 centerView;
  [vk_location(1)]
  float3 rayDir;
  [vk_location(2)]
  float3 worldPos;
  [vk_location(3)]
  float radius;
  [vk_location(4)]
  float mass;
};

[shader("vertex")]
VertexOutput main(uint vertex_id: SV_VertexID) {
  let particles = get_buffer<Particle>(pc.particle_buffer_id);
  Particle p = particles[vertex_id / 6];

  float2 corner = QUAD[vertex_id % 6];
  float radius = p.radius();

  float3 centerView = mul(pc.view, float4(p.position(), 1.0)).xyz;

  // billboard in view space (camera-aligned quad)
  float3 cornerView = centerView + float3(1.4, 0, 0) * corner.x * radius +
                      float3(0, 1.4, 0) * corner.y * radius;

  VertexOutput o;
  o.position = mul(pc.proj, float4(cornerView, 1.0));
  o.centerView = centerView;
  o.rayDir = cornerView;
  o.worldPos = p.position();
  o.radius = radius;
  o.mass = p.mass();

  return o;
}
