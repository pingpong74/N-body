struct PushConstants {
  float4x4 view;
  float4x4 proj;
  uint particle_buffer_id;
};

[vk_push_constant]
PushConstants pc;

struct VertexOutput {
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

[shader("fragment")]
float4 main(VertexOutput in) : SV_Target {
  float3 rayDir = normalize(in.rayDir);
  float3 center = in.centerView;
  float r = in.radius;

  float b = dot(rayDir, center);
  float c = dot(center, center) - r * r;
  float disc = b * b - c;
  if (disc < 0.0) {
    discard;
  }

  float t = b - sqrt(disc);
  float3 hitPos = rayDir * t;
  float3 normal = (hitPos - center) / r;

  // correct depth
  float4 clipPos = mul(pc.proj, float4(hitPos, 1.0));

  // lighting
  float3 lightDir = normalize(float3(-1, 1, 1));
  float diffuse = saturate(dot(normal, lightDir));

  float t_mass = saturate(in.mass / 2.0);
  float3 color = lerp(float3(0.2, 0.4, 1.0), float3(1.0, 0.3, 0.1), t_mass);
  color *= (0.3 + 0.7 * diffuse);

  return float4(color, 1.0);
}
