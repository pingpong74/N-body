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

  return float4(color(in.mass, in.radius, normal), 1.0);
}

float3 color(float mass, float radius, float3 normal) {
  float massNorm = saturate(mass / 2.0);
  massNorm = pow(massNorm, 1.5);
  float3 hotColor = float3(0.6, 0.8, 1.0);
  float3 midColor = float3(1.0, 1.0, 1.0);
  float3 coolColor = float3(1.0, 0.4, 0.1);
  float3 color = lerp(hotColor, midColor, saturate(massNorm * 1.5));

  color = lerp(color, coolColor, saturate((massNorm - 0.5) * 2.0));
  float radiusGlow = saturate(radius / 3.0);
  radiusGlow = pow(radiusGlow, 2.0);

  color *= (0.6 + 0.8 * radiusGlow);
  float3 lightDir = normalize(float3(-1, 1, 1));
  float diffuse = saturate(dot(normal, lightDir));
  diffuse = 0.25 + 0.75 * diffuse; // avoid flat black sides

  color *= diffuse; 
  color += 0.05 * radiusGlow;

  return pow(color, 0.9);
}
