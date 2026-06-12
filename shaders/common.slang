
module common;

public struct Particle {
  float4 velocity_mass;
  float4 position_radius;

  public float3 position() { return position_radius.xyz; }
  public float3 radius() { return position_radius.w; }

  public float3 velocity() { return velocity_mass.xyz; }
  public float mass() { return velocity_mass.w; }
};

public struct Bounds {
  public float3 max;
  public float3 min;
}
