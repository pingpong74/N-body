
module common;

public struct Particle {
  float4 velocity_mass;
  float4 position_radius;

  public float3 position() { return position_radius.xyz; }
  public float radius() { return position_radius.w; }

  public float3 velocity() { return velocity_mass.xyz; }
  public float mass() { return velocity_mass.w; }

  [mutating]
  public void set_velocity(float3 vel) {
    velocity_mass = float4(vel, velocity_mass.w);
  }

  [mutating]
  public void set_position(float3 pos) {
    position_radius = float4(pos, position_radius.w);
  }
};

public struct Bounds {
  public float3 max;
  float pad1;
  public float3 min;
  float pad2;
};

public struct Node {
  // keeping w component empty for safe alignment
  public float4 aabb_min;
  public float4 aabb_max;
  // w component is the mass
  public float4 com;
  public uint left;
  public uint right;
  public uint parent;
  public uint skip_index;
};
