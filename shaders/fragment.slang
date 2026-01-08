[shader("fragment")]
float4 main(float2 point_coord: SV_PointCoord) : SV_Target {
  if (length(point_coord - float2(0.5, 0.5)) > 0.5f) {
    discard;
  }

  return float4(1.0, 1.0, 1.0, 1.0);
}
