module sh_lighting;
import camera_uniform;

public float3 evaluateShIrradiance(float3 normal)
{
    const float x = normal.x;
    const float y = normal.y;
    const float z = normal.z;

    float basis[9];
    basis[0] = 0.282095;
    basis[1] = 0.488603 * y;
    basis[2] = 0.488603 * z;
    basis[3] = 0.488603 * x;
    basis[4] = 1.092548 * x * y;
    basis[5] = 1.092548 * y * z;
    basis[6] = 0.315392 * ((3.0 * z * z) - 1.0);
    basis[7] = 1.092548 * x * z;
    basis[8] = 0.546274 * ((x * x) - (y * y));

    float3 irradiance = float3(0.0, 0.0, 0.0);
    for (int i = 0; i < 9; ++i)
    {
        irradiance += camera.shIrradiance[i].xyz * basis[i];
    }
    return max(irradiance, float3(0.0, 0.0, 0.0));
}

public float3 evaluateShHemisphereIrradiance(float3 normal)
{
    const float upT = clamp((normal.y * 0.5) + 0.5, 0.0, 1.0);
    const float3 skyIrradiance = evaluateShIrradiance(float3(0.0, 1.0, 0.0));
    const float3 groundIrradiance = evaluateShIrradiance(float3(0.0, -1.0, 0.0));
    return lerp(groundIrradiance, skyIrradiance, upT);
}
