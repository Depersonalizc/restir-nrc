/*
 * Copyright (c) 2013-2023, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include "config.h"

#include <optix.h>

#include "system_data.h"
#include "per_ray_data.h"
#include "shader_common.h"
#include "half_common.h"
#include "random_number_generators.h"

extern "C" __constant__ SystemData sysData;

__forceinline__ __device__ void matrixMul4x4(float *a, float *b, float *c)
{
    c[0] = a[0] * b[0] + a[4] * b[1] + a[8] * b[2] + a[12] * b[3];
    c[1] = a[1] * b[0] + a[5] * b[1] + a[9] * b[2] + a[13] * b[3];
    c[2] = a[2] * b[0] + a[6] * b[1] + a[10] * b[2] + a[14] * b[3];
    c[3] = a[3] * b[0] + a[7] * b[1] + a[11] * b[2] + a[15] * b[3];

    // col 2
    c[4] = a[0] * b[4] + a[4] * b[5] + a[8] * b[6] + a[12] * b[7];
    c[5] = a[1] * b[4] + a[5] * b[5] + a[9] * b[6] + a[13] * b[7];
    c[6] = a[2] * b[4] + a[6] * b[5] + a[10] * b[6] + a[14] * b[7];
    c[7] = a[3] * b[4] + a[7] * b[5] + a[11] * b[6] + a[15] * b[7];

    // col 3
    c[8] = a[0] * b[8] + a[4] * b[9] + a[8] * b[10] + a[12] * b[11];
    c[9] = a[1] * b[8] + a[5] * b[9] + a[9] * b[10] + a[13] * b[11];
    c[10] = a[2] * b[8] + a[6] * b[9] + a[10] * b[10] + a[14] * b[11];
    c[11] = a[3] * b[8] + a[7] * b[9] + a[11] * b[10] + a[15] * b[11];

    // col 4
    c[12] = a[0] * b[12] + a[4] * b[13] + a[8] * b[14] + a[12] * b[15];
    c[13] = a[1] * b[12] + a[5] * b[13] + a[9] * b[14] + a[13] * b[15];
    c[14] = a[2] * b[12] + a[6] * b[13] + a[10] * b[14] + a[14] * b[15];
    c[15] = a[3] * b[12] + a[7] * b[13] + a[11] * b[14] + a[15] * b[15];
}

__forceinline__ __device__ void matrixMul4x4Transpose(float *a, float *b)
{
    //  0  4  8 12
    //  1  5  9 13
    //  2  6 10 14
    //  3  7 11 15

    b[0] = a[0];
    b[1] = a[4];
    b[2] = a[8];
    b[3] = a[12];

    // col 2
    b[4] = a[1];
    b[5] = a[5];
    b[6] = a[9];
    b[7] = a[13];

    // col 3
    b[8] = a[2];
    b[9] = a[6];
    b[10] = a[10];
    b[11] = a[14];

    // col 4
    b[12] = a[3];
    b[13] = a[7];
    b[14] = a[11];
    b[15] = a[15];
}

__forceinline__ __device__ void matrixVectorMul4x4(float *A, float *b, float *c)
{
    c[0] = A[0] * b[0] + A[4] * b[1] + A[8] * b[2] + A[12] * b[3];
    c[1] = A[1] * b[0] + A[5] * b[1] + A[9] * b[2] + A[13] * b[3];
    c[2] = A[2] * b[0] + A[6] * b[1] + A[10] * b[2] + A[14] * b[3];
    c[3] = A[3] * b[0] + A[7] * b[1] + A[11] * b[2] + A[15] * b[3];
}

__forceinline__ __device__ int2 pixel_from_world_coord(const float2 screen, const LensRay ray, float3 world_coord)
{
    const CameraDefinition camera = sysData.cameraDefinitions[0];

    float A[9] = {
        camera.U.x,
        camera.U.y,
        camera.U.z,
        camera.V.x,
        camera.V.y,
        camera.V.z,
        camera.W.x,
        camera.W.y,
        camera.W.z,
    };
    float3 x = world_coord - camera.P;

    //    0  1  2
    // 0  0  3  6
    // 1  1  4  7
    // 2  2  5  8

    float dt = A[0] * (A[4] * A[8] - A[5] * A[7]) -
               A[3] * (A[1] * A[8] - A[7] * A[2]) +
               A[6] * (A[1] * A[5] - A[4] * A[2]);
    float invdet = 1.f / dt;

    float minv[9];
    minv[0] = (A[4] * A[8] - A[5] * A[7]) * invdet;
    minv[3] = (A[6] * A[5] - A[3] * A[8]) * invdet;
    minv[6] = (A[3] * A[7] - A[6] * A[4]) * invdet;
    minv[1] = (A[7] * A[2] - A[1] * A[8]) * invdet;
    minv[4] = (A[0] * A[8] - A[6] * A[2]) * invdet;
    minv[7] = (A[1] * A[6] - A[0] * A[7]) * invdet;
    minv[2] = (A[1] * A[5] - A[2] * A[4]) * invdet;
    minv[5] = (A[2] * A[3] - A[0] * A[5]) * invdet;
    minv[8] = (A[0] * A[4] - A[1] * A[3]) * invdet;

    float nx = x.x * minv[0] + x.y * minv[3] + x.z * minv[6];
    float ny = x.x * minv[1] + x.y * minv[4] + x.z * minv[7];
    float nz = x.x * minv[2] + x.y * minv[5] + x.z * minv[8];
    float2 ndc;
    ndc.x = nx / nz;
    ndc.y = ny / nz;

    float2 fragment = (ndc + 1.0f) * 0.5f * screen;
    int2 pixel_index;
    pixel_index.x = (int)fragment.x;
    pixel_index.y = (int)fragment.y;

    return pixel_index;
}

__forceinline__ __device__ float3 safe_div(const float3 &a, const float3 &b)
{
    const float x = (b.x != 0.0f) ? a.x / b.x : 0.0f;
    const float y = (b.y != 0.0f) ? a.y / b.y : 0.0f;
    const float z = (b.z != 0.0f) ? a.z / b.z : 0.0f;

    return make_float3(x, y, z);
}

__forceinline__ __device__ float sampleDensity(const float3 &albedo,
                                               const float3 &throughput,
                                               const float3 &sigma_t,
                                               const float u,
                                               float3 &pdf)
{
    const float3 weights = throughput * albedo;

    const float sum = weights.x + weights.y + weights.z;

    pdf = (0.0f < sum) ? weights / sum : make_float3(1.0f / 3.0f);

    if (u < pdf.x)
    {
        return sigma_t.x;
    }
    if (u < pdf.x + pdf.y)
    {
        return sigma_t.y;
    }
    return sigma_t.z;
}

// Determine Henyey-Greenstein phase function cos(theta) of scattering direction
__forceinline__ __device__ float sampleHenyeyGreensteinCos(const float xi, const float g)
{
    // PBRT v3: Chapter 15.2.3
    if (fabsf(g) < 1e-3f) // Isotropic.
    {
        return 1.0f - 2.0f * xi;
    }

    const float s = (1.0f - g * g) / (1.0f - g + 2.0f * g * xi);
    return (1.0f + g * g - s * s) / (2.0f * g);
}

// Determine scatter reflection direction with Henyey-Greenstein phase function.
__forceinline__ __device__ void sampleVolumeScattering(const float2 xi, const float g, float3 &dir)
{
    const float cost = sampleHenyeyGreensteinCos(xi.x, g);

    float sint = 1.0f - cost * cost;
    sint = (0.0f < sint) ? sqrtf(sint) : 0.0f;

    const float phi = 2.0f * M_PIf * xi.y;

    // This vector is oriented in its own local coordinate system:
    const float3 d = make_float3(cosf(phi) * sint, sinf(phi) * sint, cost);

    // Align the vector with the incoming direction.
    const TBN tbn(dir); // Just some ortho-normal basis along dir as z-axis.

    dir = tbn.transformToWorld(d);
}

__forceinline__ __device__ float3 integrator(PerRayData &prd, int index)
{
    // The integrator starts with black radiance and full path throughput.
    prd.radiance = make_float3(0.0f);
    prd.radiance_first_hit = make_float3(0.0f);
    prd.pdf = 0.0f;
    prd.throughput = make_float3(1.0f);
    prd.flags = 0;
    prd.sigma_t = make_float3(0.0f);                  // Extinction coefficient: sigma_a + sigma_s.
    prd.walk = 0;                                     // Number of random walk steps taken through volume scattering.
    prd.eventType = mi::neuraylib::BSDF_EVENT_ABSORB; // Initialize for exit. (Otherwise miss programs do not work.)
    // Nested material handling.
    prd.idxStack = 0;
    // Small stack of four entries of which the first is vacuum.
    prd.stack[0].ior = make_float3(1.0f);     // No effective IOR.
    prd.stack[0].sigma_a = make_float3(0.0f); // No volume absorption.
    prd.stack[0].sigma_s = make_float3(0.0f); // No volume scattering.
    prd.stack[0].bias = 0.0f;                 // Isotropic volume scattering.

    // Put payload pointer into two unsigned integers. Actually const, but that's not what optixTrace() expects.
    uint2 payload = splitPointer(&prd);

    // Russian Roulette path termination after a specified number of bounces needs the current depth.
    int depth = 0; // Path segment index. Primary ray is depth == 0.
    prd.first_hit = true;

    // while (depth < sysData.pathLengths.y)
    while(depth < 1)
    {
        // if (index == 0) {
        //     printf("depth = %d\tsysData.pathLengths = %d, %d\tSPP = %d\n",
        //            depth, sysData.pathLengths.x, sysData.pathLengths.y, sysData.spp);
        // }

        // Self-intersection avoidance:
        // Offset the ray t_min value by sysData.sceneEpsilon when a geometric primitive was hit by the previous ray.
        // Primary rays and volume scattering miss events will not offset the ray t_min.
        const float epsilon = (prd.flags & FLAG_HIT) ? sysData.sceneEpsilon : 0.0f;

        prd.wo = -prd.wi;              // Direction to observer.
        prd.distance = RT_DEFAULT_MAX; // Shoot the next ray with maximum length.
        prd.flags = 0;

        // Special cases for volume scattering!
        if (0 < prd.idxStack) // Inside a volume?
        {
            // Note that this only supports homogeneous volumes so far!
            // No change in sigma_s along the random walk here.
            const float3 sigma_s = prd.stack[prd.idxStack].sigma_s;

            if (isNotNull(sigma_s)) // We're inside a volume and it has volume scattering?
            {
                // Indicate that we're inside a random walk. This changes the behavior of the miss programs.
                prd.flags |= FLAG_VOLUME_SCATTERING;

                // Random walk through scattering volume, sampling the distance.
                // Note that the entry and exit of the volume is done according to the BSDF sampling.
                // Means glass with volume scattering will still do the proper refractions.
                // When the number of random walk steps has been exceeded, the next ray is shot with distance RT_DEFAULT_MAX
                // to hit something. If that results in a transmission the scattering volume is left.
                // If not, this continues until the maximum path length has been exceeded.
                if (prd.walk < sysData.walkLength)
                {
                    const float3 albedo = safe_div(sigma_s, prd.sigma_t);
                    const float2 xi = rng2(prd.seed);

                    const float s = sampleDensity(albedo, prd.throughput, prd.sigma_t, xi.x, prd.pdfVolume);

                    // Prevent logf(0.0f) by sampling the inverse range (0.0f, 1.0f].
                    prd.distance = -logf(1.0f - xi.y) / s;
                }
            }
        }

#if (USE_SHADER_EXECUTION_REORDERING == 0 || OPTIX_VERSION < 80000)
        // Note that the primary rays and volume scattering miss cases do not offset the ray t_min by sysSceneEpsilon.
        optixTrace(sysData.topObject,
                   prd.pos, prd.wi,             // origin, direction
                   epsilon, prd.distance, 0.0f, // tmin, tmax, time
                   OptixVisibilityMask(0xFF), OPTIX_RAY_FLAG_NONE,
                   TYPE_RAY_RADIANCE, NUM_RAY_TYPES, TYPE_RAY_RADIANCE,
                   payload.x, payload.y);
#else
        // OptiX Shader Execution Reordering (SER) implementation.
        optixTraverse(sysData.topObject,
                      prd.pos, prd.wi,             // origin, direction
                      epsilon, prd.distance, 0.0f, // tmin, tmax, time
                      OptixVisibilityMask(0xFF), OPTIX_RAY_FLAG_NONE,
                      TYPE_RAY_RADIANCE, NUM_RAY_TYPES, TYPE_RAY_RADIANCE,
                      payload.x, payload.y);

        unsigned int hint = 0; // miss uses some default value. The record type itself will distinguish this case.
        if (optixHitObjectIsHit())
        {
            const int idMaterial = sysData.geometryInstanceData[optixHitObjectGetInstanceId()].ids.x;
            hint = sysData.materialDefinitionsMDL[idMaterial].indexShader; // Shader configuration only.
        }
        optixReorder(hint, sysData.numBitsShaders);

        optixInvoke(payload.x, payload.y);
#endif

        // Path termination by miss shader or sample() routines.
        if ((prd.eventType == mi::neuraylib::BSDF_EVENT_ABSORB) || isNull(prd.throughput))
        {
            break;
        }

        // Unbiased Russian Roulette path termination.
        if (sysData.pathLengths.x <= depth) // Start termination after a minimum number of bounces.
        {
            const float probability = fmaxf(prd.throughput);

            if (probability < rng(prd.seed)) // Paths with lower probability to continue are terminated earlier.
            {
                break;
            }

            prd.throughput /= probability; // Path isn't terminated. Adjust the throughput so that the average is right again.
        }

        // We're inside a volume and the scatter ray missed.
        if (prd.flags & FLAG_VOLUME_SCATTERING_MISS) // This implies FLAG_VOLUME_SCATTERING.
        {
            // Random walk through scattering volume, sampling the direction according to the phase function.
            sampleVolumeScattering(rng2(prd.seed), prd.stack[prd.idxStack].bias, prd.wi);
        }

        ++depth; // Next path segment.
        prd.first_hit = false;
    }

    return prd.radiance;
}

__forceinline__ __device__ unsigned int distribute(const uint2 launchIndex)
{
    // First calculate block coordinates of this launch index.
    // That is the launch index divided by the tile dimensions. (No operator>>() on vectors?)
    const unsigned int xBlock = launchIndex.x >> sysData.tileShift.x;
    const unsigned int yBlock = launchIndex.y >> sysData.tileShift.y;

    // Each device needs to start at a different column and each row should start with a different device.
    const unsigned int xTile = xBlock * sysData.deviceCount + ((sysData.deviceIndex + yBlock) % sysData.deviceCount);

    // The horizontal pixel coordinate is: tile coordinate * tile width + launch index % tile width.
    return xTile * sysData.tileSize.x + (launchIndex.x & (sysData.tileSize.x - 1)); // tileSize needs to be power-of-two for this modulo operation.
}

extern "C" __global__ void __raygen__path_tracer()
{
#if USE_TIME_VIEW
    clock_t clockBegin = clock();
#endif
    const uint2 theLaunchDim = make_uint2(optixGetLaunchDimensions()); // For multi-GPU tiling this is (resolution + deviceCount - 1) / deviceCount.
    const uint2 theLaunchIndex = make_uint2(optixGetLaunchIndex());

    PerRayData prd;

    // Initialize the random number generator seed from the linear pixel index and the iteration index.
    prd.seed = tea<4>(theLaunchDim.x * theLaunchIndex.y + theLaunchIndex.x, sysData.iterationIndex + sysData.rand_seed); // PERF This template really generates a lot of instructions.
    prd.launchDim = theLaunchDim;
    prd.launchIndex = theLaunchIndex;

    // Decoupling the pixel coordinates from the screen size will allow for partial rendering algorithms.
    // Resolution is the actual full rendering resolution and for the single GPU strategy, theLaunchDim == resolution.
    const float2 screen = make_float2(sysData.resolution); // == theLaunchDim for rendering strategy RS_SINGLE_GPU.
    const float2 pixel = make_float2(theLaunchIndex);
    const float2 sample = rng2(prd.seed);

    // Lens shaders
    const LensRay ray = optixDirectCall<LensRay, const float2, const float2, const float2>(sysData.typeLens, screen, pixel, sample);

    prd.pos = ray.org;
    prd.wi = ray.dir;
    float3 radiance = float3({0.0, 0.0, 0.0});

    Reservoir *ris_output_reservoir_buffer = reinterpret_cast<Reservoir *>(sysData.RISOutputReservoirBuffer);
    Reservoir *spatial_output_reservoir_buffer = reinterpret_cast<Reservoir *>(sysData.SpatialOutputReservoirBuffer);
    Reservoir *temp_reservoir_buffer = reinterpret_cast<Reservoir *>(sysData.TempReservoirBuffer);

    const unsigned int index = theLaunchIndex.y * theLaunchDim.x + theLaunchIndex.x;
    int lidx_ris = (theLaunchDim.x * theLaunchDim.y * sysData.cur_iter) + index;
    int lidx_spatial = (theLaunchDim.x * theLaunchDim.y * (sysData.cur_iter - 1)) + index;

    // don't question this too much
    const PaneFlags &pane_flags = sysData.num_panes == 1 ? sysData.pane_a_flags : sysData.num_panes == 2 ? (theLaunchIndex.x < theLaunchDim.x * 0.5) ? sysData.pane_a_flags : sysData.pane_b_flags
                                                                              : sysData.num_panes == 3   ? (theLaunchIndex.x < theLaunchDim.x * 0.33) ? sysData.pane_a_flags : (theLaunchIndex.x < theLaunchDim.x * 0.67) ? sysData.pane_b_flags
                                                                                                                                                                                                                          : sysData.pane_c_flags
                                                                                                         : sysData.pane_c_flags;

    if (pane_flags.do_reference)
    {
        prd.do_reference = true;
        prd.num_ris_samples = 0;
        prd.do_temporal_resampling = false;
        prd.do_spatial_resampling = false;
    }
    else
    {
        prd.do_reference = false;
        prd.num_ris_samples = pane_flags.ris_samples;
        prd.do_temporal_resampling = pane_flags.do_temporal_reuse;
        prd.do_spatial_resampling = pane_flags.do_spatial_reuse;
    }

    bool do_ris = prd.num_ris_samples > 0;
    // printf("CHECK GUARDS: do_ref = %i, do_ris = %i, do_temporal = %i, do_spatial = %i\n", prd.do_reference, prd.num_ris_samples, prd.do_temporal_resampling, prd.do_spatial_resampling);

    prd.launch_linear_index = lidx_ris;

    // clear out previous frame's temp buffer
    if (prd.do_temporal_resampling)
    {
        temp_reservoir_buffer[index] = Reservoir({0, 0, 0, 0});
    }
    float3 nearest_hit_current = make_float3(0.0);

    // ########################
    // HANDLE RIS LOGIC
    // ########################
    if (sysData.cur_iter != sysData.spp) {
        if (do_ris) {
            ris_output_reservoir_buffer[lidx_ris] = Reservoir({0, 0, 0, 0});
        }
        radiance = integrator(prd, index);
        // printf("AFTER INTEGRATION, radiance = %f, do_ris_resampling = %i \n", length(radiance), do_ris);

        if (do_ris) {
            nearest_hit_current = ris_output_reservoir_buffer[lidx_ris].nearest_hit;

            if (prd.do_spatial_resampling) {
                if (sysData.cur_iter == 0){
                    radiance += prd.radiance_first_hit;
                }
            }
            else {
                radiance += prd.radiance_first_hit;
            }
        }
    }

    // ########################
    //  HANDLE TEMPORAL LOGIC
    // ########################
    if (prd.do_temporal_resampling && !sysData.first_frame && sysData.cur_iter != sysData.spp) {
        Reservoir s = Reservoir({0, 0, 0, 0});

        Reservoir *current_pixel_prev_resevoir = &spatial_output_reservoir_buffer[lidx_ris]; // get current pixel's previous reservoir
        Reservoir *current_reservoir = &temp_reservoir_buffer[index];                        // choose current reservoir
        LightSample *y1 = &current_reservoir->y;

        updateReservoir(
            &s,
            y1,
            length(y1->radiance_over_pdf) * y1->pdf * current_reservoir->W * current_reservoir->M,
            &prd.seed
        );
        int2 current_pixel_prev_coord = pixel_from_world_coord(screen, ray, current_pixel_prev_resevoir->nearest_hit);
        int2 current_pixel_curr_coord = pixel_from_world_coord(screen, ray, current_reservoir->nearest_hit);
        int offset_x = theLaunchIndex.x - current_pixel_prev_coord.x;
        int offset_y = theLaunchIndex.y - current_pixel_prev_coord.y;
        int prev_coord_x = theLaunchIndex.x + offset_x;
        int prev_coord_y = theLaunchIndex.y + offset_y;

        bool prev_coord_offscreen = false;
        if (prev_coord_x < 0 || prev_coord_y < 0)
            prev_coord_offscreen = true;
        else if (prev_coord_x >= theLaunchDim.x || prev_coord_y >= theLaunchDim.y)
            prev_coord_offscreen = true;

        bool prev_coord_no_hit = true;
        if (
            current_reservoir->nearest_hit.x != 0.f &&
            current_reservoir->nearest_hit.y != 0.f &&
            current_reservoir->nearest_hit.z != 0.f
        ){
            prev_coord_no_hit = false;
        }

        bool prev_too_far = sqrt((double)(offset_x * offset_x + offset_y * offset_y)) > 30.f;

        if (!prev_coord_offscreen && !prev_coord_no_hit && !prev_too_far) {
            // select previous frame's reservoir and combine it
            // and only combine if you actually hit something (empty reservoir bad!)
            int prev_index =
                theLaunchDim.x * theLaunchDim.y * (sysData.cur_iter) +
                prev_coord_y * theLaunchDim.x + prev_coord_x; // TODO: how to calculate motion vector??

            Reservoir *prev_frame_reservoir = &spatial_output_reservoir_buffer[prev_index];

            LightSample *y2 = &prev_frame_reservoir->y;
            if (prev_frame_reservoir->M >= current_reservoir->M){
                prev_frame_reservoir->M = current_reservoir->M;
            }

            updateReservoir(
                &s,
                y2,
                length(y2->radiance_over_pdf) * y2->pdf * prev_frame_reservoir->W * prev_frame_reservoir->M,
                &prd.seed);

            s.M = current_reservoir->M + prev_frame_reservoir->M;
            s.W =
                (1.0f / (length(s.y.radiance_over_pdf) * s.y.pdf)) * // 1 / p_hat
                (1.0f / s.M) *
                s.w_sum;
            if (isnan(s.W) || s.M == 0.f) s.W = 0;
        
            s.nearest_hit = current_reservoir->nearest_hit;
            // s.y.throughput = y1->throughput;
            // s.y.bxdf = y1->bxdf;
            // s.y.weightMIS = y1->weightMIS;

            ris_output_reservoir_buffer[lidx_ris] = s;
        } else {
            ris_output_reservoir_buffer[lidx_ris] = *current_reservoir;
        }
    }

    // ########################
    // HANDLE SPATIAL LOGIC
    // ########################
    if (prd.do_spatial_resampling && sysData.cur_iter != 0){
        Reservoir updated_reservoir = ris_output_reservoir_buffer[lidx_spatial];
        float3 nearest_hit_current = updated_reservoir.nearest_hit;
        float3 current_throughput = updated_reservoir.y.throughput;
        float3 current_bxdf = updated_reservoir.y.bxdf;
        float current_weightMIS = updated_reservoir.y.weightMIS;

        if (updated_reservoir.W != 0){
            int k = 5;
            int radius = 30;
            int num_k_sampled = 0;
            int total_M = updated_reservoir.M;

            while (num_k_sampled < k){
                float2 sample = (rng2(prd.seed) - 0.5f) * radius * 2.0f;
                float squared_dist = sample.x * sample.x + sample.y * sample.y;
                if (squared_dist > radius * radius)
                    continue;

                int _x = (int)sample.x + theLaunchIndex.x;
                int _y = (int)sample.y + theLaunchIndex.y;
                if (_x < 0 || _x >= theLaunchDim.x)
                    continue;
                if (_y < 0 || _y >= theLaunchDim.y)
                    continue;
                if (_x == theLaunchIndex.x && _y == theLaunchIndex.y)
                    continue;

                unsigned int neighbor_index =
                    theLaunchDim.x * theLaunchDim.y * (sysData.cur_iter - 1) +
                    _y * theLaunchDim.x + _x;
                Reservoir *neighbor_reservoir = &ris_output_reservoir_buffer[neighbor_index];
                LightSample *y = &neighbor_reservoir->y;

                updateReservoir(
                    &updated_reservoir,
                    y,
                    length(y->radiance_over_pdf) * y->pdf * neighbor_reservoir->W * neighbor_reservoir->M,
                    &prd.seed);
                total_M += neighbor_reservoir->M;

                num_k_sampled += 1;
            }

            LightSample y = updated_reservoir.y;
            updated_reservoir.M = total_M;
            updated_reservoir.W =
                (1.0f / (length(y.radiance_over_pdf) * y.pdf)) * // 1 / p_hat
                (1.0f / updated_reservoir.M) *
                updated_reservoir.w_sum;
            updated_reservoir.nearest_hit = nearest_hit_current;

            updated_reservoir.y.bxdf = current_bxdf;
            updated_reservoir.y.throughput = current_throughput;
            updated_reservoir.y.weightMIS = current_weightMIS;

            spatial_output_reservoir_buffer[lidx_spatial] = updated_reservoir;
            radiance += current_throughput * current_bxdf * 
                y.radiance_over_pdf * y.pdf * // issue with using this pdf...
                updated_reservoir.W * sysData.numLights * current_weightMIS;
            // radiance += prd.radiance_first_hit;
        } else {
            radiance += prd.radiance_first_hit;
        }
    }

#if USE_DEBUG_EXCEPTIONS
    // DEBUG Highlight numerical errors.
    if (isnan(radiance.x) || isnan(radiance.y) || isnan(radiance.z))
    {
        radiance = make_float3(1000000.0f, 0.0f, 0.0f); // super red
    }
    else if (isinf(radiance.x) || isinf(radiance.y) || isinf(radiance.z))
    {
        radiance = make_float3(0.0f, 1000000.0f, 0.0f); // super green
    }
    else if (radiance.x < 0.0f || radiance.y < 0.0f || radiance.z < 0.0f)
    {
        radiance = make_float3(0.0f, 0.0f, 1000000.0f); // super blue
    }
#else
    // NaN values will never go away. Filter them out before they can arrive in the output buffer.
    // This only has an effect if the debug coloring above is off!
    if (!(isnan(radiance.x) || isnan(radiance.y) || isnan(radiance.z)))
#endif
    {

#if USE_FP32_OUTPUT

        float4 *buffer = reinterpret_cast<float4 *>(sysData.outputBuffer);

#if USE_TIME_VIEW
        clock_t clockEnd = clock();
        const float alpha = (clockEnd - clockBegin) * sysData.clockScale;

        float4 result = make_float4(radiance, alpha);

        if (0 < sysData.cur_iter)
        {
            const float4 dst = buffer[index]; // RGBA32F

            result = lerp(dst, result, 1.0f / float(sysData.cur_iter + 1)); // Accumulate the alpha as well.
        }
        buffer[index] = result;
#else  // if !USE_TIME_VIEW
        if (sysData.iterationIndex < sysData.spp)
        { // FIXME
            if (0 < sysData.iterationIndex)
            {
                const float4 dst = buffer[index]; // RGBA32F

                radiance = lerp(make_float3(dst), radiance, 1.0f / float(sysData.iterationIndex + 1)); // Only accumulate the radiance, alpha stays 1.0f.
            }
            buffer[index] = make_float4(radiance, 1.0f);
        }
#endif // USE_TIME_VIEW

#else // if !USE_FP32_OUPUT

        Half4 *buffer = reinterpret_cast<Half4 *>(sysData.outputBuffer);

#if USE_TIME_VIEW
        clock_t clockEnd = clock();
        float alpha = (clockEnd - clockBegin) * sysData.clockScale;

        if (0 < sysData.cur_iter)
        {
            const float t = 1.0f / float(sysData.cur_iter + 1);

            const Half4 dst = buffer[index]; // RGBA16F

            radiance.x = lerp(__half2float(dst.x), radiance.x, t);
            radiance.y = lerp(__half2float(dst.y), radiance.y, t);
            radiance.z = lerp(__half2float(dst.z), radiance.z, t);
            alpha = lerp(__half2float(dst.z), alpha, t);
        }
        buffer[index] = make_Half4(radiance, alpha);
#else  // if !USE_TIME_VIEW
        if (0 < sysData.cur_iter)
        {
            const float t = 1.0f / float(sysData.cur_iter + 1);

            const Half4 dst = buffer[index]; // RGBA16F

            radiance.x = lerp(__half2float(dst.x), radiance.x, t);
            radiance.y = lerp(__half2float(dst.y), radiance.y, t);
            radiance.z = lerp(__half2float(dst.z), radiance.z, t);
        }
        buffer[index] = make_Half4(radiance, 1.0f);
#endif // USE_TIME_VIEW

#endif // USE_FP32_OUTPUT
    }
}
