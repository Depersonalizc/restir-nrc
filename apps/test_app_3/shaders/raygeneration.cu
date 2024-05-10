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
#include "transform.h"

typedef mi::neuraylib::Shading_state_material Mdl_state;

extern "C" __constant__ SystemData sysData;

__forceinline__ __device__ int2 pixel_from_world_coord(const float2 screen, const LensRay ray, float3 world_coord)
{
    const CameraDefinition camera = sysData.cameraDefinitions[0];

    float A[9] = {
        camera.U.x, camera.U.y, camera.U.z,
        camera.V.x, camera.V.y, camera.V.z,
        camera.W.x, camera.W.y, camera.W.z,
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


__forceinline__ __device__ float3 safe_div(const float3& a, const float3& b)
{
  const float x = (b.x != 0.0f) ? a.x / b.x : 0.0f;
  const float y = (b.y != 0.0f) ? a.y / b.y : 0.0f;
  const float z = (b.z != 0.0f) ? a.z / b.z : 0.0f;

  return make_float3(x, y, z);
}

__forceinline__ __device__ float sampleDensity(const float3& albedo,
                                               const float3& throughput,
                                               const float3& sigma_t,
                                               const float   u,
                                               float3&       pdf)
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
__forceinline__ __device__ void sampleVolumeScattering(const float2 xi, const float g, float3& dir)
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


__forceinline__ __device__ float3 integrator(PerRayData& prd, int index)
{
  // The integrator starts with black radiance and full path throughput.
  prd.radiance   = make_float3(0.0f);
  prd.pdf        = 0.0f;
  prd.throughput = make_float3(1.0f);
  prd.flags      = 0;
  prd.sigma_t    = make_float3(0.0f); // Extinction coefficient: sigma_a + sigma_s.
  prd.walk       = 0;                 // Number of random walk steps taken through volume scattering. 
  prd.eventType  = mi::neuraylib::BSDF_EVENT_ABSORB; // Initialize for exit. (Otherwise miss programs do not work.)
  prd.shadow_ray = false;
  prd.prev_primitive_idx = 0;
  // Nested material handling. 
  prd.idxStack   = 0;
  // Small stack of four entries of which the first is vacuum.
  prd.stack[0].ior     = make_float3(1.0f); // No effective IOR.
  prd.stack[0].sigma_a = make_float3(0.0f); // No volume absorption.
  prd.stack[0].sigma_s = make_float3(0.0f); // No volume scattering.
  prd.stack[0].bias    = 0.0f;              // Isotropic volume scattering.

  // Put payload pointer into two unsigned integers. Actually const, but that's not what optixTrace() expects.
  uint2 payload = splitPointer(&prd);

  // Russian Roulette path termination after a specified number of bounces needs the current depth.
  int depth = 0; // Path segment index. Primary ray is depth == 0.

  while (depth < sysData.pathLengths.y)
  {
      // if (index == 0) {
      //     printf("depth = %d\tsysData.pathLengths = %d, %d\tSPP = %d\n",
      //            depth, sysData.pathLengths.x, sysData.pathLengths.y, sysData.spp);
      // }

    // Self-intersection avoidance:
    // Offset the ray t_min value by sysData.sceneEpsilon when a geometric primitive was hit by the previous ray.
    // Primary rays and volume scattering miss events will not offset the ray t_min.
    const float epsilon = (prd.flags & FLAG_HIT) ? sysData.sceneEpsilon : 0.0f;

    prd.wo       = -prd.wi;        // Direction to observer.
    prd.distance = RT_DEFAULT_MAX; // Shoot the next ray with maximum length.
    prd.flags    = 0;

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
          const float2 xi     = rng2(prd.seed);
          
          const float s = sampleDensity(albedo, prd.throughput, prd.sigma_t, xi.x, prd.pdfVolume);

          // Prevent logf(0.0f) by sampling the inverse range (0.0f, 1.0f].
          prd.distance = -logf(1.0f - xi.y) / s;
        }
      }
    }

#if (USE_SHADER_EXECUTION_REORDERING == 0 || OPTIX_VERSION < 80000)
    // Note that the primary rays and volume scattering miss cases do not offset the ray t_min by sysSceneEpsilon.
    optixTrace(sysData.topObject,
               prd.pos, prd.wi, // origin, direction
               epsilon, prd.distance, 0.0f, // tmin, tmax, time
               OptixVisibilityMask(0xFF), OPTIX_RAY_FLAG_NONE, 
               TYPE_RAY_RADIANCE, NUM_RAY_TYPES, TYPE_RAY_RADIANCE,
               payload.x, payload.y);
#else
    // OptiX Shader Execution Reordering (SER) implementation.
    optixTraverse(sysData.topObject,
                  prd.pos, prd.wi, // origin, direction
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
    prd.depth += 1;
  }
  
  return prd.radiance;
}

__forceinline__ __device__ float3 expensive_shadow_ray(Reservoir& rsv) {
    float3 origin  = rsv.nearest_hit;
    float3 dir     = rsv.y.direction;
    float3 last_wi = rsv.last_wi;
    uint32_t primitive_idx = rsv.prev_primitive_idx;

    PerRayData prd;
    prd.pos                = origin;
    prd.wi                 = dir;
    prd.prev_primitive_idx = primitive_idx;
    prd.prev_instance_id  = rsv.prev_instance_id;

    prd.shadow_ray         = true;
    prd.distance           = rsv.y.distance;
    prd.last_barycentrics  = rsv.last_barycentrics;

    prd.last_wi            = last_wi;

    uint2 payload = splitPointer(&prd);

#if (USE_SHADER_EXECUTION_REORDERING == 0 || OPTIX_VERSION < 80000)
    // Note that the primary rays and volume scattering miss cases do not offset the ray t_min by sysSceneEpsilon.
    optixTrace(sysData.topObject,
               prd.pos, prd.wi, // origin, direction
               sysData.sceneEpsilon, prd.distance, 0.0f, // tmin, tmax, time
               OptixVisibilityMask(0xFF), OPTIX_RAY_FLAG_NONE,
               TYPE_RAY_RADIANCE, NUM_RAY_TYPES, TYPE_RAY_RADIANCE,
               payload.x, payload.y);
#else
    // OptiX Shader Execution Reordering (SER) implementation.
    optixTraverse(sysData.topObject,
                  prd.pos, prd.wi, // origin, direction
                  sysData.sceneEpsilon, prd.distance+2, 0.0f, // tmin, tmax, time
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

    return prd.throughput;

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
    const uint2 theLaunchDim   = make_uint2(optixGetLaunchDimensions()); // For multi-GPU tiling this is (resolution + deviceCount - 1) / deviceCount.
    const uint2 theLaunchIndex = make_uint2(optixGetLaunchIndex());
    
    PerRayData prd;
    
    // Initialize the random number generator seed from the linear pixel index and the iteration index.
    prd.seed = tea<4>(theLaunchDim.x * theLaunchIndex.y + theLaunchIndex.x, sysData.iterationIndex); // PERF This template really generates a lot of instructions.
    prd.launchDim = theLaunchDim;
    prd.launchIndex = theLaunchIndex;
    prd.depth = 0;
    
    // Decoupling the pixel coordinates from the screen size will allow for partial rendering algorithms.
    // Resolution is the actual full rendering resolution and for the single GPU strategy, theLaunchDim == resolution.
    const float2 screen = make_float2(sysData.resolution); // == theLaunchDim for rendering strategy RS_SINGLE_GPU.
    const float2 pixel  = make_float2(theLaunchIndex);
    const float2 sample = rng2(prd.seed);
    
    // Lens shaders
    const LensRay ray = optixDirectCall<LensRay, const float2, const float2, const float2>(sysData.typeLens, screen, pixel, sample);
    
    prd.pos = ray.org;
    prd.wi  = ray.dir;
    
    float3 radiance = float3({0.0, 0.0, 0.0});
    
    Reservoir* ris_output_reservoir_buffer = reinterpret_cast<Reservoir*>(sysData.RISOutputReservoirBuffer);
    Reservoir* spatial_output_reservoir_buffer = reinterpret_cast<Reservoir*>(sysData.SpatialOutputReservoirBuffer);
    Reservoir* temp_reservoir_buffer = reinterpret_cast<Reservoir*>(sysData.TempReservoirBuffer);
    
    const unsigned int index = theLaunchIndex.y * theLaunchDim.x + theLaunchIndex.x;
    int lidx_ris = (theLaunchDim.x * theLaunchDim.y * sysData.cur_iter) + index;
    int lidx_prev = (theLaunchDim.x * theLaunchDim.y * (sysData.cur_iter - 1)) + index;

    // don't question this too much
    const PaneFlags& pane_flags = sysData.num_panes == 1 ? sysData.pane_a_flags :
                                  sysData.num_panes == 2 ? (theLaunchIndex.x < theLaunchDim.x * 0.5) ?
                                                            sysData.pane_a_flags : sysData.pane_b_flags :
                                  sysData.num_panes == 3 ? (theLaunchIndex.x < theLaunchDim.x * 0.33) ?
                                                            sysData.pane_a_flags : (theLaunchIndex.x < theLaunchDim.x * 0.67) ?
                                                            sysData.pane_b_flags : sysData.pane_c_flags
                                                            : sysData.pane_c_flags;

    if (pane_flags.do_reference) {
        prd.do_reference           = true;
        prd.num_ris_samples        = 0;
        prd.do_temporal_resampling = false;
        prd.do_spatial_resampling  = false;
    } else {
        prd.do_reference           = false;
        prd.num_ris_samples        = pane_flags.ris_samples;
        prd.do_temporal_resampling = pane_flags.do_temporal_reuse;
        prd.do_spatial_resampling  = pane_flags.do_spatial_reuse;
    }
    
    prd.launch_linear_index = lidx_ris;

    // ########################
    // HANDLE RIS LOGIC
    // ########################
    if(sysData.cur_iter != sysData.spp) {
        if (prd.num_ris_samples > 0) {
            ris_output_reservoir_buffer[lidx_ris] = Reservoir({0, 0, 0, 0});
        }
        radiance = integrator(prd, index);

        // integrator(prd, index);
    }

    if (index == 131328 && sysData.cur_iter == 0) {
        printf("\n\n\n");
    }

    // ########################
    //  HANDLE TEMPORAL LOGIC
    // ########################
    if (prd.do_temporal_resampling && !sysData.first_frame && sysData.cur_iter != sysData.spp) {
        Reservoir &current_pixel_prev_reservoir = temp_reservoir_buffer[lidx_ris]; // get current pixel's previous reservoir
        Reservoir &current_reservoir = ris_output_reservoir_buffer[lidx_ris];        // choose current reservoir
        LightSample& y1 = current_reservoir.y;

        if (index == 131328) {
            printf("\nrunning temporal reuse: sysData.cur_iter = %d\n", sysData.cur_iter);
            printf("Cur pixel prev temp reservoir w_sum= %f\tW = %f\tM = %d\n",
                   current_pixel_prev_reservoir.w_sum, current_pixel_prev_reservoir.W, current_pixel_prev_reservoir.M);
            printf("CUR RIS reservoir initial w_sum= %f\tW = %f\tM = %d\n", current_reservoir.w_sum, current_reservoir.W, current_reservoir.M);
            printf("\tCUR direction = %f,%f,%f\n", y1.direction.x, y1.direction.y, y1.direction.z);
            //printf("\tCUR light pdf = %f\tphat = %f\n", y1.pdf, length(y1.radiance_over_pdf));
            printf("\tCUR length(final_reservoir.throughput_x_bxdf) = %f\n", length(current_reservoir.throughput_x_bxdf));

        }

        // updateReservoir(
        //     &s,
        //     &y1,
        //     length(y1.radiance_over_pdf) * y1.pdf * current_reservoir.W * current_reservoir.M,
        //     &prd.seed
        // );
        int2 current_pixel_prev_coord = pixel_from_world_coord(screen, ray, current_pixel_prev_reservoir.nearest_hit);
        int2 current_pixel_curr_coord = pixel_from_world_coord(screen, ray, current_reservoir.nearest_hit);
        int offset_x = theLaunchIndex.x - current_pixel_prev_coord.x;
        int offset_y = theLaunchIndex.y - current_pixel_prev_coord.y;

        if (index == 131328) {
            printf("current_pixel_prev_coord %d, %d\ncurrent_pixel_curr_coord %d, %d\n",
                   current_pixel_prev_coord.x, current_pixel_prev_coord.y, current_pixel_curr_coord.x, current_pixel_curr_coord.y);
            printf("theLaunchIndex %d, %d\n",
                   theLaunchIndex.x, theLaunchIndex.y);
            printf("offset = %d %d\n", offset_x, offset_y);
        }

        int prev_coord_x = theLaunchIndex.x + offset_x;
        int prev_coord_y = theLaunchIndex.y + offset_y;

        bool prev_coord_offscreen = false;
        if (prev_coord_x < 0 || prev_coord_y < 0)
            prev_coord_offscreen = true;
        else if (prev_coord_x >= theLaunchDim.x || prev_coord_y >= theLaunchDim.y)
            prev_coord_offscreen = true;

        bool prev_coord_did_hit = true;
        if (
            current_reservoir.nearest_hit.x == 0.f &&
            current_reservoir.nearest_hit.y == 0.f &&
            current_reservoir.nearest_hit.z == 0.f
        ){
            prev_coord_did_hit = false;
        }

        bool prev_too_far = sqrt((double)(offset_x * offset_x + offset_y * offset_y)) > 10.0;

        if (!prev_coord_offscreen && prev_coord_did_hit && !prev_too_far) {
            // select previous frame's reservoir and combine it
            // and only combine if you actually hit something (empty reservoir bad!)
            int prev_index =
                theLaunchDim.x * theLaunchDim.y * (sysData.cur_iter) +
                prev_coord_y * theLaunchDim.x + prev_coord_x; // TODO: how to calculate motion vector??

            Reservoir& prev_frame_reservoir = temp_reservoir_buffer[prev_index];
            LightSample& y2 = prev_frame_reservoir.y;


            if (index == 131328) {
                printf("Prev frame reservoir initial w_sum= %f\tW = %f\tM = %d\n",
                       prev_frame_reservoir.w_sum, prev_frame_reservoir.W, prev_frame_reservoir.M);
                printf("PREV FRAME: idx %d reservoir w_sum = %f\tW = %f\tM = %d\n", prev_index, prev_frame_reservoir.w_sum, prev_frame_reservoir.W, prev_frame_reservoir.M);
                printf("\tPREV direction = %f,%f,%f\n", y2.direction.x, y2.direction.y, y2.direction.z);
                //printf("\tPREV light pdf = %f\tphat = %f\n", y2.pdf, length(y2.radiance_over_pdf));
                printf("\tPREV length(final_reservoir.throughput_x_bxdf) = %f\n", length(prev_frame_reservoir.throughput_x_bxdf));
            }

            // if (prev_frame_reservoir.M >= current_reservoir.M){
            //     prev_frame_reservoir.M = current_reservoir.M;
            // }

            float prv_phat = length(y2.radiance_over_pdf) * y2.pdf;
            float cur_phat = length(y1.radiance_over_pdf) * y1.pdf;

            float dist_between_hits = length(prev_frame_reservoir.nearest_hit - current_reservoir.nearest_hit);
            float wght_due_to_dist  = 1.f / (dist_between_hits + 1.f);

            float m_prev = balanceHeuristic(prev_frame_reservoir.M * prv_phat, current_reservoir.M * cur_phat) * wght_due_to_dist;
            if (prev_frame_reservoir.W > 0) {
                updateReservoir(
                    &current_reservoir,
                    &y2,
                    length(y2.radiance_over_pdf) * y2.pdf * prev_frame_reservoir.W * m_prev,
                    &prd.seed
                );
            }

            if (index == 131328) {
                printf("Temporal reuse after combination w_sum= %f\tW = %f\tM = %d\n\n",
                       current_reservoir.w_sum, current_reservoir.W, current_reservoir.M);
            }
            current_reservoir.M = min(current_reservoir.M + prev_frame_reservoir.M, 40);
            current_reservoir.W =
                (1.0f / (length(y1.radiance_over_pdf) * y1.pdf)) * // 1 / p_hat
                current_reservoir.w_sum;

            if (isnan(current_reservoir.W) || current_reservoir.M == 0.f) clear_reservoir(current_reservoir);

            if (index == 131328) {
                printf("Temporal reuse end result w_sum= %f\tW = %f\tM = %d\n\n",
                       current_reservoir.w_sum, current_reservoir.W, current_reservoir.M);
            }
        }
    }


    // ########################
    // HANDLE SPATIAL LOGIC
    // ########################
    if (prd.do_spatial_resampling && sysData.cur_iter != sysData.spp) {
        if (sysData.cur_iter == 0) {
            // No spatial reuse, simply pass the current samples forward
            spatial_output_reservoir_buffer[lidx_ris] = ris_output_reservoir_buffer[lidx_ris];;
        } else {
            if (index == 131328) {
                printf("running spatial reuse: %d\t sysData.cur_iter = %d\n", prd.do_spatial_resampling,  sysData.cur_iter);
            }
            Reservoir& updated_reservoir = ris_output_reservoir_buffer[lidx_ris];
            if (index == 131328) {
                printf("spatial reservoir TEST INTIIAL VALUE w_sum = %f\tW = %f\tM = %d\n", updated_reservoir.w_sum, updated_reservoir.W, updated_reservoir.M);
                printf("\tlight sample direction = %f,%f,%f\n", updated_reservoir.y.direction.x, updated_reservoir.y.direction.y,updated_reservoir.y.direction.z);

            }

            int k = 5;
            int radius = 30;
            int num_k_sampled = 0;
            int total_M = updated_reservoir.M;

            while(num_k_sampled < k){
                float2 sample = (rng2(prd.seed) - 0.5f) * radius * 2.0f;
                float squared_dist = sample.x * sample.x + sample.y * sample.y;
                if(squared_dist > radius * radius) continue;

                int _x = (int)sample.x + theLaunchIndex.x;
                int _y = (int)sample.y + theLaunchIndex.y;
                if(_x < 0 || _x >= theLaunchDim.x) continue;
                if(_y < 0 || _y >= theLaunchDim.y) continue;
                if(_x == theLaunchIndex.x && _y == theLaunchIndex.y) continue;

                num_k_sampled += 1;

                unsigned int neighbor_index =
                    theLaunchDim.x * theLaunchDim.y * (sysData.cur_iter - 1) +
                    _y * theLaunchDim.x + _x;
                Reservoir* neighbor_reservoir = &spatial_output_reservoir_buffer[neighbor_index];
                LightSample* y = &neighbor_reservoir->y;

                if (index == 131328) {
                    printf("NEIGHBOR: %d reservoir w_sum = %f\tW = %f\tM = %d\n", neighbor_index, neighbor_reservoir->w_sum, neighbor_reservoir->W, neighbor_reservoir->M);
                    printf("\tneighbor direction = %f,%f,%f\n", y->direction.x, y->direction.y, y->direction.z);
                    printf("\tneighbor light pdf = %f\tphat = %f\n", y->pdf, length(y->radiance_over_pdf));

                }

                float nbr_phat = length(y->radiance_over_pdf) * y->pdf;
                float cur_phat = length(updated_reservoir.y.radiance_over_pdf) * updated_reservoir.y.pdf;

                float dist_between_hits = length(neighbor_reservoir->nearest_hit - updated_reservoir.nearest_hit);
                float wght_due_to_dist  = 1.f / (dist_between_hits + 1.f);

                float m_neighbor = balanceHeuristic(neighbor_reservoir->M * nbr_phat, updated_reservoir.M * cur_phat) * wght_due_to_dist;


                if (neighbor_reservoir->W > 0) {
                    updateReservoir(
                        &updated_reservoir,
                        y,
                        length(y->radiance_over_pdf) * y->pdf * m_neighbor * neighbor_reservoir->W,
                        &prd.seed
                        );
                    total_M += neighbor_reservoir->M;
                }
            }

            LightSample& y = updated_reservoir.y;
            updated_reservoir.M = min(total_M, 40);
            updated_reservoir.W =
                (1.0f / (length(y.radiance_over_pdf) * y.pdf)) * // 1 / p_hat
                updated_reservoir.w_sum;

            if (isnan(updated_reservoir.W) || updated_reservoir.M == 0.f) clear_reservoir(updated_reservoir);

            // Keep a copy of the updated reservoir for the next-frame's temporal reuse
            spatial_output_reservoir_buffer[lidx_ris] = updated_reservoir;
            //radiance = y.f_actual * updated_reservoir.W;
        }
    }


    ////////////////////////////////
    // shoot direct lighting ray
    ////////////////////////////////
    if (prd.num_ris_samples > 0 && sysData.cur_iter != sysData.spp) {
        Reservoir& final_reservoir = ris_output_reservoir_buffer[lidx_ris];
        if (index == 131328) {
            printf("FINAL RESERVOIR w_sum = %f\tW = %f\tM = %d\n", final_reservoir.w_sum, final_reservoir.W, final_reservoir.M);
            printf("\t length(final_reservoir.throughput_x_bxdf) = %f\n", length(final_reservoir.throughput_x_bxdf));

        }
        if (final_reservoir.M > 0 && length(final_reservoir.throughput_x_bxdf) > 0) {
            LightSample& lightSample = final_reservoir.y;
            // Pass the current payload registers through to the shadow ray.
            uint2 payload = splitPointer(&prd);

            prd.flags &= ~FLAG_SHADOW;                  // Clear the shadow flag.

            int tidx = prd.launchIndex.y * prd.launchDim.x + prd.launchIndex.x;
            if (tidx == 131328) {
                printf("about to shoot shadow ray: thePrd.pos = %f,%f,%f, lightSample.direction = %f,%f,%f\n",
                       prd.pos.x,prd.pos.y,prd.pos.z, lightSample.direction.x, lightSample.direction.y, lightSample.direction.z);
                printf("prev_info_valid %d\n", final_reservoir.prev_info_valid);
            }

            // Note that the sysData.sceneEpsilon is applied on both sides of the shadow ray [t_min, t_max] interval
            // to prevent self-intersections with the actual light geometry in the scene.
            // if (final_reservoir.prev_info_valid) {
            //     if (tidx == 131328) {
            //         printf("about to shoot expensive ray. length(final_reservoir.throughput_x_bxdf) = %f\n", length(final_reservoir.throughput_x_bxdf));
            //     }
            //     final_reservoir.throughput_x_bxdf = expensive_shadow_ray(final_reservoir);
            //     if (tidx == 131328) {
            //         printf("resulting length(final_reservoir.throughput_x_bxdf) = %f\n", length(final_reservoir.throughput_x_bxdf));
            //     }
            // } else {
                optixTrace(sysData.topObject,
                           final_reservoir.nearest_hit, lightSample.direction, // origin, direction
                           sysData.sceneEpsilon, lightSample.distance - sysData.sceneEpsilon, 0.0f, // tmin, tmax, time
                           OptixVisibilityMask(0xFF), OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT, // The shadow ray type only uses anyhit programs.
                           TYPE_RAY_SHADOW, NUM_RAY_TYPES, TYPE_RAY_SHADOW,
                           payload.x, payload.y); // Pass through thePrd to the shadow ray.
            //}

            if ((prd.flags & FLAG_SHADOW) == 0) // Visibility test succeeded
            {
                float W = final_reservoir.W;
                float3 f_q =
                    lightSample.pdf * lightSample.radiance_over_pdf *
                    final_reservoir.throughput_x_bxdf * sysData.numLights;

                int tidx = prd.launchIndex.y * prd.launchDim.x + prd.launchIndex.x;
                if (tidx == 131328) {
                    printf("Point NOT in shadow: reservoir w_sum = %f\tW = %f\tM = %d\n", final_reservoir.w_sum, final_reservoir.W, final_reservoir.M);
                }
                radiance += f_q * W;
            } else {
                int tidx = prd.launchIndex.y * prd.launchDim.x + prd.launchIndex.x;
                if (tidx == 131328) {
                    printf("Zeroing out reservoir due to (prd.flags & FLAG_SHADOW) == 0 being false\n");
                }
                clear_reservoir(final_reservoir);
                spatial_output_reservoir_buffer[lidx_ris] = final_reservoir;
                ris_output_reservoir_buffer[lidx_ris] = final_reservoir;
            }
        } else {
            clear_reservoir(final_reservoir);
            spatial_output_reservoir_buffer[lidx_ris] = final_reservoir;
            ris_output_reservoir_buffer[lidx_ris] = final_reservoir;
        }
    }

    if (prd.num_ris_samples > 0 && sysData.cur_iter != 0) {
        // Forward data to the next frame
        temp_reservoir_buffer[lidx_prev] = ris_output_reservoir_buffer[lidx_prev];
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
        
        float4* buffer = reinterpret_cast<float4*>(sysData.outputBuffer);

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
#else // if !USE_TIME_VIEW
        if (sysData.iterationIndex < sysData.spp) { // FIXME
            if (0 < sysData.iterationIndex)
            {
                const float4 dst = buffer[index]; // RGBA32F

                radiance = lerp(make_float3(dst), radiance, 1.0f / float(sysData.iterationIndex + 1)); // Only accumulate the radiance, alpha stays 1.0f.
            }
            buffer[index] = make_float4(radiance, 1.0f);
        }
#endif // USE_TIME_VIEW

#else // if !USE_FP32_OUPUT
        
        Half4* buffer = reinterpret_cast<Half4*>(sysData.outputBuffer);

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
            alpha      = lerp(__half2float(dst.z), alpha,      t);
        }
        buffer[index] = make_Half4(radiance, alpha);
#else // if !USE_TIME_VIEW
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

