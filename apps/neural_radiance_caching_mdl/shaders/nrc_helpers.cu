#include "config.h"

#include <optix.h>

#include "system_data.h"
#include "neural_radiance_caching.h"
#include "vector_math.h"


// NOTE: This is a separate copy of sys data from the one managed by Optix.
extern "C" __constant__ SystemData sysData;

extern "C" __global__ void placeholder() { return; }

namespace {

__forceinline__ __device__ uint2 getLaunchIndex()
{
	return { blockDim.x * blockIdx.x + threadIdx.x, 
			 blockDim.y * blockIdx.y + threadIdx.y };
}


}

// Radiance accumulator kernel to add the radiance * throughput 
// at the end of the rendering paths to the output buffer.
extern "C" __global__ void accumulate_render_radiance(float3 *endRenderRadiance, float3 *endRenderThroughput)
{
	const auto launchIndex = getLaunchIndex();
	if (launchIndex.x >= sysData.resolution.x || launchIndex.y >= sysData.resolution.y) return;
	
	const unsigned int index = launchIndex.y * sysData.resolution.x + launchIndex.x; // Pixel index
	const auto buffer = reinterpret_cast<float4*>(sysData.outputBuffer);
	
	// Make sure endRenderRadiance doesn't contain garbage here.
	//const float3 radiance = endRenderThroughput[index] * endRenderRadiance[index];
	const float3 radiance = endRenderThroughput[index] * make_float3(1.0f);
	//const float3 radiance = endRenderRadiance[index];
	const float accWeight = 1.0f / float(sysData.pf.iterationIndex + 1);
	
	float3 dst = make_float3(buffer[index]);
	dst += (radiance * accWeight);	
	buffer[index] = make_float4(dst, 1.0f);
}

extern "C" __global__ void propagate_train_radiance(nrc::TrainingSuffixEndVertex *trainSuffixEndVertices, // [:#tiles]
													float3 *endTrainRadiance, // [:#tiles]
													nrc::TrainingRecord *trainRecords, // [65536]
													float3 *trainRadianceTargets) // [65536]
{
	const auto launchIndex = getLaunchIndex();
	if (launchIndex.x >= sysData.pf.numTiles.x || launchIndex.y >= sysData.pf.numTiles.y) return;

	const unsigned int tileIndex = launchIndex.y * sysData.pf.numTiles.x + launchIndex.x; // Tile index
	
	const auto &endVert = trainSuffixEndVertices[tileIndex];
	float3 lastRadiance = endTrainRadiance[tileIndex] * endVert.radianceMask;

	int iTo = endVert.startTrainRecord;
#if 1
	if (iTo >= sysData.nrcCB->numTrainingRecords || !(endVert.radianceMask == 0.f || endVert.radianceMask == 1.f))
	{
		printf("[Tile %d/%d] Invalid end vertex: startTrainRecord(int) = %d (/%d). radianceMask(float) = %f\n", 
			   tileIndex+1, sysData.pf.numTiles.x * sysData.pf.numTiles.y, iTo, sysData.nrcCB->numTrainingRecords, endVert.radianceMask);
		return;
	}
#endif
	while (iTo >= 0)
	{
		//if (launchIndex.x == sysData.pf.numTiles.x / 2 && launchIndex.y == sysData.pf.numTiles.y / 2)
		//	printf("%d->", iTo);

		auto &recordTo   = trainRecords[iTo];
		auto &radianceTo = trainRadianceTargets[iTo];
		
		radianceTo += recordTo.localThroughput * lastRadiance;
		
		// Go to next record
		lastRadiance = radianceTo;
		iTo = recordTo.propTo;
	}
	//if (launchIndex.x == sysData.pf.numTiles.x / 2 && launchIndex.y == sysData.pf.numTiles.y / 2)
	//	printf("[END]\n");
}
