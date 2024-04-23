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

#pragma once

#ifndef SYSTEM_DATA_H
#define SYSTEM_DATA_H

#include "config.h"

#include "camera_definition.h"
#include "light_definition.h"
#include "material_definition_mdl.h"
#include "shader_configuration.h"
#include "vertex_attributes.h"


 // Structure storing the per instance data for all instances inside the geometryInstanceData buffer below. Indexed via optixGetInstanceId().
struct GeometryInstanceData
{
	// 16 byte alignment
	// Pack the different IDs into a single int4 to load them vectorized.
	int4 ids; // .x = idMaterial, .y = idLight, .z = idObject, .w = pad
	// 8 byte alignment
	// Using CUdeviceptr here to be able to handle different attribute and index formats.
	CUdeviceptr attributes;
	CUdeviceptr indices;
};

namespace nrc
{
	constexpr int NUM_BATCHES = 4;
	constexpr int NUM_TRAINING_RECORDS_PER_FRAME = 65536;

	constexpr int TRAIN_RECORD_INDEX_NONE = -1; // Indicate primary ray
	constexpr int TRAIN_RECORD_INDEX_BUFFER_FULL = -2; // All secondary rays if buffer is full

	// Keep track of the ray path for radiance prop
	struct TrainingRecord
	{
		// 16 byte alignment

		// 8 byte alignment

		// 4 byte alignment
		int propTo/* = TRAIN_RECORD_INDEX_NONE*/; // Link to next training record in the direction of radiance prop.
		
		// Used to modulate radiance prop'd from *previous* record.
		float3 localThroughput; 

		// A radiance prop looks like this: (if propTo >= 0)
		// propFrom = index of this TrainingRecord
		// const auto &nextRec = trainingRecords[propTo];
		// trainingRadianceTargets[propTo] += nextRec.localThroughput * trainingRadianceTargets[propFrom];
	};

	struct RadianceQuery
	{
		// 16 byte alignment

		// 8 byte alignment
		float2 direction;
		float2 normal;

		// 4 byte alignment
		float3 position;
		float3 diffuse;
		float3 specular;
		float3 roughness;
	};

	struct ControlBlock
	{
		// 16 byte alignment

		// 8 byte alignment
		
		// Static. All point to fixed-length arrays of 65536 training records.
		TrainingRecord *trainingRecords = nullptr; // numTrainingRecords -> 65536
		float3 *trainingRadianceTargets = nullptr; // numTrainingRecords -> 65536
		// The results of those queries will be used to train the NRC.
		RadianceQuery *radianceQueriesTraining = nullptr; // numTrainingRecords -> 65536

		// TODO: Allocate & Resize!
		// Points to a dynamic array of (#pixels + #tiles) randiance queries. Note the #tiles is dynamic each frame.
		// Capacity is (#pixels + #2x2-tiles ~= 1.25*#pixels). Re-allocate when the render resolution changes.
		//
		// The first #pixels queries are at the end of short rendering paths, in the flattened order of pixels.
		// -- Results (potentially) used for rendering (Remember to modulate with `lastRenderingThroughput`)
		// -- For rays that are terminated early by missing into envmap, the RadianceQuery contains garbage inputs. For convenience
		//    we still query but the results won't get accumulated into the pixel buffer, since `lastRenderingThroughput` should be 0.
		// 
		// The following #tiles queries are at the end of training suffixes, in the flattened order of tiles.
		// -- Results (potentially) used for initiating radiance propagation in self-training.
		// -- For unbiased training rays, the RadianceQuery contains garbage inputs. For convenience we still query
		//    but the results won't be used to initiate radiance propagation, as indicated by the corresponding TrainTerminalVertex.
		RadianceQuery *radianceQueriesInference = nullptr;

		// TODO: Allocate & Resize!
		float3 *lastRenderThroughput = nullptr; // #pixels

		// 4 byte alignment
		int numTrainingRecords = 0;   // Number of training records generated. Upated per-frame
		
		//int maxNumTrainingRecords = NUM_TRAINING_RECORDS_PER_FRAME;
	};
}

// Data updated per frame
struct SystemDataPerFrame
{
	// 16 byte alignment

	// 8 byte alignment
	int2 tileSize = { 8, 8 };    // Example: make_int2(8, 4) for 8x4 tiles. Must be a power of two to make the division a right-shift.
	//int2 tileShift;   // Example: make_int2(3, 2) for the integer division by tile size. That actually makes the tileSize redundant. 

	// 4 byte alignment
	int iterationIndex;
	int totalSubframeIndex;  // Added: total number of subframes, counting all iterations
	int tileTrainingIndex;   // The local index of training ray within each tile. Randomly sampled from [0..tileSize) every subframe
};

struct SystemData
{
	// 16 byte alignment
	//int4 rect; // Unused, not implementing a tile renderer.

	// 8 byte alignment
	OptixTraversableHandle topObject;

	nrc::ControlBlock* nrcCB; // Single NRC control block

	// The accumulated linear color space output buffer.
	// This is always sized to the resolution, not always matching the launch dimension.
	// Using a CUdeviceptr here to allow for different buffer formats without too many casts.
	CUdeviceptr outputBuffer;
	// These buffers are used differently among the rendering strategies.
	// See: Device::compositor. Not used for this NRC demo
	CUdeviceptr tileBuffer;
	CUdeviceptr texelBuffer;

	GeometryInstanceData* geometryInstanceData; // Attributes, indices, idMaterial, idLight, idObject per instance.

	CameraDefinition* cameraDefinitions; // Currently only one camera in the array. (Allows camera motion blur in the future.)
	LightDefinition* lightDefinitions;

	MaterialDefinitionMDL* materialDefinitionsMDL;  // The MDL material parameter argument block, texture handler and index into the shader.
	DeviceShaderConfiguration* shaderConfigurations;    // Indexed by MaterialDefinitionMDL::indexShader.

	int2 resolution;  // The actual rendering resolution. Independent from the launch dimensions for some rendering strategies.
	int2 pathLengths; // .x = min path length before Russian Roulette kicks in, .y = maximum path length

	// 4 byte alignment 
	int deviceCount;   // Number of devices doing the rendering.
	int deviceIndex;   // Device index to be able to distinguish the individual devices in a multi-GPU environment.
	int samplesSqrt;
	int walkLength;   // Volume scattering random walk steps until the maximum distance is used to potentially exit the volume (could be TIR).

	float sceneEpsilon;
	float clockScale; // Only used with USE_TIME_VIEW.

	int typeLens;     // Camera type.

	int numCameras;     // Number of elements in cameraDefinitions.
	int numLights;      // Number of elements in lightDefinitions.
	int numMaterials;   // Number of elements in materialDefinitionsMDL. (Actually not used in device code.)
	int numBitsShaders; // The number of bits needed to represent the number of elements in shaderConfigurations. Used as coherence hint in SER.

	int directLighting;
	
	// Padding to 16-byte alignment
	//int pad0;
	//int pad1;

	SystemDataPerFrame pf;
};


// Helper structure to optimize the lens shader direct callable arguments.
// Return this primary ray structure instead of using references to local memory.
struct LensRay
{
	float3 org;
	float3 dir;
};
#endif // SYSTEM_DATA_H
