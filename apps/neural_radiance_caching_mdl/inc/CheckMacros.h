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

#ifndef CHECK_MACROS_H
#define CHECK_MACROS_H

#include <iostream>
#include <sstream>

#include "inc/MyAssert.h"

#define CU_CHECK(call) \
{ \
  const CUresult result = call; \
  if (result != CUDA_SUCCESS) \
  { \
    const char* name; \
    cuGetErrorName(result, &name); \
    const char *error; \
    cuGetErrorString(result, &error); \
    std::ostringstream message; \
    message << "ERROR: " << __FILE__ << "(" << __LINE__ << "): " << #call << " (" << result << ") " << name << ": " << error; \
    MY_ASSERT(!"CU_CHECK"); \
    throw std::runtime_error(message.str()); \
  } \
}

#define CU_CHECK_NO_THROW(call) \
{ \
  const CUresult result = call; \
  if (result != CUDA_SUCCESS) \
  { \
    const char* name; \
    cuGetErrorName(result, &name); \
    const char *error; \
    cuGetErrorString(result, &error); \
    std::cerr << "ERROR: " << __FILE__ << "(" << __LINE__ << "): " << #call << " (" << result << ") " << name << ": " << error << '\n'; \
    MY_ASSERT(!"CU_CHECK_NO_THROW"); \
  } \
}

#define OPTIX_CHECK(call) \
{ \
  const OptixResult result = call; \
  if (result != OPTIX_SUCCESS) \
  { \
    std::ostringstream message; \
    message << "ERROR: " << __FILE__ << "(" << __LINE__ << "): " << #call << " (" << result << ")"; \
    MY_ASSERT(!"OPTIX_CHECK"); \
    throw std::runtime_error(message.str()); \
  } \
}

#define OPTIX_CHECK_NO_THROW(call) \
{ \
  const OptixResult result = call; \
  if (result != OPTIX_SUCCESS) \
  { \
    std::cerr << "ERROR: " << __FILE__ << "(" << __LINE__ << "): " << #call << " (" << result << ")\n"; \
    MY_ASSERT(!"OPTIX_CHECK_NO_THROW"); \
  } \
}

#define CURAND_CHECK(call) \
{ \
  const curandStatus_t status = call; \
  if (status != CURAND_STATUS_SUCCESS) \
  { \
    std::ostringstream message; \
    message << "ERROR: " << __FILE__ << "(" << __LINE__ << "): " << #call << " (" << status << ")"; \
    MY_ASSERT(!"CURAND_CHECK"); \
    throw std::runtime_error(message.str()); \
  } \
}

#define CURAND_CHECK_NOTHROW(call) \
{ \
  const curandStatus_t status = call; \
  if (status != CURAND_STATUS_SUCCESS) \
  { \
    std::cerr << "ERROR: " << __FILE__ << "(" << __LINE__ << "): " << #call << " (" << status << ")\n"; \
    MY_ASSERT(!"CURAND_CHECK"); \
  } \
}

#define CUDA_CHECK(call) \
{ \
  const cudaError_t error = call; \
  if (error != cudaSuccess) \
  { \
    const char* error_name; \
    error_name = cudaGetErrorName(error); \
    const char *error_str; \
    error_str = cudaGetErrorString(error); \
    std::ostringstream message; \
    message << "ERROR: " << __FILE__ << "(" << __LINE__ << "): " << #call << " (" << error << ") " << error_name << ": " << error_str; \
    MY_ASSERT(!"CUDA_CHECK"); \
    throw std::runtime_error(message.str()); \
  } \
}

#define CUDA_CHECK_NOTHROW(call) \
{ \
  const cudaError_t error = call; \
  if (error != cudaSuccess) \
  { \
    const char* error_name; \
    error_name = cudaGetErrorName(error); \
    const char *error_str; \
    error_str = cudaGetErrorString(error); \
    std::ostringstream message; \
    message << "ERROR: " << __FILE__ << "(" << __LINE__ << "): " << #call << " (" << error << ") " << error_name << ": " << error_str; \
    MY_ASSERT(!"CUDA_CHECK"); \
  } \
}


#endif // CHECK_MACROS_H