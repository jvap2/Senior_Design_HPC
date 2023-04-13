/**
* Copyright (C) 2019-2021 Xilinx, Inc
*
* Licensed under the Apache License, Version 2.0 (the "License"). You may
* not use this file except in compliance with the License. A copy of the
* License is located at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
* WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
* License for the specific language governing permissions and limitations
* under the License.
*/

/*******************************************************************************
Description:

    This example uses the load/compute/store coding style which is generally
    the most efficient for implementing kernels using HLS. The load and store
    functions are responsible for moving data in and out of the kernel as
    efficiently as possible. The core functionality is decomposed across one
    of more compute functions. Whenever possible, the compute function should
    pass data through HLS streams and should contain a single set of nested loops.

    HLS stream objects are used to pass data between producer and consumer
    functions. Stream read and write operations have a blocking behavior which
    allows consumers and producers to synchronize with each other automatically.

    The dataflow pragma instructs the compiler to enable task-level pipelining.
    This is required for to load/compute/store functions to execute in a parallel
    and pipelined manner.

    The kernel operates on vectors of NUM_WORDS integers modeled using the hls::vector
    data type. This datatype provides intuitive support for parallelism and
    fits well the vector-add computation. The vector length is set to NUM_WORDS
    since NUM_WORDS integers amount to a total of 64 bytes, which is the maximum size of
    a kernel port. It is a good practice to match the compute bandwidth to the I/O
    bandwidth. Here the kernel loads, computes and stores NUM_WORDS integer values per
    clock cycle and is implemented as below:
                                       _____________
                                      |             |<----- Input Vector 1 from Global Memory
                                      |  load_input |       __
                                      |_____________|----->|  |
                                       _____________       |  | in1_stream
Input Vector 2 from Global Memory --->|             |      |__|
                               __     |  load_input |        |
                              |  |<---|_____________|        |
                   in2_stream |  |     _____________         |
                              |__|--->|             |<--------
                                      | compute_add |      __
                                      |_____________|---->|  |
                                       ______________     |  | out_stream
                                      |              |<---|__|
                                      | store_result |
                                      |______________|-----> Output result to Global Memory

*******************************************************************************/

// Includes
#include <hls_vector.h>
#include <hls_stream.h>
#include "assert.h"

#define MEMORY_DWIDTH 512
#define SIZEOF_WORD 4
#define NUM_WORDS ((MEMORY_DWIDTH) / (8 * SIZEOF_WORD))

#define DATA_SIZE 4096

void RGBtoGrayScale_Ver1(unsigned char* in, unsigned char* out, unsigned int h, unsigned int w, int j)
{
	unsigned int stride = h * w;//we do this so we can avoid doing excessive multiplications
	unsigned int st_case= ((stride/32)+1)*j;
	//when accessing memory
	unsigned int stride_2 = 2 * stride;//This is for blue
	unsigned char* r = in+st_case;//We are passing a pointer pointing to start of data which is red
	unsigned char* g = in + stride+st_case;//We offset the in to get to the start of the green memory location
	unsigned char* b = in + stride_2+st_case;//Offset twice the stride to start at the beginning of the blue memory
	for (int i = 0; i < stride/8 + 1; i++)
	{
#pragma HLS pipeline
		out[i] = 0.21f * (r[i]) + 0.71f * (g[i]) + .07 * (b[i]);

	}
}
