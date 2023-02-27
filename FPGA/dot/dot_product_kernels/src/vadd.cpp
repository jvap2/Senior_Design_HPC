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

//#define MEMORY_DWIDTH 512
//#define SIZEOF_WORD 4
//#define NUM_WORDS ((MEMORY_DWIDTH) / (8 * SIZEOF_WORD))
//#define uint_words 2
//#define DATA_SIZE 64
//
//// TRIPCOUNT identifier
//const int c_size = DATA_SIZE;
//
//
//static void compute_add(float* in1_stream,
//                        float* in2_stream,
//                        float* out_stream,
//                        int vSize) {
//// The kernel is operating with vector of NUM_WORDS integers. The + operator performs
//// an element-wise add, resulting in NUM_WORDS parallel additions.
//execute:
//    for (int i = 0; i < vSize; i++) {
//#pragma HLS LOOP_TRIPCOUNT min = c_size max = c_size
//        out_stream[i] = in1_stream[i] * in2_stream[i];
//    }
//}
//
//static void store_result(float* out,
//                         float* sum,
//                         int vSize) {
//float partial_sum[16];
//init: for(int i=0; i<16;i++){
//#pragma HLS unroll factor = 16
//	partial_sum[i]=0.0f;
//}
//outer_loop: for(int i=0; i<vSize; i+=16){
//#pragma HLS pipeline II=vSize/16
//	part_sum: for(int j=0; j<16;j++){
//#pragma HLS unroll factor = 4
//		partial_sum[i]+=out[i+j];
//	}
//}
//final_sum: for(int i=0; i<16;i++){
//#pragma HLS unroll factor=4
//	*sum+=partial_sum[i];
//}
//}
//
//extern "C" {
//
///*
//    Vector Addition Kernel
//
//    Arguments:
//        in1  (input)  --> Input vector 1
//        in2  (input)  --> Input vector 2
//        out  (output) --> Output vector
//        size (input)  --> Number of elements in vector
//*/
//
//void vadd(float* in1,
//          float* in2,
//          float* out,
//		  float *res,
//          int size) {
//#pragma HLS INTERFACE m_axi port = in1 bundle = gmem0
//#pragma HLS INTERFACE m_axi port = in2 bundle = gmem1
//#pragma HLS INTERFACE m_axi port = out bundle = gmem2
//#pragma HLS INTERFACE m_axi port = res bundle = gmem3
//
//
//    // Since NUM_WORDS values are processed
//    // in parallel per loop iteration, the for loop only needs to iterate 'size / NUM_WORDS' times.
////    assert(size % NUM_WORDS == 0);
//    int vSize = size / NUM_WORDS;
//#pragma HLS dataflow
//    compute_add(in1, in2, out, vSize);
//    store_result(out, res, vSize);
//}
//}
#define N 64
void vadd(float* a, float* b, float* out)
{
  float a_int[N],b_int[N];
#pragma HLS array_partition  variable=a_int dim=1 complete
#pragma HLS array_partition  variable=b_int dim=1 complete
  float product = 0;

  for(int i=0;i<N;i++) {
  #pragma HLS pipeline
    a_int[i] = a[i];
  }
  for(int i=0;i<N;i++) {
  #pragma HLS pipeline
    b_int[i] = b[i];
  }

  for(int i=0;i<N;i++) {
  #pragma HLS unroll
    product += a_int[i] * b_int[i];
  }

 *out = product;

}