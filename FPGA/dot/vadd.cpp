
#include <hls_vector.h>
#include <hls_stream.h>
#include "assert.h"

#define N 64
void dot_p(float a[N], float b[N], float out[1])
{
  float a_int[N],b_int[N];
#pragma HLS array_partition  variable=a_int type=block factor=8
#pragma HLS array_partition  variable=b_int type=block factor=8
  float product = 0;

  for(int i=0;i<N;i++) {
  #pragma HLS unroll
    a_int[i] = a[i];
  }
  for(int i=0;i<N;i++) {
  #pragma HLS unroll
    b_int[i] = b[i];
  }

  for(int i=0;i<N;i++) {
  #pragma HLS pipeline
    product += a_int[i] * b_int[i];
  }
 out[0] = product;

}
