
#include <hls_vector.h>
#include <hls_stream.h>
#include <math.h>
#include "assert.h"

#define N 128



void Dot(float a[N], float hold[N]){
#pragma HLS interface m_axi port=a bundle=gmem2 num_write_outstanding=128
#pragma HLS interface m_axi port=hold bundle=gmem num_write_outstanding=128
		float a_hold[N]{};
#pragma HLS array_partition  variable=hold type=block factor=16
#pragma HLS array_partition  variable=a_hold type=block factor=16
#pragma HLS array_partition  variable=a type=block factor=16
	copy_a: for(int i = 0 ; i<N; i++){
#pragma HLS unroll
		a_hold[i]=a[i];
	}
	dot: for(int i =0 ; i < N ; i++ ){
#pragma HLS unroll
		hold[i]=a_hold[i]*a_hold[i];
	}
}

void Reduce(float hold[N], int i){
#pragma HLS interface m_axi port=hold bundle=gmem1 num_write_outstanding=128
	reduce_1: for(int j = 0; j<i; j++){
#pragma HLS pipeline
				hold[j]+=hold[j+i];
	}
}


void L2(float a[N], float out[1]){
#pragma HLS interface m_axi port=a offset=slave bundle=gmem2 num_write_outstanding=128
#pragma HLS interface ap_ctrl_none port=out
	float hold_val[N]{};
#pragma HLS pipeline
	Dot(a,hold_val);
	for(int i=N/2;i>0;i>>=1){
		Reduce(hold_val,i);
	}
	*out=hold_val[0];
}
