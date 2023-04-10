
#include <hls_vector.h>
#include <hls_stream.h>
#include "assert.h"

#define N 2048
//void dot_p(float a[N], float b[N], float out[1])
//{
//  float a_int[N],b_int[N];
//#pragma HLS array_partition  variable=a_int type=block factor=16
//#pragma HLS array_partition  variable=b_int type=block factor=16
//  float product = 0;
//
//  for(int i=0;i<N;i++) {
//  #pragma HLS pipeline
//    a_int[i] = a[i];
//  }
//  for(int i=0;i<N;i++) {
//  #pragma HLS pipeline
//    b_int[i] = b[i];
//  }
//
//  for(int i=0;i<N;i++) {
//  #pragma HLS pipeline
//    product += a_int[i] * b_int[i];
//  }
// out[0] = product;
//
//}


//void dot_p(float a[N], float b[N], float out[1]){
//	float hold[N]{};
//	float a_hold[N]{};
//	float b_hold[N]{};
//#pragma HLS array_partition  variable=hold type=block factor=32
//#pragma HLS array_partition  variable=a_hold type=block factor=32
//#pragma HLS array_partition  variable=b_hold type=block factor=32
//	copy_a: for(int i = 0 ; i<N; i++){
//#pragma HLS unroll
//		a_hold[i]=a[i];
//	}
//	copy_b: for(int i = 0 ; i<N; i++){
//#pragma HLS UNROLL
//		b_hold[i]=b[i];
//	}
//	dot: for(int i =0 ; i < N ; i++ ){
//#pragma HLS UNROLL
//		hold[i]=a_hold[i]*b_hold[i];
//	}
//int i=256;
//reduce_1: for(int j = 0; j<256; j++){
//#pragma HLS UNROLL
//			hold[j]+=hold[j+i];
//}
//i>>=1;
//reduce_2: for(int j = 0; j<128; j++){
//#pragma HLS UNROLL
//			hold[j]+=hold[j+i];
//}
//i>>=1;
//reduce_3: for(int j = 0; j<64; j++){
//#pragma HLS UNROLL
//			hold[j]+=hold[j+i];
//}
//i>>=1;
//reduce_4: for(int j = 0; j<32; j++){
//#pragma HLS UNROLL
//			hold[j]+=hold[j+i];
//}
//i>>=1;
//reduce_5: for(int j = 0; j<16; j++){
//#pragma HLS UNROLL
//			hold[j]+=hold[j+i];
//}
//i>>=1;
//reduce_6: for(int j = 0; j<8; j++){
//#pragma HLS UNROLL
//			hold[j]+=hold[j+i];
//}
//i>>=1;
//reduce_7: for(int j = 0; j<4; j++){
//#pragma HLS UNROLL
//			hold[j]+=hold[j+i];
//}
//i>>=1;
//reduce_8: for(int j = 0; j<2; j++){
//#pragma HLS UNROLL
//			hold[j]+=hold[j+i];
//}
//out[0]=hold[0]+hold[1];
//}


void Dot(float a[N], float b[N], float hold[N]){
#pragma HLS interface m_axi port=a bundle=gmem2 num_write_outstanding=128
#pragma HLS interface m_axi port=b bundle=gmem1 num_write_outstanding=128
#pragma HLS interface m_axi port=hold bundle=gmem num_write_outstanding=128
		float a_hold[N]{};
		float b_hold[N]{};
#pragma HLS array_partition  variable=hold type=block factor=16
#pragma HLS array_partition  variable=a_hold type=block factor=16
#pragma HLS array_partition  variable=b_hold type=block factor=16
#pragma HLS array_partition  variable=a type=block factor=16
#pragma HLS array_partition  variable=b type=block factor=16
	copy_a: for(int i = 0 ; i<N; i++){
#pragma HLS unroll
		a_hold[i]=a[i];
	}
	copy_b: for(int i = 0 ; i<N; i++){
#pragma HLS unroll
		b_hold[i]=b[i];
	}
	dot: for(int i =0 ; i < N ; i++ ){
#pragma HLS unroll
		hold[i]=a_hold[i]*b_hold[i];
	}
}

void Reduce(float hold[N], int i){
#pragma HLS interface m_axi port=hold bundle=gmem1 num_write_outstanding=128
	reduce_1: for(int j = 0; j<i; j++){
#pragma HLS pipeline
				hold[j]+=hold[j+i];
	}
}


void dot_p(float a[N], float b[N], float out[1]){
#pragma HLS interface m_axi port=a offset=slave bundle=gmem2 num_write_outstanding=128
#pragma HLS interface m_axi port=b offset=slave bundle=gmem1 num_write_outstanding=128
#pragma HLS interface ap_ctrl_none port=out
	float hold_val[N]{};
#pragma HLS pipeline
	Dot(a,b,hold_val);
	for(int i=N/2;i>0;i>>=1){
		Reduce(hold_val,i);
	}
	*out=hold_val[0];
}
