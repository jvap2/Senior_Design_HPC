#include ".../src/L2.hpp"

const int L=16;
void load(float* in, hls::stream<V> & out, int size);
void L_2(hls::stream<V>,float* out, int size);


void load(float* in, hls::stream<V> & out, int size){
	main: for(i=0;i<size;i++){
		#pragma HLS pipeline II=1
		out.write(in[i]);
	}
}

void L_2(hls::stream<V>,float* out, int size){
	float p_sum[L];
	init: for(int i=0; i<L;i++){
		#pragma HLS unroll
		p_sum[i]=0;
	}
	main: for(int i=0; i<size; i+=L){
		#pragma HLS pipeline II=L
		partial_sum: for (int j=0; j<L;j++){
			#pragma HLS unroll
			V in_temp=in.read();
			compute: for(int k=0; k<V; k++){
				#pragma HLS unroll
				p_sum[j]+=powf(in_temp.read(),2.0f);
			}
		}
	}
	*out=0;
	final_sum: for(int i=0; i<L;i++){
		#pragma HLS unroll
		*out+=p_sum[i];
	}
	*out=sqrtf(*(out));
}
