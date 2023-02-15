#include "../Senior_Design_HPC/FPGA/L_2/L_2/src/L_2.hpp"

const int L=16;
void load(float* in, hls::stream<V> & out, int size);
void L_2(hls::stream<V>,float* out, int size);

extern "C" {
void fp_reduce(V *in, float *out, int size)
{
    #pragma HLS interface m_axi port=in  offset=slave num_read_outstanding=16 max_read_burst_length=16 bundle=gmem1
    #pragma HLS interface m_axi port=out offset=slave bundle=gmem2
    #pragma HLS data_pack variable=in
    #pragma HLS interface s_axilite port=in      bundle=control
    #pragma HLS interface s_axilite port=out     bundle=control
    #pragma HLS interface s_axilite port=size    bundle=control
    #pragma HLS interface s_axilite port=return  bundle=control

    static hls::stream<V> fifo;
    #pragma HLS stream variable=fifo depth=32

    #pragma HLS dataflow
    load(in, fifo, size);
    L_2(fifo, out, size);
}
}

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
