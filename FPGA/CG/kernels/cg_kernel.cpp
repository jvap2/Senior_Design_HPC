// Includes
#include <hls_vector.h>
#include <hls_stream.h>
#include "assert.h"
#include <iostream>
#include <cmath>


#define N 2048
#define nx 2048
#define ny 2048
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

void vadd_p(float a[N], float b[N], float out[N]){
	float a_int[N],b_int[N];
	#pragma HLS array_partition  variable=a_int type=block factor=8
	#pragma HLS array_partition  variable=b_int type=block factor=8

	  copy_a:for(int i=0;i<N;i++) {
	  #pragma HLS pipeline
	    a_int[i] = a[i];
	  }
	  copy_b: for(int i=0;i<N;i++) {
	  #pragma HLS pipeline
	    b_int[i] = b[i];
	  }

	  final: for(int i=0;i<N;i++) {
	  #pragma HLS unroll factor=4
	    out[i] = a_int[i] + b_int[i];
	  }
}

void MatVec_Mult(float A[N], float b[N], float b_new[N]){
	float A_comp[N]{};
	float b_comp[N]{};
#pragma HLS array partition variable=A_comp type=block factor=16
#pragma HLS array partition variable=b_comp type=block factor=16
	copy_A: for(int i=0; i<N; i++){
#pragma HLS unroll factor=4
		A_comp[i]=A[i];
	}
	copy_b: for(int i=0; i<N; i++){
#pragma HLS unroll factor=4
		b_comp[i]=b[i];
	}
	float fSum;
	row: for (int i = 0; i < ny; i++)
	{
#pragma HLS pipeline II=8
        fSum = 0.0f;
		col: for (int j = 0; j < nx; j++)
		{
			fSum += (A_comp[abs(i-j)] * b_comp[j]);
		}
        b_new[i]=fSum;
	}
}

void Copy(float new_v[N], float old_v[N]){
	float d_old[N];
	trans: for(int i=0; i<N;i++){
#pragma HLS unroll
		d_old[i]=old_v[i];
	}
	cop: for(int i=0; i<N; i++){
#pragma HLS unroll
		new_v[i]=d_old[i];
	}
}

void const_Vect_mult(float* scalar, float vect[N], float out[N]){
	float d_vect[N];
	trans: for(int i=0; i<N;i++){
#pragma HLS pipeline
		d_vect[i]=vect[i];
	}
	mult: for (int i = 0; i< N; i++){
#pragma HLS unroll
		out[i]=(scalar[0])*(d_vect[i]);
	}
}

void compt(float scal_1, float scal_2, float out[1],int flag){
	float d_scal_1{},d_scal_2{};
	d_scal_1=scal_1;
	d_scal_2=scal_2;
	if(flag==1){
		out[0]=d_scal_1/d_scal_2;
	}
	else{
		out[0]=-d_scal_1/d_scal_2;
	}

}
void final(float A[N],
		float r[N],
		float r_old[N],
		float d[N],
		float d_old[N],
		float x[N],
		float x_old[N],
		float beta[1],
		float lambda[1],
		int* iter){
	float Ad[N]{};
	float r_dot{};
	float r_old_dot{};
	float d_Ad_dot{};
	float mult_d_lam[N]{};
	float mult_d_bet[N]{};
	float Lamb_Ad[N]{};
	float neg_lamb[1]{};
	int count = *iter;
#pragma HLS interface m_axi port=A bundle=gmem2 num_write_outstanding=128
	for(int i{}; i<=(count);i++){
		MatVec_Mult(A, d_old, Ad);
		dot_p(r_old,r_old, &r_old_dot);
		dot_p(d_old,Ad,&d_Ad_dot);
		compt(r_old_dot,d_Ad_dot,lambda,1);
		compt(r_old_dot,d_Ad_dot,neg_lamb,0);
		const_Vect_mult(lambda, d_old,mult_d_lam);
		const_Vect_mult(neg_lamb, Ad,Lamb_Ad);
		vadd_p(mult_d_lam,x_old,x);
		vadd_p(Lamb_Ad,r_old,r);
		dot_p(r,r, &r_dot);
		compt(r_dot,r_old_dot,beta,1);
		const_Vect_mult(beta, d_old,mult_d_bet);
		vadd_p(mult_d_bet,r,d);
		Copy(r_old,r);
		Copy(d_old,d);
		Copy(x_old,x);
	}

}
