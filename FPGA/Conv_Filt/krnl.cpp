#include <iostream>
#include <cmath>

void Gauss_Filter(unsigned char* in, unsigned char* out, unsigned int h, unsigned int w, int k)
{
	unsigned int st_case= w*(((h-2)/32)+1)*k;
	unsigned char* in_s=in+st_case;
	if(k<31){
		for(int i=0; i<(((h-2)/32)+1);i++){
			for(int j=0;j<(w-2);j++){
	#pragma HLS pipeline
			out[i*(w-2)+j]+=.0625*in_s[i*w+j];
			out[i*(w-2)+j]+=.125*in_s[i*w+1+j];
			out[i*(w-2)+j]+=.0625*in_s[i*w+j+2];
			out[i*(w-2)+j]+=.125*in_s[(i+1)*w+j];
			out[i*(w-2)+j]+=.25*in_s[(i+1)*w+1+j];
			out[i*(w-2)+j]+=.125*in_s[(i+1)*w+2+j];
			out[i*(w-2)+j]+=.0625*in_s[(i+2)*w+j];
			out[i*(w-2)+j]+=.125*in_s[(i+2)*w+1+j];
			out[i*(w-2)+j]+=.0625*in_s[(i+2)*w+2+j];
			}
		}
	}
	else{
		for(int i=0; i<((h-2)-(((h-2)/32)+1)*31);i++){
			for(int j=0;j<(w-2);j++){
	#pragma HLS pipeline
			out[i*(w-2)+j]+=.0625*in_s[i*w+j];
			out[i*(w-2)+j]+=.125*in_s[i*w+1+j];
			out[i*(w-2)+j]+=.0625*in_s[i*w+j+2];
			out[i*(w-2)+j]+=.125*in_s[(i+1)*w+j];
			out[i*(w-2)+j]+=.25*in_s[(i+1)*w+1+j];
			out[i*(w-2)+j]+=.125*in_s[(i+1)*w+2+j];
			out[i*(w-2)+j]+=.0625*in_s[(i+2)*w+j];
			out[i*(w-2)+j]+=.125*in_s[(i+2)*w+1+j];
			out[i*(w-2)+j]+=.0625*in_s[(i+2)*w+2+j];
			}
		}

	}
}
