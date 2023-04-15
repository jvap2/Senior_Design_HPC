#include "ConvFilt.h"

void Filter(unsigned char* in, unsigned char* out, unsigned int h, unsigned int w){

    cout<<"Constructing Filter"<<endl;
    auto start = high_resolution_clock::now();
    for(int i=0; i<h-2;i++){
        for(int j=0; j<w-2;j++){
            out[i*(w-2)+j]+=.0625*in[i*w+j];
            out[i*(w-2)+j]+=.125*in[i*w+j+1];
            out[i*(w-2)+j]+=.0625*in[i*w+j+2];
            out[i*(w-2)+j]+=.125*in[(i+1)*w+j];
            out[i*(w-2)+j]+=.25*in[(i+1)*w+j+1];
            out[i*(w-2)+j]+=.125*in[(i+1)*w+j+2];
            out[i*(w-2)+j]+=.0625*in[(i+2)*w+j];
            out[i*(w-2)+j]+=.125*in[(i+2)*w+j+1];
            out[i*(w-2)+j]+=.0625*in[(i+2)*w+j+2];
        }
    }
    auto end = high_resolution_clock::now();
	auto elasped_seconds = end - start;
    cout<< "CPU Gaussian Filter execution time: "<<duration_cast<microseconds>(elasped_seconds).count()<<" us"<<endl;
}




