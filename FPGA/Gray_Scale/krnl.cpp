// Includes
#include <hls_vector.h>
#include <hls_stream.h>
#include "assert.h"

#define MEMORY_DWIDTH 512
#define SIZEOF_WORD 4
#define NUM_WORDS ((MEMORY_DWIDTH) / (8 * SIZEOF_WORD))

#define DATA_SIZE 4096

void RGBtoGrayScale_Ver1(unsigned char* in, unsigned char* out, unsigned int h, unsigned int w)
{
	unsigned int stride = h * w;//we do this so we can avoid doing excessive multiplications
	//when accessing memory
	unsigned int stride_2 = 2 * stride;//This is for blue
	unsigned char* r = in;//We are passing a pointer pointing to start of data which is red
	unsigned char* g = in + stride;//We offset the in to get to the start of the green memory location
	unsigned char* b = in + stride_2;//Offset twice the stride to start at the beginning of the blue memory
	for (int i = 0; i < stride; i++)
	{
#pragma HLS unroll
		out[i] = 0.21f * (r[i]) + 0.71f * (g[i]) + .07 * (b[i]);
	}
}
