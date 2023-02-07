#include "RGBtoGrayScale.h"

float cpu__RGBtoGrayScale_Ver0(CImg<unsigned char>& rgbImg, CImg<unsigned char>& grayImg)
{
	int height = rgbImg.height();
	int width = rgbImg.width();
	//We want to call the height and width locally so we need it a lot
	auto start = high_resolution_clock::now();
	for (int y{}; y < height;y++) {
		for (int x{};x < width;x++) {
			grayImg(x, y) = 0.21*rgbImg(x, y, 0, 0) + .71*rgbImg(x, y, 0, 1) + .07*rgbImg(x, y, 0, 2);
			//The last param is the channel, red = 0, green = 1, blue = 2. If images had depth, you specify in the 
			//third parameter
		}
	}
	auto end = high_resolution_clock::now();
	auto elapsed_seconds= end - start;
	return duration_cast<microseconds>(elapsed_seconds).count();
	//Naive Implementation
}

float cpu__RGBtoGrayScale_Ver1(unsigned char* in, unsigned char* out, unsigned int h, unsigned int w)
{
	unsigned int stride = h * w;//we do this so we can avoid doing excessive multiplications
	//when accessing memory
	unsigned int stride_2 = 2 * stride;//This is for blue
	unsigned char* r = in;//We are passing a pointer pointing to start of data which is red
	unsigned char* g = in + stride;//We offset the in to get to the start of the green memory location
	unsigned char* b = in + stride_2;//Offset twice the stride to start at the beginning of the blue memory
	auto start = high_resolution_clock::now();
	for (int i = 0; i < stride; i++)
	{
		*out = 0.21f * (*r) + 0.71f * (*g) + .07 * (*b);
		out++;
		r++;
		g++;
		b++;
	}
	auto end = high_resolution_clock::now();
	auto elasped_seconds = end - start;

	return duration_cast<microseconds>(elasped_seconds).count();
}