#include "RGBtoGrayScale.h"

int main()
{
	double cpuComputeTime = 0.0f;
	double gpuComputeTime = 0.0f;
	
	cout << "Loading Images" << endl;
	//Load a RGB image
	/*CImg<unsigned char> imgRGB = CImg<>("D:\\EE5885_Vap\\Images\\lena.jpg");*///the data type in an image is a unsigned character(1 bytes->8 bits). This means we can do from 0 to 255
	//We use the place holder to say we want to work with unsigned characters
	// One windows side, black slash is escape sequence, hence we need two backslashes. On linux side, they use forward slash
	//CImg<unsigned char> imgRGB = CImg<>("D:\\Classes\\GPGPU\\Codes\\ImageProcessing\\Images\\cat.png");
	CImg<unsigned char> imgRGB = CImg<>("/home/sureshm/Senior_Design_HPC/GPU/ImageProcessing/Gray_Scale/cowboyjoe.jpg");
	
	//Display Image
	CImgDisplay dispRGB(imgRGB, "Color Image");//We can create display object, which is CImgDisplay, and we can use dispRBG to display
	//Display Image Size
	cout << "Image Height: " << imgRGB.height()<<" pixels"<<endl;//Number of pixels in the y direction
	cout << "Image Width: " << imgRGB.width() << " pixels" << endl;//Number of pixels in the x direction
	
	//Store RGB image size in bytes
	unsigned int rgbSize = imgRGB.width() * imgRGB.height() * 3*sizeof(unsigned char);//We have to multiply by 3*sizeof unsigned due to RGB channels

	//Initialize a pointer to the RGB image data stored by CImg
	unsigned char* ptrRGB = imgRGB.data();//This gives a pointer to the data of the image

	//Create an empty image with a single channel - GrayScale
	CImg<unsigned char>imgGrayScale(imgRGB.width(),imgRGB.height());//This is done in column major order, hence for first parameter,it is width then height
	//for gray scale we only need width and height, but for RGB we need to specify more. If we wanted 100 frames, then add 100,1)
	// You will also need to specify the channel
	//Store GrayScale image size in bytes
	unsigned int graySize = imgRGB.width() * imgRGB.height() * 1 * sizeof(unsigned char);
	//Instead of 3, we only have one channel because gray scale is monochromatic
	//Initialize a pointer to the GrayScale image data stored by CImg
	unsigned char* ptrGray = imgGrayScale.data();

	//CPU Version 0
	cpuComputeTime = cpu__RGBtoGrayScale_Ver0(imgRGB, imgGrayScale);
	
	cout << "RGB to GrayScale Conversion CPU Compute Time (Ver0): " << cpuComputeTime << " usecs" << endl;
	//Display the GrayScale Image
	CImgDisplay dispGray1(imgGrayScale,"Gray Scale Image CPU Version 0");//This creates a image type
	
	//CPU Version 1disp
	cpuComputeTime = cpu__RGBtoGrayScale_Ver1(ptrRGB, ptrGray, imgRGB.height(), imgRGB.width());
	
	cout << "RGB to GrayScale Conversion CPU Compute Time (Ver1): " << cpuComputeTime << " usecs" << endl;
	////Display the GrayScale Image
	//
	CImgDisplay dispGray2(imgGrayScale, "GrayScale Image CPU V1");
	////RGB to Grayscale Conversion using GPU
	CImg<unsigned char> gpu_imgGrayScale(imgRGB.width(), imgRGB.height());
	unsigned char* gpu_ptrGray = gpu_imgGrayScale.data();
	
	//GPU Versions 0
	// gpu_RGBtoGrayScaleHelper(ptrRGB, gpu_ptrGray, rgbSize, graySize, imgRGB.height(), imgRGB.width(), 0);
	// cout << "RGB to GrayScale Conversion GPU Compute Time (Ver0): " << gpuComputeTime << " usecs" << endl;
	// CImgDisplay gpu_dispGray3(gpu_imgGrayScale, "GrayScale Image - GPU - Ver0");
	
	////GPU Versions 1
	CImg<unsigned char> gpu_imgGrayScale2(imgRGB.width(), imgRGB.height());
	gpu_ptrGray = gpu_imgGrayScale2.data();
	gpu_RGBtoGrayScaleHelper(ptrRGB, gpu_ptrGray, rgbSize, graySize, imgRGB.height(), imgRGB.width(), 1);	
	cout << "RGB to GrayScale Conversion GPU Compute Time (Ver1): " << gpuComputeTime << " usecs" << endl;
	CImgDisplay gpu_dispGray4(gpu_imgGrayScale2, "GrayScale Image - GPU - Ver1");
	

	//GPU Version 2
	// CImg<unsigned char> gpu_imgGrayScale3(imgRGB.width(), imgRGB.height());
	// gpu_ptrGray = gpu_imgGrayScale3.data();
	// gpuComputeTime = gpu_RGBtoGrayScaleHelper(ptrRGB, gpu_ptrGray, rgbSize, graySize, imgRGB.height(), imgRGB.width(), 2);
	// cout << "RGB to GrayScale Conversion GPU Compute Time (Ver2): " << gpuComputeTime << " usecs" << endl;
	// CImgDisplay gpu_dispGray5(gpu_imgGrayScale3, "GrayScale Image - GPU - Ver1");
	cin.get();
	return 0;
}