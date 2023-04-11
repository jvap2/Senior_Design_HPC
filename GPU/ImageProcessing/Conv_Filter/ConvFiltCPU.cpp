#include "ConvFilt.h"

void Filter(CImg<unsigned char>& rgbImg, CImg<unsigned char>& blurImg, int v_off, int h_off){
    int sum_r{};
    // int sum_g{};
    // int sum_b{};
    unsigned char filter_r{};
    // unsigned char filter_g{};
    // unsigned char filter_b{};
    int h = rgbImg.height();
	int w = rgbImg.width();
    int stride=h*w;
    int filt_size=v_off*h_off;
    cout<<"Constructing Filter"<<endl;
    int count = 0;
    for (int y{}; y < h;y++) {
		for (int x{};x < w;x++) {
            for(int y_off{}; y_off<v_off;y_off++){
                int row=y+y_off-(v_off/2);
                for(int x_off{}; x_off<h_off;x_off++){
                    int col=x+x_off-(h_off/2);
                    if(col>=0 && row>=0 && row<h && col<w){
                        sum_r+=(char)rgbImg(col,row,0,0);
                        // sum_g+=(char)rgbImg(col,row,0,1);
                        // sum_b+=(char)rgbImg(col,row,0,2);
                    }
                }
            }
            filter_r=(char)sum_r/filt_size;
            // filter_g=(char)sum_g/filt_size;
            // filter_b=(char)sum_b/filt_size;
            blurImg(x,y,0,0)=filter_r;
            // blurImg(x,y,0,1)=filter_g;
            // blurImg(x,y,0,2)=filter_b;
            sum_r=0;
            // sum_g=0;
            // sum_b=0;
        }
    count++;
    }
    cout<<"Done"<<endl;
}


void Sharpen_Blue(CImg<unsigned char>& rgbImg, CImg<unsigned char>& blurImg, char* mask, int v_off, int h_off){
    int sum_b{};
    int h = rgbImg.height();
	int w = rgbImg.width();
    int stride=h*w;
    int filt_size=v_off*h_off;
    cout<<"Constructing Filter"<<endl;
    int count = 0;
    for (int y{}; y < h;y++) {
		for (int x{};x < w;x++) {
            for(int y_off{}; y_off<v_off;y_off++){
                int row=y+y_off-(v_off/2);
                for(int x_off{}; x_off<h_off;x_off++){
                    int col=x+x_off-(h_off/2);
                    if(col>=0 && row>=0 && row<h && col<w){
                        sum_b+=(char)rgbImg(col,row,0,2)*mask[y_off*h_off+x_off];
                    }
                }
            }
            blurImg(x,y,0,2)=(char)sum_b;
            sum_b=0;
        }
    count++;
    }
    cout<<"Done"<<endl;
}


void cpu__RGBtoGrayScale_Ver0(CImg<unsigned char>& rgbImg, CImg<unsigned char>& grayImg)
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
	duration_cast<microseconds>(elapsed_seconds).count();
	//Naive Implementation
}