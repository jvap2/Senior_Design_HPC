#include "ConvFilt.h"

void Filter(CImg<unsigned char>& rgbImg, CImg<unsigned char>& blurImg, int v_off, int h_off){
    auto start=high_resolution_clock::now();
    unsigned char filter_r;
    unsigned char filter_g;
    unsigned char filter_b;
    int h = rgbImg.height();
	int w = rgbImg.width();
    int stride=h*w;
    int filt_size=v_off*h_off;
    for (int y{}; y < h;y++) {
		for (int x{};x < w;x++) {
            for(int y_off{}; y_off<v_off;y++){
                for(int x_off{}; x_off<h_off;x_off++){
                    int col=x+x_off-(h_off/2);
                    int row=y+y_off-(v_off/2);
                    if(col>0 || row>0 || row<h || col<w){
                        filter_r+=rgbImg(row,col,0,0);
                        filter_g+=rgbImg(row,col,0,1);
                        filter_b+=rgbImg(row,col,0,2);
                    }
                }
                filter_r/=filt_size;
                filter_g/=filt_size;
                filter_b/=filt_size;
                blurImg(x,y,0,0)=filter_r;
                blurImg(x,y,0,1)=filter_g;
                blurImg(x,y,0,2)=filter_b;
            }
        }
    }
}