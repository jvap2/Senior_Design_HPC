#include ".../src/L2.hpp"


float L_2(float* in, int size){
    float val{};
    for(int i{}; i<size;i++){
        val+=powf(in[i],2.0f);
    }
    val=sqrtf(val);
    return val;
}
