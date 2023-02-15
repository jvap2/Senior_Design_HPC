#include "Dot.h"

float Dot_Product(float* in_1,float* in_2, int size){
    float res{};
    for(int i{}; i<size; i++){
        res+=in_1[i]*in_2[i];
    }
    return res;
}