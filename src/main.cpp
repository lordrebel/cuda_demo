
#include"kernels/kernels.hpp"
#include<cuda.h>
#include<iostream>
int main(){
    getProperty(0);
    printf("------------------------------------------------------\n");
    //输出显卡0 的信息
    get_device_infos();
    //简单运行kernel 函数
    doTest();
    
}