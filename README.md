## 一个简单的cuda项目实例，用于记录cuda+cmake编程时的配置

### 简单说明

#### `注意` 安装完cuda 确保在你的bashrc中配置了如下内容
```bash
export CPATH=/usr/local/cuda-[你的cuda版本]/targets/x86_64-linux/include:$CPATH
export LD_LIBRARY_PATH=/usr/local/cuda--[你的cuda版本]/targets/x86_64-linux/lib:$LD_LIBRARY_PATH
export PATH=/usr/local/cuda--[你的cuda版本]/bin:$PATH
```

#### 编译
```
mkdir -p build && cd build && cmake .. &&cmake --build .
```
#### 运行
```
./src/demo
```
#### 文件结构
```
├── CMakeLists.txt  #设置公共的信息
├── README.md
|
└── src
    ├── CMakeLists.txt  #分别编译kernels 和主程序然后连接在一起
    ├── kernels
    │   ├── helper_cuda.h     #辅助都文件
    │   ├── helper_string.h
    │   ├── kernels.hpp
    │   └── test.cu          #测试kernel函数
    └── main.cpp
```