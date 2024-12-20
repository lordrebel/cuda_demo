一个简单的cuda项目实例，用于记录cuda+cmake编程时的配置

### 简单说明
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