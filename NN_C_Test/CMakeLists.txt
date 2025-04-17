cmake_minimum_required(VERSION 3.5)
project(MNN_Test)

# 设置 C++ 标准 (例如 C++11)
set(CMAKE_CXX_STANDARD 11)

# 指定 MNN 的头文件目录（请根据你的 MNN 安装或 build 目录调整路径）
include_directories("/home/gary/Capstone/MNN/include")

# 如果 MNN 库在 build 目录下生成了静态或动态库，
# 则需要添加链接库搜索路径
link_directories("/home/gary/Capstone/MNN/build")



# 添加可执行文件
add_executable(NN_C_Test main.cpp)

# 链接 MNN 库。注意库名可能根据你实际生成的名称而不同，
# 例如可能是 MNN 或者 libMNN.so，你需要检查 /home/gary/Capstone/MNN/build 目录下实际生成的库名称。
target_link_libraries(NN_C_Test MNN -L/lib/x86_64-linux-gnu -lmvec -lm)



