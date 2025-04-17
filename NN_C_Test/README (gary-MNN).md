1. conversion for .onnx to .mnn
go to build file (get the MNNConvert)

then open terminator(example):
./MNNConvert -f ONNX --modelFile ~/Capstone/Neural/NN/My_NN/models/gru_model.onnx --MNNModel ~/Capstone/Neural/NN/My_NN/models/gru_model.mnn --bizCode MNN


2. if you want to run the mnn on GPU, then: 
if you don't have the nvcc, then do next steps:

wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb

sudo apt-get update
sudo apt-get install cuda-toolkit-11-8      # 5~7GB

echo 'export CUDA_HOME=/usr/local/cuda-11.8' >> ~/.bashrc
echo 'export PATH=$CUDA_HOME/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

nvcc --version

then go to MNN file, run: 
mkdir build && cd build

cmake .. \
  -DMNN_BUILD_SHARED_LIBS=ON \
  -DMNN_BUILD_CONVERTER=ON \
  -DMNN_CUDA=ON \
  -DCMAKE_BUILD_TYPE=Release

make -j$(nproc)
sudo make install

Then you have finished!

3. if the file inside build are locked, run(example):
sudo chown -R $USER:$USER ~/Capstone/MNN/build
