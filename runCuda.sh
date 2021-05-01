nvcc -arch=sm_70 -g -dc -o src/ntt.o -Iinclude src/ntt.cu
nvcc -arch=sm_70 -g -dc -o src/utils.o -Iinclude src/utils.cu
nvcc -arch=sm_70 -g -dc -o src/main.o -Iinclude src/main.cu
nvcc -arch=sm_70 -g --output-file ntt_cuda src/ntt.o src/utils.o src/main.o
#g++ ntt_cuda.o src/main.o src/utils.o src/ntt.o -lcudart -o finalOut
