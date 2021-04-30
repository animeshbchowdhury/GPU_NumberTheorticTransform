CPP=$(wildcard src/*.cpp)
OBJ=$(CPP:.cpp=.o)
DEPS=$(wildcard include/*.h)

DEVICE_SRC= src/main.cpp
CUDA_SRC= src/ntt.cu src/utils.cu

CUDA_OBJ= src/ntt.o src/utils.o
DEVICE_OBJ= src/main.o

#%.o: %.cpp $(DEPS)
#	$(CXX) -c -o $@ $< -Iinclude

main.o: main.cpp $(DEPS)
	$(CXX) -c -o $@ $< -Iinclude

%.cu.o: %.cu $(DEPS)
	nvcc -arch=sm_70 -o $@ $< -Iinclude

ntt: $(OBJ) 
	$(CXX) -o $@ $^ -Iinclude

ntt_cuda: $(CUDA_OBJ) $(DEVICE_OBJ) 
	nvcc -arch=sm_70 -o $@ $^ -Iinclude

clean:
	rm -rfv ntt ntt_cuda $(OBJ) $(CUDA_OBJ) $(DEVICE_OBJ) 
