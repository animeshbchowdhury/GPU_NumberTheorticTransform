CPP=$(wildcard src/*.cpp)
OBJ=$(CPP:.cpp=.o)
DEPS=$(wildcard include/*.h)

%.o: %.cpp $(DEPS)
	$(CXX) -c -o $@ $< -Iinclude

ntt: $(OBJ) 
	$(CXX) -o $@ $^ -Iinclude

clean:
	rm -rfv ntt $(OBJ)
