compile: 
nvcc -c kernel.cu `pkg-config opencv --cflags --libs` -o kernel.o
g++ source.cpp imstack.cpp `pkg-config opencv --cflags --libs` -o source kernel.o -lstdc++ -lcuda -lcudart