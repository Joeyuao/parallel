gcc MKL.c -o cMKL -lmkl_rt
g++ MKL.cpp -o cppMKL -lmkl_rt
gcc -g -Wall -o pth_hello pth_hello.c -lpthread
g++ -O3 -fomit-frame-pointer c++_trivial.cpp -o c++_trivial
source /opt/intel/oneapi/mkl/2025.0/env/vars.sh intel64 ilp64
g++ -O -fomit-frame-pointer c++_trivial.cpp -o o_c++_trivial
g++ -O1 -fomit-frame-pointer c++_trivial.cpp -o o1_c++_trivial
g++ -O2 -fomit-frame-pointer c++_trivial.cpp -o o2_c++_trivial
g++ -O3 -fomit-frame-pointer c++_trivial.cpp -o o3_c++_trivial
g++ -Ofast -fomit-frame-pointer c++_trivial.cpp -o ofast_c++_trivial