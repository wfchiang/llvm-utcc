
all: bilateral_kernel 

INC=-I/home/wfchiang/gklee/Gklee/llvm/projects/UnsafeTypeCastingCheck 

bilateral_kernel: bilateral_kernel.c 
	gcc $(INC) -o bilateral_kernel bilateral_kernel.c 	
	clang $(INC) -O0 -emit-llvm bilateral_kernel.c -c -o bilateral_kernel.bc 

test: 
	opt -load /home/wfchiang/gklee/Gklee/llvm/Release+Asserts/lib/UnsafeTypeCastingCheck.so -utcc < bilateral_kernel.bc > /dev/null