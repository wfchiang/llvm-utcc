
all: boxFilter_kernel 

INC=-I/home/wfchiang/gklee/Gklee/llvm/projects/UnsafeTypeCastingCheck 

boxFilter_kernel: boxFilter_kernel.c 
	gcc $(INC) -o boxFilter_kernel boxFilter_kernel.c 	
	clang $(INC) -O0 -emit-llvm boxFilter_kernel.c -c -o boxFilter_kernel.bc 

test: 
	opt -load /home/wfchiang/gklee/Gklee/llvm/Release+Asserts/lib/UnsafeTypeCastingCheck.so -utcc < boxFilter_kernel.bc > /dev/null