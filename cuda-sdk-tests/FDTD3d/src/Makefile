
all: FDTD3dGPU simplified_FDTD3dGPU

INC=-I/home/wfchiang/gklee/Gklee/llvm/projects/UnsafeTypeCastingCheck 

FDTD3dGPU: FDTD3dGPU.c 
	gcc $(INC) -o FDTD3dGPU FDTD3dGPU.c 	
	clang $(INC) -O0 -emit-llvm FDTD3dGPU.c -c -o FDTD3dGPU.bc 

simplified_FDTD3dGPU: simplified_FDTD3dGPU.c 
	clang $(INC) -O0 -emit-llvm simplified_FDTD3dGPU.c -c -o simplified_FDTD3dGPU.bc 

test: 
#	opt -load /home/wfchiang/gklee/Gklee/llvm/Release+Asserts/lib/UnsafeTypeCastingCheck.so -utcc < FDTD3dGPU.bc > /dev/null
	opt -load /home/wfchiang/gklee/Gklee/llvm/Release+Asserts/lib/UnsafeTypeCastingCheck.so -utcc < simplified_FDTD3dGPU.bc > /dev/null

clean: 
	-rm FDTD3dGPU
	-rm *.bc