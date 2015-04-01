
all: BezierLineCDP 

INC=-I/home/wfchiang/gklee/Gklee/llvm/projects/UnsafeTypeCastingCheck 

BezierLineCDP: BezierLineCDP.c 
	gcc $(INC) -o BezierLineCDP BezierLineCDP.c 	
	clang $(INC) -O0 -emit-llvm BezierLineCDP.c -c -o BezierLineCDP.bc 

test: 
	opt -load /home/wfchiang/gklee/Gklee/llvm/Release+Asserts/lib/UnsafeTypeCastingCheck.so -utcc < BezierLineCDP.bc > /dev/null