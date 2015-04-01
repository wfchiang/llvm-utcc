
all: bop-phi-node func

INC=-I/home/wfchiang/gklee/Gklee/llvm/projects/UnsafeTypeCastingCheck 
OPT=-O0
IOC_FLAGS=-fsanitize=undefined -O -O0 

bop-phi-node: bop-phi-node.c 
	gcc $(INC) -o bop-phi-node bop-phi-node.c 
	clang $(INC) $(OPT) -emit-llvm bop-phi-node.c -c -o bop-phi-node.bc 

func: func.c 
	gcc $(INC) -o func func.c 
	clang $(INC) $(OPT) -emit-llvm func.c -c -o func.bc 

ioc: 
	@echo "compile by clang" 
	/home/wfchiang/lib/clang-3.3/bin/clang $(IOC_FLAGS) ioc-test.c -o ioc-test-c
	@echo "compile by clang++" 
	@cp ioc-test.c ioc-test.cpp 
	/home/wfchiang/lib/clang-3.3/bin/clang++ $(IOC_FLAGS) ioc-test.cpp -o ioc-test-cpp

test: 
#	opt -load /home/wfchiang/gklee/Gklee/llvm/Release+Asserts/lib/UnsafeTypeCastingCheck.so -utcc < bop-phi-node.bc > /dev/null

#	opt -load /home/wfchiang/gklee/Gklee/llvm/Release+Asserts/lib/UnsafeTypeCastingCheck.so -utcc < func.bc > /dev/null
	./ioc-test-c 
	./ioc-test-cpp 

clean: 
	-rm ioc-test
	-rm bop-phi-node
	-rm func
	-rm *.bc 
	-rm *~