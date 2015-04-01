
include ../Makefile.common

all: basic-c 
# all: basic-c basic-cuda 

OFLAG=-O0
IOC_FLAGS=-fsanitize=undefined -O -O0 

basic-c: basic-c.c 
	$(CLANG) $(INC) $(OFLAG) -emit-llvm basic-c.c -c -o basic-c.bc 

basic-cuda: basic-cuda.cu 
#	nvcc $(INC) -arch=sm_20 -o basic-cuda basic-cuda.cu 
	$(CLANGPP) $(INC) $(OFLAG) -x cuda -emit-llvm basic-cuda.cu -c -o basic-cuda.bc 
#	$(CLANG) $(INC) $(OFLAG) -X cuda -emit-llvm basic-cuda.cu -c -o basic-cuda.bc 

ioc: 
	@echo "compile by clang" 
	/home/wfchiang/lib/clang-3.3/bin/clang $(IOC_FLAGS) ioc-test.c -o ioc-test-c
	@echo "compile by clang++" 
	@cp ioc-test.c ioc-test.cpp 
	/home/wfchiang/lib/clang-3.3/bin/clang++ $(IOC_FLAGS) ioc-test.cpp -o ioc-test-cpp

test: 
	$(OPT) -load $(UTCCSO) -utcc < basic-c.bc > basic-c-utcc.bc 
	$(LLVMDIS) -o basic-c-utcc.dis basic-c-utcc.bc 
	$(LLI) basic-c-utcc.bc 

clean: 
	-rm basic-cuda 
	-rm *.bc 
	-rm *.o
	-rm *~