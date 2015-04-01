
include ../../Makefile.common

all: simpleZeroCopy 

INC=-I$(UTCCDIR)
LIBDIR=/home/wfchiang/lib/llvm-3.6

simpleZeroCopy: simpleZeroCopy.c 
	gcc $(INC) -o simpleZeroCopy simpleZeroCopy.c 	
	$(CLANG) $(INC) -O0 -emit-llvm simpleZeroCopy.c -c -o simpleZeroCopy.bc 

test: 
	$(OPT) -load $(UTCCSO) -utcc < simpleZeroCopy.bc > /dev/null

clean: 
	-rm simpleZeroCopy
	-rm *.o
	-rm *.bc 