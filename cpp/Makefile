CC = mpic++
CC_OPENMP = g++
RM = rm
INCLUDE_MPI = -D INCLUDE_MPI
INCLUDE_OPENMP = -D INCLUDE_OPENMP

MKLROOT = /opt/intel/mkl

UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Linux)
    CC_OPENMP = g++
endif

ifeq ($(UNAME_S),Linux)
	LINK = -L${MKLROOT}/lib/intel64 -Wl,--no-as-needed -lmkl_intel_lp64 -lmkl_gnu_thread -lmkl_core -lgomp -lpthread -lm -ldl
	INCL = -m64 -I $(MKLROOT)/include -std=c++11
else
	LINK = -L${MKLROOT}/lib -Wl,-rpath,${MKLROOT}/lib -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -lm -ldl
	INCL = -I $(MKLROOT)/include
endif

ML_SRC_FILES = $(filter-out src/ml/GLNP.cpp, $(wildcard src/ml/*.cpp))

all: lploka nystrom

lploka:
	$(CC) $(INCLUDE_MPI) src/main_lploka.cpp $(ML_SRC_FILES) src/tools/*.cpp -o build/lploka $(INCL) $(LINK)

nystrom:
	$(CC) $(INCLUDE_MPI) src/main_nystrom.cpp $(ML_SRC_FILES) src/tools/*.cpp -o build/nystrom $(INCL) $(LINK)
