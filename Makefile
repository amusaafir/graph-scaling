####Settings####
#Compilers# # Use openMPI
CC = mpicc
CPP = mpicxx
CUC = nvcc
CONVERT = convert
EXEC = sample
PROJECT_NAME = sample

#FLAGS#
#cuda
#Note: -03 does not give speed improvements over -02 in this case
DEBUG_FLAGS_CUDA = -G -g -O0 
RELEASE_FLAGS_CUDA = -O2
CUDA_TARGET = sm_52
#cc
DEBUG_FLAGS = -g -O0 
RELEASE_FLAGS = -O2


#compiler flags
GENERAL_FLAGS =
CC_FLAGS = $(GENERAL_FLAGS) -m64 -Wall -pthread -std=c99
CPP_FLAGS = $(GENERAL_FLAGS) -m64 -Wall -pthread -std=c++11
CUC_FLAGS = $(GENERAL_FLAGS) -m64 -D_FORCE_INLINES --ptxas-options=-v -arch=$(CUDA_TARGET) -rdc=true
LINK_FLAGS = -Xcompiler "-pthread" -lnvgraph -arch=$(CUDA_TARGET) -ccbin=$(CPP)

#Sources finder (used as $(shell $(SOURCE_FIND) *.ext) )
SOURCE_FIND = find . -not -path "./nbproject/*" -name

#Vars
SOURCES_C  = $(shell $(SOURCE_FIND) "*.c")
SOURCES_CU = $(shell $(SOURCE_FIND) "*.cu")
SOURCES_CP = $(shell $(SOURCE_FIND) "*.cpp")
OBJECTS_C  = $(SOURCES_C:%.c=%.o)
OBJECTS_CP = $(SOURCES_CP:%.cpp=%.o)
OBJECTS_CU = $(SOURCES_CU:%.cu=%.o)
OBJECTS_PTX= $(SOURCES_CU:%.cu=%.ptx)

SOURCES = $(SOURCES_C) $(SOURCES_CU) $(SOURCES_CP)
OBJECTS = $(OBJECTS_C) $(OBJECTS_CU) $(OBJECTS_CP)

#Rules
debug: CC_FLAGS += $(DEBUG_FLAGS)
debug: CPP_FLAGS += $(DEBUG_FLAGS)
debug: CUC_FLAGS += $(DEBUG_FLAGS_CUDA)
debug: $(EXEC)
	
release: CC_FLAGS += $(RELEASE_FLAGS) 
release: CPP_FLAGS += $(RELEASE_FLAGS)
release: CUC_FLAGS += $(RELEASE_FLAGS_CUDA)
release: $(EXEC)
	
all: release

$(EXEC):
	$(CUC) $(LINK_FLAGS) $^ -o $@

$(EXEC): $(OBJECTS)

#Generic rules
%.o: %.c
	$(CC) $(CC_FLAGS) -c $< -o $@
	
%.o: %.cpp
	$(CPP) $(CPP_FLAGS) -c $< -o $@
	
%.o: %.cu
	$(CUC) $(CUC_FLAGS) -c $< -o $@
	
%.o: %.ptx
	$(CUC) $(CUC_FLAGS) --ptx $< -o $@
	
clean:
	rm -f $(EXEC) $(OBJECTS) $(OBJECTS_PTX)
	rm -f $(PROJECT_NAME).tar.gz
	

#TAR
tar: clean
	rm -f $(PROJECT_NAME).tar.gz
	tar -czvf $(PROJECT_NAME).tar.gz $(SOURCES) $(shell find . -name *.h) makefile
