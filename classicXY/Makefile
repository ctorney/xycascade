
################################################################################
# Generic Cuda Makefile
#
# Colin Torney - October 2011
#
################################################################################

# Add source files here
EXECUTABLE	:= runSwitcher
CCFILES		:= cnpy.cpp
# Add cuda source files (compiled with nvcc)
CUFILES		:= switcherKernel.cu 


################################################################################
# Set up additional paths 
################################################################################


################################################################################
# Set up object files
################################################################################
OBJDIR := ./obj
OBJS +=  $(patsubst %.cpp,$(OBJDIR)/%.cpp.o,$(notdir $(CCFILES)))
OBJS +=  $(patsubst %.c,$(OBJDIR)/%.c.o,$(notdir $(CFILES)))
OBJS +=  $(patsubst %.cu,$(OBJDIR)/%.cu.o,$(notdir $(CUFILES)))


################################################################################
# Compilers and flags
################################################################################
CXX := g++ 
NVCC := nvcc 
CXXFLAGS= -O3 -Wall -Wno-deprecated -w
NVCCFLAGS := -arch sm_20 


################################################################################
# Include and library parameters
################################################################################
INC= -I./include 
LIBS= -lgsl -lgslcblas -lm -lcudart


################################################################################
# Build rules
################################################################################
all: executable

executable: ${OBJS} main.cpp 
	$(CXX) $(CXXFLAGS) $(INC) ${OBJS} main.cpp -o ${EXECUTABLE} $(LIBS) 

$(OBJDIR)/%.cpp.o: %.cpp
	$(CXX) -c $(CXXFLAGS) $(INC) -o $@ $<

$(OBJDIR)/%.cpp.o : $(MODULEDIR)/%.cpp
	$(CXX) -c $(CXXFLAGS) $(INC) -o $@ $<

$(OBJDIR)/%.cu.o: %.cu
	$(NVCC) -c $(NVCCFLAGS) $(INC) -o $@ $<

clean: tidy
	rm -f $(EXECUTABLE) *~

tidy:
	rm -f ${OBJS}
