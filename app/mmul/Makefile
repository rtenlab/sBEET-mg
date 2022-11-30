CUDA_PATH ?= "/usr/local/cuda"

NVCC          := $(CUDA_PATH)/bin/nvcc
# internal flags
NVCCFLAGS   := -O3 -I/usr/local/cuda/include
CCFLAGS     :=
LDFLAGS     := -lcudart -L/usr/local/cuda/lib64


PROJECT_DIR := /home/wyd/Documents/rten/gpu_energy

INCLUDES_DIR := $(PROJECT_DIR)/common/include
INCLUDES_SRC := $(PROJECT_DIR)/common/src
APPS_DIR := $(PROJECT_DIR)/app

$(info    INCLUDES_DIR is $(INCLUDES_DIR))
$(info    INCLUDES_SRC is $(INCLUDES_SRC))



INCLUDES  := -I $(INCLUDES_DIR)
LIBRARIES := -I $(APPS_DIR)
INCLUDES  += -I $(CUDA_PATH)/samples/common/inc


################################################################################
SMS ?= 52 61 72	# for Xavier AGX

ifeq ($(SMS),)
$(info >>> WARNING - no SM architectures have been specified - waiving sample <<<)
endif


################################################################################
BUILD_DIR = build
$(shell mkdir -p $(BUILD_DIR))

all: main

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

main: $(BUILD_DIR)/main.o $(PROJECT_DIR)/support.o $(BUILD_DIR)/mmul.o 
	$(NVCC) $(INCLUDES) $(LDFLAGS) $(GENCODE_FLAGS) -o $(BUILD_DIR)/$@ $+

main.o: main.cc
	$(NVCC) $(INCLUDES) $(NVCCFLAGS) $(GENCODE_FLAGS) -o $(BUILD_DIR)/$@ -c $< $(LIBRARIES)

# includes and libraries
support.o: $(INCLUDES_SRC)/support.cc
	$(NVCC) $(INCLUDES) $(NVCCFLAGS) $(GENCODE_FLAGS) -o $(PROJECT_DIR)/$@ -c $<

# apps
mmul.o: mmul.cu
	$(NVCC) $(INCLUDES) $(NVCCFLAGS) $(GENCODE_FLAGS) -o $(BUILD_DIR)/$@ -c $<


$(info    BUILD_DIR is $(BUILD_DIR))

clean:
	rm -rf build $(PROJECT_DIR)/*.o