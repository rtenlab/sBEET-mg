############################################################
# architecture
HOST_ARCH   := $(shell uname -m)
TARGET_ARCH ?= $(HOST_ARCH)
ifneq (,$(filter $(TARGET_ARCH),x86_64 aarch64 ppc64le armv7l))
    ifneq ($(TARGET_ARCH),$(HOST_ARCH))
        ifneq (,$(filter $(TARGET_ARCH),x86_64 aarch64 ppc64le))
            TARGET_SIZE := 64
        else ifneq (,$(filter $(TARGET_ARCH),armv7l))
            TARGET_SIZE := 32
        endif
    else
        TARGET_SIZE := $(shell getconf LONG_BIT)
    endif
else
    $(error ERROR - unsupported value $(TARGET_ARCH) for TARGET_ARCH!)
endif
ifneq ($(TARGET_ARCH),$(HOST_ARCH))
    ifeq (,$(filter $(HOST_ARCH)-$(TARGET_ARCH),aarch64-armv7l x86_64-armv7l x86_64-aarch64 x86_64-ppc64le))
        $(error ERROR - cross compiling from $(HOST_ARCH) to $(TARGET_ARCH) is not supported!)
    endif
endif

############################################################

CUDA_PATH ?= "/usr/local/cuda"
NVCC        := $(CUDA_PATH)/bin/nvcc
# internal flags
NVCCFLAGS   := -O3 -I/usr/local/cuda/include
CCFLAGS     :=
LDFLAGS     := -lcudart -L/usr/local/cuda/lib64 -lfmt

# Directories
PROJECT_DIR := $(shell pwd)
SRC_DIR := $(PROJECT_DIR)/src
APP_DIR := $(PROJECT_DIR)/app
# Add include lib
INCLUDES  := -I/usr/local/include
INCLUDES  := -I $(PROJECT_DIR)
INCLUDES  += -I$(CUDA_PATH)/samples/common/inc
INCLUDES  += -I$(PROJECT_DIR)/common/cuda_inc
INCLUDES  += -I $(APP_DIR)

############################################################
SMS ?= 52 61 72
ifeq ($(SMS),)
$(info >>> WARNING - no SM architectures have been specified - waiving sample <<<)
endif
ifeq ($(GENCODE_FLAGS),)
# Generate SASS code for each SM architecture listed in $(SMS)
$(foreach sm,$(SMS),$(eval GENCODE_FLAGS += -gencode arch=compute_$(sm),code=sm_$(sm)))
# Generate PTX code from the highest SM architecture in $(SMS) to guarantee forward-compatibility
HIGHEST_SM := $(lastword $(sort $(SMS)))
ifneq ($(HIGHEST_SM),)
GENCODE_FLAGS += -gencode arch=compute_$(HIGHEST_SM),code=compute_$(HIGHEST_SM)
endif
endif

############################################################

SRC_BUILD_DIR = $(SRC_DIR)/build
$(shell mkdir -p $(SRC_BUILD_DIR))
$(SRC_BUILD_DIR):
	mkdir -p $(SRC_BUILD_DIR)

# applications
APP_BUILD_DIR = $(APP_DIR)/build
$(shell mkdir -p $(APP_BUILD_DIR))
$(MODEL_BUILD_DIR):
	mkdir -p $(APP_BUILD_DIR)
# $(info    APP_BUILD_DIR: $(APP_BUILD_DIR))
APP_LIST = mmul/mmul.cu \
		stereodisparity/stereodisparity.cu \
		bfs/bfs.cu \
		hotspot/hotspot.cu \
		dxtc/dxtc.cu \
		histogram.cu
APP_SRC = $(patsubst %,$(APP_DIR)/%,$(APP_LIST))
APP_OBJ_LIST = mmul.o \
		stereodisparity.o \
		bfs.o \
		hotspot.o \
		dxtc.o \
		histogram.o
APP_OBJ = $(patsubst %,$(APP_BUILD_DIR)/%,$(APP_OBJ_LIST))

# src
MODEL_DIR = $(SRC_DIR)/model
MODEL_BUILD_DIR = $(MODEL_DIR)/build
$(shell mkdir -p $(MODEL_BUILD_DIR))
$(MODEL_BUILD_DIR):
	mkdir -p $(MODEL_BUILD_DIR)
MODEL_LIST = support.cc \
		task_model.cc \
		job_model.cc \
		gpu_slot_model.cc \
		gpu_model.cc \
		power_model.cc \
		worker.cc
MODEL_SRC = $(patsubst %,$(MODEL_DIR)/%,$(MODEL_LIST))
MODEL_OBJ_LIST = support.o \
		task_model.o \
		job_model.o \
		gpu_slot_model.o \
		gpu_model.o \
		power_model.o \
		worker.o
MODEL_OBJ = $(patsubst %,$(MODEL_BUILD_DIR)/%,$(MODEL_OBJ_LIST))

# sbeet_mg algorithm
ALG_DIR = $(SRC_DIR)/algorithm
ALG_BUILD_DIR = $(ALG_DIR)/build
$(shell mkdir -p $(ALG_BUILD_DIR))
$(ALG_BUILD_DIR):
	mkdir -p $(ALG_BUILD_DIR)
ALG_LIST = sbeet.cc \
		job_migration.cc \
		scheduler.cc
ALG_SRC = $(patsubst %,$(ALG_DIR)/%,$(ALG_LIST))
ALG_OBJ_LIST = sbeet.o \
		job_migration.o \
		scheduler.o
ALG_OBJ = $(patsubst %,$(ALG_BUILD_DIR)/%,$(ALG_OBJ_LIST))

############################################################
# src
# model
$(MODEL_BUILD_DIR)/%.o: $(MODEL_DIR)/%.cc
	$(NVCC) $(INCLUDES) $(NVCCFLAGS) $(GENCODE_FLAGS) -c $< -o $@

# algorithms
$(ALG_BUILD_DIR)/%.o: $(ALG_DIR)/%.cc
	$(NVCC) $(INCLUDES) $(NVCCFLAGS) $(GENCODE_FLAGS) -c $< -o $@

# app
$(APP_BUILD_DIR)/mmul.o: $(APP_DIR)/mmul/mmul.cu 
	$(NVCC) $(INCLUDES) $(NVCCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

$(APP_BUILD_DIR)/stereodisparity.o: $(APP_DIR)/stereodisparity/stereodisparity.cu 
	$(NVCC) $(INCLUDES) $(NVCCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

$(APP_BUILD_DIR)/hotspot.o: $(APP_DIR)/hotspot/hotspot.cu
	$(NVCC) $(INCLUDES) $(NVCCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

$(APP_BUILD_DIR)/dxtc.o: $(APP_DIR)/dxtc/dxtc.cu
	$(NVCC) $(INCLUDES) $(NVCCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

$(APP_BUILD_DIR)/bfs.o: $(APP_DIR)/bfs/bfs.cu 
	$(NVCC) $(INCLUDES) $(NVCCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

$(APP_BUILD_DIR)/histogram.o: $(APP_DIR)/histogram/histogram.cu 
	$(NVCC) $(INCLUDES) $(NVCCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

############################################################
# executables

all: main
# Build
BUILD_DIR = build
$(shell mkdir -p $(BUILD_DIR))
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

main: $(BUILD_DIR)/main.o  $(MODEL_OBJ) $(APP_OBJ) $(ALG_OBJ)
	$(NVCC) $(INCLUDES) $(LDFLAGS) $(GENCODE_FLAGS) -o $@ $+
$(BUILD_DIR)/main.o: main.cc
	$(NVCC) $(INCLUDES) $(NVCCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

clean:
	rm -rf $(APP_BUILD_DIR) $(MODEL_BUILD_DIR) $(ALG_BUILD_DIR) $(BUILD_DIR) $(LCF_BUILD_DIR) $(BCF_BUILD_DIR) src/build main profile

clean_model:
	rm -rf $(MODEL_BUILD_DIR)

clean_build:
	rm -rf $(BUILD_DIR) main

clean_app:
	rm -rf $(APP_BUILD_DIR)