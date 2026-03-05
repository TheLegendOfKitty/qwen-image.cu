NVCC = nvcc
GPU_SM ?= $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | sed 's/\.//')
CUDA_ARCH = -gencode arch=compute_$(GPU_SM),code=sm_$(GPU_SM)
LIBS = -lcublas
INCLUDES = -I.
FLAGS = -O2 -std=c++17 --expt-relaxed-constexpr --extended-lambda -diag-suppress=177 --use_fast_math

SRCS = main.cu cuda_kernels.cu text_encoder.cu transformer.cu vae_decoder.cu
TARGET = qwen-image

$(TARGET): $(SRCS) $(wildcard *.h *.cuh)
	$(NVCC) $(FLAGS) $(CUDA_ARCH) $(INCLUDES) $(SRCS) -o $@ $(LIBS) -lcusolver

quantize: quantize.cu cuda_kernels.cu $(wildcard *.h *.cuh)
	$(NVCC) $(FLAGS) $(CUDA_ARCH) $(INCLUDES) quantize.cu cuda_kernels.cu -o $@ $(LIBS) -lcusolver

clean:
	rm -f $(TARGET) quantize

.PHONY: clean
