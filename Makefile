NVCC = nvcc
CUDA_ARCH = -gencode arch=compute_110,code=sm_110
LIBS = -lcublas
INCLUDES = -I.
FLAGS = -O2 -std=c++17 --expt-relaxed-constexpr --extended-lambda -diag-suppress=177

SRCS = main.cu cuda_kernels.cu text_encoder.cu transformer.cu vae_decoder.cu
TARGET = qwen-image

$(TARGET): $(SRCS) $(wildcard *.h *.cuh)
	$(NVCC) $(FLAGS) $(CUDA_ARCH) $(INCLUDES) $(SRCS) -o $@ $(LIBS)

clean:
	rm -f $(TARGET)

.PHONY: clean
