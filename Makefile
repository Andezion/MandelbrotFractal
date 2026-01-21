CXX := g++
CXXFLAGS := -O2 -Wall -Wextra
OPENCL_CFLAGS := -I/usr/include -I/usr/local/cuda/include
OPENCL_LIB_DIR := /usr/local/cuda/targets/x86_64-linux/lib
OPENCL_LDFLAGS := -L$(OPENCL_LIB_DIR) -lOpenCL -Wl,-rpath,$(OPENCL_LIB_DIR)

all: opencl

opencl:
	$(CXX) $(CXXFLAGS) $(OPENCL_CFLAGS) \
		src/opencl/mandelbrot_opencl.cpp \
		-o build/mandelbrot_opencl \
		$(OPENCL_LDFLAGS)

clean:
	rm -f build/mandelbrot_opencl

.PHONY: all clean
