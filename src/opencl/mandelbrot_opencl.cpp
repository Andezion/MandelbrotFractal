#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include "../common/image.h"
#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>

static std::string readFile(const std::string &path) {
    std::ifstream ifs(path);
    return std::string((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
}

int main(int argc, char** argv) {
    int width = 1024, height = 768, maxIter = 1000;
    float cx = -0.6f, cy = 0.0f, scale = 3.0f / width;
    if (argc >= 3) { width = atoi(argv[1]); height = atoi(argv[2]); }
    size_t imgSize = (size_t)width * height * 3;
    std::vector<unsigned char> img(imgSize);

    cl_int err;
    cl_uint platformCount;
    clGetPlatformIDs(0, nullptr, &platformCount);
    if (platformCount == 0) { std::cerr << "No OpenCL platforms found\n"; return 1; }
    std::vector<cl_platform_id> platforms(platformCount);
    clGetPlatformIDs(platformCount, platforms.data(), nullptr);

    cl_platform_id platform = platforms[0];
    cl_device_id device;
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, nullptr);
    cl_context context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
    cl_command_queue queue = clCreateCommandQueue(context, device, 0, &err);

    std::string src = readFile("src/opencl/mandelbrot.cl");
    const char* src_c = src.c_str();
    size_t src_len = src.size();
    cl_program program = clCreateProgramWithSource(context, 1, &src_c, &src_len, &err);
    err = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        size_t logSize;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &logSize);
        std::string log(logSize, '\0');
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, logSize, &log[0], nullptr);
        std::cerr << "Build log:\n" << log << "\n";
        return 1;
    }

    cl_kernel kernel = clCreateKernel(program, "mandelbrot", &err);
    cl_mem buf = clCreateBuffer(context, CL_MEM_WRITE_ONLY, imgSize, nullptr, &err);

    err  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &buf);
    err |= clSetKernelArg(kernel, 1, sizeof(int), &width);
    err |= clSetKernelArg(kernel, 2, sizeof(int), &height);
    err |= clSetKernelArg(kernel, 3, sizeof(float), &cx);
    err |= clSetKernelArg(kernel, 4, sizeof(float), &cy);
    err |= clSetKernelArg(kernel, 5, sizeof(float), &scale);
    err |= clSetKernelArg(kernel, 6, sizeof(int), &maxIter);

    size_t global[2] = {(size_t)width, (size_t)height};
    err = clEnqueueNDRangeKernel(queue, kernel, 2, nullptr, global, nullptr, 0, nullptr, nullptr);
    clFinish(queue);

    err = clEnqueueReadBuffer(queue, buf, CL_TRUE, 0, imgSize, img.data(), 0, nullptr, nullptr);
    std::string out = "out/mandelbrot_opencl.ppm";
    if (!write_ppm(out, img.data(), width, height)) {
        std::cerr << "Failed to write " << out << "\n";
    } else {
        std::cout << "Wrote " << out << "\n";
    }

    clReleaseMemObject(buf);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    return 0;
}
