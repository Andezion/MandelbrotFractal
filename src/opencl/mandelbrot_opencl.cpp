#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>
#include <iomanip>
#include "../common/image.h"
#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>

static std::string readFile(const std::string &path) 
{
    std::ifstream ifs(path);
    return std::string((std::istreambuf_iterator<char>(ifs)), 
    std::istreambuf_iterator<char>());
}

int main(int argc, char** argv) 
{
    int width = 1024, height = 768, maxIter = 1000;
    int frames = 120;
    float cx = -0.6f, cy = 0.0f;
    double initial_scale = 3.0 / width;
    double zoom_per_frame = 0.98; 

    if (argc >= 3) { width = atoi(argv[1]); height = atoi(argv[2]); }
    if (argc >= 4) frames = atoi(argv[3]);
    if (argc >= 5) zoom_per_frame = atof(argv[4]);
    double pan_x = 0.0, pan_y = 0.0;
    if (argc >= 6) pan_x = atof(argv[5]);
    if (argc >= 7) pan_y = atof(argv[6]);
    if (argc >= 9) { cx = atof(argv[7]); cy = atof(argv[8]); }
    if (argc >= 10) maxIter = atoi(argv[9]);

    size_t imgSize = (size_t)width * height * 3;
    std::vector<unsigned char> img(imgSize);

    cl_int err;
    cl_uint platformCount;

    clGetPlatformIDs(0, nullptr, &platformCount);
    if (platformCount == 0) 
    { 
        std::cerr << "No OpenCL platforms found\n"; 
        return 1; 
    }

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
    if (err != CL_SUCCESS) 
    {
        size_t logSize;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &logSize);

        std::string log(logSize, '\0');

        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, logSize, &log[0], nullptr);
        std::cerr << "Build log:\n" << log << "\n";

        return 1;
    }

    cl_kernel kernel = clCreateKernel(program, "mandelbrot", &err);
    cl_mem buf = clCreateBuffer(context, CL_MEM_WRITE_ONLY, imgSize, nullptr, &err);

    double scale = initial_scale;
    for (int f = 0; f < frames; ++f) 
    {
        err  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &buf);
        err |= clSetKernelArg(kernel, 1, sizeof(int), &width);
        err |= clSetKernelArg(kernel, 2, sizeof(int), &height);
        float cx_f = (float)cx;
        float cy_f = (float)cy;
        float scale_f = (float)scale;
        err |= clSetKernelArg(kernel, 3, sizeof(float), &cx_f);
        err |= clSetKernelArg(kernel, 4, sizeof(float), &cy_f);
        err |= clSetKernelArg(kernel, 5, sizeof(float), &scale_f);
        err |= clSetKernelArg(kernel, 6, sizeof(int), &maxIter);

        size_t global[2] = {(size_t)width, (size_t)height};
        err = clEnqueueNDRangeKernel(queue, kernel, 2, nullptr, global, nullptr, 0, nullptr, nullptr);
        clFinish(queue);

        err = clEnqueueReadBuffer(queue, buf, CL_TRUE, 0, imgSize, img.data(), 0, nullptr, nullptr);

        char outpath[256];
        snprintf(outpath, sizeof(outpath), "out/mandelbrot_opencl_%05d.ppm", f);
        if (!write_ppm(outpath, img.data(), width, height)) {
            std::cerr << "Failed to write " << outpath << "\n";
            break;
        }
        std::cout << "Wrote " << outpath << " scale=" << std::setprecision(12) << scale << "\n";

        // apply pan (pixels per frame converted to complex-plane units)
        cx += (float)(pan_x * scale);
        cy += (float)(pan_y * scale);
        scale *= zoom_per_frame;
        if (f % 30 == 0 && f>0) maxIter = (int)(maxIter * 1.2);
    }

    clReleaseMemObject(buf);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    return 0;
}
