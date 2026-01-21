#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <thread>
#include <cmath>
#include "../common/image.h"
#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>
#include <GLFW/glfw3.h>

int main(int argc, char** argv) {
    int width = 800, height = 600;
    if (argc >= 3) { width = atoi(argv[1]); height = atoi(argv[2]); }
    float cx = -0.6f, cy = 0.0f;
    double scale = 3.0 / width;
    double zoom_per_frame = 0.98;
    if (argc >= 4) { zoom_per_frame = atof(argv[3]); }

    size_t imgSize = (size_t)width * height * 3;
    std::vector<unsigned char> img(imgSize);

    // OpenCL setup (pick first platform/device)
    cl_int err;
    cl_uint platformCount = 0;
    clGetPlatformIDs(0, nullptr, &platformCount);
    if (platformCount == 0) { std::cerr << "No OpenCL platforms\n"; return 1; }
    std::vector<cl_platform_id> platforms(platformCount);
    clGetPlatformIDs(platformCount, platforms.data(), nullptr);
    cl_platform_id platform = platforms[0];
    cl_device_id device;
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, nullptr);
    cl_context context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
    cl_command_queue queue = clCreateCommandQueue(context, device, 0, &err);

    // build program
    std::string src;
    {
        std::ifstream ifs("src/opencl/mandelbrot.cl");
        src.assign((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
    }
    const char* src_c = src.c_str();
    size_t src_len = src.size();
    cl_program program = clCreateProgramWithSource(context, 1, &src_c, &src_len, &err);
    err = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        size_t logSize; clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &logSize);
        std::string log(logSize, '\0'); clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, logSize, &log[0], nullptr);
        std::cerr << "Build log:\n" << log << "\n"; return 1;
    }

    cl_kernel kernel = clCreateKernel(program, "mandelbrot", &err);
    cl_mem buf = clCreateBuffer(context, CL_MEM_WRITE_ONLY, imgSize, nullptr, &err);

    // GLFW window
    if (!glfwInit()) { std::cerr << "glfwInit failed\n"; return 1; }
    GLFWwindow* win = glfwCreateWindow(width, height, "Mandelbrot OpenCL Viewer", nullptr, nullptr);
    if (!win) { glfwTerminate(); std::cerr << "Failed to create window\n"; return 1; }
    glfwMakeContextCurrent(win);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

    double local_scale = scale;
    while (!glfwWindowShouldClose(win)) {
        // run kernel
        err  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &buf);
        err |= clSetKernelArg(kernel, 1, sizeof(int), &width);
        err |= clSetKernelArg(kernel, 2, sizeof(int), &height);
        float cx_f = cx, cy_f = cy, scale_f = (float)local_scale;
        int maxIter = 1000;
        err |= clSetKernelArg(kernel, 3, sizeof(float), &cx_f);
        err |= clSetKernelArg(kernel, 4, sizeof(float), &cy_f);
        err |= clSetKernelArg(kernel, 5, sizeof(float), &scale_f);
        err |= clSetKernelArg(kernel, 6, sizeof(int), &maxIter);

        size_t global[2] = {(size_t)width, (size_t)height};
        err = clEnqueueNDRangeKernel(queue, kernel, 2, nullptr, global, nullptr, 0, nullptr, nullptr);
        clFinish(queue);
        clEnqueueReadBuffer(queue, buf, CL_TRUE, 0, imgSize, img.data(), 0, nullptr, nullptr);

        // draw using glDrawPixels
        glClear(GL_COLOR_BUFFER_BIT);
        glRasterPos2i(-1, -1);
        glDrawPixels(width, height, GL_RGB, GL_UNSIGNED_BYTE, img.data());
        glfwSwapBuffers(win);
        glfwPollEvents();

        local_scale *= zoom_per_frame;
        std::this_thread::sleep_for(std::chrono::milliseconds(16)); // ~60fps cap
        if (glfwGetKey(win, GLFW_KEY_ESCAPE) == GLFW_PRESS) break;
    }

    clReleaseMemObject(buf);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    glfwDestroyWindow(win);
    glfwTerminate();
    return 0;
}
