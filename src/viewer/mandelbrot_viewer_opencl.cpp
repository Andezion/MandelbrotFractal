#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <chrono>
#include <thread>
#include <cmath>
#include <cstdio>
#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>
#include <GLFW/glfw3.h>

static double g_scroll_y = 0.0;

static void scroll_callback(GLFWwindow*, double, double yoffset) {
    g_scroll_y += yoffset;
}

int main(int argc, char** argv) {
    int width = 800, height = 600;
    if (argc >= 3) { width = atoi(argv[1]); height = atoi(argv[2]); }

    double cx = -0.6, cy = 0.0;
    double scale = 3.0 / width;
    int maxIter = 256;

    size_t imgSize = (size_t)width * height * 3;
    std::vector<unsigned char> img(imgSize);

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

    std::string src;
    {
        std::ifstream ifs("src/opencl/mandelbrot.cl");
        if (!ifs.is_open()) {
            std::cerr << "Cannot open mandelbrot.cl\n";
            return 1;
        }
        src.assign((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
    }
    const char* src_c = src.c_str();
    size_t src_len = src.size();
    cl_program program = clCreateProgramWithSource(context, 1, &src_c, &src_len, &err);
    err = clBuildProgram(program, 1, &device, "-cl-opt-disable", nullptr, nullptr);
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

    if (!glfwInit()) { std::cerr << "glfwInit failed\n"; return 1; }
    GLFWwindow* win = glfwCreateWindow(width, height, "Mandelbrot Explorer", nullptr, nullptr);
    if (!win) { glfwTerminate(); std::cerr << "Failed to create window\n"; return 1; }
    glfwMakeContextCurrent(win);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glfwSetScrollCallback(win, scroll_callback);

    const double move_speed = 0.15;
    const double zoom_factor = 1.15;
    bool needs_redraw = true;

    while (!glfwWindowShouldClose(win)) {
        glfwPollEvents();

        if (glfwGetKey(win, GLFW_KEY_W) == GLFW_PRESS) {
            cy += move_speed * scale * height;
            needs_redraw = true;
        }
        if (glfwGetKey(win, GLFW_KEY_S) == GLFW_PRESS) {
            cy -= move_speed * scale * height;
            needs_redraw = true;
        }
        if (glfwGetKey(win, GLFW_KEY_A) == GLFW_PRESS) {
            cx -= move_speed * scale * width;
            needs_redraw = true;
        }
        if (glfwGetKey(win, GLFW_KEY_D) == GLFW_PRESS) {
            cx += move_speed * scale * width;
            needs_redraw = true;
        }

        if (g_scroll_y != 0.0) {
            if (g_scroll_y > 0) {
                scale /= zoom_factor;
            } else {
                scale *= zoom_factor;
            }
            double zoom_level = (3.0 / width) / scale;
            maxIter = (int)(256 + 100 * log2(zoom_level + 1));
            if (maxIter < 256) maxIter = 256;
            if (maxIter > 10000) maxIter = 10000;

            g_scroll_y = 0.0;
            needs_redraw = true;
        }

        if (glfwGetKey(win, GLFW_KEY_EQUAL) == GLFW_PRESS || glfwGetKey(win, GLFW_KEY_KP_ADD) == GLFW_PRESS) {
            scale /= zoom_factor;
            double zoom_level = (3.0 / width) / scale;
            maxIter = (int)(256 + 100 * log2(zoom_level + 1));
            if (maxIter < 256) maxIter = 256;
            if (maxIter > 10000) maxIter = 10000;
            needs_redraw = true;
        }
        if (glfwGetKey(win, GLFW_KEY_MINUS) == GLFW_PRESS || glfwGetKey(win, GLFW_KEY_KP_SUBTRACT) == GLFW_PRESS) {
            scale *= zoom_factor;
            double zoom_level = (3.0 / width) / scale;
            maxIter = (int)(256 + 100 * log2(zoom_level + 1));
            if (maxIter < 256) maxIter = 256;
            if (maxIter > 10000) maxIter = 10000;
            needs_redraw = true;
        }

        if (glfwGetKey(win, GLFW_KEY_ESCAPE) == GLFW_PRESS) break;

        if (needs_redraw) {
            // Set kernel args and compute
            double cx_d = cx, cy_d = cy, scale_d = scale;
            err  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &buf);
            err |= clSetKernelArg(kernel, 1, sizeof(int), &width);
            err |= clSetKernelArg(kernel, 2, sizeof(int), &height);
            err |= clSetKernelArg(kernel, 3, sizeof(double), &cx_d);
            err |= clSetKernelArg(kernel, 4, sizeof(double), &cy_d);
            err |= clSetKernelArg(kernel, 5, sizeof(double), &scale_d);
            err |= clSetKernelArg(kernel, 6, sizeof(int), &maxIter);

            size_t global[2] = {(size_t)width, (size_t)height};
            clEnqueueNDRangeKernel(queue, kernel, 2, nullptr, global, nullptr, 0, nullptr, nullptr);
            clFinish(queue);
            clEnqueueReadBuffer(queue, buf, CL_TRUE, 0, imgSize, img.data(), 0, nullptr, nullptr);

            glClear(GL_COLOR_BUFFER_BIT);
            glRasterPos2i(-1, -1);
            glDrawPixels(width, height, GL_RGB, GL_UNSIGNED_BYTE, img.data());
            glfwSwapBuffers(win);

            needs_redraw = false;
        } else {
            std::this_thread::sleep_for(std::chrono::milliseconds(8));
        }
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
