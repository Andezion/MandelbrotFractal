#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <string>
#include "../common/image.h"

__global__ void mandelbrotKernel(unsigned char* img, int width, int height, 
    float cx, float cy, float scale, int maxIter) 
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) 
    {
        return;
    }

    float fx = cx + (x - width / 2.0f) * scale;
    float fy = cy + (y - height / 2.0f) * scale;

    float a = 0.0f, b = 0.0f;
    int iter = 0;

    while (a * a + b * b <= 4.0f && iter < maxIter) 
    {
        float na = a*a - b*b + fx;
        float nb = 2.0f*a*b + fy;

        a = na; 
        b = nb; 
        iter++;
    }
    int idx = (y * width + x) * 3;
    unsigned char color = (unsigned char)(255.0f * iter / maxIter);
    img[idx+0] = color;
    img[idx+1] = color;
    img[idx+2] = (iter == maxIter) ? 0 : 255 - color;
}

int main(int argc, char** argv) {
    int width = 1024, height = 768, maxIter = 1000;
    float cx = -0.6f, cy = 0.0f, scale = 3.0f / width;
    if (argc >= 3) { width = atoi(argv[1]); height = atoi(argv[2]); }

    size_t imgSize = (size_t)width * height * 3;
    unsigned char* h_img = (unsigned char*)malloc(imgSize);
    unsigned char* d_img;
    cudaMalloc(&d_img, imgSize);

    dim3 block(16,16);
    dim3 grid((width + block.x - 1)/block.x, (height + block.y - 1)/block.y);
    mandelbrotKernel<<<grid, block>>>(d_img, width, height, cx, cy, scale, maxIter);
    cudaDeviceSynchronize();

    cudaMemcpy(h_img, d_img, imgSize, cudaMemcpyDeviceToHost);
    std::string out = "out/mandelbrot_cuda.ppm";
    if (!write_ppm(out, h_img, width, height)) {
        fprintf(stderr, "Failed to write %s\n", out.c_str());
    } else {
        printf("Wrote %s\n", out.c_str());
    }

    cudaFree(d_img);
    free(h_img);
    return 0;
}
