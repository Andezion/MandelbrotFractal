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
        float na = a * a - b * b + fx;
        float nb = 2.0f * a * b + fy;

        a = na; 
        b = nb; 
        iter++;
    }

    int idx = (y * width + x) * 3;
    unsigned char color = (unsigned char)(255.0f * iter / maxIter);

    img[idx + 0] = color;
    img[idx + 1] = color;
    img[idx + 2] = (iter == maxIter) ? 0 : 255 - color;
}

int main(int argc, char** argv) 
{
    int width = 1024, height = 768, maxIter = 1000;
    int frames = 120;
    float cx = -0.6f, cy = 0.0f;
    double initial_scale = 3.0 / width;
    double zoom_per_frame = 0.98; 

    if (argc >= 3) { 
        width = atoi(argv[1]); 
        height = atoi(argv[2]); 
    }
    if (argc >= 4) frames = atoi(argv[3]);
    if (argc >= 5) zoom_per_frame = atof(argv[4]);
    if (argc >= 7) { cx = atof(argv[5]); cy = atof(argv[6]); }
    if (argc >= 8) maxIter = atoi(argv[7]);

    size_t imgSize = (size_t)width * height * 3;

    unsigned char* h_img = (unsigned char*)malloc(imgSize);
    unsigned char* d_img;
    cudaMalloc(&d_img, imgSize);

    dim3 block(16,16);
    dim3 grid((width + block.x - 1)/block.x, (height + block.y - 1)/block.y);

    double scale = initial_scale;
    for (int f = 0; f < frames; ++f) 
    {
        mandelbrotKernel<<<grid, block>>>(d_img, width, height, cx, cy, (float)scale, maxIter);
        cudaDeviceSynchronize();

        cudaMemcpy(h_img, d_img, imgSize, cudaMemcpyDeviceToHost);

        char outpath[256];
        snprintf(outpath, sizeof(outpath), "out/mandelbrot_cuda_%05d.ppm", f);
        if (!write_ppm(outpath, h_img, width, height)) 
        {
            fprintf(stderr, "Failed to write %s\n", outpath);
            break;
        }
        printf("Wrote %s scale=%.12g\n", outpath, scale);

        scale *= zoom_per_frame;
        if (f % 30 == 0 && f>0) maxIter = (int)(maxIter * 1.2);
    }

    cudaFree(d_img);
    free(h_img);
    return 0;
}
