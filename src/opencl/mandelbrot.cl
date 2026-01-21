__kernel void mandelbrot(
    __global unsigned char* img,
    const int width,
    const int height,
    const float cx,
    const float cy,
    const float scale,
    const int maxIter)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    if (x >= width || y >= height) return;
    float fx = cx + (x - width/2.0f) * scale;
    float fy = cy + (y - height/2.0f) * scale;
    float a = 0.0f, b = 0.0f;
    int iter = 0;
    while (a*a + b*b <= 4.0f && iter < maxIter) {
        float na = a*a - b*b + fx;
        float nb = 2.0f*a*b + fy;
        a = na; b = nb; iter++;
    }
    int idx = (y * width + x) * 3;
    unsigned char color = (unsigned char)(255.0f * iter / maxIter);
    img[idx+0] = color;
    img[idx+1] = color;
    img[idx+2] = (iter == maxIter) ? 0 : 255 - color;
}
