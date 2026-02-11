__kernel void mandelbrot(
    __global unsigned char* img,
    const int width,
    const int height,
    const double cx,
    const double cy,
    const double scale,
    const int maxIter)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x >= width || y >= height)
    {
        return;
    }

    double fx = cx + (x - width / 2.0) * scale;
    double fy = cy + (y - height / 2.0) * scale;

    double a = 0.0, b = 0.0;

    int iter = 0;

    while (a * a + b * b <= 4.0 && iter < maxIter)
    {
        double na = a * a - b * b + fx;
        double nb = 2.0 * a * b + fy;

        a = na;
        b = nb;
        iter++;
    }

    int idx = (y * width + x) * 3;

    if (iter == maxIter)
    {
        img[idx + 0] = 0;
        img[idx + 1] = 0;
        img[idx + 2] = 0;
    }
    else
    {
        double t = (double)iter / maxIter;
        img[idx + 0] = (unsigned char)(9.0 * (1.0 - t) * t * t * t * 255.0);
        img[idx + 1] = (unsigned char)(15.0 * (1.0 - t) * (1.0 - t) * t * t * 255.0);
        img[idx + 2] = (unsigned char)(8.5 * (1.0 - t) * (1.0 - t) * (1.0 - t) * t * 255.0);
    }
}
