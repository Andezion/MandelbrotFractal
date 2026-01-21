CUDA + OpenCL Mandelbrot

Сборка:

```bash
# Собрать оба варианта
make

# Или отдельно
make cuda
make opencl
```

Запуск:

```bash
# CUDA (если доступен nvcc и GPU)
./build/mandelbrot_cuda 1024 768

# OpenCL
./build/mandelbrot_opencl 1024 768
```

Результат: `out/mandelbrot_cuda.ppm` и `out/mandelbrot_opencl.ppm`. Откройте их любым просмотрщиком изображений, поддерживающим PPM.

Примечания:
- Для CUDA нужен установленный CUDA Toolkit и `nvcc` в PATH.
- Для OpenCL нужен заголовок и библиотека OpenCL (обычно `-lOpenCL`).
