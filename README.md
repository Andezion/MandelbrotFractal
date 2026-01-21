CUDA + OpenCL Mandelbrot

Build:

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

Interactive viewer and recording
 - Build the interactive OpenCL viewer:
```bash
make viewer_opencl
```

 - Viewer CLI (args):
	 - `width` `height` — window size (default 800x600)
	 - `zoom_per_frame` — multiplicative zoom each frame (default `0.98`, <1 zooms in)
	 - `pan_x` `pan_y` — pan per frame in pixels (can be fractional; negative `pan_x` pans left)
	 - `record_file` — optional output filename (if provided, frames are piped to `ffmpeg`)
	 - `fps` — optional recording framerate (default `30`)

Examples:
```bash
# Run interactive viewer (no recording)
./build/mandelbrot_viewer_opencl 1024 768 0.98

# Run and record to MP4 (zoom 0.98, pan left 0.5 pixels/frame, 30 fps)
./build/mandelbrot_viewer_opencl 1024 768 0.98 -0.5 0 out/zoom.mp4 30
```

Notes and tips
- `pan_x`/`pan_y` are interpreted as pixels per frame and converted to complex-plane units by multiplying with the current `scale` (initially `3.0/width`). So a value like `-150` is very large (moves the center by ~-150*scale on the first frame).
- If the image becomes a flat color quickly, try:
	- smaller zoom (e.g. `0.995`),
	- smaller pan (e.g. `-0.5`),
	- larger `maxIter` in the host code to reveal deeper detail.
- Recording requires `ffmpeg` installed and available in PATH.

Build/runtime notes:
- For CUDA you need the CUDA Toolkit and `nvcc` in PATH.
- For OpenCL you need headers and a provider library (we include CUDA OpenCL headers and link to CUDA's `libOpenCL` by default).
