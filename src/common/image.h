#pragma once
#include <string>
#include <fstream>

inline bool write_ppm(const std::string &filename, 
    const unsigned char *data, int width, int height) 
{
    std::ofstream ofs(filename, std::ios::binary);
    if (!ofs) return false;
    ofs << "P6\n" << width << " " << height << "\n255\n";
    ofs.write(reinterpret_cast<const char*>(data), (size_t)width * height * 3);
    return true;
}
