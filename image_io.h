#pragma once
#include <cstdio>
#include <cstdint>
#include <string>
#include <vector>

// Write PPM image (no external dependencies)
inline bool write_ppm(const std::string& filepath, const uint8_t* pixels, int width, int height) {
    FILE* f = fopen(filepath.c_str(), "wb");
    if (!f) {
        fprintf(stderr, "Failed to open %s for writing\n", filepath.c_str());
        return false;
    }
    fprintf(f, "P6\n%d %d\n255\n", width, height);
    fwrite(pixels, 1, width * height * 3, f);
    fclose(f);
    return true;
}

// Simple BMP writer for wider compatibility
inline bool write_bmp(const std::string& filepath, const uint8_t* pixels, int width, int height) {
    FILE* f = fopen(filepath.c_str(), "wb");
    if (!f) return false;

    int row_size = ((width * 3 + 3) / 4) * 4; // rows padded to 4 bytes
    int data_size = row_size * height;
    int file_size = 54 + data_size;

    // BMP header
    uint8_t header[54] = {};
    header[0] = 'B'; header[1] = 'M';
    *(int32_t*)(header + 2) = file_size;
    *(int32_t*)(header + 10) = 54; // data offset
    *(int32_t*)(header + 14) = 40; // DIB header size
    *(int32_t*)(header + 18) = width;
    *(int32_t*)(header + 22) = height;
    *(int16_t*)(header + 26) = 1;  // planes
    *(int16_t*)(header + 28) = 24; // bits per pixel
    *(int32_t*)(header + 34) = data_size;

    fwrite(header, 1, 54, f);

    // BMP stores rows bottom-to-top, BGR
    std::vector<uint8_t> row(row_size, 0);
    for (int y = height - 1; y >= 0; y--) {
        for (int x = 0; x < width; x++) {
            row[x * 3 + 0] = pixels[(y * width + x) * 3 + 2]; // B
            row[x * 3 + 1] = pixels[(y * width + x) * 3 + 1]; // G
            row[x * 3 + 2] = pixels[(y * width + x) * 3 + 0]; // R
        }
        fwrite(row.data(), 1, row_size, f);
    }

    fclose(f);
    return true;
}
