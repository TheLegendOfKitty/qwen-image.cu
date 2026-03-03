#pragma once
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <unordered_map>
#include <vector>
#include <algorithm>

#include "json_parser.h"
#include "tensor.h"

struct TensorInfo {
    std::string name;
    DType dtype;
    std::vector<int64_t> shape;
    size_t data_offset; // offset from start of binary data section
    size_t nbytes;
    std::string filename; // which safetensors file
};

struct SafeTensorsFile {
    std::string filepath;
    uint64_t header_len;
    size_t data_start; // = 8 + header_len
};

class SafeTensorsLoader {
public:
    std::unordered_map<std::string, TensorInfo> tensors;
    std::vector<SafeTensorsFile> files;

    // Load a single safetensors file's metadata
    void load_file(const std::string& filepath) {
        FILE* f = fopen(filepath.c_str(), "rb");
        if (!f) {
            fprintf(stderr, "Failed to open: %s\n", filepath.c_str());
            exit(1);
        }

        // Read 8-byte header length
        uint64_t header_len;
        if (fread(&header_len, 8, 1, f) != 1) {
            fprintf(stderr, "Failed to read header length from %s\n", filepath.c_str());
            fclose(f);
            exit(1);
        }

        // Read header JSON
        std::string header(header_len, '\0');
        if (fread(&header[0], 1, header_len, f) != header_len) {
            fprintf(stderr, "Failed to read header from %s\n", filepath.c_str());
            fclose(f);
            exit(1);
        }
        fclose(f);

        size_t data_start = 8 + header_len;
        SafeTensorsFile sf;
        sf.filepath = filepath;
        sf.header_len = header_len;
        sf.data_start = data_start;
        files.push_back(sf);

        // Parse JSON header
        JsonParser parser;
        JsonValue root = parser.parse(header);

        for (auto& [key, val] : root.object_val) {
            if (key == "__metadata__") continue;

            TensorInfo info;
            info.name = key;
            info.filename = filepath;

            // Parse dtype
            std::string dtype_str = val["dtype"].to_str();
            if (dtype_str == "BF16" || dtype_str == "bfloat16") {
                info.dtype = DType::BF16;
            } else if (dtype_str == "F32" || dtype_str == "float32") {
                info.dtype = DType::FP32;
            } else if (dtype_str == "F16" || dtype_str == "float16") {
                // We'll treat F16 as BF16 size for loading, convert later if needed
                info.dtype = DType::BF16;
                fprintf(stderr, "Warning: tensor '%s' is F16, treating as BF16\n", key.c_str());
            } else if (dtype_str == "I8" || dtype_str == "int8") {
                info.dtype = DType::INT8;
            } else {
                fprintf(stderr, "Unknown dtype '%s' for tensor '%s'\n", dtype_str.c_str(), key.c_str());
                continue;
            }

            // Parse shape
            for (size_t i = 0; i < val["shape"].size(); i++) {
                info.shape.push_back(val["shape"][i].to_int());
            }

            // Parse data offsets
            auto& offsets = val["data_offsets"];
            size_t begin = (size_t)offsets[0].to_int();
            size_t end = (size_t)offsets[1].to_int();
            info.data_offset = begin;
            info.nbytes = end - begin;

            tensors[key] = info;
        }
    }

    // Load from sharded index file (model.safetensors.index.json)
    void load_sharded(const std::string& index_path) {
        std::ifstream ifs(index_path);
        if (!ifs.good()) {
            fprintf(stderr, "Failed to open index: %s\n", index_path.c_str());
            exit(1);
        }
        std::string content((std::istreambuf_iterator<char>(ifs)),
                           std::istreambuf_iterator<char>());
        ifs.close();

        JsonParser parser;
        JsonValue root = parser.parse(content);

        // Get directory of index file
        std::string dir = index_path;
        size_t last_sep = dir.find_last_of('/');
        if (last_sep != std::string::npos) {
            dir = dir.substr(0, last_sep + 1);
        } else {
            dir = "./";
        }

        // Collect unique filenames
        std::unordered_map<std::string, bool> loaded_files;
        auto& weight_map = root["weight_map"];
        for (auto& [tensor_name, file_val] : weight_map.object_val) {
            std::string filename = dir + file_val.to_str();
            if (!loaded_files[filename]) {
                loaded_files[filename] = true;
                load_file(filename);
            }
        }
    }

    // Auto-detect: single file or sharded
    void load(const std::string& path) {
        // Check if it's an index file
        if (path.find(".index.json") != std::string::npos) {
            load_sharded(path);
            return;
        }
        // Check if a single safetensors file
        if (path.find(".safetensors") != std::string::npos) {
            load_file(path);
            return;
        }
        // Try to find index file
        std::string index_path = path + ".index.json";
        std::ifstream test(index_path);
        if (test.good()) {
            test.close();
            load_sharded(index_path);
        } else {
            load_file(path);
        }
    }

    // Load a single tensor to GPU
    Tensor load_tensor(const std::string& name) const {
        auto it = tensors.find(name);
        if (it == tensors.end()) {
            fprintf(stderr, "Tensor not found: %s\n", name.c_str());
            exit(1);
        }
        const TensorInfo& info = it->second;

        // Find the file's data_start
        size_t data_start = 0;
        for (auto& sf : files) {
            if (sf.filepath == info.filename) {
                data_start = sf.data_start;
                break;
            }
        }

        // Read from file
        FILE* f = fopen(info.filename.c_str(), "rb");
        if (!f) {
            fprintf(stderr, "Failed to open: %s\n", info.filename.c_str());
            exit(1);
        }
        fseek(f, (long)(data_start + info.data_offset), SEEK_SET);

        std::vector<uint8_t> buf(info.nbytes);
        if (fread(buf.data(), 1, info.nbytes, f) != info.nbytes) {
            fprintf(stderr, "Failed to read tensor data: %s\n", name.c_str());
            fclose(f);
            exit(1);
        }
        fclose(f);

        // Allocate GPU tensor and upload
        Tensor t = Tensor::alloc(info.shape, info.dtype);
        t.from_host(buf.data(), info.nbytes);
        return t;
    }

    // Check if tensor exists
    bool has_tensor(const std::string& name) const {
        return tensors.find(name) != tensors.end();
    }

    // Get tensor info
    const TensorInfo& get_info(const std::string& name) const {
        return tensors.at(name);
    }

    void print_summary() const {
        fprintf(stderr, "SafeTensors: %zu tensors from %zu files\n", tensors.size(), files.size());
        size_t total = 0;
        for (auto& [name, info] : tensors) {
            total += info.nbytes;
        }
        fprintf(stderr, "  Total size: %.2f GB\n", total / (1024.0 * 1024.0 * 1024.0));
    }
};

// Convenience: load all safetensors files from a model subdirectory
inline SafeTensorsLoader load_model_dir(const std::string& dir) {
    SafeTensorsLoader loader;

    // Try index file first
    std::string index_path = dir + "/model.safetensors.index.json";
    std::ifstream test_idx(index_path);
    if (test_idx.good()) {
        test_idx.close();
        loader.load_sharded(index_path);
        return loader;
    }

    // Try diffusion_pytorch_model.safetensors.index.json
    index_path = dir + "/diffusion_pytorch_model.safetensors.index.json";
    std::ifstream test_idx2(index_path);
    if (test_idx2.good()) {
        test_idx2.close();
        loader.load_sharded(index_path);
        return loader;
    }

    // Try single file
    std::string single = dir + "/model.safetensors";
    std::ifstream test_single(single);
    if (test_single.good()) {
        test_single.close();
        loader.load_file(single);
        return loader;
    }

    single = dir + "/diffusion_pytorch_model.safetensors";
    std::ifstream test_single2(single);
    if (test_single2.good()) {
        test_single2.close();
        loader.load_file(single);
        return loader;
    }

    fprintf(stderr, "No safetensors files found in %s\n", dir.c_str());
    exit(1);
}
