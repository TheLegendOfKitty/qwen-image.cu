#pragma once
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

// Minimal safetensors writer.
// Registers tensors (name, dtype, shape, raw data) and serializes them to the
// safetensors binary format: 8-byte LE header_len + JSON header + raw data.

class SafeTensorsWriter {
public:
    struct Entry {
        std::string name;
        std::string dtype_str; // "I8", "U8", "F32", "BF16"
        std::vector<int64_t> shape;
        std::vector<uint8_t> data;
    };

    void add(const std::string& name, const std::string& dtype_str,
             const std::vector<int64_t>& shape, const void* data_ptr, size_t nbytes) {
        Entry e;
        e.name = name;
        e.dtype_str = dtype_str;
        e.shape = shape;
        e.data.resize(nbytes);
        memcpy(e.data.data(), data_ptr, nbytes);
        entries_.push_back(std::move(e));
    }

    bool write(const std::string& filepath) const {
        // Build JSON header
        // Compute data offsets
        std::vector<size_t> offsets; // begin offset for each entry
        size_t running = 0;
        for (auto& e : entries_) {
            offsets.push_back(running);
            running += e.data.size();
        }

        std::string header = "{";
        for (size_t i = 0; i < entries_.size(); i++) {
            auto& e = entries_[i];
            if (i > 0) header += ",";
            header += "\"";
            header += escape_json(e.name);
            header += "\":{\"dtype\":\"";
            header += e.dtype_str;
            header += "\",\"shape\":[";
            for (size_t j = 0; j < e.shape.size(); j++) {
                if (j > 0) header += ",";
                header += std::to_string(e.shape[j]);
            }
            header += "],\"data_offsets\":[";
            header += std::to_string(offsets[i]);
            header += ",";
            header += std::to_string(offsets[i] + e.data.size());
            header += "]}";
        }
        header += "}";

        // Pad header to 8-byte alignment (safetensors convention)
        while ((8 + header.size()) % 8 != 0) {
            header += ' ';
        }

        uint64_t header_len = header.size();

        FILE* f = fopen(filepath.c_str(), "wb");
        if (!f) {
            fprintf(stderr, "SafeTensorsWriter: failed to open %s for writing\n", filepath.c_str());
            return false;
        }

        // Write 8-byte LE header length
        fwrite(&header_len, 8, 1, f);
        // Write JSON header
        fwrite(header.data(), 1, header.size(), f);
        // Write tensor data
        for (auto& e : entries_) {
            fwrite(e.data.data(), 1, e.data.size(), f);
        }

        fclose(f);
        return true;
    }

    size_t num_entries() const { return entries_.size(); }

private:
    std::vector<Entry> entries_;

    static std::string escape_json(const std::string& s) {
        std::string out;
        for (char c : s) {
            if (c == '"') out += "\\\"";
            else if (c == '\\') out += "\\\\";
            else out += c;
        }
        return out;
    }
};
