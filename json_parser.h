#pragma once
#include <cstdint>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>
#include <stdexcept>

// Minimal JSON parser for safetensors headers
// Supports: objects, arrays, strings, numbers, booleans, null

struct JsonValue {
    enum Type { NUL, BOOL, NUMBER, STRING, ARRAY, OBJECT };
    Type type = NUL;
    bool bool_val = false;
    double number_val = 0;
    std::string string_val;
    std::vector<JsonValue> array_val;
    std::vector<std::pair<std::string, JsonValue>> object_val; // ordered

    JsonValue() : type(NUL) {}

    const JsonValue& operator[](const std::string& key) const {
        for (auto& [k, v] : object_val) {
            if (k == key) return v;
        }
        static JsonValue null_val;
        return null_val;
    }

    const JsonValue& operator[](size_t idx) const {
        return array_val[idx];
    }

    bool has(const std::string& key) const {
        for (auto& [k, v] : object_val) {
            if (k == key) return true;
        }
        return false;
    }

    int64_t to_int() const { return (int64_t)number_val; }
    double to_double() const { return number_val; }
    const std::string& to_str() const { return string_val; }
    size_t size() const {
        if (type == ARRAY) return array_val.size();
        if (type == OBJECT) return object_val.size();
        return 0;
    }
};

class JsonParser {
    const char* data;
    size_t pos;
    size_t len;

    void skip_ws() {
        while (pos < len && (data[pos] == ' ' || data[pos] == '\t' || data[pos] == '\n' || data[pos] == '\r'))
            pos++;
    }

    char peek() {
        skip_ws();
        if (pos >= len) throw std::runtime_error("JSON: unexpected end");
        return data[pos];
    }

    char next() {
        skip_ws();
        if (pos >= len) throw std::runtime_error("JSON: unexpected end");
        return data[pos++];
    }

    void expect(char c) {
        char got = next();
        if (got != c) {
            std::string msg = "JSON: expected '";
            msg += c;
            msg += "' got '";
            msg += got;
            msg += "'";
            throw std::runtime_error(msg);
        }
    }

    std::string parse_string() {
        expect('"');
        std::string result;
        while (pos < len && data[pos] != '"') {
            if (data[pos] == '\\') {
                pos++;
                if (pos >= len) throw std::runtime_error("JSON: unexpected end in string");
                switch (data[pos]) {
                    case '"': result += '"'; break;
                    case '\\': result += '\\'; break;
                    case '/': result += '/'; break;
                    case 'b': result += '\b'; break;
                    case 'f': result += '\f'; break;
                    case 'n': result += '\n'; break;
                    case 'r': result += '\r'; break;
                    case 't': result += '\t'; break;
                    case 'u': {
                        // Parse 4 hex digits as unicode codepoint
                        if (pos + 4 >= len) throw std::runtime_error("JSON: bad unicode escape");
                        char hex[5] = {data[pos+1], data[pos+2], data[pos+3], data[pos+4], 0};
                        uint32_t cp = (uint32_t)strtol(hex, nullptr, 16);
                        pos += 4;
                        // UTF-8 encode
                        if (cp < 0x80) {
                            result += (char)cp;
                        } else if (cp < 0x800) {
                            result += (char)(0xC0 | (cp >> 6));
                            result += (char)(0x80 | (cp & 0x3F));
                        } else {
                            result += (char)(0xE0 | (cp >> 12));
                            result += (char)(0x80 | ((cp >> 6) & 0x3F));
                            result += (char)(0x80 | (cp & 0x3F));
                        }
                        break;
                    }
                    default: result += data[pos]; break;
                }
            } else {
                result += data[pos];
            }
            pos++;
        }
        if (pos >= len) throw std::runtime_error("JSON: unterminated string");
        pos++; // skip closing quote
        return result;
    }

    JsonValue parse_number() {
        JsonValue v;
        v.type = JsonValue::NUMBER;
        size_t start = pos;
        if (data[pos] == '-') pos++;
        while (pos < len && data[pos] >= '0' && data[pos] <= '9') pos++;
        if (pos < len && data[pos] == '.') {
            pos++;
            while (pos < len && data[pos] >= '0' && data[pos] <= '9') pos++;
        }
        if (pos < len && (data[pos] == 'e' || data[pos] == 'E')) {
            pos++;
            if (pos < len && (data[pos] == '+' || data[pos] == '-')) pos++;
            while (pos < len && data[pos] >= '0' && data[pos] <= '9') pos++;
        }
        std::string num_str(data + start, pos - start);
        v.number_val = std::stod(num_str);
        return v;
    }

    JsonValue parse_value() {
        skip_ws();
        if (pos >= len) throw std::runtime_error("JSON: unexpected end");

        char c = data[pos];
        if (c == '"') {
            JsonValue v;
            v.type = JsonValue::STRING;
            v.string_val = parse_string();
            return v;
        }
        if (c == '{') {
            return parse_object();
        }
        if (c == '[') {
            return parse_array();
        }
        if (c == 't') { // true
            if (pos + 4 <= len && memcmp(data + pos, "true", 4) == 0) {
                pos += 4;
                JsonValue v;
                v.type = JsonValue::BOOL;
                v.bool_val = true;
                return v;
            }
            throw std::runtime_error("JSON: invalid value");
        }
        if (c == 'f') { // false
            if (pos + 5 <= len && memcmp(data + pos, "false", 5) == 0) {
                pos += 5;
                JsonValue v;
                v.type = JsonValue::BOOL;
                v.bool_val = false;
                return v;
            }
            throw std::runtime_error("JSON: invalid value");
        }
        if (c == 'n') { // null
            if (pos + 4 <= len && memcmp(data + pos, "null", 4) == 0) {
                pos += 4;
                return JsonValue();
            }
            throw std::runtime_error("JSON: invalid value");
        }
        if (c == '-' || (c >= '0' && c <= '9')) {
            return parse_number();
        }
        throw std::runtime_error(std::string("JSON: unexpected char '") + c + "'");
    }

    JsonValue parse_array() {
        expect('[');
        JsonValue v;
        v.type = JsonValue::ARRAY;
        if (peek() == ']') { pos++; return v; }
        while (true) {
            v.array_val.push_back(parse_value());
            skip_ws();
            if (pos < len && data[pos] == ',') { pos++; continue; }
            break;
        }
        expect(']');
        return v;
    }

    JsonValue parse_object() {
        expect('{');
        JsonValue v;
        v.type = JsonValue::OBJECT;
        if (peek() == '}') { pos++; return v; }
        while (true) {
            std::string key = parse_string();
            expect(':');
            JsonValue val = parse_value();
            v.object_val.emplace_back(std::move(key), std::move(val));
            skip_ws();
            if (pos < len && data[pos] == ',') { pos++; continue; }
            break;
        }
        expect('}');
        return v;
    }

public:
    JsonValue parse(const char* json_data, size_t json_len) {
        data = json_data;
        pos = 0;
        len = json_len;
        return parse_value();
    }

    JsonValue parse(const std::string& s) {
        return parse(s.data(), s.size());
    }
};
