#pragma once
#include <cstdint>
#include <fstream>
#include <string>
#include <unordered_map>
#include <vector>
#include <algorithm>
#include <sstream>

#include "json_parser.h"
#include "safetensors.h"
#include "logging.h"

class Qwen2Tokenizer {
public:
    std::unordered_map<std::string, int32_t> vocab;
    std::vector<std::pair<std::string, std::string>> merges;
    std::unordered_map<std::string, int32_t> bpe_ranks;

    // Special tokens
    static constexpr int32_t ENDOFTEXT = 151643;
    static constexpr int32_t IM_START = 151644;
    static constexpr int32_t IM_END = 151645;

    // System prompt for Qwen-Image text-to-image (stripped from encoder output)
    static constexpr const char* SYSTEM_PROMPT =
        "Describe the image by detailing the color, shape, size, texture, "
        "quantity, text, spatial relationships of the objects and background:";
    // Number of tokens in system template prefix to strip from encoder output
    // This is: <|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\n
    // = 34 tokens (verified against sd.cpp)
    static constexpr int PROMPT_TEMPLATE_ENCODE_START_IDX = 34;

    // bytes_to_unicode mapping (GPT-2 style)
    std::unordered_map<uint8_t, std::string> byte_encoder;
    std::unordered_map<std::string, uint8_t> byte_decoder;

    void init_byte_encoder() {
        // Build the GPT-2 bytes_to_unicode mapping
        std::vector<int> bs;
        // Range '!' to '~' (33 to 126)
        for (int i = 33; i <= 126; i++) bs.push_back(i);
        // Range '¡' to '¬' (161 to 172)
        for (int i = 161; i <= 172; i++) bs.push_back(i);
        // Range '®' to 'ÿ' (174 to 255)
        for (int i = 174; i <= 255; i++) bs.push_back(i);

        std::vector<int> cs(bs.begin(), bs.end());
        int n = 0;
        for (int b = 0; b < 256; b++) {
            bool found = false;
            for (int x : bs) {
                if (x == b) { found = true; break; }
            }
            if (!found) {
                bs.push_back(b);
                cs.push_back(256 + n);
                n++;
            }
        }

        for (size_t i = 0; i < bs.size(); i++) {
            // Convert cs[i] to UTF-8 string
            std::string s;
            int cp = cs[i];
            if (cp < 0x80) {
                s += (char)cp;
            } else if (cp < 0x800) {
                s += (char)(0xC0 | (cp >> 6));
                s += (char)(0x80 | (cp & 0x3F));
            } else {
                s += (char)(0xE0 | (cp >> 12));
                s += (char)(0x80 | ((cp >> 6) & 0x3F));
                s += (char)(0x80 | (cp & 0x3F));
            }
            byte_encoder[(uint8_t)bs[i]] = s;
            byte_decoder[s] = (uint8_t)bs[i];
        }
    }

    void load(const std::string& vocab_path, const std::string& merges_path) {
        init_byte_encoder();

        // Load vocab.json
        std::ifstream vf(vocab_path);
        if (!vf.good()) {
            fprintf(stderr, "Failed to open vocab: %s\n", vocab_path.c_str());
            exit(1);
        }
        std::string vcontent((std::istreambuf_iterator<char>(vf)),
                            std::istreambuf_iterator<char>());
        vf.close();

        JsonParser parser;
        JsonValue vroot = parser.parse(vcontent);
        for (auto& [key, val] : vroot.object_val) {
            vocab[key] = (int32_t)val.to_int();
        }
        LOGV("Tokenizer: loaded %zu vocab entries\n", vocab.size());

        // Load merges.txt
        std::ifstream mf(merges_path);
        if (!mf.good()) {
            fprintf(stderr, "Failed to open merges: %s\n", merges_path.c_str());
            exit(1);
        }
        std::string line;
        // Skip first line (header)
        std::getline(mf, line);
        int rank = 0;
        while (std::getline(mf, line)) {
            if (line.empty()) continue;
            size_t space = line.find(' ');
            if (space == std::string::npos) continue;
            std::string a = line.substr(0, space);
            std::string b = line.substr(space + 1);
            // Remove trailing \r if present
            if (!b.empty() && b.back() == '\r') b.pop_back();
            merges.push_back({a, b});
            bpe_ranks[a + " " + b] = rank++;
        }
        mf.close();
        LOGV("Tokenizer: loaded %zu merges\n", merges.size());
    }

    // Load vocab and merges from embedded U8 tensors in a SafeTensorsLoader
    void load(const SafeTensorsLoader& loader) {
        init_byte_encoder();

        // Read vocab.json bytes
        auto vit = loader.tensors.find("tokenizer.vocab_json");
        if (vit == loader.tensors.end()) {
            fprintf(stderr, "Tokenizer: no 'tokenizer.vocab_json' tensor found\n");
            exit(1);
        }
        const TensorInfo& vinfo = vit->second;
        std::string vcontent(vinfo.nbytes, '\0');
        {
            size_t data_start = 0;
            for (auto& sf : loader.files)
                if (sf.filepath == vinfo.filename) { data_start = sf.data_start; break; }
            FILE* f = fopen(vinfo.filename.c_str(), "rb");
            fseek(f, (long)(data_start + vinfo.data_offset), SEEK_SET);
            fread(&vcontent[0], 1, vinfo.nbytes, f);
            fclose(f);
        }
        JsonParser parser;
        JsonValue vroot = parser.parse(vcontent);
        for (auto& [key, val] : vroot.object_val)
            vocab[key] = (int32_t)val.to_int();
        LOGV("Tokenizer: loaded %zu vocab entries (from safetensors)\n", vocab.size());

        // Read merges.txt bytes
        auto mit = loader.tensors.find("tokenizer.merges_txt");
        if (mit == loader.tensors.end()) {
            fprintf(stderr, "Tokenizer: no 'tokenizer.merges_txt' tensor found\n");
            exit(1);
        }
        const TensorInfo& minfo = mit->second;
        std::string mcontent(minfo.nbytes, '\0');
        {
            size_t data_start = 0;
            for (auto& sf : loader.files)
                if (sf.filepath == minfo.filename) { data_start = sf.data_start; break; }
            FILE* f = fopen(minfo.filename.c_str(), "rb");
            fseek(f, (long)(data_start + minfo.data_offset), SEEK_SET);
            fread(&mcontent[0], 1, minfo.nbytes, f);
            fclose(f);
        }
        std::istringstream mstream(mcontent);
        std::string line;
        std::getline(mstream, line); // skip header
        int rank = 0;
        while (std::getline(mstream, line)) {
            if (line.empty()) continue;
            size_t space = line.find(' ');
            if (space == std::string::npos) continue;
            std::string a = line.substr(0, space);
            std::string b = line.substr(space + 1);
            if (!b.empty() && b.back() == '\r') b.pop_back();
            merges.push_back({a, b});
            bpe_ranks[a + " " + b] = rank++;
        }
        LOGV("Tokenizer: loaded %zu merges (from safetensors)\n", merges.size());
    }

    // Encode raw bytes to BPE-encoded unicode tokens
    std::string bytes_to_bpe_str(const std::string& text) const {
        std::string result;
        for (unsigned char c : text) {
            auto it = byte_encoder.find(c);
            if (it != byte_encoder.end()) {
                result += it->second;
            }
        }
        return result;
    }

    // Get BPE merge rank
    int get_rank(const std::string& a, const std::string& b) const {
        auto it = bpe_ranks.find(a + " " + b);
        if (it != bpe_ranks.end()) return it->second;
        return INT32_MAX;
    }

    // Apply BPE to a single word (already byte-encoded)
    std::vector<std::string> bpe(const std::string& token) const {
        // Split into individual characters (UTF-8 aware)
        std::vector<std::string> word;
        size_t i = 0;
        while (i < token.size()) {
            size_t len = 1;
            unsigned char c = token[i];
            if ((c & 0x80) == 0) len = 1;
            else if ((c & 0xE0) == 0xC0) len = 2;
            else if ((c & 0xF0) == 0xE0) len = 3;
            else if ((c & 0xF8) == 0xF0) len = 4;
            word.push_back(token.substr(i, len));
            i += len;
        }

        if (word.size() <= 1) return word;

        while (true) {
            // Find the pair with the lowest rank
            int best_rank = INT32_MAX;
            int best_i = -1;
            for (size_t j = 0; j + 1 < word.size(); j++) {
                int r = get_rank(word[j], word[j + 1]);
                if (r < best_rank) {
                    best_rank = r;
                    best_i = (int)j;
                }
            }

            if (best_i < 0) break;

            // Merge the best pair
            std::vector<std::string> new_word;
            for (size_t j = 0; j < word.size(); j++) {
                if ((int)j == best_i) {
                    new_word.push_back(word[j] + word[j + 1]);
                    j++; // skip next
                } else {
                    new_word.push_back(word[j]);
                }
            }
            word = new_word;

            if (word.size() == 1) break;
        }

        return word;
    }

    // --- Unicode-aware pre-tokenization (Qwen2 regex) ---
    // Qwen2 pattern: (?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+

    // Decode UTF-8 to codepoints
    static std::vector<uint32_t> utf8_to_codepoints(const std::string& text) {
        std::vector<uint32_t> cps;
        size_t i = 0;
        while (i < text.size()) {
            uint32_t cp = 0;
            unsigned char c = text[i];
            int len = 1;
            if ((c & 0x80) == 0) { cp = c; len = 1; }
            else if ((c & 0xE0) == 0xC0) { cp = c & 0x1F; len = 2; }
            else if ((c & 0xF0) == 0xE0) { cp = c & 0x0F; len = 3; }
            else if ((c & 0xF8) == 0xF0) { cp = c & 0x07; len = 4; }
            for (int j = 1; j < len && (i + j) < text.size(); j++)
                cp = (cp << 6) | (text[i + j] & 0x3F);
            cps.push_back(cp);
            i += len;
        }
        return cps;
    }

    static std::string codepoint_to_utf8(uint32_t cp) {
        std::string s;
        if (cp < 0x80) {
            s += (char)cp;
        } else if (cp < 0x800) {
            s += (char)(0xC0 | (cp >> 6));
            s += (char)(0x80 | (cp & 0x3F));
        } else if (cp < 0x10000) {
            s += (char)(0xE0 | (cp >> 12));
            s += (char)(0x80 | ((cp >> 6) & 0x3F));
            s += (char)(0x80 | (cp & 0x3F));
        } else {
            s += (char)(0xF0 | (cp >> 18));
            s += (char)(0x80 | ((cp >> 12) & 0x3F));
            s += (char)(0x80 | ((cp >> 6) & 0x3F));
            s += (char)(0x80 | (cp & 0x3F));
        }
        return s;
    }

    static bool is_letter(uint32_t cp) {
        // Basic check: ASCII letters + common Unicode letter ranges
        if ((cp >= 'A' && cp <= 'Z') || (cp >= 'a' && cp <= 'z')) return true;
        // Latin Extended, Greek, Cyrillic, CJK, etc.
        if (cp >= 0xC0 && cp <= 0x024F) return true;   // Latin Extended
        if (cp >= 0x0370 && cp <= 0x03FF) return true;  // Greek
        if (cp >= 0x0400 && cp <= 0x04FF) return true;  // Cyrillic
        if (cp >= 0x4E00 && cp <= 0x9FFF) return true;  // CJK Unified
        if (cp >= 0x3040 && cp <= 0x309F) return true;  // Hiragana
        if (cp >= 0x30A0 && cp <= 0x30FF) return true;  // Katakana
        if (cp >= 0xAC00 && cp <= 0xD7AF) return true;  // Korean Hangul
        if (cp >= 0x0600 && cp <= 0x06FF) return true;  // Arabic
        if (cp >= 0x0900 && cp <= 0x097F) return true;  // Devanagari
        if (cp >= 0x0080 && cp <= 0x00FF && !is_space(cp) && !is_number(cp)) {
            // Latin-1 Supplement letters
            if (cp >= 0xC0 && cp <= 0xFF && cp != 0xD7 && cp != 0xF7) return true;
        }
        return false;
    }

    static bool is_number(uint32_t cp) {
        return (cp >= '0' && cp <= '9') ||
               (cp >= 0x0660 && cp <= 0x0669) ||  // Arabic-Indic digits
               (cp >= 0xFF10 && cp <= 0xFF19);     // Fullwidth digits
    }

    static bool is_space(uint32_t cp) {
        return cp == ' ' || cp == '\t' || cp == '\n' || cp == '\r' ||
               cp == '\f' || cp == '\v' ||
               cp == 0x00A0 || cp == 0x2000 || cp == 0x2001 ||
               cp == 0x2002 || cp == 0x2003 || cp == 0x2004 ||
               cp == 0x2005 || cp == 0x2006 || cp == 0x2007 ||
               cp == 0x2008 || cp == 0x2009 || cp == 0x200A ||
               cp == 0x202F || cp == 0x205F || cp == 0x3000;
    }

    static char to_lower_ascii(char c) {
        if (c >= 'A' && c <= 'Z') return c + 32;
        return c;
    }

    // Qwen2 regex-based pre-tokenization
    static std::vector<std::string> token_split(const std::string& text) {
        std::vector<std::string> tokens;
        auto cps = utf8_to_codepoints(text);
        size_t i = 0;

        while (i < cps.size()) {
            uint32_t cp = cps[i];

            // (?i:'s|'t|'re|'ve|'m|'ll|'d)
            if (cp == '\'' && i + 1 < cps.size()) {
                char next = to_lower_ascii((char)cps[i + 1]);
                if (next == 's' || next == 't' || next == 'm' || next == 'd') {
                    std::string tok = "'";
                    tok += next;
                    tokens.push_back(tok);
                    i += 2;
                    continue;
                }
                if (i + 2 < cps.size()) {
                    char n1 = to_lower_ascii((char)cps[i + 1]);
                    char n2 = to_lower_ascii((char)cps[i + 2]);
                    std::string pair;
                    pair += n1;
                    pair += n2;
                    if (pair == "re" || pair == "ve" || pair == "ll") {
                        std::string tok = "'";
                        tok += pair;
                        tokens.push_back(tok);
                        i += 3;
                        continue;
                    }
                }
            }

            // \p{N} - single digit
            if (is_number(cp)) {
                tokens.push_back(codepoint_to_utf8(cp));
                ++i;
                continue;
            }

            // [^\r\n\p{L}\p{N}]?\p{L}+
            {
                // Non-letter/non-number/non-newline followed by letters
                if (!is_letter(cp) && cp != '\r' && cp != '\n' && !is_number(cp) &&
                    i + 1 < cps.size() && is_letter(cps[i + 1])) {
                    std::string token = codepoint_to_utf8(cp);
                    ++i;
                    while (i < cps.size() && is_letter(cps[i])) {
                        token += codepoint_to_utf8(cps[i]);
                        ++i;
                    }
                    tokens.push_back(token);
                    continue;
                }

                // Just letters
                if (is_letter(cp)) {
                    std::string token = codepoint_to_utf8(cp);
                    ++i;
                    while (i < cps.size() && is_letter(cps[i])) {
                        token += codepoint_to_utf8(cps[i]);
                        ++i;
                    }
                    tokens.push_back(token);
                    continue;
                }
            }

            // ?[^\s\p{L}\p{N}]+[\r\n]*
            {
                // Space followed by punctuation
                if (cp == ' ' && i + 1 < cps.size() &&
                    !is_space(cps[i + 1]) && !is_letter(cps[i + 1]) && !is_number(cps[i + 1])) {
                    std::string token = codepoint_to_utf8(cp);
                    token += codepoint_to_utf8(cps[i + 1]);
                    i += 2;
                    while (i < cps.size() && !is_letter(cps[i]) && !is_number(cps[i]) && !is_space(cps[i])) {
                        token += codepoint_to_utf8(cps[i]);
                        ++i;
                    }
                    while (i < cps.size() && (cps[i] == '\r' || cps[i] == '\n')) {
                        token += codepoint_to_utf8(cps[i]);
                        ++i;
                    }
                    tokens.push_back(token);
                    continue;
                }

                // Punctuation without leading space
                if (!is_letter(cp) && !is_number(cp) && !is_space(cp)) {
                    std::string token = codepoint_to_utf8(cp);
                    ++i;
                    while (i < cps.size() && !is_letter(cps[i]) && !is_number(cps[i]) && !is_space(cps[i])) {
                        token += codepoint_to_utf8(cps[i]);
                        ++i;
                    }
                    while (i < cps.size() && (cps[i] == '\r' || cps[i] == '\n')) {
                        token += codepoint_to_utf8(cps[i]);
                        ++i;
                    }
                    tokens.push_back(token);
                    continue;
                }
            }

            // \s*[\r\n]+|\s+(?!\S)|\s+
            if (is_space(cp)) {
                std::string token;
                bool saw_newline = false;
                while (i < cps.size() && is_space(cps[i])) {
                    token += codepoint_to_utf8(cps[i]);
                    if (cps[i] == '\r' || cps[i] == '\n') {
                        saw_newline = true;
                    } else if (saw_newline) {
                        break;
                    }
                    ++i;
                }
                tokens.push_back(token);
                continue;
            }

            // Fallback: single codepoint
            tokens.push_back(codepoint_to_utf8(cp));
            ++i;
        }

        return tokens;
    }

    // Tokenize text into token IDs using proper Qwen2 regex splitting
    std::vector<int32_t> encode(const std::string& text) const {
        std::vector<int32_t> tokens;

        auto words = token_split(text);

        for (auto& word : words) {
            std::string bpe_str = bytes_to_bpe_str(word);
            auto bpe_tokens = bpe(bpe_str);

            for (auto& tok : bpe_tokens) {
                auto it = vocab.find(tok);
                if (it != vocab.end()) {
                    tokens.push_back(it->second);
                } else {
                    // Unknown token - encode each byte
                    for (unsigned char c : tok) {
                        auto byte_it = byte_encoder.find(c);
                        if (byte_it != byte_encoder.end()) {
                            auto vit = vocab.find(byte_it->second);
                            if (vit != vocab.end()) {
                                tokens.push_back(vit->second);
                            }
                        }
                    }
                }
            }
        }

        return tokens;
    }

    // Format prompt for Qwen-Image chat template with system prompt
    // Returns: { tokens, prompt_template_encode_start_idx }
    // The first prompt_template_encode_start_idx tokens are the system template
    // and should be stripped from the text encoder output.
    std::pair<std::vector<int32_t>, int> tokenize_prompt(const std::string& prompt) const {
        // Format: <|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n
        std::vector<int32_t> tokens;

        // System turn
        tokens.push_back(IM_START);
        auto sys_tokens = encode(std::string("system\n") + SYSTEM_PROMPT);
        tokens.insert(tokens.end(), sys_tokens.begin(), sys_tokens.end());
        tokens.push_back(IM_END);
        auto nl = encode("\n");
        tokens.insert(tokens.end(), nl.begin(), nl.end());

        // User turn start
        tokens.push_back(IM_START);
        auto user_prefix = encode("user\n");
        tokens.insert(tokens.end(), user_prefix.begin(), user_prefix.end());

        int start_idx = (int)tokens.size();

        // User prompt content
        if (!prompt.empty()) {
            auto prompt_tokens = encode(prompt);
            tokens.insert(tokens.end(), prompt_tokens.begin(), prompt_tokens.end());
        }

        // User turn end + assistant turn start
        tokens.push_back(IM_END);
        tokens.insert(tokens.end(), nl.begin(), nl.end());
        tokens.push_back(IM_START);
        auto assistant = encode("assistant\n");
        tokens.insert(tokens.end(), assistant.begin(), assistant.end());

        return {tokens, start_idx};
    }

    // Tokenize empty prompt for unconditional (CFG)
    // Uses single space " " to match diffusers/nunchaku behavior
    std::pair<std::vector<int32_t>, int> tokenize_empty() const {
        return tokenize_prompt(" ");
    }
};
