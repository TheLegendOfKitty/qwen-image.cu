#pragma once

#include <cstdlib>
#include <cstdio>
#include <algorithm>
#include <chrono>
#include <string>

inline bool qwen_verbose_enabled() {
    const char* v = std::getenv("QWEN_VERBOSE");
    return v != nullptr && v[0] != '\0' && !(v[0] == '0' && v[1] == '\0');
}

#define LOGV(...) \
    do { \
        if (qwen_verbose_enabled()) { \
            std::fprintf(stderr, __VA_ARGS__); \
        } \
    } while (0)

struct QwenProgressState {
    bool active = false;
    std::string label;
    int total = 1;
    int current = 0;
    std::chrono::steady_clock::time_point start;
};

inline QwenProgressState& qwen_progress_state() {
    static QwenProgressState state;
    return state;
}

inline std::string qwen_format_seconds(double seconds) {
    if (seconds < 0.0) seconds = 0.0;
    int s = (int)(seconds + 0.5);
    int m = s / 60;
    s %= 60;
    char buf[32];
    if (m > 0) {
        std::snprintf(buf, sizeof(buf), "%dm %02ds", m, s);
    } else {
        std::snprintf(buf, sizeof(buf), "%ds", s);
    }
    return std::string(buf);
}

inline void qwen_progress_render(bool finish_line) {
    auto& st = qwen_progress_state();
    if (!st.active) return;

    const int width = 26;
    int filled = st.total > 0 ? (st.current * width / st.total) : width;
    filled = std::max(0, std::min(width, filled));
    std::string bar(width, '-');
    for (int i = 0; i < filled; ++i) bar[i] = '=';
    if (!finish_line && filled < width) bar[filled] = '>';

    auto now = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration<double>(now - st.start).count();
    int pct = st.total > 0 ? (100 * st.current / st.total) : 100;
    std::fprintf(stderr, "\r%s [%s] %d/%d (%3d%%) | elapsed %s",
                 st.label.c_str(), bar.c_str(), st.current, st.total, pct,
                 qwen_format_seconds(elapsed).c_str());
    if (finish_line) {
        std::fprintf(stderr, "\n");
    }
    std::fflush(stderr);
}

inline void qwen_progress_begin(const std::string& label, int total) {
    if (qwen_verbose_enabled()) return;
    auto& st = qwen_progress_state();
    if (st.active) {
        qwen_progress_render(true);
        st.active = false;
    }
    st.active = true;
    st.label = label;
    st.total = std::max(1, total);
    st.current = 0;
    st.start = std::chrono::steady_clock::now();
    qwen_progress_render(false);
}

inline void qwen_progress_tick(int delta = 1) {
    if (qwen_verbose_enabled()) return;
    auto& st = qwen_progress_state();
    if (!st.active) return;
    st.current = std::min(st.total, st.current + std::max(1, delta));
    qwen_progress_render(st.current >= st.total);
    if (st.current >= st.total) {
        st.active = false;
    }
}

inline void qwen_progress_end() {
    if (qwen_verbose_enabled()) return;
    auto& st = qwen_progress_state();
    if (!st.active) return;
    st.current = st.total;
    qwen_progress_render(true);
    st.active = false;
}
