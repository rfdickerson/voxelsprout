#include "core/log.h"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <mutex>

namespace voxelsprout::core {
namespace {

std::atomic<LogLevel> g_logLevel{LogLevel::Info};
std::once_flag g_envInitOnce;
std::mutex g_logWriteMutex;

const char* levelName(LogLevel level) {
    switch (level) {
    case LogLevel::Error:
        return "error";
    case LogLevel::Warn:
        return "warn";
    case LogLevel::Info:
        return "info";
    case LogLevel::Debug:
        return "debug";
    case LogLevel::Trace:
        return "trace";
    default:
        return "info";
    }
}

std::string makeTimestamp() {
    const auto now = std::chrono::system_clock::now();
    const auto epochMs = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch());
    const auto ms = static_cast<int>(epochMs.count() % 1000);
    const std::time_t timeValue = std::chrono::system_clock::to_time_t(now);

    std::tm localTime{};
#if defined(_WIN32)
    localtime_s(&localTime, &timeValue);
#else
    localtime_r(&timeValue, &localTime);
#endif

    char buffer[32]{};
    std::snprintf(
        buffer,
        sizeof(buffer),
        "%02d:%02d:%02d.%03d",
        localTime.tm_hour,
        localTime.tm_min,
        localTime.tm_sec,
        ms);
    return std::string(buffer);
}

void writeLine(LogLevel level, std::string_view category, std::string_view message) {
    std::lock_guard<std::mutex> lock(g_logWriteMutex);
    std::ostream& out = (level == LogLevel::Error || level == LogLevel::Warn) ? std::cerr : std::cout;
    out << "[" << makeTimestamp() << "]";
    if (!category.empty()) {
        out << "[" << category << "]";
    }
    if (level != LogLevel::Info) {
        out << "[" << levelName(level) << "]";
    }
    out << " " << message << "\n";
}

std::string toLower(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](const unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return value;
}

LogLevel parseLogLevel(const std::string& text, LogLevel fallback) {
    const std::string normalized = toLower(text);
    if (normalized == "error" || normalized == "err" || normalized == "0") {
        return LogLevel::Error;
    }
    if (normalized == "warn" || normalized == "warning" || normalized == "1") {
        return LogLevel::Warn;
    }
    if (normalized == "info" || normalized == "2") {
        return LogLevel::Info;
    }
    if (normalized == "debug" || normalized == "3") {
        return LogLevel::Debug;
    }
    if (normalized == "trace" || normalized == "4") {
        return LogLevel::Trace;
    }
    return fallback;
}

} // namespace

void setLogLevel(LogLevel level) {
    g_logLevel.store(level);
}

LogLevel logLevel() {
    initializeLogLevelFromEnvironment();
    return g_logLevel.load();
}

bool shouldLog(LogLevel level) {
    return static_cast<int>(level) <= static_cast<int>(logLevel());
}

void initializeLogLevelFromEnvironment() {
    std::call_once(g_envInitOnce, []() {
        const char* envValue = nullptr;
#if defined(_WIN32)
        std::size_t requiredLength = 0;
        char* envBuffer = nullptr;
        if (_dupenv_s(&envBuffer, &requiredLength, "VOXEL_LOG_LEVEL") != 0 || envBuffer == nullptr) {
            return;
        }
        envValue = envBuffer;
#else
        envValue = std::getenv("VOXEL_LOG_LEVEL");
#endif
        if (envValue == nullptr || envValue[0] == '\0') {
#if defined(_WIN32)
            std::free(envBuffer);
#endif
            return;
        }
        setLogLevel(parseLogLevel(envValue, LogLevel::Info));
#if defined(_WIN32)
        std::free(envBuffer);
#endif
    });
}

LogLine::LogLine(LogLevel level, std::string_view category)
    : m_level(level), m_category(category) {}

LogLine::~LogLine() {
    std::string line = m_stream.str();
    while (!line.empty() && (line.back() == '\n' || line.back() == '\r')) {
        line.pop_back();
    }
    writeLine(m_level, m_category, line);
}

std::ostream& LogLine::stream() {
    return m_stream;
}

} // namespace voxelsprout::core
