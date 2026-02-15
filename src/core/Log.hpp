#pragma once

#include <cstdint>
#include <ostream>
#include <sstream>
#include <string>
#include <string_view>

namespace core {

enum class LogLevel : uint8_t {
    Error = 0,
    Warn = 1,
    Info = 2,
    Debug = 3,
    Trace = 4
};

void setLogLevel(LogLevel level);
[[nodiscard]] LogLevel logLevel();
[[nodiscard]] bool shouldLog(LogLevel level);
void initializeLogLevelFromEnvironment();

class LogLine {
public:
    LogLine(LogLevel level, std::string_view category);
    ~LogLine();

    [[nodiscard]] std::ostream& stream();

private:
    LogLevel m_level;
    std::string m_category;
    std::ostringstream m_stream;
};

} // namespace core

#define VOX_LOG_STREAM(level, category) \
    if (!::core::shouldLog(level)) {} else ::core::LogLine((level), (category)).stream()

#define VOX_LOGE(category) VOX_LOG_STREAM(::core::LogLevel::Error, (category))
#define VOX_LOGW(category) VOX_LOG_STREAM(::core::LogLevel::Warn, (category))
#define VOX_LOGI(category) VOX_LOG_STREAM(::core::LogLevel::Info, (category))
#define VOX_LOGD(category) VOX_LOG_STREAM(::core::LogLevel::Debug, (category))
#define VOX_LOGT(category) VOX_LOG_STREAM(::core::LogLevel::Trace, (category))
