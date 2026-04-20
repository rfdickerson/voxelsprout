#pragma once

#include <cstdint>
#include <ostream>
#include <sstream>
#include <string>
#include <string_view>

namespace odai::core {

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

} // namespace odai::core

#define VOX_LOG_STREAM(level, category) \
    if (!::odai::core::shouldLog(level)) {} else ::odai::core::LogLine((level), (category)).stream()

#define VOX_LOGE(category) VOX_LOG_STREAM(::odai::core::LogLevel::Error, (category))
#define VOX_LOGW(category) VOX_LOG_STREAM(::odai::core::LogLevel::Warn, (category))
#define VOX_LOGI(category) VOX_LOG_STREAM(::odai::core::LogLevel::Info, (category))
#define VOX_LOGD(category) VOX_LOG_STREAM(::odai::core::LogLevel::Debug, (category))
#define VOX_LOGT(category) VOX_LOG_STREAM(::odai::core::LogLevel::Trace, (category))
