#ifndef ITS_LOGGER_HPP
#define ITS_LOGGER_HPP

#include "Config.hpp"
#include <spdlog/logger.h>
#include <spdlog/sinks/stdout_color_sinks.h>

NAMESPACE_BEGIN(ITS)
    namespace utils {
        FORCE_INLINE
        spdlog::logger &logger() {
            static auto default_logger = spdlog::stdout_color_mt("ITS");
            default_logger->set_pattern("[%^%l%$] %v");
            return *default_logger;
        }
    } // namespace utils
NAMESPACE_END(ITS)

#endif //ITS_LOGGER_HPP
