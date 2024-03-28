#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>

spdlog::logger &logger() {
    static auto default_logger = spdlog::stdout_color_mt("ITS");
    default_logger->set_pattern("[%^%l%$] %v");
    return *default_logger;
}

int main(){
    logger().info("123");
}