#include "logging.hpp"
#include <iostream>

using namespace matrix_profile::logging;

std::ostream& matrix_profile::logging::exec_logger = std::cout;
std::ostream& matrix_profile::logging::perf_logger = std::clog;

void matrix_profile::logging::init() {

}
