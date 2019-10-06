#ifndef CVT_CLI_HPP
#define CVT_CLI_HPP

#include <string>

/*
 * custom struct and validator for commandline argument
*/
namespace conversion_mode {
        enum Mode{TO_ASCII, TO_BIN};

        static const std::string _to_bin = "ascii-to-bin";
        static const std::string _to_ascii ="bin-to-ascii";
};

struct conversion_args {
    conversion_mode::Mode _mode;
    std::string _input_path;
    std::string _output_path;
};

// command line parser
conversion_args parse_cvt_cli(int argc, char* argv[]);

#endif // CVT_CLI_HPP
