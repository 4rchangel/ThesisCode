#include <cvt_cli.hpp>

#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>
#include <iostream>


namespace po = boost::program_options;

namespace conversion_mode{
    std::istream& operator>>(std::istream& in, conversion_mode::Mode& m){
        std::string token;
        in >> token;
        if (token == conversion_mode::_to_bin){
            m = conversion_mode::TO_BIN;
        }
        else if (token == conversion_mode::_to_ascii){
            m = conversion_mode::TO_ASCII;
        }
        else {
            in.setstate(std::ios_base::failbit);
        }
        return in;
    }
}

conversion_args parse_cvt_cli(int argc, char* argv[])
{
        // will store the user selection of conversion mode
        conversion_args args;

        // variable for cli processing
        po::variables_map varmap;
        po::options_description opt_descr("arguments");
        po::positional_options_description posopt;

        // command line options and dscription
        const char* opt_mode = "conversion_mode";
        const char* const opt_help = "help";
        const char* const opt_input = "input file path";
        const char* const opt_output = "output file path";
        const std::string output_default = "converted";
        std::stringstream mode_help;
        mode_help << "mode of conversion: " << conversion_mode::_to_bin << " or " << conversion_mode::_to_ascii;

        try{
                // specification of commandline arguments
                opt_descr.add_options()
                        (opt_help, "produce a help message")
                        (opt_mode, po::value<conversion_mode::Mode>(& args._mode)->required(), mode_help.str().c_str())
                        (opt_input, po::value<std::string>(& args._input_path)->required(), "path to the input file")
                        (opt_output, po::value<std::string>(& args._output_path)->default_value(output_default), "path to the output file")
                        ;
                posopt.add(opt_mode, 1);
                posopt.add(opt_input, 1);

                //parse the commandline
                po::store(po::command_line_parser(argc, argv).options(opt_descr).positional(posopt).run(), varmap);

                if (varmap.count(opt_help) > 0) {
                        std::cout << opt_descr << std::endl;
                        throw std::runtime_error("help requested");
                }

                //validate the specified arguments (presence, type...)
                po::notify(varmap);
        }
        catch (const po::error& exc) {
                std::cerr << "Error parsing the commandline: " << exc.what() << std::endl;
                std::cerr << "Command help: " << opt_descr << std::endl;
                throw exc;
        }

        if (!boost::filesystem::exists(args._input_path)){
            throw std::runtime_error("Input file not existing!");
        }

        if (args._output_path == output_default){
            switch (args._mode) {
                case conversion_mode::TO_ASCII:
                    args._output_path.append(".ascii");
                    break;
                case conversion_mode::TO_BIN:
                    args._output_path.append(".bin");
                    break;
            }
        }

        return args;
}
