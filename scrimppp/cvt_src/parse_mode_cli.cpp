#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>
#include <iostream>
#include <cvt_cli.hpp>

namespace po = boost::program_options;

const std::string conversion_mode::_to_bin = "ascii-to-bin";
const std::string conversion_mode::_to_ascii ="bin-to-ascii";

void validate(boost::any& v,
              std::vector<std::string> const & values,
              conversion_mode*, //overloaded argument type
              int
              )
{
        po::validators::check_first_occurrence(v);	// no assignment made so far?
        std::string const & s =po::validators::get_single_string(values); // single value allowed?
        if (s == conversion_mode::_to_bin) {
                v = boost::any(conversion_mode(conversion_mode::TO_BIN));
        }
        else if (s == conversion_mode::_to_ascii) {
                v = boost::any(conversion_mode(conversion_mode::TO_ASCII));
        }
        else {
                throw po::validation_error(po::validation_error::invalid_option_value);
        }
}

conversion_mode parse_mode_cli(int argc, char* argv[])
{
        // will store the user selection of conversion mode
        conversion_mode mode(conversion_mode::TO_BIN);

        //
        po::variables_map varmap;
        po::options_description opt_descr("arguments");
        po::positional_options_description posopt;

        // command line options and dscription
        const char* opt_mode = "conversion mode";
        const char* const opt_help = "help";
        std::stringstream mode_help;
        mode_help << "mode of conversion: " << conversion_mode::_to_bin << " or " << conversion_mode::_to_ascii;

        try{
                // specification of commandline arguments
                opt_descr.add_options()
                        (opt_help, "produce a help message")
                        (opt_mode, po::value<conversion_mode>(& mode)->required(), mode_help.str().c_str())
                        ;
                posopt.add(opt_mode, 1);

                //parse the commandline
                po::store(po::command_line_parser(argc, argv).options(opt_descr).positional(posopt).run(), varmap);

                //validate the specified arguments (presence, type...)
                po::notify(varmap);
        }
        catch (const po::error& exc) {
                std::cerr << "Error parsing the commandline: " << exc.what();
                throw exc;
        }

        return mode;
}
