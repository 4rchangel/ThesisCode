#include <cvt_cli.hpp>
#include <settings.h>
#include <logging.hpp>
#include <ScrimpSequ.hpp>
#include <binproffile.h>

#include <mpi.h>

#include <iostream>
#include <fstream>
#include <cassert>
#include <vector>
#include <boost/program_options.hpp>
using namespace matrix_profile;

void convert_profile_to_bin(const conversion_args& args);
void convert_profile_to_ascii(const conversion_args& args);

int main(int argc, char* argv[])
{
	MPI_Init(&argc, &argv);
	logging::init();

	conversion_args args;

	try {
		 args = parse_cvt_cli(argc, argv);
	}
	catch (boost::program_options::error poerr) {
		std::cerr << "Aborting due to invalid invocation..." << std::endl; // detailed info printed previously
		return 1;
	}
	catch (std::runtime_error exc)	{
		std::cerr << "Exiting after cli parsing error: " << std::endl << exc.what() << std::endl;
		return 1;
	}

	try{
		switch(args._mode){
		    case conversion_mode::TO_BIN:
			    convert_profile_to_bin(args);
			    break;
		    case conversion_mode::TO_ASCII:
			    convert_profile_to_ascii(args);
			    break;
		}
	}
	catch (std::runtime_error e) {
		std::cerr << "Runtime ERROR: " << e.what() << std::endl;
		MPI_Finalize();
		return -1;
	}
	catch (std::exception e) {
		std::cerr << "Conversion FAILED due to a exception:" << e.what() << std::endl;
		MPI_Finalize();
		return -1;
	}

	MPI_Finalize();
	return 0;
}

void convert_profile_to_bin(const conversion_args &args) {

}

void convert_profile_to_ascii(const conversion_args &args) {
	BinProfFile infile(args._input_path);
	const size_t proflen = infile.get_profile_length();
	std::vector<matrix_profile::tsa_dtype> prof(proflen);
	std::vector<matrix_profile::idx_dtype> idx(proflen);
	Scrimppp_params params;
	params.output_filename = args._output_path;
	    // set the remaining args to unknown values. Acutually they should not have a impact...
	params.algo_name = "unknown";
	params.filetype = Scrimppp_params::ASCII;
	params.prescrimp_stride = 0;
	params.query_window_len = 0;

	//TODO: better read in chunks for profeils with excessive lengths ( BinProfFile::read_ts_slice )
	infile.read_matrix_profile( prof.data(), idx.data(), proflen, true);
	ScrimpSequ::store_matrix_profile(prof, idx, params);
}
