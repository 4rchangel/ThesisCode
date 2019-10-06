#include <cvt_cli.hpp>
#include <settings.h>
#include <logging.hpp>
#include <bintsfile.hpp>

#include <mpi.h>

#include <iostream>
#include <fstream>
#include <cassert>
#include <vector>
#include <boost/program_options.hpp>


using namespace matrix_profile;

void convert_ts_to_bin(const conversion_args&);
void convert_ts_to_ascii(const conversion_args&);

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
			    convert_ts_to_bin(args);
			    break;
		    case conversion_mode::TO_ASCII:
			    convert_ts_to_ascii(args);
			    break;
		}
	}
	catch (std::exception e) {
		MPI_Finalize();
		throw e;
	}

	MPI_Finalize();
	return 0;
}

void convert_ts_to_bin(const conversion_args& args) {
	assert(args._mode == conversion_mode::TO_BIN);
	std::ifstream input(args._input_path);
	BinTsFileMPI output(args._output_path, MPI_COMM_SELF); // we gonna loop sequqntially thus COMM_SELF

	matrix_profile::tsa_dtype next_val;
	const size_t CHUNK_SIZE = 10000;
	size_t chunk_offset = 0;
	std::vector<tsa_dtype> inbuf;

	inbuf.reserve(CHUNK_SIZE);
	input >> next_val;
	int write_ctr = 0;
	while (input.good()){
		inbuf.clear();
		while (input.good() && inbuf.size() < CHUNK_SIZE) {
			inbuf.push_back(next_val);
			input >> next_val;
		}

		output.write_ts_slice(inbuf.data(), chunk_offset+inbuf.size(), inbuf.size(), chunk_offset, true);
		chunk_offset += inbuf.size();
	}
	EXEC_INFO("written " << chunk_offset << " sample values");
}


void convert_ts_to_ascii(const conversion_args&){
	throw std::runtime_error("Not yet implemented");
}
