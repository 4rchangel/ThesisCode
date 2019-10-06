#include <iostream>
#include <sstream>

#include <boost/program_options.hpp>
#include <mpi.h>

#include <ScrimpppParams.hpp>
#include <MatProfAlgo.hpp>
#include <logging.hpp>
#include <papiwrapper.hpp>
#include <settings.h>
#include <boost/filesystem.hpp>
#include <timing.h>

#ifdef LIKWID_PERFMON
    #include <likwid.h>
#endif

using namespace matrix_profile;

bool parse_args(const int argc, const char* const argvc[], Scrimppp_params& params_out);
void log_args(const Scrimppp_params& params);
void log_build_settings();

int main(int argc, char* argv[])
{
	MPI_Init(&argc, &argv);
	Timepoint starting_time = get_cur_time(); // wall-clock time
	logging::init();
	LIKWID_MARKER_INIT;
	//LIKWID_MARKER_THREADINIT for OpenMP TODO: check if this works y default, too...

#ifdef NDEBUG
	EXEC_INFO ( "Running a release build")
#else
	EXEC_INFO ( "Running a DEBUG build!" )
#endif

	if (SYNC_AFTER_STARTUP) {
		EXEC_INFO( "Syhnchonizing after startup!" );
	}
	else {
		EXEC_DEBUG( "No startup synchronization" );
	}

	// parse the command line arguments into parameters for the algorithm
	Scrimppp_params cmd_params;
	if (parse_args(argc, argv, cmd_params)) {
		std::unique_ptr<IMatProfAlgo> mat_prof_algo;

		//retrieve a algorithm instance from the factory
		try {
			mat_prof_algo = MatProfImplFactory::getInstance().create(cmd_params.algo_name);
			EXEC_INFO( "Instantiated algorithm " << cmd_params.algo_name );
		}
		catch (std::exception e) {
			//TODO: this could be integrated into the boos::program_options parsing
			EXEC_ERROR( "specified algorithm \"" << cmd_params.algo_name << "\" is not known. Valid algorithms: " );
			EXEC_ERROR( MatProfImplFactory::getInstance().get_available_implementations() );
			return -1;
		}

		// log the invocation parameters
		log_args(cmd_params);
		log_build_settings();

		// run the algorithm
		EXEC_INFO( "computing the matrix profile now...")
		if (SYNC_AFTER_STARTUP) {
			EXEC_INFO("Sync before invoking computation...");
			MPI_Barrier(MPI_COMM_WORLD);
		}
		const Timepoint init_starting_time = get_cur_time(); // wall-clock time
		mat_prof_algo->initialize(cmd_params);
		const Timepoint init_finish_time = get_cur_time(); // wall-clock time
		if (SYNC_AFTER_STARTUP && IGNORE_INIT_TIME) {
			MPI_Barrier(MPI_COMM_WORLD);
		}
		const Timepoint comp_starting_time = get_cur_time(); // wall-clock time
		mat_prof_algo->compute_matrix_profile(cmd_params);
		const Timepoint finish_time = get_cur_time(); // wall-clock time
		EXEC_TRACE( "finished matrix profile computation")

		Timespan runtime;
		if (IGNORE_INIT_TIME) {
			runtime = finish_time-comp_starting_time;
		}
		else {
			runtime = finish_time-init_starting_time;
		}

		mat_prof_algo->log_info();
		PERF_LOG( "(incl startup) overall runtime: " << finish_time-init_starting_time);
		PERF_LOG( "total runtime: " << runtime);
		PERF_LOG( "startup time: " << init_starting_time-starting_time);
		PERF_LOG( "init time: " << init_finish_time-init_starting_time);
		PERF_LOG( "algo time: " << finish_time-comp_starting_time);
		PERF_LOG( "init_idle time: " << comp_starting_time-init_finish_time);
	}
	else {
		//error parsing the command line options
		return -1;
	}
	LIKWID_MARKER_CLOSE;
	MPI_Finalize();
	return 0;
}

void log_args(const Scrimppp_params& params) {
	// always enforce logging, thus use the error log fn.
	EXEC_ERROR( " parameter summary:"
	           << " algo: "<< params.algo_name
	           << " winlen: " << params.query_window_len
	           << " prescrimp_stride: " << params.prescrimp_stride
	           << " inputfile: " << params.time_series_filename
	           << " blocklen: " << params.block_length
	           << " distributed I/O: " << params.use_distributed_io
	           << " writing chunk size " << params.writing_chunk_size
	);
}

void log_build_settings() {

#ifdef USE_INTRINSICS_KERNEL
	bool using_intrinsics_kernel = true;
#else
	bool using_intrinsics_kernel = false;
#endif

	EXEC_ERROR("BUILD settings: "
	        << "size(tsadtype) /bytes: " << sizeof(matrix_profile::tsa_dtype)
	        << "\n  size(idx_dtype) /byte: " << sizeof(matrix_profile::idx_dtype)
	        << "\n  SYNC_AFTER_STARTUP: " << SYNC_AFTER_STARTUP
	        << "\n  COLLECTIVE_DISTRIB_READ: " << COLLECTIVE_DISTRIB_READ
	        << "\n  DISTRIB_BCAST_INPUT: " << DISTRIB_BCAST_INPUT
	        << "\n  IGNORE_INIT_TIME: " << IGNORE_INIT_TIME
	        << "\n  MAX_PROFILE_LENGTH: " << MP_MAX_PROFILE_LENGTH
	        << "\n  MP_BLOCKLENGTH: " << MP_BLOCKLENGTH
	        << "\n  USING_INTRINSICS_KERNEL: " << using_intrinsics_kernel
           #ifdef __AVX2__
	        << "\n built with AVX2"
           #endif
	    );
}

namespace po = boost::program_options;

namespace matrix_profile {
    std::istream& operator>>(std::istream& in, Scrimppp_params::FILETYPE& t){
		std::string token;
		in >> token;
		if (token == std::string("BIN") ){
			t = Scrimppp_params::BIN;
		}
		else if (token == std::string("ASCII")){
			t = Scrimppp_params::ASCII;
		}
		else if (token == std::string("BY_SUFFIX")){
			t = Scrimppp_params::BY_SUFFIX;
		}
		else {
			EXEC_ERROR("invalid filetype " << token << "specified");
			in.setstate(std::ios_base::failbit);
		}
		return in;
	}
}

/**
 * @brief parse command line arguments into parameters for the scrimp++ algorithm
 * @param argc argument count as in main
 * @param argv argument values as in main
 * @param params parameters for the scrimp++ algo as specified on the caommand line
 * @return true if all required parameters to run scrimp++ had been parsed successfully
 */
bool parse_args(const int argc, const char* const argv[], Scrimppp_params& params)
{
	bool wait_debugger = false;
	po::variables_map varmap;
	po::options_description opt_descr("Allowed options");
	po::positional_options_description posopt;

	// definition of actual command line options
	const char* const opt_help = "help";
	const char* const opt_algo = "algo";
	const char* const opt_infile = "input-file";
	const char* const opt_outfile = "output-file";
	const char* const opt_winlen = "window-length";
	const char* const opt_prescr_stride = "prescrimp-stride";
	const char* const opt_dbg = "dbg";
	const char* const opt_blocklen = "blocklen";
	const char* const opt_filetype = "filetype";
	const char* const opt_distributed_io ="dist-io";
	const char* const opt_chunksize="chunksize";

	std::stringstream algo_help;
	algo_help << "name of the algorithm to run. Available: ";
	algo_help << MatProfImplFactory::getInstance().get_available_implementations();

	try  {
		//compose a description of command-line options
		opt_descr.add_options()
		        (opt_help, "produce a help message")
		        (opt_algo, po::value<std::string>(& (params.algo_name))->required(), algo_help.str().c_str())
		        (opt_infile, po::value<std::string>(& (params.time_series_filename))->required(), "file containing time series data: a series of n doubles in ASCII format, spearated by whitespaces")
		        (opt_winlen, po::value<int>(& (params.query_window_len))->required(), "size m of the matching window. I.e. the supposed motif length")
		        (opt_prescr_stride, po::value<int>(& (params.prescrimp_stride))->required(), "stride of the approximate prescrimp algorithm. Typical: window-length/4")
		        (opt_dbg, po::bool_switch(&wait_debugger)->default_value(false), "wait for user input after program start, providing the chance to attach a debugger")
		        (opt_blocklen, po::value<int>(& (params.block_length))->default_value(MP_BLOCKLENGTH), "block size for improved cache locality")
		        (opt_outfile, po::value<std::string>(& (params.output_filename))->default_value(""), "filename (/path) for storing the result")
		        (opt_filetype, po::value<Scrimppp_params::FILETYPE>(& (params.filetype))->default_value(Scrimppp_params::BY_SUFFIX), "type of file: BIN, ASCII oder automatically selected by SUFFIX")
		        (opt_distributed_io, po::value<bool>(& (params.use_distributed_io))->default_value(true), "use distributed MPI I/O. Only operational for parallel algorithms and binary I/O")
		        (opt_chunksize, po::value<int>( &params.writing_chunk_size)->default_value(0), "chunksize for binary file output (if used)")
		        ;
		//enable specification of arguments just by position
		posopt.add(opt_algo, 1).add(opt_winlen, 1).add(opt_prescr_stride, 1).add(opt_infile, 1);

		//parse the command line
		po::store(po::command_line_parser(argc, argv).options(opt_descr).positional(posopt).run(), varmap);

		//if requested, print the help
		if (varmap.count(opt_help) > 0) {
			std::cout << opt_descr << std::endl;
			std::cout << "Compile time parameter settings of the algorithm: " << std::endl
			          << "  MP_MAX_PROFILE_LENGTH: " << MP_MAX_PROFILE_LENGTH << std::endl
			          << "  default MP_BLOCK_LENGTH: " << MP_BLOCKLENGTH << std::endl;
			return false;
		}

		//validate the specified arguments (presence, type...)
		po::notify(varmap);
	}
	catch (const po::error& exc)
	{
		EXEC_ERROR( "ERROR parsing commandline arguments: " << exc.what() );
		EXEC_ERROR( opt_descr << std::endl );
		return false;
	}
	// determine the I/O mode
	if (params.filetype == Scrimppp_params::BY_SUFFIX) {
		if (params.time_series_filename.rfind(".bin") != std::string::npos) {//TODO: better check really only for the extension
			params.filetype = Scrimppp_params::BIN;
			EXEC_INFO("Selected BINARY I/O by TS file extension");
		}
		else if (params.time_series_filename.rfind(".ascii") != std::string::npos) {//TODO: same as above
			params.filetype = Scrimppp_params::ASCII;
			EXEC_INFO("Selected ASCII I/O by TS file extension");
		}
		else {
			throw std::runtime_error("file type could not be determined from suffix. Specify explicitly or rename the file...");
		}
	}

	// if not output file name was specified, choose the default one:
	if (params.output_filename.empty()) {
		std::stringstream name_stream;
		name_stream << params.algo_name << "_new_MatrixProfile_";
		name_stream << params.query_window_len;
		name_stream << boost::filesystem::path(params.time_series_filename).filename().string() << ".matprof";
		if (params.filetype == Scrimppp_params::BIN) {
			name_stream << ".bin";
		}
		if (params.filetype == Scrimppp_params::ASCII) {
			name_stream << ".ascii";
		}
		params.output_filename = name_stream.str();
	}



	if (wait_debugger) {
		std::cout << "Waiting for user input before proceeding. Chance to attach a debugger..." << std::endl << "Press ENTER to continue...";
		std::cout.flush();
		MPI_Barrier(MPI_COMM_WORLD);
		std::cin.get();
	}
	return true;
}
