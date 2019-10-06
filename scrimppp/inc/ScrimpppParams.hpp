#ifndef SCRIMPPP_HPP
#define SCRIMPPP_HPP

namespace matrix_profile {

struct Scrimppp_params {
	std::string algo_name;
	std::string time_series_filename;
	std::string output_filename;
	int query_window_len;
	int prescrimp_stride;
	int block_length;
	enum FILETYPE {BY_SUFFIX, ASCII, BIN} filetype;
	bool use_distributed_io;
	int writing_chunk_size;

	Scrimppp_params() :
	    algo_name(""),
	    time_series_filename(""),
	    output_filename("matprof"),
	    query_window_len(0),
	    prescrimp_stride(0),
	    block_length(500),
	    filetype(BY_SUFFIX),
	    use_distributed_io(true),
	    writing_chunk_size(0)
	{}
};

} //namespace matrix_profile
#endif // SCRIMPPP_H
