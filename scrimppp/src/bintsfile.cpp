#include <bintsfile.hpp>
#include <logging.hpp>
#include <partitioning_1d.h>

#include <vector>
#include <limits>
#include <assert.h>

#include <boost/assert.hpp>

using namespace matrix_profile;

BinTsFileMPI::BinTsFileMPI(const std::string& path, const MPI_Comm comm, const MPI_Info& fileinfo)
    : BinIOBase(path, comm, fileinfo)
{
}

BinTsFileMPI::~BinTsFileMPI() {
}

unsigned int BinTsFileMPI::get_file_format_version() {
	return _fmt_version;
}

unsigned long BinTsFileMPI::get_ts_len() {
	return get_series_len();
}

// I/O methods. Using raw pointers to maintain flexibility regarding the actual type (aligned vector, general vector, array...) without neessity for templating
    // read a subsequence from the file, collective operation of all processes within the communicator. Though every proc may specify a custom slice offset and length (and buffer accordingly)
void BinTsFileMPI::write_ts_slice(const matrix_profile::tsa_dtype * const slicebuffer, int total_ts_length, const int slicelen, const int sliceoffset, const bool write_header) {
	MPI_Datatype filetype;
	assert(total_ts_length > 0);

	EXEC_DEBUG("Writing to file view");
	static_assert( std::is_same<storage_type, matrix_profile::tsa_dtype>::value, "storage data type and computation type do not match, conversion required" ) ;
	write_series_slice(slicebuffer, total_ts_length, slicelen, sliceoffset, mpi_storage_type, write_header);
}

void BinTsFileMPI::write_ts(const tsa_dtype * const tsbuffer, const long total_length) {
	static_assert( std::is_same<storage_type, matrix_profile::tsa_dtype>::value, "storage data type and computation type do not match, conversion required" ) ;
	write_total_series(tsbuffer, total_length, mpi_storage_type);
}

void BinTsFileMPI::read_ts_slice(matrix_profile::tsa_dtype * const slicebuffer, const int slicelen, const int sliceoffset, const int ts_len) {
	if (ts_len < sliceoffset+slicelen){
		throw std::runtime_error("File contains less samples than requested");
	}

	EXEC_DEBUG("Reading part of the TS from a file view:" << slicelen <<" samples starting at " << sliceoffset );
	static_assert( std::is_same<storage_type, matrix_profile::tsa_dtype>::value, "File datatype and computation type do not match, onversion required");
	read_series_slice_collective(slicebuffer, slicelen, sliceoffset, ts_len, mpi_storage_type);
}

void BinTsFileMPI::read_ts_slice(matrix_profile::tsa_dtype * const slicebuffer, const int slicelen, const int sliceoffset) {
	HeaderInfo header = read_header_and_check_fmt();
	if (header._num_samples < sliceoffset+slicelen){
		throw std::runtime_error("File contains less samples than requested");
	}
	if (header._num_samples > std::numeric_limits<int>::max()){
		throw std::runtime_error("Number of time series samples in file exceeds maximum treatable number.");
	}

	read_ts_slice(slicebuffer, slicelen, sliceoffset, header._num_samples);
}

int BinTsFileMPI::read_total_ts(matrix_profile::tsa_dtype * const tsbuffer, const size_t maxlen, const bool collective_read){
	static_assert( std::is_same<storage_type, matrix_profile::tsa_dtype>::value, "File datatype and computation type do not match, onversion required");

	if (collective_read) {
		return read_total_series_collective(tsbuffer, maxlen, mpi_storage_type);
	}
	else {
		return read_total_series_individually(tsbuffer, maxlen, mpi_storage_type);
	}

}
