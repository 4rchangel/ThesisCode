#ifndef BINTSFILE_HPP
#define BINTSFILE_HPP

#include <biniobase.h>

#include <settings.h>
#include <cstddef>
#include <string>
#include <mpi.h>
#include <memory>

class BinTsFileMPI : public BinIOBase
{
    public:
	    const unsigned int _fmt_version = 2; //should be changed if anything in the file layout /behavior changes to detect inconsistencies
		using storage_type = double;
		const MPI_Datatype mpi_storage_type = MPI_DOUBLE;

		BinTsFileMPI(const std::string& path, const MPI_Comm comm = MPI_COMM_SELF, const MPI_Info& fileinfo = MPI_INFO_NULL);
		virtual ~BinTsFileMPI();

		// I/O methods. Using raw pointers to maintain flexibility regarding the actual type (aligned vector, general vector, array...) without neessity for templating
		    // read a subsequence from the file, collective operation of all processes within the communicator. Though every proc may specify a custom slice offset and length (and buffer accordingly)
		    // thus it is required, that every process in the communicator specified at opening time participates
		void write_ts_slice(const matrix_profile::tsa_dtype * const slicebuffer, const int total_ts_length, int slicelen, int sliceoffset, const bool write_header = true);
		void write_ts(const matrix_profile::tsa_dtype * const tsbuffer, const long total_length); // wrapper for writing the full ts in parallel, in case everybody has access to the full series

		void read_ts_slice(matrix_profile::tsa_dtype * const slicebuffer, const int slicelen, const int sliceoffset); //"serial" reading of a data slice with MPI I/O
		int read_total_ts(matrix_profile::tsa_dtype * const tsbuffer, const size_t maxlen, const bool collective_read); //distributed reading and data exchange

		unsigned long get_ts_len();
		virtual unsigned int get_file_format_version();

    protected:
		void read_ts_slice(matrix_profile::tsa_dtype * const slicebuffer, const int slicelen, const int sliceoffset, const int tslen); //"serial" reading of a data slice with MPI I/O
};

#endif // BINTSFILE_HPP
