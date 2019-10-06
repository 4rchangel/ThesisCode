#ifndef BINIOBASE_H
#define BINIOBASE_H

#include <settings.h>
#include <cstddef>
#include <string>
#include <mpi.h>
#include <memory>

class BinIOBase
{
    public:
	        const unsigned int _base_fmt_version = 1; //should be changed if anything in the file layout /behavior changes to detect inconsistencies
			const unsigned int _header_size = 100 * sizeof(char); //verbose but intuitive
			const char* const _mpi_file_representation = "native"; //"external32"; // lrz system (intel mpi) is stuck when usting external32,... thus just use the default

			struct HeaderInfo {
				        unsigned int _fmt_version;
						unsigned int _data_offset; // offset of the beginning of the actual time series. Gives some flexibility for additional header stuff. May be assumed constant, stored for consistency&convenience
						unsigned long _num_samples;
			    };

			BinIOBase(const std::string& path, const MPI_Comm comm = MPI_COMM_WORLD, const MPI_Info& fileinfo = MPI_INFO_NULL);
			virtual ~BinIOBase();

			    // allow exlicit opening/closing
			void open_write();
			void open_read(const bool check_header=true);
			void close();
			HeaderInfo read_header_and_check_fmt();

    protected:
			// I/O methods. Using raw pointers to maintain flexibility regarding the actual type (aligned vector, general vector, array...) without neessity for templating
			    // read a subsequence from the file, collective operation of all processes within the communicator. Though every proc may specify a custom slice offset and length (and buffer accordingly)
			    // thus it is required, that every process in the communicator specified at opening time participates
			void write_header(const size_t total_ts_length, const bool perform_write);
			void write_series_slice(const void * const slicebuffer, const int total_ts_length, int slicelen, int sliceoffset, const MPI_Datatype etype, const bool write_header = true);
			void write_total_series(const void * const tsbuffer, const long total_length, MPI_Datatype etype); // wrapper for writing the full ts in parallel, in case everybody has access to the full series
			void read_series_slice_collective(void * const slicebuffer, const int slicelen, const int sliceoffset, const MPI_Datatype etype); //"serial" reading of a data slice with MPI I/O
			int read_total_series_collective(void * const tsbuffer, const size_t maxlen, const MPI_Datatype etype); //distributed reading and data exchange
			int read_total_series_individually(void * const tsbuffer, const size_t maxlen, const MPI_Datatype etype); // using individual I/O

			unsigned long get_series_len();

			void read_series_slice_collective(void * const slicebuffer, const int slicelen, const int sliceoffset, const int ts_len, const  MPI_Datatype etype);//WARNING: does not read and check the header but assumes a correct format and the ts_len to be correct! May only be used to avoid redunant reading of the header.
			MPI_Datatype compose_header_type() const;

			virtual unsigned int get_file_format_version() = 0;

    protected:
			const std::string _path;
			enum AccessMode {READ, WRITE, FULL, UNDEFINED} _access_mode;
			MPI_File _file_handle;
			MPI_Comm _comm;
			int _rank;
			std::unique_ptr<HeaderInfo> _header;
			const MPI_Info _fileinfo;
};


#endif // BINIOBASE_H
