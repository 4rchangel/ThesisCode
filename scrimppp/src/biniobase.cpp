#include <biniobase.h>
#include <logging.hpp>
#include <partitioning_1d.h>

#include <vector>
#include <limits>
#include <assert.h>

#include <boost/assert.hpp>

using namespace matrix_profile;

BinIOBase::BinIOBase(const std::string& path, const MPI_Comm comm, const MPI_Info& fileinfo)
    : _path(path),
      _file_handle(MPI_FILE_NULL),
      _access_mode(UNDEFINED),
      _comm(comm),
      _fileinfo(fileinfo)
{
	EXEC_DEBUG("retrieving rank in file opening");
	if (comm == MPI_COMM_WORLD) {
		EXEC_DEBUG("opening file with WORLD communicator")
	}
	else if (comm == MPI_COMM_SELF) {
		EXEC_DEBUG("opening file with SELF communicator")
	}
	else if (comm == MPI_COMM_NULL) {
		EXEC_ERROR("trying to open file with NULL communicator");
	}
	else {
		EXEC_DEBUG("opening file with custom communicator");
	}
	MPI_Comm_rank(comm, &_rank);
	EXEC_DEBUG("rank in file I/O comm: " << _rank);
}

BinIOBase::~BinIOBase() {
	    if (_file_handle != MPI_FILE_NULL) {
			    close();
		}
}

void BinIOBase::close() {
	    if (_header.get() != nullptr) {
			    _header.release();
		}
		EXEC_TRACE("Closing MPI file handle")
		if (_file_handle != MPI_FILE_NULL) {
			    MPI_File_close(&_file_handle);
				_file_handle = MPI_FILE_NULL;
				_access_mode = UNDEFINED;
		}
}

BinIOBase::HeaderInfo BinIOBase::read_header_and_check_fmt(){
	    if (_file_handle==MPI_FILE_NULL or !(_access_mode==READ || _access_mode==FULL)) {
			    open_read();
		}

		if (_header.get() != nullptr){
			    return *_header;
		}

		EXEC_DEBUG("Reading binary file header");
		HeaderInfo header_buf;
		MPI_Datatype header_type = compose_header_type();
		MPI_Type_commit(&header_type);
		MPI_File_read_at(_file_handle, 0, &header_buf, 1, header_type, MPI_STATUS_IGNORE);
		MPI_Type_free(&header_type);

		const unsigned int fmt_ver = get_file_format_version();

		if (header_buf._fmt_version != fmt_ver) {
			    throw std::runtime_error("The binary input file uses a format different from the one the program was built for.");
		}
		if (header_buf._data_offset != this->_header_size) {
			    throw std::runtime_error("The binary input file uses a format different from the one the program was built for.");
		}

		_header = std::make_unique<HeaderInfo>(header_buf);
		return header_buf;
}

void BinIOBase::open_write() {
	    if (_file_handle != MPI_FILE_NULL) {
			    EXEC_ERROR("Reopening file!!! Closing first...")
				close();
		}

		EXEC_DEBUG("Opening file with MPI for writing...")
		MPI_File_open(_comm, _path.c_str(), MPI_MODE_WRONLY | MPI_MODE_CREATE, _fileinfo, &_file_handle);
		_access_mode = WRITE;
}

void BinIOBase::open_read(const bool check_header) {
	    if (_file_handle != MPI_FILE_NULL) {
			    EXEC_ERROR("Reopening file!!! Closing first...")
				close();
		}

		EXEC_DEBUG("Opening file with MPI for reading...");
		MPI_File_open(_comm, _path.c_str(), MPI_MODE_RDONLY, _fileinfo, &_file_handle);
		_access_mode = READ;
		EXEC_DEBUG("successfully opened the file");
}

MPI_Datatype BinIOBase::compose_header_type() const {
	    MPI_Datatype dtype;
		const int entrynum = 3;
		int blocklens[] = {1,1,1};
		MPI_Aint displacements[] = {
		        offsetof(HeaderInfo, _fmt_version),
		        offsetof(HeaderInfo, _data_offset),
		        offsetof(HeaderInfo, _num_samples)
		    };
		MPI_Datatype types[] = {MPI_UINT32_T, MPI_UINT32_T, MPI_UINT64_T};
		static_assert( (sizeof(HeaderInfo::_fmt_version) == 4), "Invalid MPI type declaration, assumed 32 bit int" );
		static_assert( (sizeof(HeaderInfo::_num_samples) == 8), "Invalid MPI type declaration, assumed 64 bit for ulong" );

		EXEC_TRACE("Creating MPI datatype for file header");
		MPI_Type_create_struct(entrynum, blocklens, displacements, types, &dtype);
#ifndef NDEBUG
		MPI_Aint extent=0, lb=0;
		MPI_Type_get_extent(dtype, &lb, &extent);
		BOOST_ASSERT_MSG( extent==sizeof(HeaderInfo), "MPI Type extent does not match the header info struct");
#endif
		return dtype;
}

void BinIOBase::write_header(const size_t total_ts_length, const bool perform_write) {
	// make sure the file is opened
	if (_file_handle==MPI_FILE_NULL or !(_access_mode==WRITE || _access_mode==FULL)) {
		    open_write();
	}

	HeaderInfo header {
		    ._fmt_version = get_file_format_version(),
		    ._data_offset = _header_size,
		    ._num_samples = total_ts_length
	};
	EXEC_DEBUG("composing binary file header information");
	MPI_Datatype header_dtype = compose_header_type();
	MPI_Type_commit(&header_dtype);
	if(perform_write) { EXEC_DEBUG("set the header view");}
	MPI_File_set_view(_file_handle, 0, header_dtype, header_dtype, _mpi_file_representation, MPI_INFO_NULL); // IMPORTANT, as writing in chunks could have modified the wirting position
	if (perform_write) {
		EXEC_DEBUG("write header now");
		MPI_File_write_at(_file_handle, 0, &header, 1, header_dtype, MPI_STATUS_IGNORE);
	}
	MPI_Type_free(&header_dtype);
}

// I/O methods. Using raw pointers to maintain flexibility regarding the actual type (aligned vector, general vector, array...) without neessity for templating
    // read a subsequence from the file, collective operation of all processes within the communicator. Though every proc may specify a custom slice offset and length (and buffer accordingly)
void BinIOBase::write_series_slice(const void * const slicebuffer, int total_ts_length, const int slicelen, const int sliceoffset, const MPI_Datatype etype, const bool write_header) {
	    MPI_Datatype filetype;
		assert(total_ts_length > 0);

		// make sure the file is opened
		if (_file_handle==MPI_FILE_NULL or !(_access_mode==WRITE || _access_mode==FULL)) {
			    open_write();
		}

		BinIOBase::write_header(total_ts_length, write_header);

		EXEC_DEBUG("Writing to file view");

		MPI_Type_create_subarray(1, &total_ts_length, &slicelen, &sliceoffset, MPI_ORDER_C, etype, &filetype);
		MPI_Type_commit(&filetype);
		MPI_File_set_view(_file_handle, _header_size, etype, filetype, _mpi_file_representation, MPI_INFO_NULL );
		MPI_File_write_all(_file_handle, slicebuffer, slicelen, etype, MPI_STATUS_IGNORE);
		MPI_Type_free(&filetype);
/*
 * actually causing a deadlock, if more than 1 process is used in distributed I/O, therefore removed
#ifndef NDEBUG
		EXEC_INFO("Checking written header...")
		read_header_and_check_fmt();
		EXEC_DEBUG("Validated written header!");
#endif
*/
}

void BinIOBase::write_total_series(const void * const tsbuffer, const long total_length, const MPI_Datatype etype) {
	    int commsize, rank;

		// partition the writing work:
		MPI_Comm_size(_comm, &commsize);
		MPI_Comm_rank(_comm, &rank);
		const PartitioningInfo partinf= split_work(commsize, total_length);
		const Partition1D local_workload = get_partition(rank, partinf);

		MPI_File_set_size(_file_handle, total_length);
		EXEC_DEBUG("Writing part of the TS: from " << local_workload._first_id << " to " << local_workload._last_id);
		write_series_slice(tsbuffer, total_length, local_workload._last_id-local_workload._first_id+1, local_workload._first_id, etype, rank==0);
}

void BinIOBase::read_series_slice_collective(void * const slicebuffer, const int slicelen, const int sliceoffset, const int ts_len, MPI_Datatype etype) {
	    MPI_Datatype filetype;

		if (_file_handle==MPI_FILE_NULL or !(_access_mode==READ || _access_mode==FULL)) {
			open_read();
		}

		if (ts_len < sliceoffset+slicelen){
			    throw std::runtime_error("File contains less samples than requested");
		}

		EXEC_DEBUG("Reading part of the TS from a file view:" << slicelen <<" samples starting at " << sliceoffset );
		MPI_Type_create_subarray(1, &ts_len, &slicelen, &sliceoffset, MPI_ORDER_C, etype, &filetype);
		MPI_Type_commit(&filetype);
		MPI_File_set_view(_file_handle, _header_size, etype, filetype, _mpi_file_representation, MPI_INFO_NULL);
		MPI_File_read_all(_file_handle, slicebuffer, slicelen, etype, MPI_STATUS_IGNORE);
		MPI_Type_free(&filetype);
}

void BinIOBase::read_series_slice_collective(void * const slicebuffer, const int slicelen, const int sliceoffset, const MPI_Datatype etype) {
	    if (_file_handle==MPI_FILE_NULL or !(_access_mode==READ || _access_mode==FULL)) {
			    open_read();
		}
		HeaderInfo header = read_header_and_check_fmt();
		if (header._num_samples < sliceoffset+slicelen){
			    throw std::runtime_error("File contains less samples than requested");
		}
		if (header._num_samples > std::numeric_limits<int>::max()){
			    throw std::runtime_error("Number of time series samples in file exceeds maximum treatable number.");
		}

		read_series_slice_collective(slicebuffer, slicelen, sliceoffset, header._num_samples, etype);
}

int BinIOBase::read_total_series_collective(void * const tsbuffer, const size_t maxlen, const MPI_Datatype etype){
	    // perform a distributed read of sections and spread the result afterwards
	    MPI_Comm comm = _comm;
		int rank, commsize;
		MPI_Comm_size(comm, &commsize);
		MPI_Comm_rank(comm, &rank);

		EXEC_DEBUG("parallel loading of binary file with " << commsize << "processes");
		//read the header to get total size and create work split
		const HeaderInfo fhead = read_header_and_check_fmt();
		if (maxlen < fhead._num_samples) {
			    throw std::runtime_error("input buffer too small for the input timeseries");
		}
		if (fhead._num_samples > std::numeric_limits<int>::max()){
			    throw std::runtime_error("Number of time series samples in file exceeds maximum treatable number.");
		}
		const int tslen = static_cast<int>(fhead._num_samples);

		//partition the work.
		    // TODO: check whether it is really a good idea, if everybody participates in parallel I/O. Maybe just a subset would be better
		const PartitioningInfo pinfo = split_work(commsize, fhead._num_samples);
		const Partition1D localpart = get_partition(rank, pinfo);
		const int partlen = localpart._last_id-localpart._first_id+1;
		MPI_Aint lb, extent;
		MPI_Type_get_extent(etype, &lb, &extent);
		void * part_buf = tsbuffer + localpart._first_id*extent;

		std::vector<int> rcv_counts(commsize);
		std::vector<int> displ(commsize);
		int displr=0;
		for(int r = 0; r < commsize; ++r) {
			    rcv_counts[r] = get_workload(r, pinfo);
				displ[r] = displr;
				displr += rcv_counts[r];
		}

		//distributed read of slices
		EXEC_DEBUG("Read slice as part of parallel read: start " << localpart._first_id << " len" << partlen);
		read_series_slice_collective(part_buf, partlen, localpart._first_id, tslen, etype);

		//exchange all the local slices
		MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, tsbuffer, rcv_counts.data(), displ.data(), matrix_profile::ts_mpitype, comm);

		return fhead._num_samples;
}

int BinIOBase::read_total_series_individually(void * const tsbuffer, const size_t maxlen, const MPI_Datatype etype){
	    // perform a distributed read of sections and spread the result afterwards
	    MPI_Datatype filetype;

		EXEC_DEBUG("individual reading of binary file");
		//read the header to get total size and create work split
		const HeaderInfo fhead = read_header_and_check_fmt();
		if (maxlen < fhead._num_samples) {
			    throw std::runtime_error("input buffer too small for the input timeseries");
		}
		if (fhead._num_samples > std::numeric_limits<int>::max()){
			    throw std::runtime_error("Number of time series samples in file exceeds maximum treatable number.");
		}
		const int tslen = static_cast<int>(fhead._num_samples);

		//individual read of the whole series
		const int zero=0;
		EXEC_DEBUG("Read total series with individual read");
		MPI_Type_create_subarray(1, &tslen, &tslen, &zero, MPI_ORDER_C, etype, &filetype);
		MPI_Type_commit(&filetype);
		MPI_File_set_view(_file_handle, _header_size, etype, filetype, _mpi_file_representation, MPI_INFO_NULL);
		MPI_File_read(_file_handle, tsbuffer, tslen, etype, MPI_STATUS_IGNORE);
		MPI_Type_free(&filetype);

		return fhead._num_samples;
}

unsigned long BinIOBase::get_series_len() {
	    if (_file_handle==MPI_FILE_NULL or !(_access_mode==READ || _access_mode==FULL)) {
			    open_read();
		}

		return read_header_and_check_fmt()._num_samples;
}
