#include <binproffile.h>
#include <logging.hpp>

#include <limits>
#include <assert.h>
#include <vector>
#include <partitioning_1d.h>

BinProfFile::BinProfFile(const std::string& path, const MPI_Comm comm, const MPI_Info& fileinfo)
    : BinIOBase(path, comm, fileinfo), _requested_chunksize(0)
{
	// Initialize the MPI Datatype as stored in the file
	const int entries = 2;
	const int bl[entries] = {1,1};
	static_assert( std::is_same<profile_store_dtype, double>::value, "Assuming the profile distances to be stored as doubles");
	static_assert( std::is_same<idx_store_dtype, unsigned long>::value, "Assuming the profile indices to be stored as unsigned long integers");
	static_assert( (sizeof(idx_store_dtype) == 8), "long integer size of 8 bytes assumed");
	const MPI_Datatype types[entries] = {MPI_DOUBLE, MPI_UINT64_T};
	const MPI_Aint disps[entries] = {
	        offsetof(ProfileEntry, _prof),
	        offsetof(ProfileEntry, _idx),
	    };

	MPI_Type_create_struct(entries, bl, disps, types, &_mpi_prof_type);
	MPI_Type_commit(&_mpi_prof_type);

    #ifndef NDEBUG
	    MPI_Aint extent, lb;
		MPI_Type_get_extent(_mpi_prof_type, &lb, &extent);
		EXEC_DEBUG("MPI I/O storage type extent " << extent << " lb " << lb << " offsets: " << disps[0] << ", " << disps[1]);
    #endif

	EXEC_DEBUG("Opened file " << path << " binary MPI i/o ");
}

BinProfFile::~BinProfFile() {
	MPI_Type_free(&_mpi_prof_type);
}


matrix_profile::idx_dtype BinProfFile::get_profile_length() {
	size_t len = get_series_len();
	if (len > std::numeric_limits<matrix_profile::idx_dtype>::max()) {
		throw std::runtime_error("the length of the stored matrix profile exceeds the index types maximum");
	}
	return len;
}

void BinProfFile::write_matrix_profile_slice(
        const matrix_profile::tsa_dtype * const profile_buf,
        const matrix_profile::idx_dtype * const idx_buf,
        const matrix_profile::idx_dtype slicelen,
        const matrix_profile::idx_dtype sliceoffset,
        const matrix_profile::idx_dtype proflen,
        bool write_header,
        const int chunknum
        ) {
	const idx_store_dtype MAX_CHUNKSIZE= (slicelen%chunknum==0)? (slicelen/chunknum) : (slicelen/chunknum)+1;
	int num_large_chunks = (slicelen%chunknum)>0? (slicelen%chunknum) : chunknum; // if all chunks are of equal size, all of them are 'large'
	std::vector<ProfileEntry> entrybuf(MAX_CHUNKSIZE);

	EXEC_DEBUG("rank " << _rank << " chunknum: " << chunknum << " max chunksize " << MAX_CHUNKSIZE << " slicelen " << slicelen );

	MPI_Datatype filetype;
	assert(proflen > 0);
	// TODO: in debug communicate the chunknum and verify, that it is identical among processes

	// make sure the file is opened
	if (_file_handle==MPI_FILE_NULL or !(_access_mode==WRITE || _access_mode==FULL)) {
		    open_write();
	}

	if (write_header) {
		EXEC_DEBUG("Rank" << _rank << " writing the file header");
	}
	// EVERYONE needs to call write header: the used set_file_view is a collective, thus we can not omit it
	BinIOBase::write_header(proflen, write_header);
	EXEC_DEBUG("Rank" << _rank << " finished writing the header");

	if (std::numeric_limits<int>::max() < proflen) {
		throw std::runtime_error("Profile length exceeds int range limit in file write");
	}
	const int slice_lens[1] = {static_cast<int>(slicelen)}; //TODO: add limit check!
	const int arr_lens[1] = {static_cast<int>(proflen)};
	const int ofsts[1] = {static_cast<int>(sliceoffset)};

	EXEC_DEBUG("Subarray creation: total len: " << arr_lens[0] << " slicelen: " << slice_lens[0] << " sliceoffset: " << ofsts[0]);
	MPI_Type_create_subarray(1, arr_lens, slice_lens, ofsts, MPI_ORDER_C, _mpi_prof_type, &filetype);
	MPI_Type_commit(&filetype);
	MPI_File_set_view(_file_handle, _header_size, _mpi_prof_type, filetype, _mpi_file_representation, MPI_INFO_NULL );

	EXEC_DEBUG("Rank " << _rank << " writing to file view a profile slice of len " << slicelen);
	int elem_ctr = 0;
	int write_ctr = 0;
	while (elem_ctr<slicelen) {
		int chunkelemctr = 0;
		int chunkstart = elem_ctr;
		const int chunksize = (write_ctr<num_large_chunks)? MAX_CHUNKSIZE : MAX_CHUNKSIZE-1;
		while (elem_ctr<slicelen && chunkelemctr < chunksize ) {
			entrybuf[chunkelemctr]._prof = static_cast<profile_store_dtype>(profile_buf[elem_ctr]);
			entrybuf[chunkelemctr]._idx  = static_cast<idx_store_dtype>(idx_buf[elem_ctr]);
			chunkelemctr += 1;
			elem_ctr +=1;
		}

    #ifndef NDEBUG
		    MPI_Offset offset, disp;
			MPI_File_get_position(_file_handle, &offset);
			MPI_File_get_byte_offset(_file_handle, offset, &disp);
			EXEC_DEBUG("Individual write by rank " << _rank << ", chunk " << write_ctr << " from idx " << chunkstart+sliceoffset << " to " << elem_ctr+sliceoffset-1 << "( length " << chunkelemctr << " ) at displacement " << disp);
    #endif
		MPI_File_write_all(_file_handle, entrybuf.data(), chunkelemctr, _mpi_prof_type, MPI_STATUS_IGNORE);//TODO: use the async one to overlap with the previous loop...
		write_ctr+=1;
	}
	EXEC_DEBUG("world rank " << _rank << " done with writing");
	MPI_Type_free(&filetype);
/*
 * debug check is causing a deadlock, if more than 1 process is used in distributed I/O, therefore removed
	#ifndef NDEBUG
		if (write_header) {
			EXEC_DEBUG("Checking written header...");
			read_header_and_check_fmt();
			EXEC_DEBUG("Validated written header!");
		}
	#endif
*/
}

matrix_profile::idx_dtype BinProfFile::read_matrix_profile(matrix_profile::tsa_dtype * const profile_buf, matrix_profile::idx_dtype * const idx_buf, matrix_profile::idx_dtype buflen, const bool read_collective) {
	const size_t proflen = get_series_len();
	int readlen=0;

	std::vector<ProfileEntry> entrybuf(proflen);
	assert(entrybuf.size() == proflen);
	if (read_collective) {
		readlen = read_total_series_collective(entrybuf.data(), proflen, _mpi_prof_type);
	}
	else {
		readlen = read_total_series_individually(entrybuf.data(), proflen, _mpi_prof_type);
	}

	assert(readlen == proflen);

	for (int i = 0; i < readlen; ++i) {
		profile_buf[i] = static_cast<matrix_profile::tsa_dtype>(entrybuf[i]._prof);
		idx_buf[i] = static_cast<matrix_profile::idx_dtype>(entrybuf[i]._idx);
	}
}

void BinProfFile::write_matrix_profile(
        const matrix_profile::tsa_dtype * const profile_buf,
        const matrix_profile::idx_dtype * const idx_buf,
        matrix_profile::idx_dtype proflen
        )
{
	// perform a distributed write of different sections
	int rank, commsize;
	MPI_Comm_size(_comm, &commsize);
	MPI_Comm_rank(_comm, &rank);

	EXEC_DEBUG("distributed parallel writing of binary matrix profile file with " << commsize << "processes");

	//partition the work.
	    // TODO: check whether it is really a good idea, if everybody participates in parallel I/O. Maybe just a subset would be better
	const PartitioningInfo pinfo = split_work(commsize, proflen);
	const Partition1D localpart = get_partition(rank, pinfo);
	const int partlen = localpart._last_id-localpart._first_id+1;
	const int chunknum = (_requested_chunksize>0)? (pinfo._size_full_load / _requested_chunksize) : 1;

	const matrix_profile::tsa_dtype * const profslicebuf = profile_buf + localpart._first_id;
	const matrix_profile::idx_dtype * const idxslicebuf = idx_buf + localpart._first_id;

	//distributed read of slices
	EXEC_TRACE("Write slice as part of parallel Write: start " << localpart._first_id << " len" << partlen);
	write_matrix_profile_slice(profslicebuf, idxslicebuf, partlen, localpart._first_id, proflen, rank==0, chunknum); //let the master take care of the header...
}
