#include <ScrimpDistribPar.hpp>
#include <logging.hpp>
#include <partitioning_1d.h>
#include <checkerboard_partitioning.h>
#include <binproffile.h>
#include <timing.h>
#include <kernels.h>

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <limits>
#include <algorithm>
#include <string>
#include <sstream>
#include <cassert>
#include <memory>
#include <cstddef>

#include <boost/assert.hpp>
#include <boost/align.hpp>
#include <boost/filesystem.hpp>

#include <mpi.h>

#include <papiwrapper.hpp>

using namespace matrix_profile;

using DiagPartition = Partition1D;

void MPI_Wait_inputwrapper(MPI_Request* requ, MPI_Status* status) {
	MPI_Wait(requ, status);
}

template<class T, class... Args>
inline aligned_ptr<T> make_aligned(Args&&... args)
{
  auto p = boost::alignment::
    aligned_alloc(alignof(T), sizeof(T));
  if (!p) {
	throw std::bad_alloc();
  }
  try {
	auto q = ::new(p) T(std::forward<Args>(args)...);
	return aligned_ptr<T>(q);
  } catch (...) {
	boost::alignment::aligned_free(p);
	throw;
  }
}
// continue own code
static FactoryRegistration<ScrimpDistribPar> s_trivParReg("distrib_par");
static const int notification_interval_iter = 10000;


// some helpers for measuring / tracking runtimes
#define TIMEPOINT( varname ) const matrix_profile::Timepoint varname = matrix_profile::get_cur_time();

Timespan track_sync_time(MPI_Comm comm = MPI_COMM_WORLD) {
	const Timepoint tstart = get_cur_time();
	MPI_Barrier(comm);
	return get_cur_time()-tstart;
}

#if SYNC_AND_TRACK_IDLE > 0
    #define TRACK_SYNCTIME( varname, comm ) Timespan varname = track_sync_time( comm );
    #define LOG_SYNCTIME( varname ) _timing_info._synctime_map.insert( SynctimeMap::value_type(#varname, varname) )
#else
    #define TRACK_SYNCTIME( x, y ) ;
    #define LOG_SYNCTIME( varname ) ;
#endif

// Communication functions
/**
 * @brief ScrimpDistribPar::init_dotproducts computes the initial dotproducts
 * @param num_diags umber
 * @param outbuf at least of length num_dotproducts, will contain the dotproducts
 * @param A at least of length num_dotproducts+m, "vertical time series". Dotproducts with all of its subsequences will be computed
 * @param B at least of length m. the single subsequence, which is the second factor in the dotrpoduct
 * @param windowSize window length m, dimensionality for dotproduct
 */
void ScrimpDistribPar::init_dotproducts(idx_dtype num_dotproducts, tsa_dtype outbuf[], const tsa_dtype A[], const tsa_dtype B[], const int windowSize)
{
	assert(outbuf != nullptr);
	assert(A != nullptr);
	assert(B != nullptr);

	for (idx_dtype diag = 0; diag < num_dotproducts; ++diag)
	{
		tsa_dtype accu = 0;
		for (int k = 0; k < windowSize; ++k) {
			accu += A[k+diag] * B[k];
		}
		outbuf[diag] = accu;
	}
}

void invalidate_n_entries(tsa_dtype* time_series, tsa_dtype* mu, tsa_dtype* s, const int n) {
	for (int i=0; i < n; ++i) {
		time_series[i] = 0.0;
		mu[i] = 0.0;
		s[i] = 0.0;
	}
}

int worldcollective_read_input_length(const std::string& filename) {
	BinTsFileMPI tsfile(filename, MPI_COMM_WORLD);
	return tsfile.get_ts_len();
}

static void decompose_datatype(
        const void * const invec,
        const void * const inoutvec,
        MPI_Datatype *datatype,
        tsa_dtype** inoutprof, idx_dtype** inout_idx,
        tsa_dtype** in2_prof, idx_dtype** in2_idx,
        idx_dtype& blocklen
        )
{
	EXEC_TRACE("reducing a MatProfSOA!");
	// retrieve the profile length: This is equal to the block length of the datastructure, not the length specified in the reduction!
	int num_int, num_add, num_dtype, combiner;
	MPI_Aint inout_addr, in2_addr;
	MPI_Get_address(invec, &in2_addr);
	MPI_Get_address(inoutvec, &inout_addr);
	MPI_Type_get_envelope(*datatype, &num_int, &num_add ,&num_dtype, &combiner);

	if (num_int>3 || num_add>2 || num_dtype >2){
		EXEC_ERROR("Invalid dataype sizes retrieved: num_i " << num_int
		           << " num_add " << num_add
		           << " num_dtype " << num_dtype
		           << "aborting as buffers are too small");
		MPI_Abort(MPI_COMM_WORLD, 1);
	}

	int blocklens[3];
	MPI_Aint addresses[2];
	MPI_Datatype dtypes[2];
	MPI_Type_get_contents(*datatype, num_int, num_add, num_dtype, blocklens, addresses, dtypes);

	assert(blocklens[1] == blocklens[2]);


	//"shortcuts" for coding conveniencebb
	*inoutprof = reinterpret_cast<tsa_dtype*>(MPI_Aint_add(inout_addr, addresses[0]) );;
	*inout_idx = reinterpret_cast<idx_dtype*>(MPI_Aint_add(inout_addr, addresses[1]) );
	*in2_prof = reinterpret_cast<tsa_dtype*>(MPI_Aint_add(in2_addr, addresses[0]) );
	*in2_idx = reinterpret_cast<idx_dtype*>(MPI_Aint_add(in2_addr, addresses[1]) );
	blocklen = blocklens[1];
}

void ScrimpDistribPar::MPI_MatProfSOA_reduction(void * invec, void *inoutvec, int *len, MPI_Datatype *datatype) {
	tsa_dtype *inout_prof, *in2_prof;
	idx_dtype *inout_idx, *in2_idx;
	idx_dtype blocklen;

	assert(*len == 1); // only reduction with size 1 is supported: Otherwise strides between SOAs needed to be considered....
	decompose_datatype(invec, inoutvec, datatype, &inout_prof, &inout_idx, &in2_prof, &in2_idx, blocklen);
	EXEC_TRACE("reduction with len " << *len << " and blocklen " << blocklen);

	//finally the matrix profile reduction
	const int limit = blocklen; // avoid unnecessary reads
	for (int i = 0; i < limit; ++i) {
		if (in2_prof[i] < inout_prof[i]) {
			inout_prof[i] = in2_prof[i];
			inout_idx[i] = in2_idx[i];
		}
	}
}

void merge_score_profiles(tsa_dtype* inoutprof, idx_dtype* inout_idx, const tsa_dtype* const in2_prof, const idx_dtype* const in2_idx, const int len) {
	for (int i = 0; i < len; ++i) {
		if (in2_prof[i] > inoutprof[i]) {
			inoutprof[i] = in2_prof[i];
			inout_idx[i] = in2_idx[i];
		}
	}
}

void ScrimpDistribPar::MatProfSOA_score_reduction(void * invec, void *inoutvec, int *len, MPI_Datatype *datatype) {
	tsa_dtype *inout_prof, *in2_prof;
	idx_dtype *inout_idx, *in2_idx;
	idx_dtype blocklen;

	if (*len==0) {
		EXEC_ERROR("length 0 reduction requested...");
		return;
	}

	EXEC_DEBUG("score reduction with len " << *len );
	assert(*len == 1); // only reduction with size 1 is supported: Otherwise strides between SOAs needed to be considered....

	decompose_datatype(invec, inoutvec, datatype, &inout_prof, &inout_idx, &in2_prof, &in2_idx, blocklen);
	EXEC_TRACE("score reduction with len " << *len << " and blocklen " << blocklen);
	merge_score_profiles(inout_prof, inout_idx, in2_prof, in2_idx, blocklen);
}

/**
 * @brief ScrimpDistribPar::eval_tile_blocked evaluate a tile consisting of the first tilelen diagonals of a triangular matrix section
 */
void ScrimpDistribPar::eval_tile_blocked(
        tsa_dtype prof_colmin[],
        idx_dtype idx_colmin[],
        tsa_dtype prof_rowmin[],
        idx_dtype idx_rowmin[],
        tsa_dtype tmpQ[],
        const idx_dtype blocklen,
        const idx_dtype trianglen,
        const idx_dtype tilelen,
        const tsa_dtype A_hor[],
        const tsa_dtype A_vert[],
        const idx_dtype windowSize,
        const tsa_dtype s_hor[],
        const tsa_dtype mu_hor[],
        const tsa_dtype s_vert[],
        const tsa_dtype mu_vert[],
        const idx_dtype baserow,
        const idx_dtype basecol
    )
{
	assert(tilelen <= trianglen);
	for ( idx_dtype block_offset = 0; block_offset < tilelen; block_offset+=blocklen) {
		const idx_dtype ndiags = std::min(blocklen, tilelen-block_offset);
		eval_diag_block_triangle(
		    prof_colmin,
		    idx_colmin,
		    prof_rowmin+block_offset,
		    idx_rowmin+block_offset,
		    tmpQ+block_offset,
		    ndiags,
		    trianglen-block_offset,
		    A_hor,
		    A_vert+block_offset,
		    windowSize,
		    s_hor,
		    mu_hor,
		    s_vert+block_offset,
		    mu_vert+block_offset,
		    baserow+block_offset,
		    basecol
		    );
	}
}

static bool is_square_num(const int num) {
	int sqrt = std::sqrt(num);
	return (num == sqrt*sqrt);
}

void get_row_communicator(const int row, const int world_size, MPI_Comm* comm, const int tag=11) {
	std::vector<int> ranks_rowwise = get_world_ranks_in_row(row, world_size);
	MPI_Group worldgroup, rowgroup;

	std::sort(ranks_rowwise.begin(), ranks_rowwise.end(), std::greater<>()); //sort descending. By that the "diagonal" process (i/o one) will have rank 0.  //NOTE: actually a simple reverse should do the trick,as get_rank_info._world_ranks_in_col compose the vector sequentially in ascending order
	EXEC_TRACE("create groups for row communicator");
	MPI_Comm_group(MPI_COMM_WORLD, &worldgroup);
	MPI_Group_incl(worldgroup, ranks_rowwise.size(), ranks_rowwise.data(), &rowgroup);
	EXEC_TRACE("create row communicator");
	MPI_Comm_create_group(MPI_COMM_WORLD, rowgroup, tag, comm);
}

void get_col_communicator(const int col, const int world_size, MPI_Comm* comm, const int tag=12) {
	std::vector<int> ranks_colwise = get_world_ranks_in_col(col, world_size);
	MPI_Group worldgroup, colgroup;

	std::sort(ranks_colwise.begin(), ranks_colwise.end()); //sort ascending. By that the "diagonal" process (i/o one) will have rank 0. //NOTE: actually not required due to the fact, that get_rank_info._world_ranks_in_col compose the vector sequentially
	EXEC_TRACE("create groups for col communicator");
	MPI_Comm_group(MPI_COMM_WORLD, &worldgroup);
	MPI_Group_incl(worldgroup, ranks_colwise.size(), ranks_colwise.data(), &colgroup);
	EXEC_TRACE("create column communicator");
	MPI_Comm_create_group(MPI_COMM_WORLD, colgroup, tag, comm);
}

void get_main_result_comm(const int result_col, const int world_size, MPI_Comm* comm, const int commtag, int& rank_result_accu) {
	const std::vector<int> ranks_rowwise = get_world_ranks_in_row(result_col, world_size);
	const std::vector<int> ranks_colwise = get_world_ranks_in_col(result_col, world_size);
	MPI_Group worldgroup, colwise, rowwise, commgroup;

	EXEC_TRACE("create world group");
	MPI_Comm_group(MPI_COMM_WORLD, &worldgroup);
	EXEC_TRACE("create rowwise group");
	MPI_Group_incl(worldgroup, ranks_rowwise.size(), ranks_rowwise.data(), &rowwise);
	EXEC_TRACE("create colwise group");
	MPI_Group_incl(worldgroup, ranks_colwise.size(), ranks_colwise.data(), &colwise);
	EXEC_TRACE("create union group");
	MPI_Group_union(rowwise, colwise, &commgroup);

    #ifndef NDEBUG
	    int groupsize;
		MPI_Group_size(commgroup, &groupsize);
		EXEC_DEBUG("create main result communicator from group of size " << groupsize);
    #endif

	MPI_Comm_create_group(MPI_COMM_WORLD, commgroup, commtag, comm);

	    // get the rank of the guy within the communicator, who will accumulate all the results
	//int accu_col_rank = *std::max_element(ranks_colwise.begin(), ranks_colwise.end()); // preivous version with I/O i the very last row
	int accu_col_rank = *std::min_element(ranks_colwise.begin(), ranks_colwise.end());
	int accu_comm_rank[1];
	MPI_Group_translate_ranks(worldgroup, 1, &accu_col_rank, commgroup, accu_comm_rank);
	EXEC_DEBUG("translated world rank " << accu_col_rank << " to " << accu_comm_rank[0] << " as receiver of slice result for col " << result_col);

	rank_result_accu=accu_comm_rank[0];//TODO: set to a ptoentially more meaninfully chosen value...
}

void get_exlusion_result_comm(const int result_col, const int world_size, MPI_Comm* comm, const int commtag, int& rank_result_accu) {
	const int num_cols = std::sqrt(world_size);
	assert(result_col >0);
	assert(result_col < num_cols);

	const std::vector<int> ranks_rowwise = get_world_ranks_in_row(result_col-1, world_size);
	const std::vector<int> ranks_colwise = get_world_ranks_in_col(result_col, world_size);

	MPI_Group worldgroup, colwise, rowwise, commgroup;

	EXEC_TRACE("create world group");
	MPI_Comm_group(MPI_COMM_WORLD, &worldgroup);
	EXEC_TRACE("create rowwise group");
	MPI_Group_incl(worldgroup, ranks_rowwise.size(), ranks_rowwise.data(), &rowwise);
	EXEC_TRACE("create colwise group");
	MPI_Group_incl(worldgroup, ranks_colwise.size(), ranks_colwise.data(), &colwise);
	EXEC_TRACE("create union group");
	MPI_Group_union(rowwise, colwise, &commgroup);

    #ifndef NDEBUG
	    int groupsize;
		MPI_Group_size(commgroup, &groupsize);
		std::stringstream ss;
		for (auto it = ranks_rowwise.begin(); it < ranks_rowwise.end(); ++it)
		{
			ss << *it << " ";
		}
		for (auto it = ranks_colwise.begin(); it < ranks_colwise.end(); ++it)
		{
			ss << *it << " ";
		}

		EXEC_DEBUG("create exclusion result communicator from group of size " << groupsize << " with world ranks " << ss.str());
    #endif

	MPI_Comm_create_group(MPI_COMM_WORLD, commgroup, commtag, comm);

	    // get the rank of the guy within the communicator, who will accumulate all the results
	//int accu_rank_info._world_rank = *std::max_element(ranks_colwise.begin(), ranks_colwise.end()); // previous version with I/O in the very last row
	int accu_row_rank = *std::min_element(ranks_colwise.begin(), ranks_colwise.end());
	int accu_comm_rank[1];
	MPI_Group_translate_ranks(worldgroup, 1, &accu_row_rank, commgroup, accu_comm_rank);
	EXEC_DEBUG("translated world rank " << accu_row_rank << " to " << accu_comm_rank[0] << " as receiver of exclusion result for col " << result_col);

	rank_result_accu = accu_comm_rank[0];
}

void get_result_io_comm(const int world_size, MPI_Comm* comm, const int tag = 13) {
	 int world_rank;
	EXEC_DEBUG("retrieve rank");
	 MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

	EXEC_DEBUG("world rank " << world_rank  << "check responsibility");
	 if (! is_io_responsible(world_size, world_rank)) {
		EXEC_DEBUG("NOT responsible");
		*comm = MPI_COMM_NULL;
		return;
	}
	else {
		EXEC_DEBUG("get ranks along diag");
		const std::vector<int> ioranks= get_world_ranks_along_diag(world_size);
		EXEC_DEBUG("build world group");
		MPI_Group worldgroup, iogroup;
		MPI_Comm_group(MPI_COMM_WORLD, &worldgroup);
		EXEC_DEBUG("build comm group");
		MPI_Group_incl(worldgroup, ioranks.size(), ioranks.data(), &iogroup);
		EXEC_DEBUG("create the communicator");
		MPI_Comm_create_group(MPI_COMM_WORLD, iogroup, tag, comm);
		EXEC_DEBUG("got the comm!");
	}
}

void get_excess_comm(const int world_size, MPI_Comm* comm, const int commtag, int& rank_result_accu) {
	const int num_rows = std::sqrt(world_size);
	const std::vector<int> ranks = get_world_ranks_in_row(num_rows-1, world_size); // all in the last row are participating
	MPI_Group worldgroup, commgroup;

	EXEC_TRACE("create world group");
	MPI_Comm_group(MPI_COMM_WORLD, &worldgroup);
	EXEC_TRACE("create colwise group");
	MPI_Group_incl(worldgroup, ranks.size(), ranks.data(), &commgroup);

	MPI_Comm_create_group(MPI_COMM_WORLD, commgroup, commtag, comm);

	    // get the rank of the guy within the communicator, who will accumulate all the results
	int accu_world_rank = world_size-1;
	int accu_comm_rank[1];
	MPI_Group_translate_ranks(worldgroup, 1, &accu_world_rank, commgroup, accu_comm_rank);
	EXEC_DEBUG("translated world rank " << accu_world_rank << " to " << accu_comm_rank[0] << " as receiver of the very last excess (excl. zone bottom)");

    #ifndef NDEBUG
	    int size;
		MPI_Comm_size(*comm, &size);
		EXEC_DEBUG( " created excess communicator of size "<< size);
    #endif

	rank_result_accu=accu_comm_rank[0];//TODO: set to potentially more meaninfully chosen value...
}

static std::string idx_string(const idx_dtype* ptr, const int n) {
	std::stringstream ss;
	for (int i = 0; i < n; ++i) {
		ss << ptr[i] << " ";
	}
	return ss.str();
}

static std::string ts_string(const tsa_dtype* ptr, const int n) {
	std::stringstream ss;
	for (int i = 0; i < n; ++i) {
		ss << ptr[i] << " ";
	}
	return ss.str();
}

void ScrimpDistribPar::commit_mpi_SOA_prof_type(MPI_Datatype* ptr_mpi_type, tsa_dtype* profile_buf, idx_dtype* idx_buf, int profile_length)
{
	const int blocklen[] = {profile_length,profile_length};
	MPI_Aint profBufAddr, idxBufAddr;
	MPI_Aint disps[2];
	//assertion for DOUBLE usage!!! TODO employ some preprocessor or template magic to automatically use a diffferent MPI Datatype...
	static_assert(sizeof(tsa_dtype) == sizeof(double), "NO FLOAT POSSIBLE without modifing MPI Datatype!" );
	static_assert(std::is_same<idx_dtype, long>::value, "Index datatype does not match the MPI Datatype!");
	const MPI_Datatype dtypes[] = {MPI_DOUBLE, MPI_LONG};
	// official way of handling addresses
	MPI_Get_address(profile_buf, &profBufAddr);
	MPI_Get_address(idx_buf, &idxBufAddr);
	disps[0] = 0;
	disps[1] = MPI_Aint_diff(idxBufAddr, profBufAddr);

	MPI_Type_create_struct(2, blocklen, disps, dtypes, ptr_mpi_type);
	MPI_Type_commit(ptr_mpi_type);
}

void ScrimpDistribPar::initialize(const Scrimppp_params &params) {
	const bool BCAST_INPUT = DISTRIB_BCAST_INPUT;
	_init_start_time = get_cur_time();

	EXEC_DEBUG("Initialization...")
	EXEC_DEBUG("Memory allocation...");
	_hor_result = make_aligned<MatProfSOA>();
	_vert_result = make_aligned<MatProfSOA>();

	EXEC_DEBUG("Communicator initialization...");
	MPI_Comm_size(MPI_COMM_WORLD, &_rank_info._world_size);
	MPI_Comm_rank(MPI_COMM_WORLD, &_rank_info._world_rank);

	_tile_info._tile_id = _rank_info._world_rank;
	_tile_info._tile_num = _rank_info._world_size;

	get_partition_coords(_tile_info._tile_id, _tile_info._partition_row, _tile_info._partition_col, _tile_info._is_upper_triangle);
	const int num_cols = std::sqrt(_tile_info._tile_num);

	EXEC_DEBUG("IO comm creation");
	get_result_io_comm(_rank_info._world_size, &_prof_io_comm);
	EXEC_DEBUG("finished result comm creation");
	if (BCAST_INPUT) {
		_ts_io_comm = _prof_io_comm;
		if (_ts_io_comm == MPI_COMM_NULL) {
			_ts_io_comm = MPI_COMM_SELF; // if the proc does not participate in File output it will still do a individual read to retrieve the input length. Thus a valid communicator is required...
		}
	}
	else {
		_ts_io_comm = MPI_COMM_WORLD; // if the proc does not participate in File I/O it will still do a individual read to retrieve the input length. As everyone does it, we can use the world commmunicator
	}

	EXEC_DEBUG("main result comm creation");
	for ( int i = 0; i < num_cols; ++i){
		if (i == _tile_info._partition_row && (_tile_info._partition_col != _tile_info._partition_row) )
		{
			EXEC_DEBUG("world rank " << _rank_info._world_rank << " create communicator for main slice " << i);
			get_main_result_comm(i, _rank_info._world_size, &_row_result_comm, 1, _rank_info._accu_results_rowcomm);
		}
		if ( (i == _tile_info._partition_col))
		{
			EXEC_DEBUG("world rank " << _rank_info._world_rank << " create communicator for main slice " << i);
			get_main_result_comm(i, _rank_info._world_size, &_col_result_comm, 1, _rank_info._accu_results_colcomm);
		}
	}

	EXEC_DEBUG("exlusion_result_comm_creation");
	for (int i = 1; i<num_cols; ++i) {
		if (i==1+_tile_info._partition_row) {
			EXEC_DEBUG("world rank " << _rank_info._world_rank << " creates row comm for exclusion slice " << i);
			get_exlusion_result_comm(i, _rank_info._world_size, &_exclusion_rowcomm, 2, _rank_info._exclusion_row_target);
		}
		if (i==_tile_info._partition_col) {
			EXEC_DEBUG("world rank " << _rank_info._world_rank << " creates col comm for exlusion slice " << i);
			get_exlusion_result_comm(i, _rank_info._world_size, &_exclusion_colcomm, 2, _rank_info._eexclusion_col_target);
		}
	}

	EXEC_DEBUG("Grid communicator creation");
	get_row_communicator(_tile_info._partition_row, _rank_info._world_size, &_row_comm); // NOTE: silently assuming that the I/O nodes end up with proper rank 0 within the communicators..
	get_col_communicator(_tile_info._partition_col, _rank_info._world_size, &_col_comm); // cleaner way woul be to propagate the I/O node rank through parameters TODO

	if (_tile_info._partition_row == num_cols-1) {
		get_excess_comm(_rank_info._world_size, &_final_excesscomm, 3, _rank_info._final_exess_target);
	}


	EXEC_DEBUG("retrieving row and col comm ranks");
	MPI_Comm_rank(_row_comm, &_rank_info._rank_in_row);
	MPI_Comm_rank(_col_comm, &_rank_info._rank_in_col);
	_finished_initialization = true;
}

void ScrimpDistribPar::compute_matrix_profile(const Scrimppp_params& params) {
	const bool BCAST_INPUT = DISTRIB_BCAST_INPUT;
	Timepoint tstart = get_cur_time();
	if ( ! IGNORE_INIT_TIME) {
		tstart = _init_start_time;
	}

	/*EXEC_DEBUG("Test null-type freeing")
	MPI_Datatype nulltype=MPI_DATATYPE_NULL;
	MPI_Type_free(&nulltype);
	EXEC_DEBUG("Passed the free test")*/


	int windowSize = params.query_window_len;
	int exclusionZone = windowSize / 4;

	const int BLOCKING_SIZE = params.block_length;

	assert(params.use_distributed_io == true);//I/O from a single node is not supported in this algorithm...
	assert(_finished_initialization);

	//EXEC_INFO( "MPI INFO: comm size: " << _rank_info._world_size << " world rank: " << _rank_info._world_rank);
	//EXEC_INFO( "Blocking size: " << BLOCKING_SIZE)
	_log_info.blocklen = BLOCKING_SIZE;

	const int num_cols = std::sqrt(_tile_info._tile_num);

	if (!is_square_num(_tile_info._tile_num)) { //TODO: maybe better check the number of procs directly
		throw std::runtime_error("number of processors is no square number, which is required for the checkerboard partitioning.");
	}

	if (params.filetype != Scrimppp_params::BIN) {
		throw std::runtime_error("Only binary input supported by this algorithm.");
	}

	int rank_in_colcomm=-1, rank_in_rowcom=-1;
	int rank_in_exclusion_rowcomm=-1, rank_in_exclusion_colcomm=-1;

	TIMEPOINT( tinit)

	const bool shall_read_ts_slice = BCAST_INPUT? (_ts_io_comm!=MPI_COMM_NULL && _ts_io_comm!=MPI_COMM_SELF) : true;
	TIMEPOINT( tsetup1 );

	//const int input_series_len = worldcollective_read_input_length(params.time_series_filename);
	int input_series_len = 0;

	BinTsFileMPI tsfile(params.time_series_filename, _ts_io_comm);

	if (shall_read_ts_slice) {
		tsfile.open_read(false);
	}

	if (_rank_info._world_rank==0) {
		input_series_len = tsfile.get_ts_len();
	}
	TIMEPOINT( intermediate_reading );

	MPI_Bcast(&input_series_len, 1, MPI_INT, 0, MPI_COMM_WORLD);

//	input_series_len=tsfile.get_ts_len();

	TIMEPOINT( tinread );

	const int floored_work_len = (input_series_len -windowSize -exclusionZone)/num_cols;
	const int delta = (input_series_len -windowSize-exclusionZone -  num_cols*floored_work_len);
	const int input_padding = (delta>0)? num_cols-delta : 0;
	const int time_series_len = input_series_len + input_padding;
	_log_info.time_series_len = time_series_len;
	const int profile_length = time_series_len - windowSize + 1;

	const int work_triang_len = profile_length - exclusionZone-1;
	const int tile_length = work_triang_len / num_cols;
	const int max_prof_slice_len = tile_length+exclusionZone+1;

	//EXEC_INFO ( "input time series of length: " << input_series_len );
	_log_info.input_series_len = input_series_len;
	//EXEC_INFO ( "apply padding: " << input_padding );
	_log_info.input_padding = input_padding;

	//EXEC_INFO(" running distributed parallelization: applying padding of " << input_padding << " samples to process tiles with base length " << tile_length
	//         << " input len " << input_series_len << " time_series_len " << time_series_len << " profile_length " << profile_length << " worklen " << work_triang_len << " tile length " << tile_length);
	_log_info.tile_length = tile_length;
	_log_info.profile_length = profile_length;
	_log_info.work_triang_len = work_triang_len;
	// validation of input parameters and input file...
	if (max_prof_slice_len > RESULT_ALLOC_LEN) { // check, whether the data fit into the preallocated result storage
		throw std::runtime_error("resulting profile length EXCEEDS MAXIMUM LENGTH. Recompile with different allocation size or choose a smaller problem");
	}
	else if(profile_length<=0) {
		throw std::runtime_error("parameters chosen badly: resulting matrix profile of length 0. done.");
	}
	if ( work_triang_len % num_cols != 0) {
		EXEC_ERROR("triangle base size " << work_triang_len << " can not be divided into " << num_cols << " equal columns")
		throw std::runtime_error("checkerboard partitioning is impossible, as problem size is no multiple of the column number.");
	}

	//Initialize Matrix Profile and Matrix Profile Index
	    //TODO: is it better (faster) to fill only up to the required number, not the whole preallocated structure !? (space: first touch allocation policy!?)
	/*_hor_result->profile.fill(-1.0);
	_hor_result->index.fill(-1);
	_vert_result->profile.fill(-1.0);
	_vert_result->index.fill(-1);*/
	    // partial filling only...
	std::fill_n(_hor_result->profile.begin(), tile_length+exclusionZone, -1.0);
	std::fill_n(_hor_result->index.begin(), tile_length+exclusionZone, -1);
	std::fill_n(_vert_result->profile.begin(), tile_length+exclusionZone, -1.0);
	std::fill_n(_vert_result->index.begin(), tile_length+exclusionZone, -1);

	// MPI Setup. NOTE: Moving this to the initialize method also requires moving the read of the input length!
	    //create the MPI dataype for the result
	commit_mpi_SOA_prof_type(&_mpi_types.hor_prof_mpitype, _hor_result->profile.data(), _hor_result->index.data(), tile_length +exclusionZone);
	commit_mpi_SOA_prof_type(&_mpi_types.vert_prof_mpitype, _vert_result->profile.data(), _vert_result->index.data(), tile_length); //tile_length +exclusionZone); // TODO: double-check length

	if (_tile_info._partition_row < num_cols-1) {
		commit_mpi_SOA_prof_type(&_mpi_types.vert_prof_mainslice_mpitype, _vert_result->profile.data(), _vert_result->index.data(), tile_length-exclusionZone-1);
	}
	else {
		_mpi_types.vert_prof_mainslice_mpitype = MPI_DATATYPE_NULL;
	}
	commit_mpi_SOA_prof_type(&_mpi_types.hor_prof_mainslice_mpitype, _hor_result->profile.data()+exclusionZone+1, _hor_result->index.data()+exclusionZone+1, tile_length-exclusionZone-1);

/*	else {
		if (_tile_info._partition_row == _tile_info._partition_col) {
			commit_mpi_SOA_prof_type(&_mpi_types.hor_prof_mainslice_mpitype, _hor_result->profile.data()+exclusionZone+1, _hor_result->index.data()+exclusionZone+1, tile_length);
		}
		else {
			commit_mpi_SOA_prof_type(&_mpi_types.hor_prof_mainslice_mpitype, _hor_result->profile.data()+exclusionZone+1, _hor_result->index.data()+exclusionZone+1, tile_length-exclusionZone-1);
		}
		commit_mpi_SOA_prof_type(&_mpi_types.vert_prof_mainslice_mpitype, _vert_result->profile.data(), _vert_result->index.data(), tile_length);
	}*/

	commit_mpi_SOA_prof_type(&_mpi_types.hor_prof_exclusionslice_mpitype, _hor_result->profile.data(), _hor_result->index.data(), exclusionZone+1);
	commit_mpi_SOA_prof_type(&_mpi_types.vert_prof_exclusionslice_mpitype, _vert_result->profile.data()+tile_length-exclusionZone-1, _vert_result->index.data()+tile_length-exclusionZone-1, exclusionZone+1);

	if (_tile_info._partition_row == num_cols-1) {
		if (_tile_info._partition_col == num_cols-1) {
			commit_mpi_SOA_prof_type(&_mpi_types.final_excess_slice_mpitype, _hor_result->profile.data()+exclusionZone+1, _hor_result->index.data()+exclusionZone+1, tile_length);
		}
		else {
			commit_mpi_SOA_prof_type(&_mpi_types.final_excess_slice_mpitype, _vert_result->profile.data(), _vert_result->index.data(), tile_length);
		}
	}
	else  {
		_mpi_types.final_excess_slice_mpitype = MPI_DATATYPE_NULL;
	}

	    // create the reduction operation for the custom datatype
	MPI_Op_create(MatProfSOA_score_reduction, 1, &_mpi_types.score_reduction_op);

	const idx_dtype col_offset_to_lower_tile = (_tile_info._is_upper_triangle?1:0);
	const idx_dtype col_offset = _tile_info._partition_col*tile_length;
	const idx_dtype row_offset = _tile_info._partition_row*tile_length +exclusionZone+1;
	//const idx_dtype hor_result_offset = col_offset_to_lower_tile;
	//const idx_dtype vert_result_offset = exclusionZone+1;
	const bool is_processing_padding = _tile_info._partition_row==num_cols; // num_cols==num_rows, the ones in the last row will operate on the padding
	_log_info.is_processing_padding = is_processing_padding;

	const idx_dtype ts_slice_len = tile_length+windowSize-1;
	aligned_tsdtype_vec A, B;
	aligned_tsdtype_vec mu_A(ts_slice_len), s_A(ts_slice_len);
	aligned_tsdtype_vec mu_B(ts_slice_len), s_B(ts_slice_len);
	aligned_tsdtype_vec dotproducts(ts_slice_len);
	// reserve space to pad the time series to account for additional read in the kernel
	A.reserve(ts_slice_len+1);
	B.reserve(ts_slice_len+1+exclusionZone+1);
	A.resize(ts_slice_len);
	B.resize(ts_slice_len);

	/*if (partition_col != partition_row) {
		get_main_result_comm(partition_row, world_size, &row_result_comm, 0);
	}*/

	if (_tile_info._partition_row == num_cols-1) {
		EXEC_DEBUG("retrieve rank in excess comm");
		MPI_Comm_rank(_final_excesscomm, &_rank_info._rank_in_final_excesscomm);
	}

	if (_tile_info._partition_col != _tile_info._partition_row) {
		MPI_Comm_rank(_row_result_comm, &rank_in_rowcom);
		EXEC_DEBUG("world rank " << _rank_info._world_rank << " has rank " << rank_in_rowcom << " in row_result_comm")
	}
	MPI_Comm_rank(_col_result_comm, &rank_in_colcomm);
	EXEC_DEBUG("world rank " << _rank_info._world_rank << " has rank " << rank_in_colcomm << " in col_result_comm")

	if (_tile_info._partition_row < num_cols-1) {
		EXEC_DEBUG("world rank " << _rank_info._world_rank << " retrieving exclusion rowrank")
		MPI_Comm_rank(_exclusion_rowcomm, &rank_in_exclusion_rowcomm);
	}
	if (_tile_info._partition_col>0) {
		EXEC_DEBUG("world rank " << _rank_info._world_rank << " retrieving exclusion columnrank")
		MPI_Comm_rank(_exclusion_colcomm, &rank_in_exclusion_colcomm);
	}

	EXEC_DEBUG("finished communicator creation!");

#ifndef NDEBUG
	MPI_Barrier(MPI_COMM_WORLD);
	int rowcomm_size, colcomm_size;
	MPI_Comm_size(_row_comm, &rowcomm_size);
	MPI_Comm_size(_col_comm, &colcomm_size);
	EXEC_DEBUG("world rank " << _rank_info._world_rank << " at STARTOFF Barrier: partition (" << _tile_info._partition_col << "/" << _tile_info._partition_row <<
	           ") row comm size " << rowcomm_size << " col comm size " << colcomm_size);
	std::cout.flush();
	MPI_Barrier(MPI_COMM_WORLD);
#endif
	TRACK_SYNCTIME( idle_setup, MPI_COMM_WORLD );
	TIMEPOINT( tsetup2 );

	// read the time series input slies
	if (shall_read_ts_slice) {
		if (_tile_info._partition_col < num_cols -1) { // all readers, except for the one of the last column read regularly
			const int read_len = (BCAST_INPUT)? ts_slice_len+exclusionZone+1 : ts_slice_len;
			B.resize(read_len);
			tsfile.read_ts_slice(B.data(), read_len, col_offset);
		}
		else { // padding needs to be applied for the last ts slice. As padding can not be read (as not physically existing), fewer values are to be read, than for the other processes
			const int read_len = (BCAST_INPUT)? ts_slice_len+exclusionZone+1-input_padding : ts_slice_len-input_padding;
			B.resize(read_len+input_padding); // set size to the padded length
			tsfile.read_ts_slice(B.data(), read_len, col_offset); // but read only as many values as there are in the time series
		}

		if (!BCAST_INPUT) {
			// if the input is not spread by broadcasting, every process needs to read both its input chunks ( in theory the ones along the diagonal can avoid it, though for symmetry i stuck to everyone reading
			if (_tile_info._partition_row < num_cols-1) {
				tsfile.read_ts_slice(A.data(), ts_slice_len, row_offset);
			}
			else { // padding needs to be applied for the last ts slice. Padding can not be read (as not physically existing)
				const int read_len = ts_slice_len-input_padding;
				tsfile.read_ts_slice(A.data(), read_len, row_offset);
			}
		}
	}
	else if (!BCAST_INPUT) {
		throw std::runtime_error(" some process had not read any input, though no scattering of the input is performed! Thus required input is missing...");
	}

	TRACK_SYNCTIME( idle_reading, MPI_COMM_WORLD );
	TIMEPOINT( tfileread );

	if (BCAST_INPUT) { // broadast the read input from the io processes to the other ones
		EXEC_DEBUG(" spread the ts input values by broadcasting...");
		MPI_Request col_bcast_requ=MPI_REQUEST_NULL, row_bcast_requ=MPI_REQUEST_NULL;
		MPI_Ibcast(B.data(), ts_slice_len, ts_mpitype, _rank_info.rank_input_in_colcomm, _col_comm, &col_bcast_requ);
		if ( shall_read_ts_slice) {
			EXEC_DEBUG(" copying overlapping input slice, offset: " << row_offset-col_offset << " copy len " << ts_slice_len);
			//EXEC_DEBUG(" B values " << ts_string(B.data(), row_offset-col_offset+10) << std::endl << " A values " << ts_string(A.data(), 10));
			memcpy(A.data(), B.data()+row_offset-col_offset, (ts_slice_len)*sizeof(tsa_dtype) );
			//EXEC_DEBUG(" B values " << ts_string(B.data(), row_offset-col_offset+10) << std::endl << " A values " << ts_string(A.data(), 10));
		}
		MPI_Ibcast(A.data(), ts_slice_len, ts_mpitype, _rank_info.rank_input_in_rowcomm, _row_comm, &row_bcast_requ);
		if (true || ! shall_read_ts_slice) { //TODO: somehow the communication was "stuck", i.e. the other workers did not recieve anything, while the io procs were already computing. Thus i also let also the I/O node wait for the broadcast to complete. Maybe one could investigate, what is going wrong, when proceeding asynchronously
		// the I/O processes already have loaded all required data and could proceed asynchronously with the computation. All the other ones need to wait
			MPI_Wait_inputwrapper(&col_bcast_requ, MPI_STATUS_IGNORE);
			MPI_Wait_inputwrapper(&row_bcast_requ, MPI_STATUS_IGNORE);
		}
	}
	else {
		EXEC_DEBUG("everyone read immediately, no broadcasting");
	}

	TRACK_SYNCTIME( idle_bcast, MPI_COMM_WORLD );
	TIMEPOINT( tinput );

	precompute_window_statistics(windowSize, A, tile_length, mu_A, s_A, ts_slice_len);
	precompute_window_statistics(windowSize, B, tile_length, mu_B, s_B, ts_slice_len);

	if (_tile_info._partition_col == num_cols-1) {
		// make sure that the required padding does not impact the result
		const int read_len = ts_slice_len-input_padding;
		invalidate_n_entries(
		    B.data()+read_len,
		    mu_B.data()+read_len,
		    s_B.data()+read_len,
		    input_padding
		    );
	}
	if (_tile_info._partition_row == num_cols-1) {
		// make sure that the required padding does not impact the result
		const int read_len = ts_slice_len-input_padding;
		invalidate_n_entries(
		    A.data()+read_len,
		    mu_A.data()+read_len,
		    s_A.data()+read_len,
		    input_padding
		    );
	}

	// padding to avoid invalid read (valgrind) in the last iteration of the kernel... unelegant. TODO: adapting the kernel might be nicer, on the other hand it could cause some overhead
	A.push_back(0.0);
	B.push_back(0.0);

	TIMEPOINT( tprecomputations );

	if (_tile_info._is_upper_triangle) {
		init_dotproducts(tile_length, dotproducts.data(), B.data(), A.data(), windowSize);
	}
	else {
		init_dotproducts(tile_length, dotproducts.data(), A.data(), B.data(), windowSize);
	}

	TIMEPOINT( tdotprods );

	if (_tile_info._is_upper_triangle) {
		EXEC_DEBUG("evaluating upper tile " << _tile_info._tile_id << "/" << _tile_info._tile_num << " starting at " << row_offset << "," << col_offset);
		eval_tile_blocked(
		            _vert_result->profile.data(),
		            _vert_result->index.data(),
		            _hor_result->profile.data() +col_offset_to_lower_tile,
		            _hor_result->index.data() +col_offset_to_lower_tile,
		            dotproducts.data() + col_offset_to_lower_tile,
		            params.block_length, // all the triangle as a single block for debugging
		            tile_length-col_offset_to_lower_tile,
		            tile_length-col_offset_to_lower_tile,
		            A.data(),
		            B.data() +col_offset_to_lower_tile,
		            windowSize,
		            s_A.data(),
		            mu_A.data(),
		            s_B.data() +col_offset_to_lower_tile,
		            mu_B.data() +col_offset_to_lower_tile,
		            col_offset +col_offset_to_lower_tile,
		            row_offset
		            );
	}
	else {
		EXEC_DEBUG("evaluating lower tile " << _tile_info._tile_id << "/" << _tile_info._tile_num << " starting at " << row_offset << "," << col_offset);
		eval_tile_blocked(
		            _hor_result->profile.data(),
		            _hor_result->index.data(),
		            _vert_result->profile.data(),
		            _vert_result->index.data(),
		            dotproducts.data(),
		            params.block_length, // all the triangle as a single block for debugging
		            tile_length,
		            tile_length,
		            B.data(),
		            A.data(),
		            windowSize,
		            s_B.data(),
		            mu_B.data(),
		            s_A.data(),
		            mu_A.data(),
		            row_offset,
		            col_offset
		            );
	}

	TRACK_SYNCTIME( idle_eval, MPI_COMM_WORLD );
	TIMEPOINT( tevaluations );

	if (_tile_info._partition_col == _tile_info._partition_row) {
		EXEC_DEBUG("rank " << _rank_info._world_rank << " merging local slice main results");
		merge_score_profiles(
		        _hor_result->profile.data() +exclusionZone+1,
		        _hor_result->index.data() +exclusionZone+1,
		        _vert_result->profile.data(),
		        _vert_result->index.data(),
		        tile_length-exclusionZone-1
		    );
	}
	if (_tile_info._partition_col == num_cols-1) {
		EXEC_DEBUG("rank " << _rank_info._world_rank << " merging local exclusion results");
		merge_score_profiles(
		        _hor_result->profile.data()+tile_length,
		        _hor_result->index.data()+tile_length,
		        _vert_result->profile.data()+tile_length-exclusionZone-1,
		        _vert_result->index.data()+tile_length-exclusionZone-1,
		        exclusionZone+1
		    );
	}

	TIMEPOINT( tmerge );

	MPI_Request row_request, col_request;
	MPI_Request row_exclusion_request, col_exclusion_request;
	MPI_Request final_excess_request;
	row_request = MPI_REQUEST_NULL;
	col_request = MPI_REQUEST_NULL;
	row_exclusion_request = MPI_REQUEST_NULL;
	col_exclusion_request = MPI_REQUEST_NULL;
	final_excess_request = MPI_REQUEST_NULL;

	// row-wise main slice reduction for all tiles except  for those in the last row
	if (_tile_info._partition_col != _tile_info._partition_row && _tile_info._partition_row<num_cols-1)
	{
		EXEC_DEBUG("rank " << _rank_info._world_rank << " participating in row reduction");
		if (rank_in_rowcom == _rank_info._accu_results_rowcomm) { // as we want the result to be accumulated in the columns bottom-most  processor, this shall no longer happen after fixing the ranks
			MPI_Ireduce(MPI_IN_PLACE, _vert_result.get(), 1, _mpi_types.vert_prof_mainslice_mpitype, _mpi_types.score_reduction_op, _rank_info._accu_results_rowcomm, _row_result_comm,  &row_request);
		}
		else {
			MPI_Ireduce(_vert_result.get(), _vert_result.get(), 1, _mpi_types.vert_prof_mainslice_mpitype, _mpi_types.score_reduction_op, _rank_info._accu_results_rowcomm, _row_result_comm, &row_request);
		}
	}
	else if (_tile_info._partition_row == num_cols-1) // in the last row, we perform the extended reduction including the excess reduction
	{
		if (_tile_info._partition_col != num_cols-1) // the diagonal tile receives the stuff as a "vertical_result"
		{
			EXEC_DEBUG("SENDING in excess main reduction, world rank: "  << _rank_info._world_rank)
			MPI_Ireduce(_vert_result.get(), _vert_result.get(), 1, _mpi_types.final_excess_slice_mpitype, _mpi_types.score_reduction_op, _rank_info._accu_results_rowcomm, _row_result_comm, &row_request);
		}
	}

	//col-wise main slice reduction
	if (_tile_info._partition_col>0 && _tile_info._partition_col!=num_cols-1) {
		if (rank_in_colcomm == _rank_info._accu_results_colcomm) {
			EXEC_DEBUG("rank " << _rank_info._world_rank << " participating in col reduction");
			MPI_Ireduce(MPI_IN_PLACE, _hor_result->profile.data()+exclusionZone+1, 1, _mpi_types.hor_prof_mainslice_mpitype, _mpi_types.score_reduction_op, _rank_info._accu_results_colcomm, _col_result_comm, &col_request);
		}
		else {
			MPI_Ireduce(_hor_result->profile.data()+exclusionZone+1, _hor_result->profile.data()+exclusionZone+1, 1, _mpi_types.hor_prof_mainslice_mpitype, _mpi_types.score_reduction_op, _rank_info._accu_results_colcomm, _col_result_comm, &col_request);
		}
	}
	else if (_tile_info._partition_col==num_cols-1) {
		// only the diagonal I/O tile is in the last column.
		assert(rank_in_colcomm == _rank_info._accu_results_colcomm);
		EXEC_DEBUG("last tile receiving excess main reduction, world rank " << _rank_info._world_rank);
		MPI_Ireduce(MPI_IN_PLACE, _hor_result->profile.data()+exclusionZone+1, 1, _mpi_types.final_excess_slice_mpitype, _mpi_types.score_reduction_op, _rank_info._accu_results_colcomm, _col_result_comm, &col_request);
	}
	else {//_partition_col==0
		// in column zero no row-wise reduction is involved. Thus we can reduce the full profile...
		if (rank_in_colcomm == _rank_info._accu_results_colcomm) {
			EXEC_DEBUG("rank " << _rank_info._world_rank << " participating in col reduction");
			MPI_Ireduce(MPI_IN_PLACE, _hor_result->profile.data(), 1, _mpi_types.hor_prof_mpitype, _mpi_types.score_reduction_op, _rank_info._accu_results_colcomm, _col_result_comm, &col_request);
		}
		else {
			MPI_Ireduce(_hor_result->profile.data(), _hor_result->profile.data(), 1, _mpi_types.hor_prof_mpitype, _mpi_types.score_reduction_op, _rank_info._accu_results_colcomm, _col_result_comm, &col_request);
		}
	}

	// row-wise exclusion slice reduction
	if (_tile_info._partition_row < num_cols-1) { //last row requires very special treatment, i.e. does not participate in a excusion reduction
		if (_rank_info._exclusion_row_target == rank_in_exclusion_rowcomm) {
			MPI_Ireduce(MPI_IN_PLACE, _vert_result->profile.data()+tile_length-exclusionZone-1,
			            1,
			            _mpi_types.vert_prof_exclusionslice_mpitype, _mpi_types.score_reduction_op, _rank_info._exclusion_row_target, _exclusion_rowcomm, &row_exclusion_request);
			EXEC_DEBUG("world rank " << _rank_info._world_rank << " RECEIVING row reduction result");
		}
		else {
			MPI_Ireduce(_vert_result->profile.data()+tile_length-exclusionZone-1,
			            _vert_result->profile.data()+tile_length-exclusionZone-1,
			            1,
			            _mpi_types.vert_prof_exclusionslice_mpitype, _mpi_types.score_reduction_op, _rank_info._exclusion_row_target, _exclusion_rowcomm, &row_exclusion_request);
			EXEC_DEBUG("world rank " << _rank_info._world_rank << " SENDING row reduction result");
		}
	}
	// col-wise exclusion slice reduction
	if (_tile_info._partition_col > 0) {
		if (_rank_info._eexclusion_col_target == rank_in_exclusion_colcomm) {
			MPI_Ireduce(
			    MPI_IN_PLACE,
			    _hor_result->profile.data(),
			    1,
			    _mpi_types.hor_prof_exclusionslice_mpitype, _mpi_types.score_reduction_op, _rank_info._eexclusion_col_target, _exclusion_colcomm, &col_exclusion_request);
			EXEC_DEBUG("world rank " << _rank_info._world_rank << " RECEIVING col reduction result");
		}
		else
		{
			MPI_Ireduce(
			    _hor_result->profile.data(),
			    _hor_result->profile.data(),
			    1,
			    _mpi_types.hor_prof_exclusionslice_mpitype, _mpi_types.score_reduction_op, _rank_info._eexclusion_col_target, _exclusion_colcomm, &col_exclusion_request);
			EXEC_DEBUG("world rank " << _rank_info._world_rank << " SENDING in col reduction");
		}
	}

	// reduction of the very last excess among all processes in the last row ("vertically below the exclusion zone")
	/*if (_tile_info._partition_row == num_cols-1) //num_cols==num_rows...
	{
		if (_rank_info._rank_in_final_excesscomm == _rank_info._final_exess_target) {
			EXEC_DEBUG("world rank " << _rank_info._world_rank << " receiving in final excess reduction as rank " << _rank_info._rank_in_final_excesscomm);
			MPI_Ireduce(MPI_IN_PLACE, _hor_result->profile.data()+tile_length, 1, _mpi_types.final_excess_slice_mpitype, _mpi_types.score_reduction_op, _rank_info._final_exess_target, _final_excesscomm, &final_excess_request);
		}
		else {
			EXEC_DEBUG("world rank " << _rank_info._world_rank << " participating in final excess reduction with rank " << _rank_info._rank_in_final_excesscomm);
			MPI_Ireduce(
				_vert_result->profile.data()+tile_length-exclusionZone-1,
				_vert_result->profile.data()+tile_length-exclusionZone-1,
				1, _mpi_types.vert_prof_exclusionslice_mpitype, _mpi_types.score_reduction_op, _rank_info._final_exess_target, _final_excesscomm, &final_excess_request);
		}
	}*/

	MPI_Wait(&row_request, MPI_STATUS_IGNORE);
	MPI_Wait(&col_request, MPI_STATUS_IGNORE);
	MPI_Wait(&col_exclusion_request, MPI_STATUS_IGNORE);
	MPI_Wait(&row_exclusion_request, MPI_STATUS_IGNORE);
	MPI_Wait(&final_excess_request, MPI_STATUS_IGNORE);
	EXEC_DEBUG("world rank " << _rank_info._world_rank << " finished all reductions");

	TRACK_SYNCTIME( idle_reductions, MPI_COMM_WORLD );
	TIMEPOINT( tresultreductions );

	EXEC_DEBUG("apply transformation from score to distance values")
	const idx_dtype local_result_slice_len = (_tile_info._partition_col==(num_cols-1))?tile_length+exclusionZone+1:tile_length;
	tsa_dtype twice_m = 2.0*static_cast<tsa_dtype>(windowSize);
	auto profile = (_hor_result->profile).data();
	for (idx_dtype i = 0; i<local_result_slice_len; ++i) {
		profile[i] = twice_m - 2.0 * profile[i];
	}

	TIMEPOINT( tpostprocessing)

	if (rank_in_colcomm == _rank_info._accu_results_colcomm ) {
		EXEC_DEBUG(" world rank " << _rank_info._world_rank << "participating in result file output");

		if ( _prof_io_comm == MPI_COMM_NULL) {
			// validation, that all the I/O responsible processes actually are in the I/O communicator
			 EXEC_ERROR("world rank " << _rank_info._world_rank << " aka " << rank_in_colcomm << " in colcomm is not in the I/O communicator" )
			throw std::runtime_error("not in the I/O communicator");
		}
		int io_rank;
		MPI_Comm_rank(_prof_io_comm, &io_rank);

		const int write_length = (_tile_info._partition_col<num_cols-1)? local_result_slice_len : local_result_slice_len-input_padding;
		const int result_len = profile_length-input_padding;
		const int chunknum=(params.writing_chunk_size>0)? tile_length/params.writing_chunk_size : 1; // needs to be identical among all processes!

		MPI_Info outputinfo;
		MPI_Info_create(&outputinfo);
		MPI_Info_set(outputinfo, "access_style", "write_once");
		BinProfFile outputfile(params.output_filename, _prof_io_comm, outputinfo);
		outputfile.write_matrix_profile_slice(
		            _hor_result->profile.data(),
		            _hor_result->index.data(),
		            write_length,
		            col_offset,
		            result_len,
		            io_rank==0,
		            chunknum // NOTE: number of chunks MUST be identical among all processes
		            );
		MPI_Info_free(&outputinfo);
	}
	else if ( _prof_io_comm != MPI_COMM_NULL) {
		// validation of correct communicator construction
		EXEC_ERROR( "world rank " << _rank_info._world_rank << "is in io communicator, but not declared as responsible fore I/O");
		throw std::runtime_error("invalid communicator structure: wrong proc in I/O communicator");
	}

	TRACK_SYNCTIME( idle_writing, MPI_COMM_WORLD )
	TIMEPOINT( tfilewrite )

	// MPI cleanup
	MPI_Op_free(&_mpi_types.score_reduction_op);

#define FREE_NOTNULL_MPITYPE( typevar ) if (typevar != MPI_DATATYPE_NULL ) {MPI_Type_free(&typevar);}

	FREE_NOTNULL_MPITYPE(_mpi_types.final_excess_slice_mpitype);
	FREE_NOTNULL_MPITYPE(_mpi_types.vert_prof_exclusionslice_mpitype);
	FREE_NOTNULL_MPITYPE(_mpi_types.hor_prof_exclusionslice_mpitype);
	FREE_NOTNULL_MPITYPE(_mpi_types.vert_prof_mainslice_mpitype);
	FREE_NOTNULL_MPITYPE(_mpi_types.hor_prof_mainslice_mpitype);
	FREE_NOTNULL_MPITYPE(_mpi_types.vert_prof_mpitype);
	FREE_NOTNULL_MPITYPE(_mpi_types.hor_prof_mpitype);


	TIMEPOINT( tmpiclean)

	_timing_info.setup_time = (tsetup2-tinread) + (tsetup1-tstart);
	_timing_info.comm_setup_time = tsetup1-tinit;
	_timing_info.cleanup_time = tmpiclean-tfilewrite;
	_timing_info.bcast_time = tinput-tfileread;
	_timing_info.comm_time = _timing_info.bcast_time + (tresultreductions-tevaluations);
	_timing_info.dotproduct_time = tdotprods - tprecomputations;
	_timing_info.evaluation_time = tevaluations-tprecomputations;
	_timing_info.precomp_time = tprecomputations-tinput;
	_timing_info.comp_time = _timing_info.evaluation_time + _timing_info.precomp_time + (tpostprocessing-tresultreductions);
	_timing_info.work_time = _timing_info.comp_time + _timing_info.comm_time + _timing_info.setup_time + _timing_info.cleanup_time;
	_timing_info.reading_time = (tfileread - tsetup2) + (tinread-tsetup1);
	_timing_info.io_time = _timing_info.reading_time + (tfilewrite-tpostprocessing); //Actually also subsuming some setup time between tfileread and tstart

	LOG_SYNCTIME( idle_setup );
	LOG_SYNCTIME( idle_eval );
	LOG_SYNCTIME( idle_reading );
	LOG_SYNCTIME( idle_bcast );
	LOG_SYNCTIME( idle_reductions );
	LOG_SYNCTIME( idle_writing );

	// log exhaustive timing information
#define STORE_TIME_TRACE( timer, description ) _timing_info._time_trace_map.insert( TracepointMap::value_type(description, timer));
	STORE_TIME_TRACE( tstart, "tstart");
	STORE_TIME_TRACE( tsetup1, "tsetup1");
	STORE_TIME_TRACE( intermediate_reading, "interm_reading" );
	STORE_TIME_TRACE( tinread, "tinread");
	STORE_TIME_TRACE( tsetup2, "tsetup2");
	STORE_TIME_TRACE( tfileread, "tfileread");
	STORE_TIME_TRACE( tinput, "tinput" );
	STORE_TIME_TRACE( tprecomputations, "tprecomputations");
	STORE_TIME_TRACE( tdotprods, "tdotproducts");
	STORE_TIME_TRACE( tevaluations, "tevaluations");
	STORE_TIME_TRACE( tmerge, "tmerge");
	STORE_TIME_TRACE( tresultreductions, "tresultreductions");
	STORE_TIME_TRACE( tpostprocessing, "tpostprocessing");
	STORE_TIME_TRACE( tfilewrite, "tfilewrite");
	STORE_TIME_TRACE( tmpiclean, "mpi_cleanup")

    #ifdef PROFILING
	    PERF_LOG("number of matrix profile updates: " << _profile_updates);
	    PERF_LOG("number of evaluations: " << _eval_ctr);
    #endif
}


void ScrimpDistribPar::log_info() {
	EXEC_INFO( "MPI INFO: comm size: " << _rank_info._world_size << " world rank: " << _rank_info._world_rank);
	EXEC_INFO( "Blocking size: " << _log_info.blocklen)
	EXEC_INFO ( "input time series of length: " << _log_info.input_series_len );
	EXEC_INFO ( "apply padding: " << _log_info.input_padding );
	EXEC_INFO(" running distributed parallelization: applying padding of " << _log_info.input_padding << " samples to process tiles with base length " << _log_info.tile_length
	         << " input len " << _log_info.input_series_len << " time_series_len " << _log_info.time_series_len << " profile_length " << _log_info.profile_length << " worklen " << _log_info.work_triang_len << " tile length " << _log_info.tile_length);

	// log computation performance
	PERF_LOG ( "evaluation time: " << _timing_info.evaluation_time );
	PERF_LOG ( "dotproduct time: " << _timing_info.dotproduct_time );
	PERF_LOG ( "local comp time (eval+precomp): " << _timing_info.comp_time );
	PERF_LOG ( "communication time: " << _timing_info.comm_time );
	PERF_LOG ( "local working time: " << _timing_info.work_time );
	PERF_LOG ( "I/O time: " << _timing_info.io_time );
	PERF_LOG ( "reading time: " << _timing_info.reading_time)
	PERF_LOG ( "input communication time: " << _timing_info.bcast_time );
	PERF_LOG ( "setup time: " << _timing_info.setup_time );
	PERF_TRACE( "comm_setup time: " << _timing_info.comm_setup_time );
	PERF_LOG ( "cleanup time: " << _timing_info.cleanup_time );

	const size_t true_work_len = _log_info.tile_length-(_log_info.is_processing_padding?_log_info.input_padding:0);
	const size_t true_local_eval_amount = true_work_len*true_work_len;
	const size_t true_total_work_amount = (_log_info.work_triang_len-_log_info.input_padding)*(_log_info.work_triang_len-_log_info.input_padding);
	PERF_LOG ( "local throughput evaluations: " << std::setprecision(6) << true_local_eval_amount / get_seconds(_timing_info.evaluation_time) << " matrix entries/second" );
	PERF_LOG ( "throughput evaluations: " << std::setprecision(6) << true_total_work_amount / get_seconds(_timing_info.evaluation_time) << " matrix entries/second (estimate based on local)" );
	PERF_LOG ( "local throughput computations: " << std::setprecision(6) << true_local_eval_amount / get_seconds(_timing_info.comp_time) << " matrix entries/second" );
	PERF_LOG ( "throughput computations: " << std::setprecision(6) << true_total_work_amount / get_seconds(_timing_info.comp_time) << " matrix entries/second (estimate based on local)" );

	for (auto iter = _timing_info._time_trace_map.begin(); iter != _timing_info._time_trace_map.end(); ++iter) {
		PERF_TRACE ( "timepoint after " << iter->first << ": " << iter->second );
	}

	for (auto iter = _timing_info._synctime_map.begin(); iter != _timing_info._synctime_map.end(); ++iter) {
		PERF_LOG(" sync time " << iter->first << ": " << iter->second);
	}
}
