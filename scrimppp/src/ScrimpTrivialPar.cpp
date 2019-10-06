#include <ScrimpTrivialPar.hpp>
#include <logging.hpp>
#include <timing.h>
#include <partitioning_1d.h>
#include <binproffile.h>

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
#include <kernels.h>

#include <boost/assert.hpp>
#include <boost/align.hpp>
#include <boost/filesystem.hpp>

#include <mpi.h>

#include <papiwrapper.hpp>

using namespace matrix_profile;

using DiagPartition = Partition1D;

// easy aligned allocation.
// code from https://www.boost.org/doc/libs/1_57_0/doc/html/align/examples.html#align.examples.aligned_ptr, licensed under the Boost Software License - Version 1.0
// see copy of the license terms provided under /licenses/boostSoftwareLicense.txt
template<class T>
using aligned_ptr = std::unique_ptr<T,
  boost::alignment::aligned_delete>;

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
// condinue own code


#define TIMEPOINT( varname ) const Timepoint varname = matrix_profile::get_cur_time();

using namespace matrix_profile;

static FactoryRegistration<ScrimpTrivialPar> s_trivParReg("scrimp_triv_par");
static const int notification_interval_iter = 10000;


void ScrimpTrivialPar::MPI_MatProfSOA_reduction(void * invec, void *inoutvec, int *len, MPI_Datatype *datatype) {
	MatProfSOA* inout = static_cast<MatProfSOA*>(inoutvec);
	const MatProfSOA* const in2 = static_cast<const MatProfSOA* const>(invec);

	EXEC_TRACE("reducing a MatProfSOA!");
	// retrieve the profile length: This is equal to the block length of the datastructure, not the length specified in the reduction!
	int num_int, num_add, num_dtype, combiner;
	MPI_Type_get_envelope(*datatype, &num_int, &num_add ,&num_dtype, &combiner);

	if (num_int>3 || num_add>3 || num_dtype >3){
		EXEC_ERROR("Invalid dataype sizes retrieved: num_i " << num_int
		           << " num_add " << num_add
		           << " num_dtype " << num_dtype
		           << "aborting as buffers are too small");
		MPI_Abort(MPI_COMM_WORLD, 1);
	}

	int blocklens[3];
	MPI_Aint addresses[3];
	MPI_Datatype dtypes[3];
	MPI_Type_get_contents(*datatype, num_int, num_add, num_dtype, blocklens, addresses, dtypes);

	EXEC_DEBUG("reduction with len " << *len << " and blocklen " << blocklens[1] <<", " << blocklens[2]);
	assert(blocklens[1] == blocklens[2]);
	assert(*len == 1); // only reduction with size 1 is supported: Otherwise strides between SOAs needed to be considered....

	//"shortcuts" for coding convenience
	tsa_dtype* inoutprof = inout->profile.data();
	idx_dtype* inout_idx = inout->index.data();
	const tsa_dtype* const in2_prof = in2->profile.data();
	const idx_dtype* const in2_idx = in2->index.data();

	//finally the reduction
	const idx_dtype limit = blocklens[1]; // avoid unnecessary reads
	for (idx_dtype i = 0; i < limit; ++i) {
		if (in2_prof[i] < inoutprof[i]) {
			inoutprof[i] = in2_prof[i];
			inout_idx[i] = in2_idx[i];
		}
	}
}

void ScrimpTrivialPar::eval_diagonal_block(MatProfSOA& result,
           const int blocklen,
           const aligned_tsdtype_vec& A,
           aligned_tsdtype_vec& initial_zs,
           const int windowSize,
           const aligned_tsdtype_vec& ASigmaInv,
           const aligned_tsdtype_vec& AMeanScaledSigSqrM,
           const int base_diag
           )
{
EXEC_TRACE("evaluate block of diagonals: first " << base_diag << " last " << base_diag+blocklen-1);
    const int profileLength = AMeanScaledSigSqrM.size();
	eval_diag_block_triangle(
	            result.profile.data(),
	            result.index.data(),
	            result.profile.data()+base_diag,
	            result.index.data()+base_diag,
	            initial_zs.data()+base_diag,
	            blocklen,
	            profileLength-base_diag,
	            A.data(),
	            A.data()+base_diag,
	            windowSize,
	            ASigmaInv.data(),
	            AMeanScaledSigSqrM.data(),
	            ASigmaInv.data()+base_diag,
	            AMeanScaledSigSqrM.data()+base_diag,
	            base_diag,
	            0
	    );
}

void ScrimpTrivialPar::compute_matrix_profile(const Scrimppp_params& params) {
	TIMEPOINT( tstart );
	aligned_tsdtype_vec A = fetch_time_series<aligned_tsdtype_vec::allocator_type>(params); //load the time series data
	TIMEPOINT( tfileread );
	aligned_tsdtype_vec AMean(A.size());
	aligned_tsdtype_vec ASigma(A.size());
	aligned_tsdtype_vec dotproducts(A.size());
	idx_dtype windowSize = params.query_window_len;
	idx_dtype exclusionZone = windowSize / 4;
	idx_dtype timeSeriesLength = A.size();
	idx_dtype profile_length = timeSeriesLength - windowSize + 1;
	auto result = make_aligned<MatProfSOA>();
	const int BLOCKING_SIZE = params.block_length;
	const bool distrib_io = (params.filetype == Scrimppp_params::BIN) && params.use_distributed_io;

	if (params.filetype != Scrimppp_params::BIN && params.use_distributed_io) {
		EXEC_ERROR("distributed I/O is NOT supported with ASCII files. Falling back to master I/O!");
	}

	// code instrumentaion (if enabled)
	PerfCounters ctrs_diaginit("diagonal initialization");
	PerfCounters ctrs_eval("block evaluations");
	PerfCounters ctrs_mpi_comm("MPI communication");

	//Initialize Matrix Profile and Matrix Profile Index
	result->profile.fill(-1.0);
	result->index.fill(-1);

	int world_size;
	int world_rank;
	MPI_Datatype mpi_result_type;
	MPI_Op reduction_op;
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);
	MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);



	if (profile_length > RESULT_ALLOC_LEN) { // check, whether the data fit into the preallocated result storage
		throw std::runtime_error("resulting profile length EXCEEDS MAXIMUM LENGTH. Recompile with different allocation size or choose a smaller problem");
	}
	else if(profile_length<=0) {
		throw std::runtime_error("parameters chosen badly: resulting matrix profile of length 0. done.");
	}

	//create the MPI dataype for the result
	const int blocklen[] = {static_cast<int>(profile_length),static_cast<int>(profile_length)};
	const MPI_Aint disps[] = {(MPI_Aint)((size_t)((result->profile).data()) - (size_t)(result.get())),
	                          (MPI_Aint)((size_t)((result->index).data()) - (size_t)(result.get())) };
	//assertion for DOUBLE usage!!! TODO employ some preprocessor or tempalte magic...
	static_assert(sizeof(tsa_dtype) == sizeof(double), "NO FLOAT POSSIBLE without modifing MPI Datatype!" );
	static_assert(std::is_same<idx_dtype, long>::value, "Mismatch between idx_dtype and respective mpi type MPI_LONG" );
	const MPI_Datatype dtypes[] = {MPI_DOUBLE, MPI_LONG};
	MPI_Type_create_struct(2, blocklen, disps, dtypes, &mpi_result_type);
	MPI_Type_commit(&mpi_result_type);

	//create the reduction operation for the custom datatype
	MPI_Op_create(MPI_MatProfSOA_reduction, 1, &reduction_op);

	// values regarding work partitioning
	const int num_partitions = 2*world_size; //number of partitions. 2 partitions per process for the sake of balancing
	const int diags_to_process_world = profile_length-exclusionZone-1; //total number of diagonals in the "adjacency" matrix to process
	// likely the partitions can not be of exactly even size. We assume the first ones to be of partition_size_full_load and the remaining ones to be one element smaller
	const PartitioningInfo partinfo = {
	    ._num_partitions = 2*world_size,
	    ._num_partitions_full_load = diags_to_process_world - ((diags_to_process_world/num_partitions) * num_partitions),
	    ._size_full_load = (diags_to_process_world/num_partitions) + 1
	};

	/*const bool proc_has_reduced_load = world_rank<num_reduced_load;
	const int diags_to_process_proc = (proc_has_reduced_load?diags_to_process_reduced_load:diags_to_process_full_load);
	const int proc_offset = (proc_has_reduced_load? world_rank*diags_to_process_reduced_load
													: diags_to_process_reduced_load*num_reduced_load
														+diags_to_process_full_load*(world_rank-num_reduced_load));*/

	// retrieve the local partitions: each process gets 2. A "long" one in the upper half of the triangle and a "short" one in the lower half of the triangle
	std::vector<DiagPartition> local_partitions;
	    // the upper partition is the "process-rank"-th one
	DiagPartition part_offsets = get_partition(world_rank, partinfo);
//NOTE: causes a integer conversion warning. It is not of interest, as using more than MAX_INT diagonals is far beyond the memory limit of a single node (currently)
	local_partitions.push_back( {exclusionZone+1+part_offsets._first_id, std::min(profile_length-1, exclusionZone+1+part_offsets._last_id)} );
	part_offsets = get_partition(num_partitions-world_rank-1, partinfo);
//NOTE: causes a integer conversion warning. It is not of interest, as using more than MAX_INT diagonals is far beyond the memory limit of a single node
	local_partitions.push_back( {exclusionZone+1+part_offsets._first_id, std::min(profile_length-1, exclusionZone+1+part_offsets._last_id)} );

	//sum up, how many diags the local proc has to evaluate
	int diags_to_process_proc = 0;
	for (auto partit = local_partitions.begin(); partit < local_partitions.end(); ++partit) {
		diags_to_process_proc += partit->_last_id - partit->_first_id +1;
	}

#ifndef NDEBUG
	// sum up over all processes
	int num_processed_diags = 0;
	MPI_Reduce(&diags_to_process_proc, &num_processed_diags, 1, MPI_INTEGER, MPI_SUM, 0, MPI_COMM_WORLD);
	if (world_rank == 0) {
		BOOST_ASSERT_MSG(num_processed_diags==diags_to_process_world, "work partitioning of the diagonals is ivalid");
	}
	EXEC_INFO("partitioning is fine!");
#endif

//	EXEC_INFO("Processing all the matrix profile!");
//	local_partitions.clear();
//	local_partitions.push_back( {exclusionZone+1, ProfileLength-1} );
/*
	const int tmprank=0;
	EXEC_INFO("Processing the rank" << tmprank << " partition of the matrix profile!");
	local_partitions.clear();
	part_offsets = get_partition(tmprank, partinfo);
	local_partitions.push_back( {exclusionZone+1+part_offsets._first_id, std::min(ProfileLength-1, exclusionZone+1+part_offsets._last_id)} );
	part_offsets = get_partition(num_partitions-tmprank-1, partinfo);
	local_partitions.push_back( {exclusionZone+1+part_offsets._first_id, std::min(ProfileLength-1, exclusionZone+1+part_offsets._last_id)} );
*/

	//validation of parameters
	if (timeSeriesLength < windowSize) {
		throw std::invalid_argument("ERROR: Time series is shorter than the window length, can not proceed");
	}

	TIMEPOINT( tsetup );

	//precompute the mean and standard deviations of the sliding windows along the time series
	precompute_window_statistics(windowSize, A, profile_length, AMean, ASigma);

	//start time measurment
	TIMEPOINT( tprecomputations );

	// randomly shuffle the comutation order of diagonal blocks (according to the blocking size).
	// In order to do so split the local partition into blocks and shuffle them.
	std::vector<DiagPartition> eval_blocks;
	eval_blocks.reserve((diags_to_process_proc/BLOCKING_SIZE) +1);

	for (auto partit = local_partitions.begin(); partit < local_partitions.end(); ++partit) {
		    // split the partitions into block which preserve cache locality
		const int partlen = partit->_last_id - partit->_first_id;
		const int num_prolonged_blocks = std::max(partlen % BLOCKING_SIZE, partlen/BLOCKING_SIZE) ;
		int prev_block_end = partit->_first_id-1;

		EXEC_TRACE("splitting into blocks a partition with start: " << partit->_first_id << " and last: " << partit->_last_id);

		for ( int blocki = 0; prev_block_end < partit->_last_id; ++blocki) {
			DiagPartition part;
			part._first_id = prev_block_end+1;
			if (blocki < num_prolonged_blocks) {
				part._last_id = std::min(part._first_id + BLOCKING_SIZE-1, partit->_last_id);
			}
			else {
				part._last_id = std::min(part._first_id+BLOCKING_SIZE-1, partit->_last_id);
			}
			eval_blocks.push_back( part );
			prev_block_end = part._last_id;

			{
				const ScopedPerfAccumulator monitor(ctrs_diaginit);
				init_diagonals(part._first_id, part._last_id, dotproducts, A, windowSize);
			}
		}
	}
	//std::random_shuffle(eval_blocks.begin(), eval_blocks.end());

	TIMEPOINT( tpartitioning )
	EXEC_DEBUG( "Proc " << world_rank << " shall process " << diags_to_process_proc << " diagonals");

	//init_all_diagonals(dotproducts, A, windowSize);

	//iteratively evaluate the diagonals of the distance matrix
	int progressctr=0;
	int next_notification_thresh = notification_interval_iter;
	for (auto blockiter = eval_blocks.begin(); blockiter < eval_blocks.end(); ++blockiter) {
		const int blocklen = blockiter->_last_id +1 - blockiter->_first_id;

		// compute distance of the first entry for each entry in the block

		    // might be more efficient, if performed with MASS for ech partition (when splitting them up into blocks)


		EXEC_DEBUG("Rank " << world_rank << " evaluating diags " << blockiter->_first_id << " to " << blockiter->_last_id);
		{
			const ScopedPerfAccumulator monitor(ctrs_eval);
			eval_diagonal_block(*result, blocklen, A, dotproducts, windowSize,  ASigma, AMean, blockiter->_first_id);
			progressctr+=blocklen;
		}
		//Show time per 10000 iterations
		if (progressctr > next_notification_thresh && false) //disable in or der for the logging not to disturb the timing
		{
			const auto tcur = get_cur_time();
			matrix_profile::Timespan time_elapsed = tcur - tpartitioning;
			EXEC_INFO ("finished " << progressctr << " iterations in: " << std::setprecision(4) << time_elapsed);
			next_notification_thresh += notification_interval_iter;
		}
	}

	TIMEPOINT( tevaluations)
	EXEC_DEBUG("apply transformation from score to distance values")
	// apply a correction of the distance values, as we dropped a factor of 2 to avoid unnecessary computations
	tsa_dtype twice_m = 2.0*static_cast<tsa_dtype>(windowSize);
	auto profile = (result->profile).data();
	for (int i = 0; i<=profile_length; ++i) {
		profile[i] = twice_m - 2.0 * profile[i];
	}
	TIMEPOINT( tpostprocessing)
	{
		const ScopedPerfAccumulator monitor(ctrs_mpi_comm);
		EXEC_DEBUG("reduce the processes individual results")
		//update matrix profile and matrix profile index if the current distance value is smaller
		if (distrib_io) { //distributed I/O => everyone needs the result => Allreduce
			// actually only a fraction is required: the partition which will be written. That would require several reductions (one for each partition), where the receiver is the process responsible for the partition...
			MPI_Allreduce(MPI_IN_PLACE, result.get(), 1, mpi_result_type, reduction_op, MPI_COMM_WORLD); // in case of distributed binary I/O everyone needs the result
		}
		else { // reduction into master only, in case it is the only one writing...
			if (world_rank != 0) {
				MPI_Reduce(result.get(), result.get(), 1, mpi_result_type, reduction_op, 0, MPI_COMM_WORLD); // Reduction of length 1, as the datatype is defined as ProfileLength!
			}
			else {
				MPI_Reduce(MPI_IN_PLACE, result.get(), 1, mpi_result_type, reduction_op, 0, MPI_COMM_WORLD); // Reduction of length 1, as the datatype is defined as ProfileLength!
			}
		}
		EXEC_DEBUG( "rank " << world_rank << " done with reduction");
	}

	TIMEPOINT( tcommunication)
	// MPI_Barrier(MPI_COMM_WORLD); //just for "debugging"

	//store the result
	if (world_rank == 0 || distrib_io ){ // storing in parallel if BIN type specified...
		EXEC_DEBUG("Rank " << world_rank << "storing the result");
		store_matrix_profile(*result, params, profile_length, distrib_io);
	}

	TIMEPOINT( tfilewrite )

	// MPI cleanup
	MPI_Op_free(&reduction_op);
	MPI_Type_free(&mpi_result_type);

	TIMEPOINT( tmpiclean)

	_timing_info.dotproduct_time = (tpartitioning-tprecomputations);
	_timing_info.setup_time = (tsetup-tfileread) + (tmpiclean-tfilewrite);
	_timing_info.comm_time = (tcommunication-tpostprocessing) ;
	_timing_info.evaluation_time = tevaluations-tpartitioning;
	_timing_info.precomp_time = tprecomputations-tsetup;
	_timing_info.comp_time = _timing_info.evaluation_time + _timing_info.dotproduct_time +_timing_info.precomp_time + (tpostprocessing-tevaluations);
	_timing_info.work_time = _timing_info.comp_time + _timing_info.comm_time + _timing_info.setup_time;
	_timing_info.io_time = (tfileread - tstart) + (tfilewrite-tcommunication);

#define STORE_TIME_TRACE( timer, description ) _timing_info._time_trace_map.insert( TracepointMap::value_type(description, timer));
	STORE_TIME_TRACE( tstart, "start");
	STORE_TIME_TRACE( tfileread, "fileread");
	STORE_TIME_TRACE( tsetup, "setup");
	STORE_TIME_TRACE( tprecomputations, "precomputations");
	STORE_TIME_TRACE( tpartitioning, "partitioning");
	STORE_TIME_TRACE( tevaluations, "matrix_eval");
	STORE_TIME_TRACE( tpostprocessing, "postprocessing");
	STORE_TIME_TRACE( tcommunication, "communication");
	STORE_TIME_TRACE( tfilewrite, "filwrite");
	STORE_TIME_TRACE( tmpiclean, "mpi_cleanup");

	ctrs_diaginit.log_perf();
	ctrs_eval.log_perf();
	ctrs_mpi_comm.log_perf();

	_log_info.profile_length = profile_length;
	_log_info.windowSize = windowSize;
	_log_info.exclusionZone =  exclusionZone;
	_log_info.timeSeriesLength = timeSeriesLength;
	_log_info.world_size = world_size;
	_log_info.world_rank = world_rank;
	_log_info.BLOCKING_SIZE = BLOCKING_SIZE;
	_log_info.diags_to_process_proc = diags_to_process_proc;
}

void ScrimpTrivialPar::store_matrix_profile(const MatProfSOA& result, const Scrimppp_params& params, const idx_dtype profileLength, const bool distributed_io)
{
	ScrimpSequ::store_matrix_profile(result.profile.data(), result.index.data(), profileLength, params, distributed_io);
}

void ScrimpTrivialPar::log_info() {
	EXEC_INFO( "MPI INFO: comm size: " << _log_info.world_size << " world rank: " << _log_info.world_rank);
	EXEC_INFO( "Blocking size: " << _log_info.BLOCKING_SIZE)
    #ifdef PROFILING
	    PERF_LOG("number of matrix profile updates: " << _profile_updates);
	    PERF_LOG("number of evaluations: " << _eval_ctr);
    #endif

	// log computation performance
	const double triang_len = _log_info.profile_length-_log_info.exclusionZone;
	auto timerprecision = std::numeric_limits<tsa_dtype>::digits10 + 2;
	PERF_LOG ( "evaluation time: " << std::setprecision(timerprecision) << _timing_info.evaluation_time);
	PERF_LOG ( "local comp time (eval+precomp): " << std::setprecision(timerprecision) << _timing_info.comp_time );
	PERF_LOG ( "communication time: " << std::setprecision(timerprecision) << _timing_info.comm_time );
	PERF_LOG ( "local working time: " << std::setprecision(timerprecision) << _timing_info.work_time );
	PERF_LOG ( "I/O time: " << std::setprecision(timerprecision) << _timing_info.io_time );
	PERF_LOG ( "dotproduct time: " << _timing_info.dotproduct_time );

	PERF_LOG ( "local throughput evaluations: " << std::setprecision(6) << _log_info.diags_to_process_proc * triang_len / get_seconds(_timing_info.evaluation_time) << " matrix entries/second" );
	PERF_LOG ( "throughput evaluations: " << std::setprecision(6) << triang_len * triang_len / get_seconds(_timing_info.evaluation_time) << " matrix entries/second (estimate based on local)" );
	PERF_LOG ( "local throughput computations: " << std::setprecision(6) << _log_info.diags_to_process_proc * triang_len / get_seconds(_timing_info.comp_time) << " matrix entries/second" );
	PERF_LOG ( "throughput computations: " << std::setprecision(6) << triang_len * triang_len / get_seconds(_timing_info.comp_time) << " matrix entries/second (estimate based on local)" );

	for (auto iter = _timing_info._time_trace_map.begin(); iter != _timing_info._time_trace_map.end(); ++iter) {
		PERF_TRACE ( "timepoint after " << iter->first << ": " << iter->second );
	}

	for (auto iter = _timing_info._synctime_map.begin(); iter != _timing_info._synctime_map.end(); ++iter) {
		PERF_LOG(" sync time " << iter->first << ": " << iter->second);
	}
}
