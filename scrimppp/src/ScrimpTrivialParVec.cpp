#include <ScrimpTrivialParVec.hpp>
#include <logging.hpp>
#include <partitioning_1d.h>
#include <binproffile.h>
#include <settings.h>

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
#include <chrono>
#include <cassert>
#include <memory>
#include <cstddef>

#include <boost/assert.hpp>
#include <boost/align.hpp>
#include <boost/filesystem.hpp>

#include <mpi.h>

#include <papiwrapper.hpp>

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


#define TIMEPOINT( varname ) const std::chrono::steady_clock::time_point varname = std::chrono::steady_clock::now();

using namespace matrix_profile;

static FactoryRegistration<ScrimpTrivialParVec> s_trivParReg("triv_par_vec");
static const int notification_interval_iter = 10000;

void commit_mpi_SOA_prof_type(MPI_Datatype* ptr_mpi_type, tsa_dtype* profile_buf, idx_dtype* idx_buf, int profile_length)
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

void decompose_datatype(
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

void ScrimpTrivialParVec::MPI_MatProfSOA_reduction(void * invec, void *inoutvec, int *len, MPI_Datatype *datatype) {
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

void ScrimpTrivialParVec::compute_matrix_profile(const Scrimppp_params& params) {
	TIMEPOINT( tstart );
	aligned_tsdtype_vec A = fetch_time_series<aligned_tsdtype_vec::allocator_type>(params); //load the time series data
	TIMEPOINT( tfileread );
	aligned_tsdtype_vec AMean(A.size());
	aligned_tsdtype_vec ASigma(A.size());
	aligned_tsdtype_vec dotproducts(A.size());
	int windowSize = params.query_window_len;
	int exclusionZone = windowSize / 4;
	int timeSeriesLength = A.size();
	int profile_length = timeSeriesLength - windowSize + 1;
	aligned_tsdtype_vec profile(profile_length, -1.0);
	aligned_int_vec profileIndex(profile_length, 0);
	const int BLOCKING_SIZE = params.block_length;
	const bool distrib_io = (params.filetype == Scrimppp_params::BIN) && params.use_distributed_io;

	if (params.filetype != Scrimppp_params::BIN && params.use_distributed_io) {
		EXEC_ERROR("distributed I/O is NOT supported with ASCII files. Falling back to master I/O!");
	}

	// code instrumentaion (if enabled)
	PerfCounters ctrs_diaginit("diagonal initialization");
	PerfCounters ctrs_eval("block evaluations");
	PerfCounters ctrs_mpi_comm("MPI communication");

	// retrieve basic MPI information
	int world_size;
	int world_rank;
	MPI_Datatype mpi_result_type;
	MPI_Op reduction_op;
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);
	MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

	EXEC_INFO( "MPI INFO: comm size: " << world_size << " world rank: " << world_rank);
	EXEC_INFO( "Blocking size: " << BLOCKING_SIZE)

	if (profile_length > RESULT_ALLOC_LEN) { // check, whether the data fit into the preallocated result storage
		throw std::runtime_error("resulting profile length EXCEEDS MAXIMUM LENGTH. Recompile with different allocation size or choose a smaller problem");
	}
	else if(profile_length<=0) {
		throw std::runtime_error("parameters chosen badly: resulting matrix profile of length 0. done.");
	}

	//create the MPI dataype for the result
	commit_mpi_SOA_prof_type(&mpi_result_type, profile.data(), profileIndex.data(), profile_length );

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
	local_partitions.push_back( {exclusionZone+1+part_offsets._first_id, std::min(profile_length-1, exclusionZone+1+part_offsets._last_id)} );
	part_offsets = get_partition(num_partitions-world_rank-1, partinfo);
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
		{
			const ScopedPerfAccumulator monitor(ctrs_diaginit);
			init_diagonals(blockiter->_first_id, blockiter->_last_id, dotproducts, A, windowSize);
		}

		EXEC_DEBUG("Rank " << world_rank << " evaluating diags " << blockiter->_first_id << " to " << blockiter->_last_id);
		{
			const ScopedPerfAccumulator monitor(ctrs_eval);
			eval_diagonal_block(blocklen, profileIndex, A, dotproducts, windowSize, ASigma, AMean, blockiter->_first_id, profile);

			progressctr+=blocklen;
		}
		//Show time per 10000 iterations
		if (progressctr > next_notification_thresh && false) //disable in or der for the logging not to disturb the timing
		{
			const auto tcur = std::chrono::steady_clock::now();
			std::chrono::duration<float> time_elapsed = tcur - tpartitioning;
			EXEC_INFO ("finished " << progressctr << " iterations in: " << std::setprecision(4) << time_elapsed.count() << " seconds.");
			next_notification_thresh += notification_interval_iter;
		}
	}

	TIMEPOINT( tevaluations)
	EXEC_DEBUG("apply transformation from score to distance values")
	// apply a correction of the distance values, as we dropped a factor of 2 to avoid unnecessary computations
	tsa_dtype twice_m = 2.0*static_cast<tsa_dtype>(windowSize);
	for (int i = 0; i<=profile_length; ++i) {
		profile[i] = twice_m - 2.0 * profile[i];
	}
	TIMEPOINT( tpostprocessing)
	{
		const ScopedPerfAccumulator monitor(ctrs_mpi_comm);
		EXEC_INFO("reduce the processes individual results")
		//update matrix profile and matrix profile index if the current distance value is smaller
		if (distrib_io) { //distributed I/O => everyone needs the result => Allreduce
			// actually only a fraction is required: the partition which will be written. That would require several reductions (one for each partition), where the receiver is the process responsible for the partition...
			MPI_Allreduce(MPI_IN_PLACE, profile.data(), 1, mpi_result_type, reduction_op, MPI_COMM_WORLD); // in case of distributed binary I/O everyone needs the result
		}
		else { // reduction into master only, in case it is the only one writing...
			if (world_rank != 0) {
				MPI_Reduce(profile.data(), profile.data(), 1, mpi_result_type, reduction_op, 0, MPI_COMM_WORLD); // Reduction of length 1, as the datatype is defined as ProfileLength!
			}
			else {
				MPI_Reduce(MPI_IN_PLACE, profile.data(), 1, mpi_result_type, reduction_op, 0, MPI_COMM_WORLD); // Reduction of length 1, as the datatype is defined as ProfileLength!
			}
		}
		EXEC_INFO( "rank " << world_rank << " done with reduction");
	}

	TIMEPOINT( tcommunication)
	// MPI_Barrier(MPI_COMM_WORLD); //just for "debugging"

	//store the result
	if (world_rank == 0 || distrib_io ){ // storing in parallel if BIN type specified...
		EXEC_DEBUG("Rank " << world_rank << "storing the result");
		store_matrix_profile(profile.data(), profileIndex.data(), profile_length, params, distrib_io);
	}

	TIMEPOINT( tfilewrite )

	// MPI cleanup
	MPI_Op_free(&reduction_op);
	MPI_Type_free(&mpi_result_type);

	TIMEPOINT( tmpiclean)

	std::chrono::duration<double> setup_time, comp_time, comm_time, evaluation_time, io_time, work_time, precomp_time;
	setup_time = tsetup-tfileread + tmpiclean-tfilewrite + tpartitioning-tprecomputations;
	comm_time = tcommunication-tpostprocessing ;
	evaluation_time = tevaluations-tpartitioning;
	precomp_time = tprecomputations-tsetup;
	comp_time = evaluation_time + precomp_time + tpostprocessing-tevaluations;
	work_time = comp_time + comm_time + setup_time;
	io_time = tfileread - tstart + tfilewrite-tcommunication;
	// log computation performance
	const double triang_len = profile_length-exclusionZone;
	auto timerprecision = std::numeric_limits<tsa_dtype>::digits10 + 2;
	PERF_LOG ( "evaluation time: " << std::setprecision(timerprecision) << evaluation_time.count() << " seconds." );
	PERF_LOG ( "local comp time (eval+precomp): " << std::setprecision(timerprecision) << comp_time.count() << " seconds." );
	PERF_LOG ( "communication time: " << std::setprecision(timerprecision) << comm_time.count() << " seconds" );
	PERF_LOG ( "local working time: " << std::setprecision(timerprecision) << work_time.count() << " seconds" );
	PERF_LOG ( "I/O time: " << std::setprecision(timerprecision) << io_time.count() << " seconds" );

	PERF_LOG ( "local throughput evaluations: " << std::setprecision(6) << diags_to_process_proc * triang_len / evaluation_time.count() << " matrix entries/second" );
	PERF_LOG ( "throughput evaluations: " << std::setprecision(6) << triang_len * triang_len / evaluation_time.count() << " matrix entries/second (estimate based on local)" );
	PERF_LOG ( "local throughput computations: " << std::setprecision(6) << diags_to_process_proc * triang_len / comp_time.count() << " matrix entries/second" );
	PERF_LOG ( "throughput computations: " << std::setprecision(6) << triang_len * triang_len / comp_time.count() << " matrix entries/second (estimate based on local)" );

	using unit_seconds = std::chrono::duration<double>;
	// log exhaustive timing information
#define TIME_TRACE( timer, description ) PERF_TRACE ( "timepoint after " << description << ": " << \
	    std::setprecision(timerprecision) << std::chrono::duration_cast< unit_seconds >(timer.time_since_epoch()).count() << " seconds");
	TIME_TRACE( tstart, "start");
	TIME_TRACE( tfileread, "fileread");
	TIME_TRACE( tsetup, "setup");
	TIME_TRACE( tprecomputations, "precomputations");
	TIME_TRACE( tpartitioning, "partitioning");
	TIME_TRACE( tevaluations, "matrix_eval");
	TIME_TRACE( tpostprocessing, "postprocessing");
	TIME_TRACE( tcommunication, "communication");
	TIME_TRACE( tfilewrite, "postprocessing");
	TIME_TRACE( tmpiclean, "mpi_cleanup")


    #ifdef PROFILING
	    PERF_LOG("number of matrix profile updates: " << _profile_updates);
	    PERF_LOG("number of evaluations: " << _eval_ctr);
    #endif
	ctrs_diaginit.log_perf();
	ctrs_eval.log_perf();
	ctrs_mpi_comm.log_perf();
}
