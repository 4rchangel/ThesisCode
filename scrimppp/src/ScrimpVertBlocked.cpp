#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <limits>
#include <vector>
#include <algorithm>
#include <string>
#include <sstream>
#include <chrono>

#include <boost/filesystem.hpp>

#include <ScrimpVertBlocked.hpp>
#include <logging.hpp>
#include <papiwrapper.hpp>
#include <kernels.h>

using namespace matrix_profile;

static FactoryRegistration<ScrimpVertBlocked> s_sequRegistr("scrimp_vert_blocked");
static const int notification_interval_iter = 10000;

void ScrimpVertBlocked::eval_diagonal_block(const int blocklen, aligned_int_vec& profileIndex, const aligned_tsdtype_vec& A, aligned_tsdtype_vec& initial_zs, const int windowSize, const aligned_tsdtype_vec& ASigmaInv, const aligned_tsdtype_vec& AMeanScaledSigSqrM, const int base_diag, aligned_tsdtype_vec& profile)
{
	const int profileLength = AMeanScaledSigSqrM.size();

#ifndef USE_INTRINSICS_KERNEL
	// scalar kernel. Intel compiler manages to auto-vectorize it...
	EXEC_DEBUG("invoking plain C vert blocked kernel");
	//iteration in diagonal direction for all of the blocked diagonals.
	    //the loop is expressed in terms og the row-coordinate of the first diagonal in the block
	for (int j=base_diag; j<profileLength; j++)
	{
		const int i=j-base_diag;
		tsa_dtype profile_i = profile[i];
		int index_i = profileIndex [i];

		//iteration over all diagonals in the block. Handling incomplete blocks with the "iterlimit"
		const int iterlimit = std::min(blocklen, profileLength-j);
		for (int blockiter=0; blockiter < iterlimit; ++blockiter)
		{
			const tsa_dtype corrScore = initial_zs[base_diag + blockiter]* (ASigmaInv[j+blockiter] * ASigmaInv[i]) - AMeanScaledSigSqrM[j+blockiter] * AMeanScaledSigSqrM[i] ;
			EXEC_TRACE ( "eval i: " << i << " j: " << j+blockiter << " lastz" << initial_zs[blockiter]);

			initial_zs[base_diag+blockiter] += A[j+blockiter+windowSize]*A[i+windowSize]  - A[j+blockiter]*A[i];

			if(corrScore > profile[j+blockiter]) {
				profile[j+blockiter] = corrScore;
				profileIndex [j+blockiter] = i;
			}

			if (corrScore > profile_i) {
				profile_i = corrScore;
				index_i = j+blockiter;
			}
		}
		//integration of the result in i direction into memory
		if (profile_i > profile[i]) {
			profile[i] = profile_i;
			profileIndex[i] = index_i;
		}
	}

#else
	    eval_diag_block_triangle(
		            profile.data(),
		            profileIndex.data(),
		            profile.data()+base_diag,
		            profileIndex.data()+base_diag,
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
#endif
}

void ScrimpVertBlocked::compute_matrix_profile(const Scrimppp_params& params)
{
	std::chrono::high_resolution_clock::time_point tstart, tend;
	std::chrono::duration<double> time_elapsed;
	aligned_tsdtype_vec A = fetch_time_series<aligned_tsdtype_vec::allocator_type>(params); //load the time series data
	aligned_tsdtype_vec AMeanScaledSqrtM(A.size());
	aligned_tsdtype_vec ASigmaInv(A.size());
	int windowSize = params.query_window_len;
	int exclusionZone = windowSize / 4;
	int timeSeriesLength = A.size();
	int ProfileLength = timeSeriesLength - windowSize + 1;
	const int BLOCKING_SIZE=params.block_length; // for convinience, after switching from compile time constant to optional runtime argument
	BOOST_ASSERT_MSG( ProfileLength>0, "No profile results from sepcified args. Choose a smaller window length / longer time series");
	//Initialize Matrix Profile and Matrix Profile Index
	aligned_tsdtype_vec profile(ProfileLength, -1.0);
	aligned_int_vec profileIndex(ProfileLength, 0);
	aligned_tsdtype_vec initial_zs(timeSeriesLength); // stores products between two Timeseries values with a distinct offset
	std::vector<int> idx; // store indices of the diagonals, defining their evaluation order
	idx.reserve(ProfileLength-exclusionZone-1);

	//several monitors for performance measurement
	PerfCounters setup_perf("setup");
	PerfCounters init_diag_perf("diagonal initialization");
	PerfCounters eval_diag_perf("diagonal evaluation");

	//validation of parameters
	if (timeSeriesLength < windowSize) {
		throw std::invalid_argument("ERROR: Time series is shorter than the window length, can not proceed");
	}

	EXEC_INFO( "Sequential SCRIMP matrix profile computation with profile length " << ProfileLength << " and window size " << windowSize);
	EXEC_INFO( "Blocking size: " << BLOCKING_SIZE)


	{
		ScopedPerfAccumulator monitor(setup_perf);
		//precompute the mean and standard deviations of the sliding windows along the time series
		precompute_window_statistics(windowSize, A, ProfileLength, AMeanScaledSqrtM, ASigmaInv);

	}
	// compute the first correlation values in the diagonals (i.e. compute the correlation between the first windows)
	{
		ScopedPerfAccumulator monitor(init_diag_perf);
		init_all_diagonals(initial_zs, A, windowSize);
	}
	{
		ScopedPerfAccumulator monitor(setup_perf);
		/******************** SCRIMP ********************/
		//Random shuffle the computation order of the diagonals of the distance matrix
		int i;
		for (i = exclusionZone+1; i <= ProfileLength; i+=BLOCKING_SIZE) {
			idx.push_back(i);
		}
		std::random_shuffle(idx.begin(), idx.end());
	}


	//start time measurment
	tstart = std::chrono::high_resolution_clock::now();

	//iteratively evaluate the diagonals of the distance matrix
	for (int ri = 0; ri < idx.size(); ri++)
	    {
		//select a random diagonal
		int diag = idx[ri];

		//evaluate the second to the last distance values along the diagonal in the matrix and update the matrix profile/matrix profile index.
		{
			ScopedPerfAccumulator monitor(eval_diag_perf);
			eval_diagonal_block(BLOCKING_SIZE, profileIndex, A, initial_zs, windowSize, ASigmaInv, AMeanScaledSqrtM, diag, profile);
		}

		//Show time per 10000 iterations
/*		if ((ri+1) % notification_interval_iter == 0)
		{
			tend = std::chrono::high_resolution_clock::now();
			time_elapsed = tend - tstart;
			EXEC_INFO ( "finished " << ri+1 << " blocked iterations after " << std::setprecision(std::numeric_limits<tsa_dtype>::digits10 + 2) << time_elapsed.count() << " seconds.");
		}*/
	}

	// apply a correction of the distance values, as we dropped a factor of 2 to avoid unnecessary computations
	tsa_dtype twice_m = 2.0*static_cast<tsa_dtype>(windowSize);
	for (auto iter=profile.begin(); iter<profile.end(); ++iter) {
		(*iter) = twice_m - 2.0 * (*iter);
	}

	// end timer
	// tend = time(0);
	tend = std::chrono::high_resolution_clock::now();
	time_elapsed = tend - tstart;

	PERF_LOG ( "total computation time: " << std::setprecision(std::numeric_limits<tsa_dtype>::digits10 + 2) << time_elapsed.count() << " seconds." );
	double triang_len = ProfileLength-exclusionZone;
	PERF_LOG ( "throughput computations: " << triang_len * triang_len / time_elapsed.count() << " matrix entries/second");

	//store the result
	store_matrix_profile(profile, profileIndex, params);

	setup_perf.log_perf();
	init_diag_perf.log_perf();
	eval_diag_perf.log_perf();
#ifdef PROFILING
	PERF_LOG ( "number of matrix profile updates: " << _profileUpdateCounter);
#endif

}
