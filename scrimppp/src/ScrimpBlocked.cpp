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
#include <timing.h>

#include <boost/filesystem.hpp>

#include <ScrimpBlocked.hpp>
#include <logging.hpp>
#include <papiwrapper.hpp>

using namespace matrix_profile;

static FactoryRegistration<ScrimpSequBlocked> s_sequRegistr("scrimp_blocked");
static const int notification_interval_iter = 10000;
static const int PRIMARY_BLOCKING_SIZE = 500;
static const int SECONDARY_BLOCKING_SIZE = 50;

template<int NUM_DIAGS, int BLOCKLEN> void ScrimpSequBlocked::eval_diagonal_block(aligned_int_vec& profileIndex, const aligned_tsdtype_vec& A, aligned_tsdtype_vec& initial_zs, const int windowSize, const aligned_tsdtype_vec& ASigmaInv, const aligned_tsdtype_vec& AMeanScaledSigSqrM, const int diag, aligned_tsdtype_vec& profile)
{
	const int profileLength = AMeanScaledSigSqrM.size();
	tsa_dtype corrScore;
#ifdef PROFILING
	long updateCtr = 0;
#endif

	for (int blockstart = 0; blockstart < profileLength; blockstart+= BLOCKLEN)
	{
		int block_lim = std::min(blockstart+BLOCKLEN, profileLength);
		const int diag_lim = std::min(NUM_DIAGS, profileLength-diag-blockstart);

		for (int diag_iter = 0; diag_iter<diag_lim; ++diag_iter) {
			int ilimit =  std::min(profileLength-diag-diag_iter, block_lim);
			tsa_dtype tmpz = initial_zs[diag+diag_iter]; //rather use a local, to avoid innecessary writes to the referenced memory

//std::cout << "diag " << diag+diag_iter << " lastz: " << tmpz << std::endl;
			for (int i=blockstart; i<ilimit; i++)
			{
				int j=i+diag+diag_iter;

//std::cout << "eval i: " << i << " j: " << j << "  ilimit:" << ilimit << " blockstart: " << blockstart << " lastz" << tmpz << std::endl;
				corrScore = (tmpz* (ASigmaInv[j] * ASigmaInv[i]) - AMeanScaledSigSqrM[j] * AMeanScaledSigSqrM[i]) ;
				tmpz += A[j+windowSize]*A[i+windowSize]  - A[j]*A[i];

				if (corrScore > profile[j])
				{
					profile[j] = corrScore;
					profileIndex [j] = i;
                    #ifdef PROFILING
					updateCtr+=1;
                    #endif
				}
				if (corrScore > profile[i])
				{
					profile[i] = corrScore;
					profileIndex [i] = j;
                    #ifdef PROFILING
					updateCtr+=1;
                    #endif
				}
			}
			initial_zs[diag+diag_iter]=tmpz;
		}
	}

#ifdef PROFILING
	_profileUpdateCounter += updateCtr;
#endif
}

void ScrimpSequBlocked::compute_matrix_profile(const Scrimppp_params& params)
{
	Timepoint tstart, tend;
	Timespan time_elapsed;
	aligned_tsdtype_vec A = fetch_time_series<aligned_tsdtype_vec::allocator_type>(params); //load the time series data
	aligned_tsdtype_vec AMeanScaledSqrtM(A.size());
	aligned_tsdtype_vec ASigmaInv(A.size());
	int windowSize = params.query_window_len;
	int exclusionZone = windowSize / 4;
	int timeSeriesLength = A.size();
	int ProfileLength = timeSeriesLength - windowSize + 1;
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

	EXEC_INFO( "Sequential blocked SCRIMP matrix profile computation with profile length " << ProfileLength << " window size " << windowSize);
	EXEC_INFO ( "diagonal BLOCK size" << PRIMARY_BLOCKING_SIZE << " (secondary) blocking toegether " << SECONDARY_BLOCKING_SIZE << " diagonals");

	{
		ScopedPerfAccumulator monitor(setup_perf);
		//precompute the mean and standard deviations of the sliding windows along the time series
		precompute_window_statistics(windowSize, A, ProfileLength, AMeanScaledSqrtM, ASigmaInv);

		//start time measurment
		tstart = get_cur_time();
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
		for (i = exclusionZone+1; i <= ProfileLength-SECONDARY_BLOCKING_SIZE; i+=SECONDARY_BLOCKING_SIZE) {
			idx.push_back(i);
		}
		std::random_shuffle(idx.begin(), idx.end());
		//sequentially evaluate diagonals, which did not fit the vector size
		for (i;i<ProfileLength; ++i)
		{
			eval_diagonal_block<1, PRIMARY_BLOCKING_SIZE>(profileIndex, A, initial_zs, windowSize, ASigmaInv, AMeanScaledSqrtM, i, profile);
		}
	}

	//iteratively evaluate the diagonals of the distance matrix
	for (int ri = 0; ri < idx.size(); ri++)
	    {
		//select a random diagonal
		int diag = idx[ri];

		//evaluate the second to the last distance values along the diagonal in the matrix and update the matrix profile/matrix profile index.
		{
			ScopedPerfAccumulator monitor(eval_diag_perf);
			eval_diagonal_block<SECONDARY_BLOCKING_SIZE, PRIMARY_BLOCKING_SIZE>(profileIndex, A, initial_zs, windowSize, ASigmaInv, AMeanScaledSqrtM, diag, profile);
		}

		//Show time per 10000 iterations
		if ((ri+1) % notification_interval_iter == 0)
		{
			tend = get_cur_time();
			time_elapsed = tend - tstart;
			EXEC_INFO ( "finished " << ri+1 << " blocked iterations after " << std::setprecision(std::numeric_limits<tsa_dtype>::digits10 + 2) << time_elapsed << " seconds.");
		}
	}

	// apply a correction of the distance values, as we dropped a factor of 2 to avoid unnecessary computations
	tsa_dtype twice_m = 2.0*static_cast<tsa_dtype>(windowSize);
	for (auto iter=profile.begin(); iter<profile.end(); ++iter) {
		(*iter) = twice_m - 2.0 * (*iter);
	}

	// end timer
	// tend = time(0);
	tend = get_cur_time();
	time_elapsed = tend - tstart;

	PERF_LOG ( "total computation time: " << std::setprecision(std::numeric_limits<tsa_dtype>::digits10 + 2) << time_elapsed << " seconds." );
	const double triang_len = ProfileLength-exclusionZone;
	PERF_LOG ( "throughput computations: " << triang_len * triang_len / get_seconds(time_elapsed) << " matrix entries/second");

	//store the result
	store_matrix_profile(profile, profileIndex, params);

	setup_perf.log_perf();
	init_diag_perf.log_perf();
	eval_diag_perf.log_perf();
#ifdef PROFILING
	PERF_LOG ( "number of matrix profile updates: " << _profileUpdateCounter);
#endif

}
