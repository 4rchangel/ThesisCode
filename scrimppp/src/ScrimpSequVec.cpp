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
#include <boost/circular_buffer.hpp>

#include <ScrimpSequVec.hpp>
#include <logging.hpp>
#include <papiwrapper.hpp>

using namespace matrix_profile;

static FactoryRegistration<ScrimpSequVec> s_sequRegistr("scrimp_sequ_vec");
static const int notification_interval_iter = 10000;

PerfCounters cumsum_perf("cumulative_sum");
PerfCounters eval_perf("vectorized evaluation");

tsa_dtype ScrimpSequVec::init_diagonal(const aligned_tsdtype_vec& ASigmaInv, aligned_tsdtype_vec& cumDotproduct, aligned_tsdtype_vec& profile, const int diag, const aligned_tsdtype_vec& A, aligned_int_vec& profileIndex, const int windowSize, const aligned_tsdtype_vec& AMeanScaledSigSqrM)
{
	tsa_dtype corr_score=0;
	tsa_dtype lastz=0;

	//evaluate the fist distance value in the current diagonal
	for (int k = 0; k < windowSize; k++)
	{
		cumDotproduct[k+diag]=lastz;
		lastz += A[k+diag]*A[k];
	}
	//j is the column index, i is the row index of the current distance value in the distance matrix

	return lastz;
}

void ScrimpSequVec::eval_diagonal(idx_dtype* const profileIndex, const tsa_dtype* const A, tsa_dtype& lastz, const idx_dtype windowSize, tsa_dtype* const cumDotproduct, tsa_dtype* const ASigmaInv, const tsa_dtype* const AMeanScaledSigSqrM, const idx_dtype diag, tsa_dtype* const profile, const idx_dtype profileLength)
{
	{
		ScopedPerfAccumulator monitor(cumsum_perf);
		tsa_dtype cumProd = lastz; //TODO: rename lastz. The idea of this line is toavoid unnecessary writes to the input reference....
		for (idx_dtype j=diag; j<profileLength; j++)
		{
			cumDotproduct[j+windowSize]=cumProd;
			cumProd += A[j+windowSize]*A[j+windowSize-diag];
		}
	}
	{
		ScopedPerfAccumulator monitor(eval_perf);
        #pragma omp simd aligned(profile, profileIndex, ASigmaInv, AMeanScaledSigSqrM, cumDotproduct : alignment)
		for (idx_dtype j=diag; j<profileLength; j++)
		{
			idx_dtype i=j-diag;
			tsa_dtype corrScore = ( (cumDotproduct[j+windowSize] - cumDotproduct [j]) * (ASigmaInv[j] * ASigmaInv[i]) - AMeanScaledSigSqrM[j] * AMeanScaledSigSqrM[i]) ;

			bool update_j = (corrScore > profile[j]);
			profile[j] = update_j?corrScore:profile[j];
			profileIndex [j] = update_j?i:profileIndex [j];

			bool update_i = (corrScore > profile[i]);
			profile[i] = update_i?corrScore:profile[i];
			profileIndex [i] = update_i?j:profileIndex[i];
		}
	}
}

void ScrimpSequVec::compute_matrix_profile(const Scrimppp_params& params)
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
	//Initialize Matrix Profile and Matrix Profile Index
	aligned_tsdtype_vec profile(ProfileLength, 0.0);
	aligned_int_vec profileIndex(ProfileLength, 0);
	aligned_tsdtype_vec dotproduct(timeSeriesLength); // stores products between two Timeseries values with a distinct offset
	aligned_int_vec idx; // store indices of the diagonals, defining their evaluation order
	idx.reserve(ProfileLength-exclusionZone-1);

	//several monitors for performance measurement
	PerfCounters setup_perf("setup performance");
	PerfCounters init_diag_perf("diagonal initialization");
	//PerfCounters eval_diag_perf("diagonal evaluation");

	//validation of parameters
	if (timeSeriesLength < windowSize) {
		throw std::invalid_argument("ERROR: Time series is shorter than the window length, can not proceed");
	}

	EXEC_INFO( "Sequential SCRIMP matrix profile computation with profile length " << ProfileLength << " and window size " << windowSize);


	{
		ScopedPerfAccumulator monitor(setup_perf);
		//precompute the mean and standard deviations of the sliding windows along the time series
		precompute_window_statistics(windowSize, A, ProfileLength, AMeanScaledSqrtM, ASigmaInv);

		//start time measurment
		tstart = std::chrono::high_resolution_clock::now();

		/******************** SCRIMP ********************/
		//Random shuffle the computation order of the diagonals of the distance matrix
		for (int i = exclusionZone+1; i < ProfileLength; i++) {
			idx.push_back(i);
		}
		std::random_shuffle(idx.begin(), idx.end());
	}

	//iteratively evaluate the diagonals of the distance matrix
	for (int ri = 0; ri < idx.size(); ri++)
	    {
		//select a random diagonal
		int diag = idx[ri];
		tsa_dtype lastz=0; //the dot product of a subsequence


		// compute the first distance value in the diagonal (i.e. compute the distance between the first windows)
		{
			ScopedPerfAccumulator monitor(init_diag_perf);
			lastz = init_diagonal(ASigmaInv, dotproduct, profile, diag, A, profileIndex, windowSize, AMeanScaledSqrtM);
		}

		//evaluate the second to the last distance values along the diagonal in the matrix and update the matrix profile/matrix profile index.
		{
			//ScopedPerfAccumulator monitor(eval_diag_perf);
			// eval_diagonal(profileIndex, A, lastz, windowSize, dotproduct, ASigmaInv, AMeanScaledSqrtM, diag, profile);
			eval_diagonal(profileIndex.data(), A.data(), lastz, windowSize, dotproduct.data(), ASigmaInv.data(), AMeanScaledSqrtM.data(), diag, profile.data(), AMeanScaledSqrtM.size());
		}

		//Show time per 10000 iterations
		if ((ri+1) % notification_interval_iter == 0)
		{
			tend = std::chrono::high_resolution_clock::now();
			time_elapsed = tend - tstart;
			EXEC_INFO ( "finished " << ri+1 << " iterations after " << std::setprecision(std::numeric_limits<tsa_dtype>::digits10 + 2) << time_elapsed.count() << " seconds.");
		}
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
	const double triang_len = ProfileLength-exclusionZone;
	PERF_LOG ( "throughput computations: " << triang_len * triang_len / time_elapsed.count() << " matrix entries/second");

	//store the result
	store_matrix_profile(profile, profileIndex, params);

	setup_perf.log_perf();
	init_diag_perf.log_perf();
	cumsum_perf.log_perf();
	eval_perf.log_perf();
	//eval_diag_perf.log_perf();
#ifdef PROFILING
	PERF_LOG ( "number of matrix profile updates: " << _profileUpdateCounter);
#endif

}
