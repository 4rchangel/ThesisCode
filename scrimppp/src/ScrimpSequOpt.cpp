/*
This is a modified version SCRIMP with optimized arithmetics.
The code builds on SCRIMP++, as published by Zhua, Yeh, Zimmerman et al. at https://sites.google.com/site/scrimpplusplus/ and contains parts from their code

Details of the SCRIMP algorithm can be found at:
(author information ommited for ICDM review),
"SCRIMP++: Motif Discovery at Interactive Speeds", submitted to ICDM 2018.
*/
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

#include <ScrimpSequOpt.hpp>
#include <logging.hpp>
#include <papiwrapper.hpp>

using namespace matrix_profile;

static FactoryRegistration<ScrimpSequOpt> s_sequRegistr("scrimp_sequ_opt");
static const int notification_interval_iter = 10000;

void ScrimpSequOpt::precompute_window_statistics(const int windowSize,
    const aligned_tsdtype_vec& A,
    const int ProfileLength,
    aligned_tsdtype_vec& AMeanScaledSigSqrM,
    aligned_tsdtype_vec& ASigmaInv,
    const idx_dtype ts_len)
{
	//TODO: refactore raw pointers, const method
	const size_t timeSeriesLength =(ts_len==-1)?A.size():ts_len;
	std::vector<tsa_dtype> AMean(A.size()-windowSize+1);
	AMeanScaledSigSqrM.resize(A.size()-windowSize+1);
	ASigmaInv.resize(A.size()-windowSize+1);
	const tsa_dtype sqrt_m = sqrt(static_cast<tsa_dtype>(windowSize));

	tsa_dtype* ACumSum = new tsa_dtype[timeSeriesLength];
	ACumSum[0] = A[0];
	for (int i = 1; i < timeSeriesLength; i++)
		ACumSum[i] = A[i] + ACumSum[i - 1];
	tsa_dtype* ASqCumSum = new tsa_dtype[timeSeriesLength];
	ASqCumSum[0] = A[0] * A[0];
	for (int i = 1; i < timeSeriesLength; i++)
		ASqCumSum[i] = A[i] * A[i] + ASqCumSum[i - 1];
	tsa_dtype* ASum = new tsa_dtype[ProfileLength];
	ASum[0] = ACumSum[windowSize - 1];
	for (int i = 0; i < timeSeriesLength - windowSize; i++)
		ASum[i + 1] = ACumSum[windowSize + i] - ACumSum[i];
	tsa_dtype* ASumSq = new tsa_dtype[ProfileLength];
	ASumSq[0] = ASqCumSum[windowSize - 1];
	for (int i = 0; i < timeSeriesLength - windowSize; i++)
		ASumSq[i + 1] = ASqCumSum[windowSize + i] - ASqCumSum[i];
	for (int i = 0; i < ProfileLength; i++){
		    AMean[i] = ASum[i] / windowSize;
	    }
	tsa_dtype* ASigmaSq = new tsa_dtype[ProfileLength];
	for (int i = 0; i < ProfileLength; i++)
		ASigmaSq[i] = ASumSq[i] / windowSize - AMean[i] * AMean[i];

	for (int i = 0; i < ProfileLength; i++) {
		ASigmaInv[i] = 1.0/sqrt(ASigmaSq[i]);
		AMeanScaledSigSqrM[i] = AMean[i]*ASigmaInv[i]*sqrt_m;
	}
	delete [] ACumSum;
	delete [] ASqCumSum;
	delete [] ASum;
	delete [] ASumSq;
	delete [] ASigmaSq;
}

void ScrimpSequOpt::init_diagonals(const int first_diag, const int last_diag, aligned_tsdtype_vec& initial_zs, const aligned_tsdtype_vec& A, const int windowSize)
{
	const size_t profileLength = A.size() - windowSize+1;
//	initial_zs.reserve(profileLength);
	assert(initial_zs.size() >= profileLength);
	//evaluate the fist distance value in the current diagonal
	for (size_t diag = first_diag; diag <= last_diag; ++diag) {
		tsa_dtype lastz=0;
		for (int k = 0; k < windowSize; k++)
		{
			lastz += A[k+diag]*A[k];
		}
		initial_zs[diag] = lastz;
//std::cout << "inited " << diag << " with " << lastz << std::endl;
	}
}

void ScrimpSequOpt::init_all_diagonals(aligned_tsdtype_vec& initial_zs, const aligned_tsdtype_vec& A, const int windowSize) {
	//init diagonals 0 to profileLength-1
	init_diagonals(0, A.size()-windowSize+1, initial_zs, A, windowSize);
}

void ScrimpSequOpt::eval_diagonal(aligned_int_vec& profileIndex, const aligned_tsdtype_vec& A, const aligned_tsdtype_vec& initial_zs, const int windowSize, const aligned_tsdtype_vec& ASigmaInv, const aligned_tsdtype_vec& AMeanScaledSigSqrM, const int diag, aligned_tsdtype_vec& profile)
{
	const int profileLength = AMeanScaledSigSqrM.size();
	tsa_dtype corrScore;
#ifdef PROFILING
	long updateCtr = 0;
#endif

	tsa_dtype tmpz = initial_zs[diag]; //rather use a local, to avoid innecessary writes to the referenced memory
	for (int j=diag; j<profileLength; j++)
	{
		int i=j-diag;

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

#ifdef PROFILING
	_profileUpdateCounter += updateCtr;
#endif
}

void ScrimpSequOpt::compute_matrix_profile(const Scrimppp_params& params)
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

	// compute the first correlation values in the diagonals (i.e. compute the correlation between the first windows)
	{
		ScopedPerfAccumulator monitor(init_diag_perf);
		init_all_diagonals(initial_zs, A, windowSize);
	}
	//iteratively evaluate the diagonals of the distance matrix
	for (int ri = 0; ri < idx.size(); ri++)
	    {
		//select a random diagonal
		int diag = idx[ri];

		//evaluate the second to the last distance values along the diagonal in the matrix and update the matrix profile/matrix profile index.
		{
			ScopedPerfAccumulator monitor(eval_diag_perf);
			eval_diagonal(profileIndex, A, initial_zs, windowSize, ASigmaInv, AMeanScaledSqrtM, diag, profile);
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
	eval_diag_perf.log_perf();
#ifdef PROFILING
	PERF_LOG ( "number of matrix profile updates: " << _profileUpdateCounter);
#endif

}
