/*
This is a modified version of SCRIMP++, as published by Zhua, Yeh, Zimmerman et al. at https://sites.google.com/site/scrimpplusplus/
It is provided as a baseline for our work and republished with kind permission of Prof. Eamonn Keogh
All rights belong the original authors and they shall be asked for licensing, if required.
In particular the PreSCRIMP computation was removed to provide a fair comparision of implmentations, as we rely only on the SCRIMP part.
A few more modifications to the code were made to adapt it in our framework

Details of the SCRIMP++ algorithm can be found at:
(author information ommited for ICDM review),
"SCRIMP++: Motif Discovery at Interactive Speeds", submitted to ICDM 2018.
*/
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <string>
#include <timing.h>

#include <ScrimpSequ.hpp>
#include <papiwrapper.hpp>

using namespace matrix_profile;

static FactoryRegistration<ScrimpSequ> s_sequRegistr("scrimp_sequ");
static const int notification_interval_iter = 10000;

void ScrimpSequ::initialize(const Scrimppp_params &params){

}

void ScrimpSequ::precompute_window_statistics(const int windowSize, const std::vector<tsa_dtype>& A, const int ProfileLength, std::vector<tsa_dtype>& AMean, std::vector<tsa_dtype>& ASigma)
{
	//TODO: refactore raw pointers, const method
	size_t timeSeriesLength = A.size();
	AMean.resize(A.size()-windowSize+1);
	ASigma.resize(A.size()-windowSize+1);

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
	for (int i = 0; i < ProfileLength; i++)
		AMean[i] = ASum[i] / windowSize;
	tsa_dtype* ASigmaSq = new tsa_dtype[ProfileLength];
	for (int i = 0; i < ProfileLength; i++)
		ASigmaSq[i] = ASumSq[i] / windowSize - AMean[i] * AMean[i];

	for (int i = 0; i < ProfileLength; i++)
		ASigma[i] = sqrt(ASigmaSq[i]);
	delete [] ACumSum;
	delete [] ASqCumSum;
	delete [] ASum;
	delete [] ASumSq;
	delete [] ASigmaSq;
}

matrix_profile::tsa_dtype ScrimpSequ::init_diagonal(const std::vector<tsa_dtype>& ASigma, const std::vector<tsa_dtype>& dotproduct, std::vector<tsa_dtype>& profile, const idx_dtype diag, const std::vector<tsa_dtype>& A, std::vector<idx_dtype>& profileIndex, const idx_dtype windowSize, const std::vector<tsa_dtype>& AMean)
{
	tsa_dtype distance=0;
	tsa_dtype lastz=0;

	//evaluate the fist distance value in the current diagonal
	for (int k = 0; k < windowSize; k++)
		lastz += dotproduct[k+diag];
	//j is the column index, i is the row index of the current distance value in the distance matrix
	int j=diag, i=j-diag;
	//evaluate the distance based on the dot product
	distance = 2 * (windowSize - (lastz - windowSize * AMean[j] * AMean[i]) / (ASigma[j] * ASigma[i]));
	EXEC_TRACE( "eval i: " << i << " j: " << j << " lastz " << lastz);

	//update matrix profile and matrix profile index if the current distance value is smaller
	if (distance < profile[j])
	{
		profile[j] = distance;
		profileIndex [j] = i;
	}
	if (distance < profile[i])
	{
		profile[i] = distance;
		profileIndex [i] = j;
	}

	return lastz;
}

void ScrimpSequ::eval_diagonal(std::vector<idx_dtype>& profileIndex, tsa_dtype& lastz, const idx_dtype windowSize, const std::vector<tsa_dtype>& dotproduct, const std::vector<tsa_dtype>& ASigma, const std::vector<tsa_dtype>& AMean, const idx_dtype diag, std::vector<tsa_dtype>& profile)
{
	tsa_dtype distance;
	const int profileLength = AMean.size();
    #ifdef PROFILING
	long updateCtr = 0;
    #endif
	EXEC_TRACE( "start evaluation of diag " << diag << " lastz " << lastz);
	for (int j=diag+1; j<profileLength; j++)
	{
		int i=j-diag;
		lastz = lastz + dotproduct[j+windowSize-1] - dotproduct [j-1];
		distance = 2 * (windowSize - (lastz - windowSize * AMean[j] * AMean[i]) / (ASigma[j] * ASigma[i]));
		EXEC_TRACE( "eval i: " << i << " j: " << j << " lastz " << lastz);

		if (distance < profile[j])
		{
			profile[j] = distance;
			profileIndex [j] = i;
            #ifdef PROFILING
			updateCtr+=1;
            #endif
		}
		if (distance < profile[i])
		{
			profile[i] = distance;
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

void ScrimpSequ::precompute_dotproducts(std::vector<tsa_dtype>& dotproduct, const int diag, const std::vector<tsa_dtype>& A)
{
	for (int j=diag; j < A.size(); j++)
		dotproduct[j]=A[j]*A[j-diag];
}

void ScrimpSequ::compute_matrix_profile(const Scrimppp_params& params)
{
	Timepoint tstart, tend;
	Timespan time_elapsed;

	std::vector<tsa_dtype> A = fetch_time_series<std::vector<tsa_dtype>::allocator_type>(params); //load the time series data
	std::vector<tsa_dtype> AMean(A.size());
	std::vector<tsa_dtype> ASigma(A.size());
	int windowSize = params.query_window_len;
	int exclusionZone = windowSize / 4;
	int timeSeriesLength = A.size();
	int ProfileLength = timeSeriesLength - windowSize + 1;
	//Initialize Matrix Profile and Matrix Profile Index
	std::vector<tsa_dtype> profile(ProfileLength, std::numeric_limits<tsa_dtype>::infinity());
	std::vector<idx_dtype> profileIndex(ProfileLength, 0);
	std::vector<tsa_dtype> dotproduct(timeSeriesLength); // stores products between two Timeseries values with a distinct offset
	std::vector<idx_dtype> idx; // store indices of the diagonals, defining their evaluation order
	idx.reserve(ProfileLength-exclusionZone-1);

	//several monitors for performance measurement
	PerfCounters setup_perf("setup");
	PerfCounters dotprod_precomp_perf("dotproduct precomputation");
	PerfCounters init_diag_perf("diagonal initialization ");
	PerfCounters eval_diag_perf("diagonal evaluation");

	//validation of parameters
	if (timeSeriesLength < windowSize) {
		throw std::invalid_argument("ERROR: Time series is shorter than the window length, can not proceed");
	}

	EXEC_INFO( "Sequential SCRIMP matrix profile computation with profile length " << ProfileLength << " and window size " << windowSize);


	{
		ScopedPerfAccumulator monitor(setup_perf);
		//precompute the mean and standard deviations of the sliding windows along the time series
		precompute_window_statistics(windowSize, A, ProfileLength, AMean, ASigma);

		//start time measurment
		tstart = get_cur_time();

		/******************** SCRIMP ********************/
		//Random shuffle the computation order of the diagonals of the distance matrix
		for (int i = exclusionZone+1; i < ProfileLength; i++) {
			idx.push_back(i);
		}

#ifdef _NDEBUG //omitt shuffling in th debug build for facilitated comparision
		std::random_shuffle(idx.begin(), idx.end());
#endif
	}

	//iteratively evaluate the diagonals of the distance matrix
	for (int ri = 0; ri < idx.size(); ri++)
	    {
		//select a random diagonal
		int diag = idx[ri];
		tsa_dtype lastz=0; //the dot product of a subsequence

		//calculate the dot product of every two time series values that ar diag away
		{
			ScopedPerfAccumulator monitor(dotprod_precomp_perf);
			precompute_dotproducts(dotproduct, diag, A);
		}


		// compute the first distance value in the diagonal (i.e. compute the distance between the first windows)
		{
			ScopedPerfAccumulator monitor(init_diag_perf);
			lastz = init_diagonal(ASigma, dotproduct, profile, diag, A, profileIndex, windowSize, AMean);
		}

		//evaluate the second to the last distance values along the diagonal in the matrix and update the matrix profile/matrix profile index.
		{
			ScopedPerfAccumulator monitor(eval_diag_perf);
			eval_diagonal(profileIndex, lastz, windowSize, dotproduct, ASigma, AMean, diag, profile);
		}

		//Show time per 10000 iterations
		if ((ri+1) % notification_interval_iter == 0)
		{
			tend = get_cur_time();
			time_elapsed = tend - tstart;
			EXEC_INFO ( "finished " << ri+1 << " iterations after " << time_elapsed);
		}
	}

	// end timer
	// tend = time(0);
	tend = get_cur_time();
	time_elapsed = tend - tstart;

	PERF_LOG ( "total computation time: " << std::setprecision(std::numeric_limits<tsa_dtype>::digits10 + 2) << time_elapsed << " seconds." );
	double triang_len = ProfileLength-exclusionZone;
	PERF_LOG ( "throughput computations: " << triang_len * triang_len / get_seconds(time_elapsed) << " matrix entries/second");

	//store the result
	store_matrix_profile(profile, profileIndex, params);

	setup_perf.log_perf();
	init_diag_perf.log_perf();
	dotprod_precomp_perf.log_perf();
	eval_diag_perf.log_perf();
#ifdef PROFILING
	PERF_LOG ( "number of matrix profile updates: " << _profileUpdateCounter);
#endif

}

void ScrimpSequ::store_matrix_profile(
        const tsa_dtype* const profile,
        const idx_dtype* const profileIndex,
        const idx_dtype profileLength,
        const Scrimppp_params& params,
        const bool distributed_io
        )
{
	// validate the parameter
	if (params.output_filename.empty()) {
		throw std::runtime_error("Empty output file name specified!");
	}

	switch (params.filetype) {
	    case Scrimppp_params::ASCII:
	        {
		        EXEC_DEBUG("storing length " << profileLength << " profile in ASCII mode");
				// open the file
				std::fstream profileOutFile(params.output_filename.c_str(), std::ios_base::out);
				// Write final Matrix Profile and Matrix Profile Index to file.
				for (int i = 0; i < profileLength; i++)
				{
					tsa_dtype distance =  sqrt(abs(profile[i]));
					profileOutFile << std::setprecision(std::numeric_limits<tsa_dtype>::digits10 + 2) << distance << " " << std::setprecision(std::numeric_limits<int>::digits10+1) << profileIndex[i] << std::endl;
				}
				profileOutFile.close();
	        }
		    break;
	    case Scrimppp_params::BIN:
	        {
		        MPI_Info outputinfo = MPI_INFO_NULL;
				MPI_Info_create(&outputinfo);
				MPI_Info_set(outputinfo, "access_style", "write_once");
				EXEC_DEBUG("storing length " << profileLength << " profile in BIN mode");
				const MPI_Comm comm = distributed_io?MPI_COMM_WORLD:MPI_COMM_SELF;
				if (!distributed_io) {
					EXEC_INFO( "storing the matrix profile sequentially with a single proc" );
				}
				else {
					EXEC_INFO( "storing the matrix profile with parallel I/O" );
				}
				BinProfFile ofile(params.output_filename, comm, outputinfo);

				if (params.writing_chunk_size>0) {
					EXEC_DEBUG("setting binary output chunksize to " << params.writing_chunk_size);
					ofile.set_chunksize(params.writing_chunk_size);
				}

				ofile.write_matrix_profile(profile, profileIndex, profileLength);
				MPI_Info_free(&outputinfo);
				break;
	        }
	    default:
		    throw std::runtime_error("Unhandled file type detected.");
		    break;
	}
}
