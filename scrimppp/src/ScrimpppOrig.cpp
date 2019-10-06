/*
This is SCRIMP++, as published by Zhua, Yeh, Zimmerman et al. at https://sites.google.com/site/scrimpplusplus/
It is provided as a baseline for our work and republished with kind permission of Prof. Eamonn Keogh
All rights belong the original authors and they shall be asked for licensing, if required.
Few modifications to the code were made to adapt it in our framework

Details of the SCRIMP++ algorithm can be found at:
(author information ommited for ICDM review),
"SCRIMP++: Motif Discovery at Interactive Speeds", submitted to ICDM 2018.
*/
#include <stdio.h>
#include <stdlib.h>
#include <fftw3.h>
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
#include <logging.hpp>

#include <ScrimpppOrig.hpp>
#include <boost/filesystem.hpp>

using namespace matrix_profile;

static FactoryRegistration<ScrimpppOrig> s_origRegistration("scrimppp_orig");
static const int notification_interval_iter = 10000;

void ScrimpppOrig::initialize(const Scrimppp_params &params) {

}

void ScrimpppOrig::compute_matrix_profile(const Scrimppp_params& params)
{
	// start timer
	//time_t tstart, tend;
	//tstart = time(0);
	Timepoint tstart, tend;
	Timespan time_elapsed;


	// read time series and subsequence length (windowSize).
	std::fstream timeSeriesFile(params.time_series_filename, std::ios_base::in);

	int windowSize = params.query_window_len;
	int stepSize = floor(params.prescrimp_stride*windowSize);

	const std::string preoutfilename = params.output_filename + "_prescrimp";

	if (!timeSeriesFile.is_open())
	{
		throw std::runtime_error("Could not open input file");
	}

	std::vector<double> A;
	double tempval;
	int timeSeriesLength = 0;
	while (timeSeriesFile >> tempval)
	{
		A.push_back(tempval);
		timeSeriesLength++;
	}
	EXEC_INFO( "loaded time series of length " << timeSeriesLength );

	if (timeSeriesLength < windowSize) {
		throw std::runtime_error("ERROR: Time series is shorter than the window length, can not proceed");
	}

	timeSeriesFile.close();

	// set exclusion zone
	int exclusionZone = windowSize / 4;

	// set Matrix Profile Length
	int ProfileLength = timeSeriesLength - windowSize + 1;

	// preprocess, statistics, get the mean and standard deviation of every subsequence in the time series
	double* ACumSum = new double[timeSeriesLength];
	ACumSum[0] = A[0];
	for (int i = 1; i < timeSeriesLength; i++)
		ACumSum[i] = A[i] + ACumSum[i - 1];
	double* ASqCumSum = new double[timeSeriesLength];
	ASqCumSum[0] = A[0] * A[0];
	for (int i = 1; i < timeSeriesLength; i++)
		ASqCumSum[i] = A[i] * A[i] + ASqCumSum[i - 1];
	double* ASum = new double[ProfileLength];
	ASum[0] = ACumSum[windowSize - 1];
	for (int i = 0; i < timeSeriesLength - windowSize; i++)
		ASum[i + 1] = ACumSum[windowSize + i] - ACumSum[i];
	double* ASumSq = new double[ProfileLength];
	ASumSq[0] = ASqCumSum[windowSize - 1];
	for (int i = 0; i < timeSeriesLength - windowSize; i++)
		ASumSq[i + 1] = ASqCumSum[windowSize + i] - ASqCumSum[i];
	double* AMean = new double[ProfileLength];
	for (int i = 0; i < ProfileLength; i++)
		AMean[i] = ASum[i] / windowSize;
	double* ASigmaSq = new double[ProfileLength];
	for (int i = 0; i < ProfileLength; i++)
		ASigmaSq[i] = ASumSq[i] / windowSize - AMean[i] * AMean[i];
	double* ASigma = new double[ProfileLength];
	for (int i = 0; i < ProfileLength; i++)
		ASigma[i] = sqrt(ASigmaSq[i]);
	delete [] ACumSum;
	delete [] ASqCumSum;
	delete [] ASum;
	delete [] ASumSq;
	delete [] ASigmaSq;

	//Initialize Matrix Profile and Matrix Profile Index
	double* profile = new double[ProfileLength];
	int* profileIndex = new int[ProfileLength];
	for (int i=0; i<ProfileLength; i++)
	{
		profile[i]=std::numeric_limits<double>::infinity();
		profileIndex[i]=0;
	}
	tstart = get_cur_time();

	//int fftsize = pow(2,ceil(log2(timeSeriesLength)));
	int fftsize = timeSeriesLength; //fftsize must be at least 2*windowSize
	fftsize = fftsize > 2 * windowSize ? fftsize : 2 * windowSize;
	EXEC_TRACE ( "length of fft input: " << fftsize );


	/*******************************PreSCRIMP***************************************/

	fftw_plan plan;
	fftw_complex* ATime = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * fftsize);
	fftw_complex* AFreq = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * fftsize);

	for (int i = 0; i < fftsize; i++)
	{
		ATime[i][1] = 0;
		if (i < timeSeriesLength)
			ATime[i][0] = A[i];
		else
			ATime[i][0] = 0;
	}

	plan = fftw_plan_dft_1d(fftsize, ATime, AFreq, FFTW_FORWARD, FFTW_ESTIMATE);
	fftw_execute(plan);
	fftw_free(ATime);

	fftw_complex* queryTime = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * fftsize);
	fftw_complex* queryFreq = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * fftsize);
	fftw_complex* AQueryTime = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * fftsize);
	fftw_complex* AQueryFreq = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * fftsize);

	//Sample subsequences with a fixed stepSize, then random shuffle their computation order
	std::vector<int> idx;
	for (int i = 0; i < timeSeriesLength - windowSize + 1; i += stepSize)
		idx.push_back(i);
	std::random_shuffle(idx.begin(), idx.end());

	double* query = new double[windowSize];

	for (int idx_i = 0; idx_i < idx.size(); idx_i++)
	{
		int i = idx[idx_i];
		for (int j = 0; j < windowSize; j++)
		{
			query[j] = A[i + j];
		}
		double queryMean = AMean[i];
		double queryStd = ASigma[i];

		for (int j = 0; j < fftsize; j++)
		{
			queryTime[j][1] = 0;

			if (j < windowSize)
				queryTime[j][0] = query[windowSize - j - 1];
			else
				queryTime[j][0] = 0;
		}

		plan = fftw_plan_dft_1d(fftsize, queryTime, queryFreq, FFTW_FORWARD, FFTW_ESTIMATE);
		fftw_execute(plan);

		for (int j = 0; j < fftsize; j++)
		{
			AQueryFreq[j][0] = AFreq[j][0] * queryFreq[j][0] - AFreq[j][1] * queryFreq[j][1];
			AQueryFreq[j][1] = AFreq[j][1] * queryFreq[j][0] + AFreq[j][0] * queryFreq[j][1];
		}

		plan = fftw_plan_dft_1d(fftsize, AQueryFreq, AQueryTime, FFTW_BACKWARD, FFTW_ESTIMATE);
		fftw_execute(plan);

		int exclusionZoneStart = i - exclusionZone;
		int exclusionZoneEnd = i + exclusionZone;
		double minimumDistance = std::numeric_limits<double>::infinity();
		int minimumDistanceIndex;
		for (int j = 0; j < timeSeriesLength - windowSize + 1; j++)
		{
			double distance;
			if ((j > exclusionZoneStart) && (j < exclusionZoneEnd))
				distance = std::numeric_limits<double>::infinity();
			else
			{
				distance = 2 * (windowSize - (AQueryTime[windowSize + j - 1][0] / fftsize - windowSize * AMean[j] * queryMean) / (ASigma[j] * queryStd));
			}

			if (distance < minimumDistance)
			{
				minimumDistance = distance;
				minimumDistanceIndex = j;
			}

			if (distance < profile[j])
			{
				profile[j] = distance;
				profileIndex[j] = i;
			}
		}
		profile[i] = minimumDistance;
		profileIndex[i] = minimumDistanceIndex;

		int j = profileIndex[i];
		double lastz = (windowSize - profile[i] / 2) * (ASigma[j] * ASigma[i]) + windowSize * AMean[j] * AMean[i];
		double lastzz = lastz;
		double distance;
		for (int k = 1; k < stepSize && i + k < timeSeriesLength - windowSize + 1 && j + k < timeSeriesLength - windowSize + 1; k++)
		{
			lastz = lastz - A[i + k - 1] * A[j + k - 1] + A[i + k + windowSize - 1] * A[j + k + windowSize - 1];
			distance = 2 * (windowSize - (lastz - windowSize * AMean[j + k] * AMean[i + k]) / (ASigma[j + k] * ASigma[i + k]));
			if (distance < profile[i + k])
			{
				profile[i + k] = distance;
				profileIndex[i + k] = j + k;
			}
			if (distance < profile[j + k])
			{
				profile[j + k] = distance;
				profileIndex[j + k] = i + k;
			}
		}
		lastz = lastzz;
		for (int k = 1; k < stepSize && i - k >= 0 && j - k >= 0; k++)
		{
			lastz = lastz - A[i - k + windowSize] * A[j - k + windowSize] + A[i - k] * A[j - k];
			distance = 2 * (windowSize - (lastz - windowSize * AMean[j - k] * AMean[i - k]) / (ASigma[j - k] * ASigma[i - k]));
			if (distance < profile[i - k])
			{
				profile[i - k] = distance;
				profileIndex[i - k] = j - k;
			}
			if (distance < profile[j - k])
			{
				profile[j - k] = distance;
				profileIndex[j - k] = i - k;
			}
		}
	}

	fftw_destroy_plan(plan);
	fftw_free(AFreq);
	fftw_free(queryTime);
	fftw_free(queryFreq);
	fftw_free(AQueryTime);
	fftw_free(AQueryFreq);
	delete[] query;

	tend = get_cur_time();
	time_elapsed = tend - tstart;

	// output
	EXEC_INFO("finished prescrimp")
	PERF_LOG( "Time for PreSCRIMP: " << std::setprecision(std::numeric_limits<double>::digits10 + 2) << time_elapsed );

	std::fstream preprofileOutFile(preoutfilename.c_str(), std::ios_base::out);

	// Write PreSCRIMP Matrix Profile and Matrix Profile Index to file.
	for (int i = 0; i < timeSeriesLength - windowSize + 1; i++)
	{
		preprofileOutFile << std::setprecision(std::numeric_limits<double>::digits10 + 2) << sqrt(abs(profile[i])) << " " << std::setprecision(std::numeric_limits<int>::digits10 + 1) << profileIndex[i] << std::endl;
	}

	preprofileOutFile.close();

	/******************** SCRIMP ********************/

	//Random shuffle the computation order of the diagonals of the distance matrix
std::srand(1);//TODO: remove. Introduced for constistent evaluation order among all algorithms during debugging!
    idx.clear();
	for (int i = exclusionZone+1; i < ProfileLength; i++)
		idx.push_back(i);
	std::random_shuffle(idx.begin(), idx.end());

	double* dotproduct = new double[timeSeriesLength];

	//iteratively evaluate the diagonals of the distance matrix
	for (int ri = 0; ri < idx.size(); ri++)
	    {
		//select a random diagonal
		int diag = idx[ri];

		//calculate the dot product of every two time series values that ar diag away
		for (int j=diag; j < timeSeriesLength; j++)
			dotproduct[j]=A[j]*A[j-diag];

		//evaluate the fist distance value in the current diagonal
		double distance;
		double lastz=0; //the dot product of a subsequence
		for (int k = 0; k < windowSize; k++)
			lastz += dotproduct[k+diag];

		//j is the column index, i is the row index of the current distance value in the distance matrix
		int j=diag, i=j-diag;

		//evaluate the distance based on the dot product
		distance = 2 * (windowSize - (lastz - windowSize * AMean[j] * AMean[i]) / (ASigma[j] * ASigma[i]));

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

//std::cout << "diag " << diag << " lastz " << lastz << std::endl;
		//evaluate the second to the last distance values along the diagonal and update the matrix profile/matrix profile index.
		for (j=diag+1; j<ProfileLength; j++)
		{
			i=j-diag;
			lastz = lastz + dotproduct[j+windowSize-1] - dotproduct [j-1];
			distance = 2 * (windowSize - (lastz - windowSize * AMean[j] * AMean[i]) / (ASigma[j] * ASigma[i]));

//std::cout << "eval i: " << i << " j: " << j << " lastz" << lastz << std::endl;
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
		}

		//Show time per 10000 iterations
		if ( (ri+1) % notification_interval_iter == 0)
		{
			tend = get_cur_time();
			time_elapsed = tend - tstart;
			//std::cout << "Time spent: " << std::setprecision(std::numeric_limits<double>::digits10 + 2) << difftime(time(0), tstart) << " seconds." << std::endl;
			PERF_TRACE( "completed " << notification_interval_iter << " iterations in: " << time_elapsed );
		}

		//The following commented section is to produce provisional results. Basically, if you would like to enable interrupt and look at the current matrix profile/matrix profile index, you can uncomment this section, and revise line 182 according to the intterupt mechanism you're using.

		/*if (interrupt_detected) //revise this line to enable your interrupt mechanism
		{
			std::fstream prov_profileOutFile(outfilename_provisional.c_str(), std::ios_base::out);

			// Write Current Matrix Profile and Matrix Profile Index to file.
			for (int k = 0; k < timeSeriesLength - windowSize + 1; k++)
				prov_profileOutFile << std::setprecision(std::numeric_limits<double>::digits10 + 2) << sqrt(abs(profile[k])) << " " << std::setprecision(std::numeric_limits<int>::max()) << profileIndex[k] << std::endl;
			prov_profileOutFile.close();
		}
		*/
	}

	// end timer
	//tend = time(0);
	tend = get_cur_time();
	time_elapsed = tend - tstart;

	PERF_LOG ( "total computation time: " << time_elapsed << " seconds." );
	const double triang_len = ProfileLength-exclusionZone;
	PERF_LOG ( "throughput computations: " << triang_len * triang_len / get_seconds(time_elapsed) << " matrix entries/second");
	EXEC_INFO( "Writing result to file");

	if (params.output_filename.empty()) {
		throw std::runtime_error("Empty output file name specified!");
	}
	std::fstream profileOutFile(params.output_filename.c_str(), std::ios_base::out);

	// Write final Matrix Profile and Matrix Profile Index to file.
	for (int i = 0; i < timeSeriesLength - windowSize + 1; i++)
	{
		profile[i] = sqrt(abs(profile[i]));
		profileOutFile << std::setprecision(std::numeric_limits<double>::digits10 + 2) << profile[i] << " " << std::setprecision(std::numeric_limits<int>::digits10+1) << profileIndex[i] << std::endl;
	}

	profileOutFile.close();

	delete [] dotproduct;
	delete [] AMean;
	delete [] ASigma;
	delete [] profile;
	delete [] profileIndex;
}
