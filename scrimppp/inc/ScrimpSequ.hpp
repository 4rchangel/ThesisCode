#ifndef SCRIMPSEQU_HPP
#define SCRIMPSEQU_HPP

#include <MatProfAlgo.hpp>
#include <vector>

#include <fstream>
#include <sstream>
#include <limits>
#include <iomanip>
#include <math.h>

#include <logging.hpp>
#include <settings.h>
#include <bintsfile.hpp>
#include <binproffile.h>

#include <boost/filesystem.hpp>

#if defined(ENABLE_GPROF) ||  defined(PROFILING)
    #define NOINLINE_IF_GPROF_EN __attribute__((noinline))
#else
    #define NOINLINE_IF_GPROF_EN
#endif

namespace matrix_profile {

class ScrimpSequ : public IMatProfAlgo {
    public:
	    ScrimpSequ() : _profileUpdateCounter(0){}
		virtual void initialize(const Scrimppp_params& params);
		virtual void compute_matrix_profile(const Scrimppp_params& params);
		//made the storing method public for reuse in ascii-bin conversion: TODO: best completly extract from the class
		static void store_matrix_profile(const tsa_dtype* const profile, const idx_dtype* const profileIndex, const idx_dtype profileLength, const Scrimppp_params& params, const bool distributed_io=false);
		template<class tsa_alloc, class int_alloc> static void store_matrix_profile(const std::vector<tsa_dtype, tsa_alloc>& profile, const std::vector<idx_dtype, int_alloc>& profileIndex, const Scrimppp_params& params);
    protected:
		static void precompute_window_statistics(const int windowSize, const std::vector<tsa_dtype>& A, const int ProfileLength, std::vector<tsa_dtype>& AMean, std::vector<tsa_dtype>& ASigma);
		template<class allocator> static std::vector<tsa_dtype, allocator> fetch_time_series(const Scrimppp_params& params);
		tsa_dtype init_diagonal(const std::vector<tsa_dtype>& ASigma, const std::vector<tsa_dtype>& dotproduct, std::vector<tsa_dtype>& profile, const idx_dtype diag, const std::vector<tsa_dtype>& A, std::vector<idx_dtype>& profileIndex, const idx_dtype windowSize, const std::vector<tsa_dtype>& AMean) NOINLINE_IF_GPROF_EN;
		void eval_diagonal(std::vector<idx_dtype>& profileIndex, tsa_dtype& lastz, const idx_dtype windowSize, const std::vector<tsa_dtype>& dotproduct, const std::vector<tsa_dtype>& ASigma, const std::vector<tsa_dtype>& AMean, const idx_dtype diag, std::vector<tsa_dtype>& profile) NOINLINE_IF_GPROF_EN;
		void precompute_dotproducts(std::vector<tsa_dtype>& dotproduct, const int diag, const std::vector<tsa_dtype>& A) NOINLINE_IF_GPROF_EN;

    protected:
		long _profileUpdateCounter;
};


template<class allocator> std::vector<matrix_profile::tsa_dtype, allocator> fetch_time_series_ascii(const Scrimppp_params& params)
{
	std::vector<tsa_dtype, allocator> A;
	tsa_dtype tempval;
	std::fstream timeSeriesFile(params.time_series_filename, std::ios_base::in);
	if (!timeSeriesFile.is_open()) {
		throw std::runtime_error("Could not open input file");
	}
	while (timeSeriesFile >> tempval) {
		A.push_back(tempval);
	}
	EXEC_INFO ( "Read ASCII time series of length: " << A.size() );
	timeSeriesFile.close();
	return A;
}

template<class allocator> std::vector<matrix_profile::tsa_dtype, allocator> fetch_time_series_bin(const Scrimppp_params& params)
{
	std::vector<tsa_dtype, allocator> A;
	const bool distrib_read=(COLLECTIVE_DISTRIB_READ&&params.use_distributed_io);
	MPI_Info mpiinfo;
	if (distrib_read) {
		MPI_Info_create(&mpiinfo);
		MPI_Info_set(mpiinfo, "access_style", "read_once");
	}
	BinTsFileMPI file(params.time_series_filename, distrib_read?MPI_COMM_WORLD:MPI_COMM_SELF, mpiinfo);
	const unsigned long tslen = file.get_ts_len();
	A.resize(tslen);
	if (distrib_read) {
		EXEC_INFO("distributed reading with COMM_WORLD")
	}
	else {
		EXEC_INFO("simulatneous binary MPI reading with COMM_SELF");
	}
	const int readlen = file.read_total_ts(A.data(), tslen, distrib_read);
	assert(A.size() == readlen);
	EXEC_INFO ( "Read BINary time series of length: " << readlen );
	if (mpiinfo != MPI_INFO_NULL) {
		MPI_Info_free(&mpiinfo);
	}
	return A;
}

template<class allocator> std::vector<matrix_profile::tsa_dtype, allocator> ScrimpSequ::fetch_time_series(const Scrimppp_params& params)
{
	std::vector<matrix_profile::tsa_dtype, allocator> T;

	switch (params.filetype) {
	case Scrimppp_params::ASCII:
		T = fetch_time_series_ascii<allocator>(params);
		break;
	case Scrimppp_params::BIN:
		T = fetch_time_series_bin<allocator>(params);
		break;
	default:
		throw std::runtime_error("Unhandled file type detected.");
		break;
	}

	EXEC_DEBUG("First 5 TS values: ");
	for (int i=0; i < 5 && i < T.size(); ++i) {
		EXEC_DEBUG( T[i] );
	}
	return T;
}

template<class tsa_alloc, class int_alloc> void ScrimpSequ::store_matrix_profile(const std::vector<tsa_dtype, tsa_alloc>& profile, const std::vector<idx_dtype, int_alloc>& profileIndex, const Scrimppp_params& params)
{
	assert(profile.size() == profileIndex.size());

	ScrimpSequ::store_matrix_profile(profile.data(), profileIndex.data(), profile.size(), params);
}

} // namespace matrix_profile
#endif
