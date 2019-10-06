#ifndef SCRIMPTRIVIALPAR_HPP
#define SCRIMPTRIVIALPAR_HPP

#include <ScrimpVertBlocked.hpp>
#include <mpi.h>
#include <settings.h>
#include <timing.h>

namespace matrix_profile {

const int RESULT_ALLOC_LEN=MP_MAX_PROFILE_LENGTH;

class ScrimpTrivialPar : public ScrimpVertBlocked
{
    public:
	    struct alignas(16) MatProfSOA {
			// alignment could have been solved more elegant with aligned aligned storage...
			std::array<tsa_dtype, RESULT_ALLOC_LEN> profile;
			std::array<idx_dtype, RESULT_ALLOC_LEN> index;
		};
		ScrimpTrivialPar() : ScrimpVertBlocked(), _profile_updates(0){}

    public:
		virtual void compute_matrix_profile(const Scrimppp_params& params);
		virtual void log_info();
    protected:

		void store_matrix_profile(const MatProfSOA& result, const Scrimppp_params& params, const idx_dtype profileLength, const bool distributed_io);
		void eval_diagonal_block(MatProfSOA& result, const int blocklen, const aligned_tsdtype_vec& A, aligned_tsdtype_vec& initial_zs, const int windowSize, const aligned_tsdtype_vec& ASigmaInv, const aligned_tsdtype_vec& AMeanScaledSigSqrM, const int base_diag) NOINLINE_IF_GPROF_EN;

		static void MPI_MatProfSOA_reduction(void* invec, void* inoutvec, int *len,
		                              MPI_Datatype *datatype);
    private:
		long _profile_updates;
		long _eval_ctr;


		using TracepointMap = std::map<std::string, Timepoint>;
		using SynctimeMap = std::map<std::string, Timespan>;
		struct TimeLoggingInfo {
			    matrix_profile::Timespan setup_time, comp_time, comm_time, evaluation_time, io_time, work_time, precomp_time, dotproduct_time;
			TracepointMap  _time_trace_map;
			SynctimeMap _synctime_map;
		} _timing_info;

		struct LoggingInfo {
			idx_dtype profile_length;
			idx_dtype exclusionZone;
			idx_dtype windowSize;
			idx_dtype timeSeriesLength;
			int world_size;
			int world_rank;
			int BLOCKING_SIZE;
			idx_dtype diags_to_process_proc;
		} _log_info;
};

} // namespace matrix_profile
#endif // SCRIMPTRIVIALPAR_HPP
