#ifndef ScrimpDistribPar_HPP
#define ScrimpDistribPar_HPP

#include <ScrimpVertBlocked.hpp>
#include <mpi.h>
#include <settings.h>
#include <kernels.h>
#include <timing.h>

namespace matrix_profile {

const int RESULT_ALLOC_LEN=MP_MAX_PROFILE_LENGTH;
// easy aligned allocation.
// code from https://www.boost.org/doc/libs/1_57_0/doc/html/align/examples.html#align.examples.aligned_ptr, licensed under the Boost Software License - Version 1.0
// see copy of the license terms provided under /licenses/boostSoftwareLicense.txt
template<class T>
using aligned_ptr = std::unique_ptr<T,
  boost::alignment::aligned_delete>;

class ScrimpDistribPar : public ScrimpVertBlocked
{
    public:
	        struct alignas(16) MatProfSOA {
				        // alignment could have been solved more elegant with aligned aligned storage...
				        std::array<tsa_dtype, RESULT_ALLOC_LEN> profile;
						std::array<idx_dtype, RESULT_ALLOC_LEN> index;
			    };
			    ScrimpDistribPar() : ScrimpVertBlocked(), _profile_updates(0){};

    public:
				virtual void compute_matrix_profile(const Scrimppp_params& params);
				virtual void initialize(const Scrimppp_params& params);
				virtual void log_info();
    protected:
				void init_dotproducts(idx_dtype num_dotproducts, tsa_dtype outbuf[], const tsa_dtype A[], const tsa_dtype B[], const int windowSize) NOINLINE_IF_GPROF_EN;

				void eval_tile_blocked(
				        tsa_dtype prof_colmin[],
				        idx_dtype idx_colmin[],
				        tsa_dtype prof_rowmin[],
				        idx_dtype idx_rowmin[],
				        tsa_dtype tmpQ[],
				        const idx_dtype blocklen,
				        const idx_dtype trianglen,
				        const idx_dtype tilelen,
				        const tsa_dtype A_hor[],
				        const tsa_dtype A_vert[],
				        const idx_dtype windowSize,
				        const tsa_dtype s_hor[],
				        const tsa_dtype mu_hor[],
				        const tsa_dtype s_vert[],
				        const tsa_dtype mu_vert[],
				        const idx_dtype baserow,
				        const idx_dtype basecol
				        ) NOINLINE_IF_GPROF_EN;

				static void MPI_MatProfSOA_reduction(void* invec, void* inoutvec, int *len,
				                              MPI_Datatype *datatype);
				static void MatProfSOA_score_reduction(void* invec, void* inoutvec, int *len,
				                              MPI_Datatype *datatype);
				void commit_mpi_SOA_prof_type(MPI_Datatype* ptr_total_prof_mpi_type, tsa_dtype* profile_buf, idx_dtype* idx_buf, int profile_length);
    private:
				bool _finished_initialization=false;
				long _profile_updates;
				long _eval_ctr;

				aligned_ptr<MatProfSOA> _hor_result;
				aligned_ptr<MatProfSOA> _vert_result;

				struct RankInfo{
					const int rank_input_in_rowcomm=0, rank_input_in_colcomm=0;

					int _world_size, _world_rank;
					int _rank_in_row, _rank_in_col;
					int _accu_results_rowcomm=-2, _accu_results_colcomm=-2;
					int _exclusion_row_target=-2, _eexclusion_col_target=-2;
					int _rank_in_final_excesscomm=-1, _final_exess_target=-2;
				} _rank_info;

				struct TileInfo{
					int _tile_id, _tile_num; // id of the local work tile and overall number of tiles
					idx_dtype _partition_row, _partition_col; // row and column coordinate of the local work tile. Note: There might be a upper and a lower tile with the same coordinates...
					bool _is_upper_triangle;
				} _tile_info;

				struct MpiTypes{
					MPI_Datatype hor_prof_mpitype, vert_prof_mpitype =MPI_DATATYPE_NULL;
					MPI_Datatype vert_prof_mainslice_mpitype, hor_prof_mainslice_mpitype =MPI_DATATYPE_NULL;
					MPI_Datatype vert_prof_exclusionslice_mpitype, hor_prof_exclusionslice_mpitype =MPI_DATATYPE_NULL;
					MPI_Datatype final_excess_slice_mpitype =MPI_DATATYPE_NULL;
					MPI_Op score_reduction_op =MPI_NO_OP;
				} _mpi_types;

				Timepoint _init_start_time;

				MPI_Comm _exclusion_rowcomm = MPI_COMM_NULL, _exclusion_colcomm = MPI_COMM_NULL;
				MPI_Comm _final_excesscomm = MPI_COMM_NULL;
				MPI_Comm _row_comm=MPI_COMM_NULL, _col_comm=MPI_COMM_NULL; //	rowise/colwise communicators for the reduction of the result (possibly spreading of the input
				MPI_Comm _row_result_comm=MPI_COMM_NULL, _col_result_comm=MPI_COMM_NULL;
				MPI_Comm _prof_io_comm=MPI_COMM_NULL, _ts_io_comm=MPI_COMM_NULL;

				struct LoggingInfo {
					int input_padding;
					int input_series_len;
					int time_series_len; // may contain additional padding
					int tile_length;
					int blocklen;
					idx_dtype profile_length;
					idx_dtype work_triang_len;
					bool is_processing_padding;
				} _log_info;

				using TracepointMap = std::map<std::string, Timepoint>;
				using SynctimeMap = std::map<std::string, Timespan>;
				struct TimeLoggingInfo {
					Timespan comp_time, comm_time, bcast_time, evaluation_time, io_time, comm_setup_time,
					        reading_time, work_time, precomp_time, setup_time, cleanup_time, dotproduct_time;
					TracepointMap  _time_trace_map;
					SynctimeMap _synctime_map;
				} _timing_info;
};

} // namespace matrix_profile
#endif // ScrimpDistribPar_HPP
