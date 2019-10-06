#ifndef SCRIMPSEQUOPT_HPP
#define SCRIMPSEQUOPT_HPP

#include <ScrimpSequ.hpp>
#include <boost/align.hpp>

namespace matrix_profile {

class ScrimpSequOpt : public ScrimpSequ {
    public:
	        ScrimpSequOpt() {}
			virtual void compute_matrix_profile(const Scrimppp_params& params);

    protected:
			static const size_t align_stride = 1;
			static void precompute_window_statistics(const int windowSize, const aligned_tsdtype_vec &A, const int ProfileLength, aligned_tsdtype_vec &AMean, aligned_tsdtype_vec &ASigmaInv, const idx_dtype ts_len=-1) NOINLINE_IF_GPROF_EN;
			void init_diagonals(const int first_diag, const int last_diag, aligned_tsdtype_vec& initial_zs, const aligned_tsdtype_vec& A, const int windowSize) NOINLINE_IF_GPROF_EN;
			void init_all_diagonals(aligned_tsdtype_vec& initial_zs, const aligned_tsdtype_vec& A, const int windowSize) NOINLINE_IF_GPROF_EN;
			void eval_diagonal(aligned_int_vec& profileIndex, const aligned_tsdtype_vec& A, const aligned_tsdtype_vec& initial_zs, const int windowSize, const aligned_tsdtype_vec& ASigmaInv, const aligned_tsdtype_vec& AMeanScaledSigSqrM, const int diag, aligned_tsdtype_vec& profile) NOINLINE_IF_GPROF_EN;
};

} // namespace matrix_profile
#endif
