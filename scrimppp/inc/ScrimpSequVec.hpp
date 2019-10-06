#ifndef SCRIMPSEQUVEC_HPP
#define SCRIMPSEQUVEC_HPP

#include <ScrimpSequOpt.hpp>

namespace matrix_profile {

class ScrimpSequVec : public ScrimpSequOpt {
    public:
	    ScrimpSequVec() {}
		virtual void compute_matrix_profile(const Scrimppp_params& params);

    protected:
		typedef double tsa_dtype; // data type of a single time series element
		tsa_dtype init_diagonal(const aligned_tsdtype_vec &ASigma, aligned_tsdtype_vec &dotproduct, aligned_tsdtype_vec &profile, const int diag, const aligned_tsdtype_vec &A, aligned_int_vec &profileIndex, const int windowSize, const aligned_tsdtype_vec &AMean) NOINLINE_IF_GPROF_EN;
		//void eval_diagonal(aligned_int_vec& profileIndex, const aligned_tsdtype_vec& A, tsa_dtype& lastz, const int windowSize, aligned_tsdtype_vec& cumDotproduct, const aligned_tsdtype_vec& ASigmaInv, const aligned_tsdtype_vec& AMeanScaledSigSqrM, const int diag, aligned_tsdtype_vec& profile) NOINLINE_IF_GPROF_EN;
		void eval_diagonal(idx_dtype* const profileIndex, const tsa_dtype* const A, tsa_dtype& lastz, const idx_dtype windowSize, tsa_dtype* const cumDotproduct, tsa_dtype* const ASigmaInv, const tsa_dtype* const AMeanScaledSigSqrM, const idx_dtype diag, tsa_dtype* const profile, const idx_dtype profileLength);
};

} // namespace matrix_profile
#endif
