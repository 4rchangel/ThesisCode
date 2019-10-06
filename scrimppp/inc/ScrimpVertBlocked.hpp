#ifndef SCRIMPVERTBLOCKED_HPP
#define SCRIMPVERTBLOCKED_HPP

#include <ScrimpSequOpt.hpp>

namespace matrix_profile {

class ScrimpVertBlocked : public ScrimpSequOpt {
    public:
	    ScrimpVertBlocked() {}
		virtual void compute_matrix_profile(const Scrimppp_params& params);

    protected:
		void eval_diagonal_block(const int blocklen, aligned_int_vec& profileIndex, const aligned_tsdtype_vec& A, aligned_tsdtype_vec& initial_zs, const int windowSize, const aligned_tsdtype_vec& ASigmaInv, const aligned_tsdtype_vec& AMeanScaledSigSqrM, const int diag, aligned_tsdtype_vec& profile) NOINLINE_IF_GPROF_EN;
};

} // namespace matrix_profile
#endif // multiple incluseion guard
