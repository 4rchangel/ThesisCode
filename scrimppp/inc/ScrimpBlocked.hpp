#ifndef SCRIMPSEQBLOCKED_HPP
#define SCRIMPSEQBLOCKED_HPP

#include <ScrimpSequOpt.hpp>

namespace matrix_profile {

class ScrimpSequBlocked : public ScrimpSequOpt {
    public:
        ScrimpSequBlocked() {}
        virtual void compute_matrix_profile(const Scrimppp_params& params);

    protected:
        template<int NUM_DIAGS, int BLOCKLEN> void eval_diagonal_block(aligned_int_vec& profileIndex, const aligned_tsdtype_vec& A, aligned_tsdtype_vec& initial_zs, const int windowSize, const aligned_tsdtype_vec& ASigmaInv, const aligned_tsdtype_vec& AMeanScaledSigSqrM, const int diag, aligned_tsdtype_vec& profile) NOINLINE_IF_GPROF_EN;
};

} // namespace matrix_profile
#endif // multiple incluseion guard
