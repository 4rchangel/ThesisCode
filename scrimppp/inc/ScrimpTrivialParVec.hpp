#ifndef SCRIMPTRIVIALPAR_HPP
#define SCRIMPTRIVIALPAR_HPP

#include <ScrimpVertBlocked.hpp>
#include <mpi.h>
#include <settings.h>

namespace matrix_profile {

const int RESULT_ALLOC_LEN=MP_MAX_PROFILE_LENGTH;

class ScrimpTrivialParVec : public ScrimpVertBlocked
{
    public:
                ScrimpTrivialParVec() : ScrimpVertBlocked(), _profile_updates(0){}
    public:
                virtual void compute_matrix_profile(const Scrimppp_params& params);
    protected:
//                void store_matrix_profile(const MatProfSOA& result, const Scrimppp_params& params, const idx_dtype profileLength, const bool distributed_io)
                static void MPI_MatProfSOA_reduction(void* invec, void* inoutvec, int *len,
                                              MPI_Datatype *datatype);
    private:
                long _profile_updates;
                long _eval_ctr;
};

} // namespace matrix_profile
#endif // SCRIMPTRIVIALPAR_HPP
