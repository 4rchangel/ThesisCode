#ifndef SETTINGS_H
#define SETTINGS_H

#include <mpi.h>
#include <vector>
#include <boost/align.hpp>

// define a default maximum result length, if not set in the build configuration.
    // affects only SOA implementations (TrivialPar)
#ifndef MP_MAX_PROFILE_LENGTH
    #define MP_MAX_PROFILE_LENGTH 100000
#endif

// default value for blocking size, if not specified in the build configuration
#ifndef MP_BLOCKLENGTH
        #define MP_BLOCKLENGTH 500
#endif

#ifndef SYNC_AND_TRACK_IDLE
    #define SYNC_AND_TRACK_IDLE 0
#endif

namespace matrix_profile {
        using tsa_dtype = double; // data type of a single time series element
        using idx_dtype = long; //TODO: use this alias instead of ints everywhere...

        static const size_t alignment = 16;
		using aligned_tsdtype_vec = std::vector<tsa_dtype, boost::alignment::aligned_allocator<tsa_dtype, alignment> >;
		using aligned_int_vec = std::vector<idx_dtype, boost::alignment::aligned_allocator<idx_dtype, alignment> >;

		const MPI_Datatype ts_mpitype = MPI_DOUBLE;
		const bool SYNC_AFTER_STARTUP=true;
		const bool COLLECTIVE_DISTRIB_READ=true; // only regarding scrimp_triv_par: read slices and scatter them (alternatively have everyone reading)
		const bool DISTRIB_BCAST_INPUT=true; // only affecting the distrib_par version: one rank reading and broadcasting the input or (if false), evry rank is reading all its inputs itself

#ifdef CMAKE_IGNORE_INIT_TIME
		const bool IGNORE_INIT_TIME=CMAKE_IGNORE_INIT_TIME;
#else
		const bool IGNORE_INIT_TIME=true;
#endif
}

#ifdef APPLY_MPIFIX
// supermucs ibm mpi is lacking some definitions
// took the following ones from ompi
#define MPI_Aint_add(base, disp) ((MPI_Aint) ((char *) (base) + (disp)))
#define MPI_Aint_diff(addr1, addr2) ((MPI_Aint) ((char *) (addr1) - (char *) (addr2)))
#define PMPI_Aint_add(base, disp) MPI_Aint_add(base, disp)
#define PMPI_Aint_diff(addr1, addr2) MPI_Aint_diff(addr1, addr2)
#endif

#endif // SETTINGS_H
