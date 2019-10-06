#ifndef BINPROFFILE_H
#define BINPROFFILE_H

#include <biniobase.h>

#include <mpi.h>

class BinProfFile : BinIOBase
{
    protected:
	    using profile_store_dtype = double;
	    using idx_store_dtype = unsigned long;
	    struct ProfileEntry {
			profile_store_dtype _prof;
			idx_store_dtype _idx;
		    };

    protected:
		MPI_Datatype _mpi_prof_type;
		idx_store_dtype _requested_chunksize; // a value <= zero will be interpreted as writing everything as a single chunk

    public:
		BinProfFile(const std::string& path, const MPI_Comm = MPI_COMM_WORLD, const MPI_Info& fileinfo = MPI_INFO_NULL);
		~BinProfFile();
		void set_chunksize(const idx_store_dtype chunksize) {_requested_chunksize=chunksize;} // currently only affects writing of full profile

		// reding in the complete sereis from the file, either with individual or collective I/O....
		matrix_profile::idx_dtype read_matrix_profile(
		        matrix_profile::tsa_dtype * const profile_buf,
		        matrix_profile::idx_dtype * const idx_buf,
		        matrix_profile::idx_dtype buflen,
		        const bool read_collective);

		//distributed parallel collective write, assuming the full profile available everywhere in the communicator.
		    // i.e. no broadcasting is performed
		    // all processes in the communicator (constructor) need to participate
		void write_matrix_profile(const matrix_profile::tsa_dtype * const profile_buf,
		                          const matrix_profile::idx_dtype * const idx_buf,
		                          matrix_profile::idx_dtype proflen);
		// collective operation writing the locally specified slice with collective MPI I/O. All processes in the communicator (constructor) need to participate
		void write_matrix_profile_slice(
		        const matrix_profile::tsa_dtype * const profile_buf, const matrix_profile::idx_dtype * const idx_buf,
		        const matrix_profile::idx_dtype slicelen, const matrix_profile::idx_dtype sliceoffset,
		        const matrix_profile::idx_dtype proflen, // requred to be non-zero on all processes
		        bool write_header,
		        const int chunknum=1 // is required to be identical among all processes! Otherwise some processes will be stuck waiting for the other ones...
		        );
		//TODO determine optimal chunk size/number

		matrix_profile::idx_dtype get_profile_length();
		virtual unsigned int get_file_format_version(){return 1;}
};

#endif // BINPROFFILE_H
