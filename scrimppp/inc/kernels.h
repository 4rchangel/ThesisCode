#ifndef KERNELS_H
#define KERNELS_H

#include <settings.h>

namespace matrix_profile {
    void eval_diag_block_triangle(
	        tsa_dtype prof_colmin[],
	        idx_dtype idx_colmin[],
	        tsa_dtype prof_rowmin[],
	        idx_dtype idx_rowmin[],
	        tsa_dtype tmpQ[],
	        const idx_dtype blocklen,
	        const idx_dtype trianglen,
	        const tsa_dtype A_hor[],
	        const tsa_dtype A_vert[],
	        const idx_dtype windowSize,
	        const tsa_dtype s_hor[],
	        const tsa_dtype mu_hor[],
	        const tsa_dtype s_vert[],
	        const tsa_dtype mu_vert[],
	        const idx_dtype baserow, // row of the top left corner
	        const idx_dtype basecol  // col of the top left corner
	        );
}

#endif // KERNELS_H
