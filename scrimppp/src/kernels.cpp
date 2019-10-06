#include <kernels.h>

#include <immintrin.h>

#include <logging.hpp>

using namespace matrix_profile;


    void eval_diag_block_triangle_ABBA_vec(
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
	        const idx_dtype baserow,
	        const idx_dtype basecol
	    )
{
#ifndef __AVX2__
		EXEC_TRACE("evaluate block of diagonals in triangle. Triangle length: " << trianglen << " blocklen " << blocklen);
		const idx_dtype VECLEN = 4;
		static_assert(sizeof(tsa_dtype) == 8,  "only 64bit signed double precision tsa_dtype supported");
		static_assert(sizeof(idx_dtype) == 8,  "only 64bit signed integer datatype supported for index");
		static_assert(sizeof(idx_dtype) == sizeof(double), "casting between m256d and m256i for convnience");

		//iteration in diagonal direction for all of the blocked diagonals.
		    //the loop is expressed in terms of the column-coordinate
		for (idx_dtype j=0; j<trianglen; j++)
		{
			    tsa_dtype profile_j = prof_colmin[j];
				idx_dtype index_j = idx_colmin[j];
				__m256d mm_profile_j_veclane = _mm256_broadcast_sd(prof_colmin + j); // tsa_dtype profile_j = prof_colmin[j];
				__m256d mm_index_j_veclane_dalias = _mm256_broadcast_sd( (double*)(idx_colmin + j) ); // idx_dtype index_j = idx_colmin[j]; //TODO: replace double-casting hack with proper load/bcast
				//__m256i mm_index_j_veclane = _mm256_castpd_si256(mm_index_j_veclane_dalias);
				const idx_dtype col_in_mat = j+basecol;

				//iteration over all diagonals in the block. Handling incomplete blocks with the "iterlimit"
				const idx_dtype iterlimit = j+std::min(blocklen, trianglen-j);
				idx_dtype veci=j;
				for (; veci < iterlimit-VECLEN+1; veci+=VECLEN)
				{
					    //for (idx_dtype i=veci; i<veci+VECLEN; i+=VECLEN) {
					    //_mm256i mm_i = ;
					    const idx_dtype vecdiag = veci-j; //const idx_dtype diag = i-j;
						__m256d mm_tmp;
						__m256d mm_corrScore;
//			const tsa_dtype corrScore = tmpQ[diag]* (s_vert[i] * s_hor[j]) - mu_vert[i] * mu_hor[j];
						{
							    __m256d mm_svert = _mm256_loadu_pd( s_vert + veci );
								__m256d mm_shor = _mm256_broadcast_sd( s_hor + j);
								mm_corrScore = _mm256_mul_pd(mm_svert, mm_shor);
						}
						{
							    __m256d mm_muvert = _mm256_loadu_pd(mu_vert +veci );
								__m256d mm_muhor = _mm256_broadcast_sd(mu_hor +j );
								mm_tmp = _mm256_mul_pd(mm_muhor, mm_muvert);
						}
						__m256d mm_tmpQ = _mm256_loadu_pd( tmpQ + vecdiag);
						{
							    mm_corrScore = _mm256_mul_pd(mm_tmpQ, mm_corrScore);
								mm_corrScore = _mm256_sub_pd(mm_corrScore, mm_tmp);
								//mm_corrScore = _mm256_fnmadd_pd(mm_tmpQ, mm_corrScore, mm_tmp); //TODO: check replacement of previous ops by FMA
						}

						// update dotproducts tmpQ
						{
							    __m256d mm_Avertfar = _mm256_loadu_pd(A_vert + windowSize + veci);
								__m256d mm_Ahorfar = _mm256_broadcast_sd(A_hor + windowSize + j);
								__m256d mm_Avert = _mm256_loadu_pd(A_vert + veci);
								__m256d mm_Ahor = _mm256_broadcast_sd(A_hor + j);
								mm_tmp = _mm256_mul_pd(mm_Avertfar, mm_Ahorfar);
								mm_tmpQ = _mm256_add_pd(mm_tmpQ, mm_tmp );
								mm_tmp = _mm256_mul_pd(mm_Avert, mm_Ahor);
								mm_tmpQ = _mm256_sub_pd(mm_tmpQ, mm_tmp );
						}

						// update vertical profile result, i.e. looking for new minimu in rows
						{
							    __m256d mm_vertprof = _mm256_loadu_pd(prof_rowmin + veci);
								__m256d mm_vert_mask_pd = _mm256_cmp_pd(mm_corrScore, mm_vertprof, _CMP_GT_OS);
								__m256i mm_vert_mask_alias = _mm256_castpd_si256(mm_vert_mask_pd);
								__m256i mm_tmp_idx = _mm256_set1_epi64x( col_in_mat );
								_mm256_maskstore_pd(prof_rowmin +veci, mm_vert_mask_alias, mm_corrScore);
								//static_assert(sizeof(idx_dtype) == sizeof(long long int), "intrinsic type hack, assuming equal length of long long and long");
								//_mm256_maskstore_epi64((long long*)(idx_rowmin+veci), mm_vert_mask_alias, mm_tmp_idx);
								auto mm_tmp_vertidx_double_alias = _mm256_castsi256_pd(mm_tmp_idx);
								_mm256_maskstore_pd((double*)(idx_rowmin+veci), mm_vert_mask_alias, mm_tmp_vertidx_double_alias);// TODO: use AVX2 int version instead of the double-hack
						}

						//update horizontal profile result, i.e. looking for new minima in cols
						// generating a vertical result for each vector lane.
						// we will look for a minimum among the at the very end of a block
						// also at this point, we only store the indices of the base lane! Need to add appropriate offsets later!
						{
							    auto mm_i = _mm256_set1_epi64x( veci );
								auto mm_i_dalias = _mm256_castsi256_pd(mm_i);
								__m256d mm_lane_mask_pd = _mm256_cmp_pd(mm_corrScore, mm_profile_j_veclane, _CMP_GT_OS);
								//__m256i mm_lane_mask_alias = _mm256_castpd_si256(mm_lane_mask_pd);
								mm_profile_j_veclane = _mm256_blendv_pd (mm_profile_j_veclane, mm_corrScore, mm_lane_mask_pd);
								mm_index_j_veclane_dalias = _mm256_blendv_pd(mm_index_j_veclane_dalias, mm_i_dalias, mm_lane_mask_pd);
						}

						tsa_dtype scoreArr[VECLEN];
						tsa_dtype QArr[VECLEN];
						_mm256_storeu_pd(scoreArr, mm_corrScore);
						_mm256_storeu_pd(QArr, mm_tmpQ);

						// store back the updated dotproducts into memory
						// IMPORTANT PART OF THE VCTORIZED ALGO!
						_mm256_storeu_pd(tmpQ + vecdiag, mm_tmpQ);
				}
				//integration of the profile_j results among the vector lanes
				{
					    tsa_dtype avx_prof[VECLEN] __attribute__ ((aligned (16)));
						idx_dtype avx_index[VECLEN] __attribute__ ((aligned (16)));

						_mm256_store_pd(avx_prof, mm_profile_j_veclane); // get the data from the AVX registers in a commodity array
						static_assert(sizeof(idx_dtype) == 8,  "only 64bit signed integer datatype supported");

						__m256i mm_index_j_veclane = _mm256_castpd_si256(mm_index_j_veclane_dalias);
						_mm256_store_si256((__m256i*)avx_index, mm_index_j_veclane );


						//"horizontal simd reduction", into a register
						for (size_t veciter=0; veciter<VECLEN; ++veciter) {
							    //EXEC_TRACE("avx_prof " << avx_prof[veciter] << " id " << avx_index[veciter] << " profile_j: "<<profile_j);
							    if (avx_prof[veciter] > profile_j) {
									    profile_j = avx_prof[veciter];
										index_j = avx_index[veciter]+veciter+baserow;
								}
						}
				}
				// non-vectorized remainder loop
				for (idx_dtype i=veci; i < iterlimit; ++i)
				{
					    const idx_dtype diag = i-j;
						const tsa_dtype corrScore = tmpQ[diag]* (s_vert[i] * s_hor[j]) - mu_vert[i] * mu_hor[j];
						EXEC_TRACE ("eval i: " << j+basecol << " j: " << i+baserow  << " lastz " << tmpQ[diag] << " mu_h_j " << mu_hor[j]); //i and j naming of the output follows the convention in the scrimp_squ version for debugging: i denotes colum-, j row offsets

						tmpQ[diag] += A_vert[i+windowSize]*A_hor[j+windowSize]  ; //- A_vert[i]*A_hor[j];
						tmpQ[diag] -= A_vert[i]*A_hor[j];

						if (corrScore > prof_rowmin[i]) {
							    prof_rowmin[i] = corrScore;
								idx_rowmin[i] = j+basecol;
						}

						if (corrScore > profile_j) {
							    profile_j = corrScore;
								index_j = i+baserow;
						}
				}
				//integration of the result in i direction into memory
				if (profile_j > prof_colmin[j]) {
					    prof_colmin[j] = profile_j;
						idx_colmin[j] = index_j;
				}
		}
#else
// AVX2 version!

		EXEC_TRACE("evaluate block of diagonals in triangle. Triangle length: " << trianglen << " blocklen " << blocklen);
		const idx_dtype VECLEN = 4;
		static_assert(sizeof(tsa_dtype) == 8,  "only 64bit signed double precision tsa_dtype supported");
		static_assert(sizeof(idx_dtype) == 8,  "only 64bit signed integer datatype supported for index");
		static_assert(sizeof(idx_dtype) == sizeof(double), "casting between m256d and m256i for convnience");

		//iteration in diagonal direction for all of the blocked diagonals.
		    //the loop is expressed in terms of the column-coordinate
		for (idx_dtype j=0; j<trianglen; j++)
		{
			    tsa_dtype profile_j = prof_colmin[j];
				idx_dtype index_j = idx_colmin[j];
				__m256d mm_profile_j_veclane = _mm256_broadcast_sd(prof_colmin + j); // tsa_dtype profile_j = prof_colmin[j];
				__m256d mm_index_j_veclane_dalias = _mm256_broadcast_sd( (double*)(idx_colmin + j) ); // idx_dtype index_j = idx_colmin[j]; //TODO: replace double-casting hack with proper load/bcast
				//__m256i mm_index_j_veclane = _mm256_castpd_si256(mm_index_j_veclane_dalias);
				const idx_dtype col_in_mat = j+basecol;
				__m256d mm_muhor = _mm256_broadcast_sd( mu_hor +j );
				__m256d mm_shor = _mm256_broadcast_sd( s_hor + j);
				__m256d mm_Ahor = _mm256_broadcast_sd(A_hor + j);
				__m256d mm_Ahorfar = _mm256_broadcast_sd(A_hor + windowSize + j);

				//iteration over all diagonals in the block. Handling incomplete blocks with the "iterlimit"
				const idx_dtype iterlimit = j+std::min(blocklen, trianglen-j);
				const idx_dtype veciterlimit = iterlimit-VECLEN+1;
				idx_dtype veci=j;
				for (; veci < veciterlimit; veci+=VECLEN)
				{
					    //for (idx_dtype i=veci; i<veci+VECLEN; i+=VECLEN) {
					    //_mm256i mm_i = ;
					    const idx_dtype vecdiag = veci-j; //const idx_dtype diag = i-j;
						__m256d mm_tmp; // temporary
						__m256d mm_corrScore;
						__m256d mm_tmpQ; // store 4 tmpQs, starting at offset vecdiag
//			const tsa_dtype corrScore = tmpQ[diag]* (s_vert[i] * s_hor[j]) - mu_vert[i] * mu_hor[j];
						        __m256d mm_svert = _mm256_loadu_pd( s_vert + veci );
								mm_corrScore = _mm256_mul_pd(mm_svert, mm_shor);
								__m256d mm_muvert = _mm256_loadu_pd(mu_vert +veci );
								mm_tmp = _mm256_mul_pd(mm_muhor, mm_muvert);
								mm_tmpQ = _mm256_loadu_pd( tmpQ + vecdiag);
								//_mm_prefetch( tmpQ + vecdiag +VECLEN, 1);

//                               mm_corrScore = _mm256_mul_pd(mm_tmpQ, mm_corrScore);
//                               mm_corrScore = _mm256_sub_pd(mm_corrScore, mm_tmp);
								 mm_corrScore = -_mm256_fnmadd_pd(mm_tmpQ, mm_corrScore, mm_tmp); //TODO: check replacement of previous ops by FMA, FOR SOME REASON IT DOES NOT WORK WITH THIS LINE. mybe because of multiple usage of mm_corrScore?

								// update dotproducts tmpQ
								__m256d mm_Avertfar = _mm256_loadu_pd(A_vert + windowSize + veci);
								__m256d mm_Avert = _mm256_loadu_pd(A_vert + veci);

								mm_tmpQ = _mm256_fmadd_pd(mm_Avertfar, mm_Ahorfar, mm_tmpQ);
								mm_tmpQ = _mm256_fnmadd_pd(mm_Avert, mm_Ahor, mm_tmpQ);
								//_mm_prefetch(A_vert + veci + VECLEN, 1);
								//_mm_prefetch(A_vert + veci + windowSize + VECLEN, 1);
								_mm256_storeu_pd(tmpQ + vecdiag, mm_tmpQ);

								// update vertical profile result, i.e. looking for new minimu in rows
								__m256d mm_vertprof = _mm256_loadu_pd(prof_rowmin + veci);
								__m256d mm_vert_mask_pd = _mm256_cmp_pd(mm_corrScore, mm_vertprof, _CMP_GT_OS);
								__m256i mm_vert_mask_alias = _mm256_castpd_si256(mm_vert_mask_pd);
								__m256i mm_tmp_idx = _mm256_set1_epi64x( col_in_mat );
								_mm256_maskstore_pd(prof_rowmin +veci, mm_vert_mask_alias, mm_corrScore);
								static_assert(sizeof(idx_dtype) == sizeof(long long int), "intrinsic type hack, assuming equal length of long long and long");
								_mm256_maskstore_epi64((long long*)(idx_rowmin+veci), mm_vert_mask_alias, mm_tmp_idx);

								//update horizontal profile result, i.e. looking for new minima in cols
								// generating a vertical result for each vector lane.
								// we will look for a minimum among the at the very end of a block
								// also at this point, we only store the indices of the base lane! Need to add appropriate offsets later!
								auto mm_i = _mm256_set1_epi64x( veci );
								auto mm_i_dalias = _mm256_castsi256_pd(mm_i);
								__m256d mm_lane_mask_pd = _mm256_cmp_pd(mm_corrScore, mm_profile_j_veclane, _CMP_GT_OS);
								mm_profile_j_veclane = _mm256_blendv_pd (mm_profile_j_veclane, mm_corrScore, mm_lane_mask_pd);
								mm_index_j_veclane_dalias = _mm256_blendv_pd(mm_index_j_veclane_dalias, mm_i_dalias, mm_lane_mask_pd);

				}
				//integration of the profile_j results among the vector lanes
				{
					    __m256i mm_index_j_veclane = _mm256_castpd_si256(mm_index_j_veclane_dalias);
						// max score reduction among the vector lanes.
						__m256i laneidx = _mm256_set_epi64x(3,2,1,0);
						mm_index_j_veclane = _mm256_add_epi64(mm_index_j_veclane, laneidx);
						mm_index_j_veclane_dalias = _mm256_castsi256_pd(mm_index_j_veclane);

						// permutation within lower hand higher 128bit subvectors
						auto mm_cmpprof = _mm256_permute_pd(mm_profile_j_veclane, 0x05);
						auto mm_cmpidx = _mm256_permute_pd(mm_index_j_veclane_dalias, 0x05);
						auto maxmask= _mm256_cmp_pd(mm_cmpprof, mm_profile_j_veclane, _CMP_GT_OS);
						// first of two reductions
						mm_profile_j_veclane = _mm256_blendv_pd( mm_profile_j_veclane, mm_cmpprof, maxmask);
						//mm_index_j_veclane_dalias = _mm256_castsi256_pd(mm_index_j_veclane);
						mm_index_j_veclane_dalias = _mm256_blendv_pd(mm_index_j_veclane_dalias, mm_cmpidx, maxmask);

						// permutation in order to compare higher to lower part...
						mm_cmpprof = _mm256_permute2f128_pd( mm_profile_j_veclane, mm_profile_j_veclane,  0x01);
						mm_cmpidx = _mm256_permute2f128_pd( mm_index_j_veclane_dalias, mm_index_j_veclane_dalias, 0x01);
						maxmask = _mm256_cmp_pd(mm_cmpprof, mm_profile_j_veclane, _CMP_GT_OS);
						// reduction (actually we are only interested in the very first value. could now use scalars aswell
						// first of two reductions
						mm_profile_j_veclane = _mm256_blendv_pd( mm_profile_j_veclane, mm_cmpprof, maxmask);
						mm_index_j_veclane_dalias = _mm256_blendv_pd(mm_index_j_veclane_dalias, mm_cmpidx, maxmask);

						//write back into the "normal" variables
						auto profj128 = _mm256_extractf128_pd(mm_profile_j_veclane, 0x0);
						_mm_store_sd(&profile_j, profj128);
						auto idx128 = _mm256_extractf128_pd(mm_index_j_veclane_dalias, 0x0);
						_mm_store_sd((double*) &index_j, idx128);
						index_j += baserow;

				}
				// non-vectorized remainder loop
				for (idx_dtype i=veci; i < iterlimit; ++i)
				{
					    const idx_dtype diag = i-j;
						const tsa_dtype corrScore = tmpQ[diag]* (s_vert[i] * s_hor[j]) - mu_vert[i] * mu_hor[j];
						EXEC_TRACE ("eval i: " << j+basecol << " j: " << i+baserow  << " lastz " << tmpQ[diag] << " mu_h_j " << mu_hor[j]); //i and j naming of the output follows the convention in the scrimp_squ version for debugging: i denotes colum-, j row offsets

						tmpQ[diag] += A_vert[i+windowSize]*A_hor[j+windowSize]  ; //- A_vert[i]*A_hor[j];
						tmpQ[diag] -= A_vert[i]*A_hor[j];

						if (corrScore > prof_rowmin[i]) {
							    prof_rowmin[i] = corrScore;
								idx_rowmin[i] = j+basecol;
						}

						if (corrScore > profile_j) {
							    profile_j = corrScore;
								index_j = i+baserow;
						}
				}
				//integration of the result in i direction into memory
				if (profile_j > prof_colmin[j]) {
					    prof_colmin[j] = profile_j;
						idx_colmin[j] = index_j;
				}
		}
#endif
}

void eval_diag_block_triangle_ABBA_scalar(
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
        const idx_dtype baserow,
        const idx_dtype basecol
    )
{
	    EXEC_TRACE("evaluate block of diagonals in triangle. Triangle length: " << trianglen << " blocklen " << blocklen);
		const idx_dtype VECLEN = 8;

		//iteration in diagonal direction for all of the blocked diagonals.
		    //the loop is expressed in terms of the column-coordinate
		for (idx_dtype j=0; j<trianglen; j++)
		{
			    tsa_dtype profile_j = prof_colmin[j];
				idx_dtype index_j = idx_colmin[j];

				//iteration over all diagonals in the block. Handling incomplete blocks with the "iterlimit"
				const int iterlimit = j+std::min(blocklen, trianglen-j);
				for (idx_dtype i=j; i < iterlimit; ++i)
				{
					    const idx_dtype diag = i-j;
						const tsa_dtype corrScore = tmpQ[diag]* (s_vert[i] * s_hor[j]) - mu_vert[i] * mu_hor[j];
						EXEC_TRACE ("eval i: " << j+basecol << " j: " << i+baserow  << " lastz " << tmpQ[diag] << " mu_h_j " << mu_hor[j]); //i and j naming of the output follows the convention in the scrimp_squ version for debugging: i denotes colum-, j row offsets

						tmpQ[diag] += A_vert[i+windowSize]*A_hor[j+windowSize]  ; //- A_vert[i]*A_hor[j];
						tmpQ[diag] -= A_vert[i]*A_hor[j];

						if (corrScore > prof_rowmin[i]) {
							    prof_rowmin[i] = corrScore;
								idx_rowmin[i] = j+basecol;
						}

						if (corrScore > profile_j) {
							    profile_j = corrScore;
								index_j = i+baserow;
						}
				}
				//integration of the result in i direction into memory
				if (profile_j > prof_colmin[j]) {
					    prof_colmin[j] = profile_j;
						idx_colmin[j] = index_j;
				}
		}
}

void matrix_profile::eval_diag_block_triangle(
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
        ) {
#ifdef  USE_INTRINSICS_KERNEL
	EXEC_DEBUG("invoking intrinsics kernel");
	eval_diag_block_triangle_ABBA_vec(
#else
	EXEC_DEBUG("invoking scalar kernel");
	eval_diag_block_triangle_ABBA_scalar(
#endif
	        prof_colmin,
	        idx_colmin,
	        prof_rowmin,
	        idx_rowmin,
	        tmpQ,
	        blocklen,
	        trianglen,
	        A_hor,
	        A_vert,
	        windowSize,
	        s_hor,
	        mu_hor,
	        s_vert,
	        mu_vert,
	        baserow, // row of the top left corner
	        basecol  // col of the top left corner
	        );
}
