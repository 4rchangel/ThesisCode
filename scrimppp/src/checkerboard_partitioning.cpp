#include <checkerboard_partitioning.h>

#include <math.h>
#include <cmath>

using namespace matrix_profile;

void get_partition_coords(
        const matrix_profile::idx_dtype world_rank,
        matrix_profile::idx_dtype& row,
        matrix_profile::idx_dtype& col,
        bool& is_upper
        )
{
	row = 0;
	idx_dtype first_in_row = 0;
	idx_dtype next_first = 1;

	// compute the row by comparing the rank to the rank of the first rank in the row
	    // a simpler/faster approach might be done by computing the floor of the square root of the rank
	    // this was just implemented quicker and should be neglectable as only performed once...
	while (next_first <= world_rank) {
		row += 1;
		first_in_row = row*row;
		next_first = first_in_row + 2*row +1;
	}

	// compute the column: there are always 2 triangles per column,
	// thus division by to of the offset into the row is the solution
	col = (world_rank-first_in_row)/2;

	// in even rows upper partitions have odd ranks,
	    // in odd rows upper partitions have even ranks
	if (0 == row%2) {
		is_upper = (1 == world_rank%2);
	}
	else {
		is_upper = (0 == world_rank%2);
	}
}

std::vector<int> get_world_ranks_in_row(const int partition_row, const int world_size) {
	const int first_in_row = partition_row*partition_row;
	const int limit = (partition_row+1)*(partition_row+1);
	std::vector<int> accu;

	for (int iter = first_in_row; iter < limit; ++iter) {
		accu.push_back(iter);
	}

	return accu;
}

std::vector<int> get_world_ranks_in_col(const int partition_col, const int world_size) {
	const int first_in_col = (partition_col+1)*(1+partition_col)-1;
	const int offset_in_row = first_in_col - partition_col*partition_col;
	const int collimit = std::sqrt(world_size);
	std::vector<int> accu;

	accu.push_back(first_in_col);
	for (int rowi = partition_col+1; rowi < collimit; ++rowi) {
		const int first_in_row = (rowi*rowi);
		accu.push_back(first_in_row+offset_in_row);
		accu.push_back(first_in_row+offset_in_row+1);
	}

	return accu;
}

std::vector<int> get_world_ranks_along_diag(const int world_size) {
	std::vector<int> ranks;
	int next_diag_rank = 0;

	for (int rowi = 2; next_diag_rank< world_size; ++rowi) {
		ranks.push_back(next_diag_rank);
		next_diag_rank = (rowi*rowi)-1;
	}

	return ranks;
}

bool is_io_responsible(const int world_size, const int world_rank) {
	// the ranks along the main diagonal are responsible for I/O
	matrix_profile::idx_dtype row, col;
	bool is_upper;
	get_partition_coords(world_rank, row, col, is_upper);
	return (row==col);
}
