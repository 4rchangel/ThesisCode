#ifndef CHECKERBOARD_PARTITIONING_H
#define CHECKERBOARD_PARTITIONING_H

#include <settings.h>
#include <vector>

void get_partition_coords(
        const matrix_profile::idx_dtype rank,
        matrix_profile::idx_dtype& row,
        matrix_profile::idx_dtype& col,
        bool& is_upper
        );

std::vector<int> get_world_ranks_in_row(const int partition_row, const int world_size);
std::vector<int> get_world_ranks_in_col(const int partition_col, const int world_size);

std::vector<int> get_world_ranks_along_diag(const int world_size);
bool is_io_responsible(const int world_size, const int world_rank);

#endif // CHECKERBOARD_PARTITIONING_H
