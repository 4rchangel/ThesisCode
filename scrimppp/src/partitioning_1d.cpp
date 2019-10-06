#include "partitioning_1d.h"
#include <boost/assert.hpp>

/*
 * get the first and last elements within a partition
 * assuming 0 <= part_id < partinfo._num_partitions
*/
Partition1D get_partition(const int part_id, const PartitioningInfo& partinfo) {
	    Partition1D part;
		BOOST_ASSERT_MSG(part_id>=0, "negative partition id specified");
		BOOST_ASSERT_MSG(part_id < partinfo._num_partitions, "requested partition with higher id than existing partitions");

		// compute first element
		int overhead = part_id - partinfo._num_partitions_full_load;
		if (overhead >0) {
			    part._first_id= part_id * partinfo._size_full_load - overhead;
		}
		else {
			    part._first_id= part_id *partinfo._size_full_load;
		}

		// compute last element
		if (part_id < partinfo._num_partitions_full_load) {
			    part._last_id = part._first_id + partinfo._size_full_load-1; // -1 as last element is inclusive in the range
		}
		else {
			    part._last_id = part._first_id + partinfo._size_full_load-2; // -2 as last element is inclusive and the partition is of reduced (-1) size
		}

		return part;
}

PartitioningInfo split_work(const int num_partitions, const int ids_to_process) {
	return {
		._num_partitions = num_partitions,
		._num_partitions_full_load = ids_to_process - ((ids_to_process/num_partitions) * num_partitions),
		._size_full_load = (ids_to_process/num_partitions) + 1
	};
}

int get_workload(const int part_id, const PartitioningInfo& partinfo) {
	if (part_id < partinfo._num_partitions_full_load) {
		return partinfo._size_full_load;
	}
	else {
		return partinfo._size_full_load-1;
	}
}

