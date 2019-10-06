#ifndef PARTITIONING_1D_H
#define PARTITIONING_1D_H

struct Partition1D {
	    int _first_id;
		int _last_id;
};

struct PartitioningInfo {
	    int _num_partitions;
		int _num_partitions_full_load;
		int _size_full_load;
};


PartitioningInfo split_work(const int num_partitions, const int ids_to_process);
Partition1D get_partition(const int part_id, const PartitioningInfo& partinfo);
int get_workload(const int part_id, const PartitioningInfo& partinfo);

#endif // PARTITIONING_1D_H
