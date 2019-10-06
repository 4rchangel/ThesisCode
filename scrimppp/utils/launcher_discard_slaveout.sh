#!/bin/bash
# MPI application launcher, usage like mpirun -n 1 launcher_discard_slaves.sh ./exec_path/application app_param_1 app_param_2 ...
# input param 1: directory for creation of log files
# input param 2...: program invocation, i.e. executable with arguments
# inspired by https://software.intel.com/en-us/forums/intel-moderncode-for-parallel-architectures/topic/747597
# all stdout and stderr outputs from the workers will be discarded to /dev/null
# only the masters output is "kept" to stdout

# INPUT: command line to be invoked with appropiately redirected outputs

if [ -n "${PMI_RANK}" ]; then # rank in case of intel MPI
	rank=${PMI_RANK}
elif [ -n "${OMPI_COMM_WORLD_RANK}" ]; then
	rank="${OMPI_COMM_WORLD_RANK}"
else
	echo "MPI rank could not be detrmined by launcher script. Terminating... check which environment variables are set by mpirun"
	exit	
fi

if [ "${rank}" -gt 0 ]; then
	$2 ${@:3} 2>/dev/null 1>/dev/null
else
	$2 ${@:3} 
fi

# quick and dirty way to figur out the environment variables of the active environment
#env | grep MPI | grep RANK

