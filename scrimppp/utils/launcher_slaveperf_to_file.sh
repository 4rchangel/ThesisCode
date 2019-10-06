#!/bin/bash
# MPI application launcher, usage like mpirun -n 1 launcher_discard_slaves.sh ./tmp_log_dir ./exec_path/application app_param_1 app_param_2 ....
# input param 1: directory for creation of log files
# input param 2...: program invocation, i.e. executable with arguments
# inspired by https://software.intel.com/en-us/forums/intel-moderncode-for-parallel-architectures/topic/747597
# all stdout and stderr outputs from the workers will be discarded to /dev/null
# only the masters output is "kept" to stdout

# INPUT: command line to be invoked

if [ -n "${PMI_RANK}" ]; then # rank in case of intel MPI
	rank=${PMI_RANK}
elif [ -n "${OMPI_COMM_WORLD_RANK}" ]; then # rank in case of OpenMPI
	rank="${OMPI_COMM_WORLD_RANK}"
elif [ -n "${MP_CHILD}" ]; then #rank in case of IBM MPI
	rank="${MP_CHILD}"
else
	echo "MPI rank could not be detrmined by launcher script. Terminating... check which environment variables are set by mpirun"
	exit	
fi

if [ "${rank}" -gt 0 ]; then
	$2 ${@:3} 2>"${1}/log_r${rank}.txt" 1>/dev/null  # keep only the error and perfomance logs from slave processes
else
	$2 ${@:3} # keep all logs from the master at the standard output
fi

