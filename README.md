# Matrix Profile on HPC
This repository contains parallel implementations of the Matrix Profile algorithm for high performance computing clusters based on MPI, forming the core part of my thesis (<https://mediatum.ub.tum.de/1471292>).
My code parts are provided under the 3-Clause BSD License. The repository also contains a few marked foreign code sections, underlying the licenses as stated by their owners.

## Credit for Foreign Work
* The work started off with the SCRIMP++ C++ code from Zhu, Yeh, Zimmerman et al. at <https://sites.google.com/site/scrimpplusplus/>
In particular the scrimppp/srcScrimpppOrig.cpp and scrimppp/src/ScrimpSequ.cpp contain minor modifications of their work, in order provide a baseline and validation opportunity within a unified framework. With kind permission of Prof. Eamonn Keogh the code is republished here, still all rights belong to the original authors and interested users should concact him in case of licensing questions on it.
* Credits for the cmake/FindPAPI.cmake goes to "LLNL", licensed under the LGPL (see information in the file), cmake/FindLIKWID.cmake  is derived from it.
* Credit for cmake/FindFFTW.cmake goes to Jed Brown, licensed under a BSD 2-clause license (see information in the file)
* Data analysis of results may be conducted for example with the original GUI provided by Prof. Keogh (2016/01 <https://www.cs.ucr.edu/~eamonn/MatrixProfile.html>) by applying minor modifications for data loading.

## Build
CMake (>3.6) is used a a build tool. For a manual build, switch to some empty build directory and (for a release build) execute:
```shell	
cmake -DCMAKE_BUILD_TYPE=RELEASE <path_to_repository>/scrimppp/CMakeLists.txt
make
```

### Dependencies
The project generally depends on following libraries:
* MPI: tested with Intel MPI 2018 Update 2, OpenMPI 2.1.1, MPICH 3.2.1
* Boost: tested with 1.65
* FFTW3: tested with version 3.3.8, see <http://www.fftw.org/>
usage of FFTW3 can be switched off by specifying the CMake option "-DNO_ORIG=1". In this case the algorithm variant scrimppp_orig with the original SCRIMP++ implementation is unavailable and all FFTW dependencies are removed

optionally required for building with enabled source-code instrumentation are:
* LIKWID: tested with LIKWID 4.3.0, see <https://github.com/RRZE-HPC/likwid>
* PAPI: tested with PAPI 5.6.1, see <http://icl.utk.edu/papi/software/>

In case that the FFTW or PAPI libraries had been installed into custom user locations instead of the system default, the respective locations need to be specified before the build like this:
```shell
cmake -D FFTW_ROOT=<path_to_fftw_installation> PAPI_ROOT=<path_to_papi_installation> BOOST_ROOT=...
```
where <path_to_xyz_installation> is the path containing the lib/ and include/ subdirectories of the respective libraries. For example on the lrz linux cluster a release build can be run with:
```shell
cmake -DCMAKE_BUILD_TYPE=Release -D FFTW_ROOT=$FFTW_BASE -D BOOST_ROOT=$BOOST_BASE -D PAPI_ROOT=$PAPI_BASE ../scrimppp/
```
Usage of provided build wrappers in lrz_utils/build_helpers.sh avoids such manual settings for the lrz systems

### CMake Build options
Various build variants can be specified when running cmake. The most important settings are:
* -DMAX_PROFILE_LENGTH=1234 choose the allocation size for the result structure-of-arrays. Limits the maximum (per-node) problem size. In turn it is limited by the nodes available RAM. The default value is 10000000
* -DUSE_INTRINSICS_KERNEL=OFF use the scalar / auto-vectorized kernel instead of the manual vector-intrinsics one (the latter is the default)
* -DIGNORE_INIT_TIME=1 specifies that runtime measurement is started after program initialization is completed, i.e. before invocation of the algorithms main routine. It is set by default and can be turned off by setting it to an empty string (-DINGNORE_INIT_TIME='')
* -DSYNC_AND_TRACK_IDLE=ON can be used to add synchronization barriers and track the time spent in there, as used in the thesis in combination with Score-P to investigate bottlenecks
* -DENABLE_DEBUG_INFO=ON turns on the generation of debug information (also when using a release build). Required when applying Score-P instrumentation!
* -DENABLE_VECTORIZATION_REPORT=1 Generate a vectorization/optimization report during compilation with
* -DENABLE_GPROF=1 Enable gprof instrumentation of the program to generate profiling data during a program run
* -DLIKWID_STATS=1 Enable source code instrumentation with LIKWID marker API (adds LIKWID dependency)
* -DNO_ORIG=1 removes the baseline SCRIMP++ implementation and the dependency on FFTW. Useful if baseline is not required but the build environment has no FFTW installation
* -DFIX_CXX11_ABI=1 When linking against libraries built with a older compiler (like boost on one of the clusters i worked on), -D FIX_CXX11_ABI will fix the resulting errors.
* -DAPPLY_MPIFIX=1 adds preprocessor definitions for MPI address operations  missing in the IBM MPI headers on one of the systems i worked on
* -DEXEC_LOG_LEVEL=1 can be used to change amount of logging information. Default 1. WARNING: higher levels (like 2 == debug) add logging within the measured timings and for this reason should not be used while measuring performance
* -DPAPI_CACHE_STATS=1 or -DPAPI_INSTR_STATS=1 or -DPAPI__BRANCH_STATS=1 Add source code instrumentation with papi, using counters to track cache / instructions / branching (adds PAPI dependency)

## Program Execution
Program execution parameters can be obtained with the --help switch. Positional specification of some arguments is possible as well. The following exemplifies the execution of the original sequential crimpplusplus algorithm with a window length of 400 (time points) and a Prescrimp stride of 100 (time points) on the test data in the repository (which contains a generated random-walk time-series)
```shell
mpirun -n 1 ./scrimppp scrimppp_orig 400 100 <path to input data file>
```

A detailed list of command-line arguments is available via 
```shell
./scrimppp -h
```

the most important available algorithms are:
```shell
scrimp_sequ # original SCRIMP algorithm and kernel as a baseline with minor modifications for usage in our framework
scrimp_sequ_opt # kernel with arithmetic optimizations, still original SCRIMP iteration scheme
scrimp_vert_blocked # iteration scheme with vertical blocking. -DUSE_INTRINSICS_KERNEL=ON/OFF specifies, whether the auto-vectorized or manually vectorized kernel version is used
scrimp_triv_par # trivial parallelization , holding the full input series on each compute node. -DUSE_INTRINISCS_KERNEL=OFF can be used to switch to a scalar kernel
distrib_par # distributed parallelization DUSE_INTRINSICS_KERNEL=OFF can be used to switch to a scalar kernel
```
further available variants which were developement dead-ends due to bad performance
```
scrimp_sequ_vec # ivestigations on vectorization along a diagonal.
triv_par_vec # triv_par variant using std::vectors as buffers instead of SOA struct to use auto-vec kernel in trivial parallelization
```

## Experimental data
### Time series data
In our experiments we rely on random walk input data. Basic file format is a ASCII file with floating point valuse of the form 123.456 separated by spaces. A example file is given in scrimppp/data_testing/test.ascii. The parallelizations rely on a binary file format as specified in the thesis (ASCII files are also supported as input for all sequential algorithm implementations and scrimp_triv_par).
Two scripts are provided for generation of ASCII data:
* scrimppp/utils/random_ts_gen.py is a high level variant for generation of random walk series specifying only the time series and motif length
* scrimppp/utils/gnerate_ts.py allows to specify explitcly the number of different motifs and occurrences.
In order to convert ASCII-format series to the binary data format the project contains code of conversion utiliy names cvt_ts. Usage (also see --help argument):
```shell
# store the ASCII input series from test.ascii in a binary file
./builddirectory/cvt_ts ascii-to-bin scrimppp/data_testing/test.ascii --output binarytest.bin
```
### Matrix profile results
Similarly to the input series, the result files are of binary format and could/should be converted to ASCII processes for persistent and portable storage or as input for further processing.
```shell
# store the binary matrix profile result as a ASCII file
./builddirectory/cvt_matprof bin-to-ascii someResult.bin --output theAsciiResult.ascii
```
