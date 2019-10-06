cmake_minimum_required(VERSION 3.6)
project(scrimppp LANGUAGES CXX)

# define the target application
add_executable(scrimppp src/main.cpp)
add_executable(cvt_ts cvt_src/cvt_ts.cpp)
add_executable(cvt_matprof cvt_src/cvt_matprof.cpp)

# enable c++14
set_property(TARGET scrimppp cvt_ts cvt_matprof PROPERTY CXX_STANDARD 14)

# determine, whether to build and include the original FFTW implementation
option (NO_ORIG "Optionally exclude the original Scrimppp implementation (and the according FFTW dependency)" OFF) # default log level: 1=INFO

# search for CMake modules in the system default and in the cmake subfolder of the source directory
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake;${CMAKE_MODULE_PATH}")
# MPI dependency
find_package(MPI REQUIRED)
target_include_directories(scrimppp  PRIVATE ${MPI_INCLUDE_PATH})
if (NOT NO_ORIG)
    message ( " Including the original algorithm, linking against FFTW" )
    find_package(FFTW REQUIRED) # does not work, but simple linker option suffices for compilation
    include_directories(scrimppp PRIVATE ${FFTW_INCLUDES} ) # does not work, but simple linker option suffices for compilation
endif()
# boost
# set(Boost_USE_STATIC_LIBS ON )
# set(Boost_USE_STATIC_RUNTIME OFF )
find_package(Boost REQUIRED COMPONENTS program_options filesystem system)
# NOTE: PAPI and LIKWID will are searched for later on, only if needed...

# at least my local mpi_ch installation has problems with c++11: some spaces in preprocessor macros are missing. Suppress it by compiler flag
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-literal-suffix")
# target_compile_options(scrimppp PUBLIC -Wno-literals-suffix)

# ----------------------------------------------------------------------------------------------
# add project sources
# ----------------------------------------------------------------------------------------------
# general project-specific header directory
target_include_directories(scrimppp PRIVATE inc)

# source files
target_sources(scrimppp
    PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}/src/MatProfAlgo
        ${CMAKE_CURRENT_LIST_DIR}/src/ScrimpSequ
        ${CMAKE_CURRENT_LIST_DIR}/src/ScrimpSequOpt
        ${CMAKE_CURRENT_LIST_DIR}/src/ScrimpBlocked
        ${CMAKE_CURRENT_LIST_DIR}/src/ScrimpVertBlocked
        ${CMAKE_CURRENT_LIST_DIR}/src/ScrimpSequVec
        ${CMAKE_CURRENT_LIST_DIR}/src/ScrimpTrivialPar
        ${CMAKE_CURRENT_LIST_DIR}/src/ScrimpTrivialParVec
        ${CMAKE_CURRENT_LIST_DIR}/src/ScrimpDistribPar
        ${CMAKE_CURRENT_LIST_DIR}/src/logging
        ${CMAKE_CURRENT_LIST_DIR}/src/papiwrapper
        ${CMAKE_CURRENT_LIST_DIR}/src/biniobase
        ${CMAKE_CURRENT_LIST_DIR}/src/bintsfile
        ${CMAKE_CURRENT_LIST_DIR}/src/binproffile
        ${CMAKE_CURRENT_LIST_DIR}/src/partitioning_1d
        ${CMAKE_CURRENT_LIST_DIR}/src/checkerboard_partitioning
        ${CMAKE_CURRENT_LIST_DIR}/src/timing
        ${CMAKE_CURRENT_LIST_DIR}/src/kernels
    )
if ( NOT NO_ORIG )
    target_sources(scrimppp
        PRIVATE
            ${CMAKE_CURRENT_LIST_DIR}/src/ScrimpppOrig
    )
endif()
# ----------------------------------------------------------------------------------------------

# link libraries
# ----------------------------------------------------------------------------------------------
if ( NOT NO_ORIG )
    target_link_libraries(scrimppp PRIVATE ${FFTW_LIBRARIES})
endif()
target_link_libraries(scrimppp PRIVATE ${MPI_LIBRARIES})
target_link_libraries(scrimppp PRIVATE Boost::boost Boost::program_options Boost::filesystem Boost::system)
# ----------------------------------------------------------------------------------------------
# compiler flags, see https://cmake.org/pipermail/cmake/2011-June/045037.html
# ----------------------------------------------------------------------------------------------
if(MPI_COMPILE_FLAGS)
   target_compile_options(scrimppp PRIVATE "${MPI_COMPILE_FLAGS}")
endif()
if(MPI_LINK_FLAGS)
  target_link_libraries(scrimppp PRIVATE "${MPI_LINK_FLAGS}") # yes target_link_libraries is the official way to set linker options...
endif()

# needed, when the linked libraries (like boost on the lrz linux cluster) had been built with an old compiler
if (FIX_CXX11_ABI)
        target_compile_definitions(scrimppp PRIVATE _GLIBCXX_USE_CXX11_ABI=0)
        target_compile_definitions(cvt_ts PRIVATE _GLIBCXX_USE_CXX11_ABI=0)
        target_compile_definitions(cvt_matprof PRIVATE _GLIBCXX_USE_CXX11_ABI=0)
endif()

# compiler specific flags for vectorization
target_compile_options(scrimppp PRIVATE "-march=native")
if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    # compile with OpenMP (for vecotirzation and maybe hybrid parallelization)
    target_compile_options(scrimppp PRIVATE "-fopenmp")

    # gcc vectorization flags, see https://gcc.gnu.org/projects/tree-ssa/vectorization.html
    # set_target_properties(scrimppp PROPERTIES COMPILE_FLAGS "-ffast-math -msse2")
    target_compile_options(scrimppp PRIVATE "-ftree-vectorize" "-ftree-loop-if-convert" "-ffast-math") # actually they should be part of -O3
        if (ENABLE_VECTORIZATION_REPORT)
                target_compile_options(scrimppp PRIVATE  "-fopt-info-loop-vec-all")
        endif()
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
    if (ENABLE_VECTORIZATION_REPORT)
        target_compile_options(scrimppp PRIVATE "-qopt-report=5")
    endif()
    # compile with OpenMP (for vecotirzation and maybe hybrid parallelization)
    target_compile_options(scrimppp PRIVATE "-qopenmp")
    target_link_libraries(scrimppp PRIVATE "-qopenmp")
endif()

# option to enable profiling with gprof
if(ENABLE_GPROF)
    target_compile_options(scrimppp PRIVATE -pg)
    target_link_libraries(scrimppp PRIVATE -pg)
    target_compile_definitions(scrimppp PRIVATE ENABLE_GPROF)
    target_compile_definitions(scrimppp PRIVATE PROFILING)
endif()

# selection of PAPI profiling options
if(LIKWID_STATS)
    find_package(LIKWID REQUIRED)
    target_include_directories(scrimppp PRIVATE ${LIKWID_INCLUDE_DIRS})
    target_compile_definitions(scrimppp PRIVATE LIKWID_PERFMON)
    target_link_libraries(scrimppp PRIVATE ${LIKWID_LIBRARIES})
    target_compile_definitions(scrimppp PRIVATE PROFILING)
elseif(PAPI_CACHE_STATS)
    set(LINK_PAPI 1)
    target_compile_definitions(scrimppp PRIVATE PAPI_CACHE_STATS)
    target_compile_definitions(scrimppp PRIVATE PROFILING PAPI_PERFMON)
elseif(PAPI_INSTR_STATS)
    set(LINK_PAPI 1)
    target_compile_definitions(scrimppp PRIVATE PAPI_INSTR_STATS)
    target_compile_definitions(scrimppp PRIVATE PROFILING PAPI_PERFMON)
elseif(PAPI_BRANCH_STATS)
    set(LINK_PAPI 1)
    target_compile_definitions(scrimppp PRIVATE PAPI_BRANCH_STATS)
    target_compile_definitions(scrimppp PRIVATE PROFILING PAPI_PERFMON)
endif()

# profile properties of the algorithm
if (PROFILING)
        target_compile_definitions(scrimppp PRIVATE PROFILING)
endif()

# link against PAPI, iff required
if (LINK_PAPI)
    find_package(PAPI REQUIRED)
    target_include_directories(scrimppp PRIVATE ${PAPI_INCLUDE_DIRS})
    target_link_libraries(scrimppp PRIVATE ${PAPI_LIBRARIES})
endif()

# selection of log levels
set (EXEC_LOG_LEVEL 1 CACHE STRING "Logging level: 0-3 (ERROR, INFO, DEBUG, TRACE)") # default log level: 1=INFO
target_compile_definitions(scrimppp PRIVATE MP_LOG_LEVEL=${EXEC_LOG_LEVEL})
set (PERF_LOG_LEVEL 3 CACHE STRING "Logging level: 1,3 (INFO, TRACE)") # default log level: 3=TRACE
target_compile_definitions(scrimppp PRIVATE MP_PERF_LOG_LEVEL=${PERF_LOG_LEVEL})

# setting of algorithmic parameters
set (MAX_PROFILE_LENGTH 10000000 CACHE STRING "Maximum result length. Determines the fixed result buffer allocation size (in number of entries)")
target_compile_definitions(scrimppp PRIVATE MP_MAX_PROFILE_LENGTH=${MAX_PROFILE_LENGTH})
message( "MAX_PROFILE_LENGTH (SOA allocation size) set to ${MAX_PROFILE_LENGTH}" )

option(SYNC_AND_TRACK_IDLE "Enable synchronization an tracking of the idle times in distrib_par" OFF)
if (SYNC_AND_TRACK_IDLE)
    target_compile_definitions(scrimppp PRIVATE SYNC_AND_TRACK_IDLE=1)
    message("ENABLED synchronization in distrib par")
else()
    target_compile_definitions(scrimppp PRIVATE SYNC_AND_TRACK_IDLE=0)
    message("no synchronization and idle time tracking in distrib_par")
endif()

if (APPLY_MPIFIX)
    target_compile_definitions(scrimppp PRIVATE APPLY_MPIFIX=1)
    target_compile_definitions(cvt_ts PRIVATE APPLY_MPIFIX=1)
    target_compile_definitions(cvt_matprof PRIVATE APPLY_MPIFIX=1)
    message("apply MPI fix for ibm mpi")
endif()

option(IGNORE_INIT_TIME "Ignore the initialization time (i.e. MPI communicator setup) in the reported runtime measurment" ON)
if (IGNORE_INIT_TIME)
    message("Ignoring the initialization time")
    target_compile_definitions(scrimppp PRIVATE CMAKE_IGNORE_INIT_TIME=true)
else()
    message("Taking the initialization time into account")
    target_compile_definitions(scrimppp PRIVATE CMAKE_IGNORE_INIT_TIME=false)
endif()

option(USE_INTRINSICS_KERNEL "Use the gerneral ABBA kernel vectorized with intrinsics" ON)
if (USE_INTRINSICS_KERNEL)
    message("using the vectorized kernel based on intrinsics")
    target_compile_definitions(scrimppp PRIVATE USE_INTRINSICS_KERNEL=1)
else()
    message("using the plain (typically scalar) kernel")
endif()

set (BLOCKLENGTH 500 CACHE STRING "Block length for cache locality: 1... default: 500. Can be overriden at runtimewith --blocklen argument")
target_compile_definitions(scrimppp PRIVATE MP_BLOCKLENGTH=${BLOCKLENGTH})

if(INTEL_TRACE_COLLECTOR)
target_compile_options(scrimppp PRIVATE "-trace")
endif()

if (DISABLE_PREFETCHING)
        target_compile_options(scrimppp PRIVATE "-qopt-prefetch=0")
endif()

option (ENABLE_DEBUG_INFO "Generate gcc debug symbols, i.e. compile with -g" OFF)
if (ENABLE_DEBUG_INFO)
        message( "compiling with debug information" )
        target_compile_options(scrimppp PRIVATE "-g")
endif()


# conversion utitlity targets
target_include_directories(cvt_ts PRIVATE inc cvt_src ${MPI_INCLUDE_PATH})
target_include_directories(cvt_matprof PRIVATE inc cvt_src ${MPI_INCLUDE_PATH})
# time series converter
target_sources(cvt_ts
    PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}/cvt_src/cvt_ts.cpp
        ${CMAKE_CURRENT_LIST_DIR}/cvt_src/cvt_cli.cpp
        ${CMAKE_CURRENT_LIST_DIR}/src/biniobase
        ${CMAKE_CURRENT_LIST_DIR}/src/bintsfile
        ${CMAKE_CURRENT_LIST_DIR}/src/binproffile
        ${CMAKE_CURRENT_LIST_DIR}/src/logging
        ${CMAKE_CURRENT_LIST_DIR}/src/partitioning_1d
    )
target_link_libraries(cvt_ts PRIVATE ${MPI_LIBRARIES})
target_link_libraries(cvt_ts PRIVATE Boost::boost Boost::program_options Boost::filesystem Boost::system)
# matrixprofile converter
target_sources(cvt_matprof
    PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}/cvt_src/cvt_matprof.cpp
        ${CMAKE_CURRENT_LIST_DIR}/cvt_src/cvt_cli.cpp
        ${CMAKE_CURRENT_LIST_DIR}/src/biniobase
        ${CMAKE_CURRENT_LIST_DIR}/src/bintsfile
        ${CMAKE_CURRENT_LIST_DIR}/src/binproffile
        ${CMAKE_CURRENT_LIST_DIR}/src/logging
        ${CMAKE_CURRENT_LIST_DIR}/src/partitioning_1d
 # for the sake of the store_matrix_profile method
        ${CMAKE_CURRENT_LIST_DIR}/src/ScrimpSequ
        ${CMAKE_CURRENT_LIST_DIR}/src/MatProfAlgo
        ${CMAKE_CURRENT_LIST_DIR}/src/timing
    )
target_link_libraries(cvt_matprof PRIVATE ${MPI_LIBRARIES})
target_link_libraries(cvt_matprof PRIVATE Boost::boost Boost::program_options Boost::filesystem Boost::system)
