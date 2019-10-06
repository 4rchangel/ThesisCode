# This file contains code from the Lawrence Livermore National Laboratory and all rights belong to them
# the code is licensed under the GNU Lesser GPL terms, as provided under ../../licenses/license_LGPL_FindPAPI.txt
# original source repository: https://github.com/LLNL/perf-dump/blob/master/cmake/FindPAPI.cmake
# change note: added header on 06.10.2019

# Try to find PAPI headers and libraries.
#
# Usage of this module as follows:
#
#     find_package(PAPI)
#
# Variables used by this module, they can change the default behaviour and need
# to be set before calling find_package:
#
#  PAPI_ROOT         Set this variable to the root installation of
#                      libpapi if the module has problems finding the
#                      proper installation path.
#
# Variables defined by this module:
#
#  PAPI_FOUND              System has PAPI libraries and headers
#  PAPI_LIBRARIES          The PAPI library
#  PAPI_INCLUDE_DIRS       The location of PAPI headers

find_path(PAPI_ROOT
    NAMES include/papi.h
)

find_library(PAPI_LIBRARIES
    # Pick the static library first for easier run-time linking.
    NAMES libpapi.so libpapi.a papi
    HINTS ${PAPI_ROOT}/lib # ${HILTIDEPS}/lib
)

find_path(PAPI_INCLUDE_DIRS
    NAMES papi.h
    HINTS ${PAPI_ROOT}/include # ${HILTIDEPS}/include
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(PAPI DEFAULT_MSG
    PAPI_LIBRARIES
    PAPI_INCLUDE_DIRS
)

mark_as_advanced(
    PAPI_PREFIX_DIRS
    PAPI_LIBRARIES
    PAPI_INCLUDE_DIRS
)

