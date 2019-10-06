# Modified version from the FindPAPI file of Lawrence Livermore National Laboratory
# the code is licensed under the GNU Lesser GPL terms, as provided under ../../licenses/license_LGPL_FindPAPI.txt
# original source repository: https://github.com/LLNL/perf-dump/blob/master/cmake/FindPAPI.cmake
# source repository: https://github.com/LLNL/perf-dump/blob/master/cmake/FindPAPI.cmake
# change note: adapted names, parameters and dependency paths in order to load LIKWID on 04/2019
# change note: added header on 06.10.2019

# Try to find LIKWID headers and libraries.
#
# Usage of this module as follows:
#
#     find_package(LIKWID)
#
# Variables used by this module, they can change the default behaviour and need
# to be set before calling find_package:
#
#  LIKWID_ROOT         Set this variable to the root installation of
#                      likwid if the module has problems finding the
#                      proper installation path.
#
# Variables defined by this module:
#
#  LIKWID_FOUND              System has LICKWID libraries and headers
#  LIKWID_LIBRARIES          The LIKWID libraries
#  LIKWID_INCLUDE_DIRS       The location of LIKWID headers

find_path(LIKWID_ROOT
    NAMES include/likwid.h
)

find_library(LIKWID_LIBRARIES
    # Pick the static library first for easier run-time linking.
    NAMES likwid
    HINTS ${LIKWID_ROOT}/lib
)

find_path(LIKWID_INCLUDE_DIRS
    NAMES likwid.h
    HINTS ${LIKWID_ROOT}/include ${HILTIDEPS}/include
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(LIKWID DEFAULT_MSG
    LIKWID_LIBRARIES
    LIKWID_INCLUDE_DIRS
)

mark_as_advanced(
    LIKWID_PREFIX_DIRS
    LIKWID_LIBRARIES
    LIKWID_INCLUDE_DIRS
)

