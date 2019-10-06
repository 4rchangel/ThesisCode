# Credit for this file goes to Jed Brown
# licensed under the BSD 2-Clause "Simplified" License, copy provided in ../../licenses/license_FindFFTW.txt
# original code on https://github.com/jedbrown/cmake-modules/blob/master/FindFFTW.cmake

# - Find FFTW
# Find the native FFTW includes and library
#
#  FFTW_ROOT	      specify fftw root installation directory, if not found automatically
#
#  FFTW_INCLUDES    - where to find fftw3.h
#  FFTW_LIBRARIES   - List of libraries when using FFTW.
#  FFTW_FOUND       - True if FFTW found.

if (FFTW_INCLUDES)
  # Already in cache, be silent
  set (FFTW_FIND_QUIETLY TRUE)
endif (FFTW_INCLUDES)

find_path (FFTW_ROOT
        NAMES include/fftw3.h
)

find_path (FFTW_INCLUDES
        NAMES fftw3.h
        HINTS ${FFTW_ROOT}/include
)

find_library (FFTW_LIBRARIES
        NAMES fftw3
        HINTS ${FFTW_ROOT}/lib 
)

# handle the QUIETLY and REQUIRED arguments and set FFTW_FOUND to TRUE if
# all listed variables are TRUE
include (FindPackageHandleStandardArgs)
find_package_handle_standard_args (FFTW DEFAULT_MSG FFTW_LIBRARIES FFTW_INCLUDES)

mark_as_advanced (FFTW_LIBRARIES FFTW_INCLUDES)
