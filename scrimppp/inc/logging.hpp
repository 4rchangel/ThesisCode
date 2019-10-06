#ifndef LOGGING_H
#define LOGGING_H

#include <ostream>

namespace matrix_profile {
namespace logging {
    void init();

	extern std::ostream& exec_logger;
	extern std::ostream& perf_logger;
} // end of namespace logging
} // end of namespace matrix_profile

// different levels for logging information about the program execution
#define MP_LOG_LEVEL_ERROR 0
#define MP_LOG_LEVEL_INFO  1
#define MP_LOG_LEVEL_DEBUG 2
#define MP_LOG_LEVEL_TRACE 3

// default level of execution logging
#ifndef MP_LOG_LEVEL
    #define MP_LOG_LEVEL MP_LOG_LEVEL_INFO
#endif

// default level of performance logging
#ifndef MP_PERF_LOG_LEVEL
    #define MP_PERF_LOG_LEVEL MP_LOG_LEVEL_INFO
#endif


// macros for loggin performance results
#if MP_PERF_LOG_LEVEL >= MP_LOG_LEVEL_INFO
    #define PERF_LOG( stream_msg ) (matrix_profile::logging::perf_logger << stream_msg << std::endl);
#endif

#if MP_PERF_LOG_LEVEL >= MP_LOG_LEVEL_TRACE
    #define PERF_TRACE( stream_msg ) (matrix_profile::logging::perf_logger << stream_msg << std::endl);
#else
    #define PERF_TRACE( stream_msg )
#endif


//macros for general logging
#if MP_LOG_LEVEL >= MP_LOG_LEVEL_ERROR
    #define EXEC_ERROR( stream_msg ) (matrix_profile::logging::exec_logger << stream_msg << std::endl);
#else
    #define EXEC_ERROR( ... )
#endif

#if MP_LOG_LEVEL >= MP_LOG_LEVEL_INFO
    #define EXEC_INFO( stream_msg ) (matrix_profile::logging::exec_logger << stream_msg << std::endl);
#else
    #define EXEC_INFO( ... )
#endif

#if MP_LOG_LEVEL >= MP_LOG_LEVEL_DEBUG
    #define EXEC_DEBUG( stream_msg ) (matrix_profile::logging::exec_logger << stream_msg << std::endl);
#else
    #define EXEC_DEBUG( stream_msg )
#endif

#if MP_LOG_LEVEL >= MP_LOG_LEVEL_TRACE
    #define EXEC_TRACE( stream_msg ) (matrix_profile::logging::exec_logger << stream_msg << std::endl);
#else
    #define EXEC_TRACE( ... )
#endif


#endif // LOGGING_H
