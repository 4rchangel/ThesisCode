#include <papiwrapper.hpp>


#include <boost/assert.hpp>
#include <sstream>
#include <chrono>
#include <logging.hpp>

using namespace matrix_profile;

#ifdef LIKWID_PERFMON
#include <likwid.h>
#include <algorithm>

std::string replace_spaces(const std::string& input) {
	std::string output(input);
	std::replace(output.begin(), output.end(), ' ', '_');
	return output;
}
LikwidCounters::LikwidCounters(std::string descriptor)
    : _marker_name( replace_spaces(descriptor) ), //likwid does not support marker tages with spaces...
      IPerfCounter(descriptor)
{
	LIKWID_MARKER_REGISTER( _marker_name.c_str()); // TODO: might be better at different locations
	// invocation is optional, but will degade performance at first "start" invocation
	// might be necessary to be dropped...
}
void LikwidCounters::start_counters() {
	LIKWID_MARKER_START(_marker_name.c_str());
}
void LikwidCounters::stop_and_accu_counters() {
	LIKWID_MARKER_STOP(_marker_name.c_str());
}
void LikwidCounters::stop_and_read_counters() {
	LIKWID_MARKER_STOP(_marker_name.c_str());
}
void LikwidCounters::log_perf(){
	PERF_LOG("skip explicit likwid logging.");
	//skip
}
#endif //LIKWID_PERFMON

#ifdef PAPI_PERFMON
#include "papi.h"

inline void assert_papi_ok(int papi_ret_val, const std::string& descr) {
	if (papi_ret_val != PAPI_OK) {
		std::stringstream ss;
		ss << "PAPI Error " << descr << " errno: " << papi_ret_val << ": " << std::string(PAPI_strerror(papi_ret_val));
		throw std::runtime_error(ss.str());
	}
}

template<int N> PapiWrapper<N>::PapiWrapper(const EvtArray& evts, const std::string& descriptor) :
    _events(evts), _delta_t(0), _accu_count(0),
    IPerfCounter(descriptor)
{
	_ctrs.fill(0);
	int num_hwctrs = PAPI_num_counters();
	if (num_hwctrs <= 0) {
		if (PAPI_is_initialized()) {
			throw std::runtime_error("PAPI counters seem to be unavailable");
		}
		else
		{
			throw std::runtime_error("PAPI is not initialized");
		}
	}
	EXEC_TRACE( "Constructing IPapiWrapper, available hardware counters: " << num_hwctrs);
	BOOST_ASSERT_MSG ( num_hwctrs >= N, "too few papi counters available to track performance");
}

template<int N> void PapiWrapper<N>::start_counters() {
	int ret = PAPI_start_counters(_events.data(), N);
	assert_papi_ok(ret, "starting counters");
	_start_time = std::chrono::steady_clock::now();
}

template <int N> void PapiWrapper<N>::stop_and_read_counters() {
	auto tstop = std::chrono::steady_clock::now();
	int ret = PAPI_stop_counters(_ctrs.data(), N);
	assert_papi_ok(ret, "stopping counters");
	_delta_t = (tstop-_start_time);
	_accu_count = 1;
}

template <int N> void PapiWrapper<N>::stop_and_accu_counters() {
	//take time
	auto tstop = std::chrono::steady_clock::now();

	// read and accumulate counter values
	int ret = PAPI_accum_counters(_ctrs.data(), N);
	assert_papi_ok(ret, "accumulating counters");

	// stop the counters, as the accum_counters call doesn't do that
	std::array<long long, N> tmp; //need to specify a specific storage location for counter results
	ret = PAPI_stop_counters(tmp.data(), N);
	assert_papi_ok(ret, "stopping counters");

	// accumulate up the time
	_delta_t += (tstop-_start_time);
	_accu_count += 1;
}

CacheCounters::CacheCounters(const std::string& descriptor) :
    PapiWrapper(EvtArray({PAPI_L2_TCA, PAPI_L2_TCM, PAPI_L3_TCA, PAPI_L3_TCM}), descriptor) {

}
void CacheCounters::log_perf() {
	std::stringstream ss;
	ss << "Cache stats from " << _descr << ": ";
	ss << "  L2 accesses: " << _ctrs[0];
	ss << "  L2 misses: " << _ctrs[1];
	ss << "  L2 miss rate: " << _ctrs[1]/static_cast<double>(_ctrs[0]);
	ss << "  L3 accesses: " << _ctrs[2];
	ss << "  L3 misses: " << _ctrs[3];
	ss << "  L3 miss rate: " << _ctrs[3]/static_cast<double>(_ctrs[2]);
//	ss << "  L1 data misses: " << _ctrs[4];
	ss << "  time total/ms:  " << std::chrono::duration_cast<std::chrono::milliseconds>(_delta_t).count();
	ss << "  time average /us: " << (std::chrono::duration_cast<std::chrono::microseconds>(_delta_t).count()/static_cast<double>(_accu_count));
	PERF_LOG( ss.str() );
}

OperationCounters::OperationCounters(const std::string& descriptor) :
    PapiWrapper(EvtArray({PAPI_FP_OPS, PAPI_TOT_CYC, PAPI_TOT_INS, PAPI_RES_STL}), descriptor) {
}

void OperationCounters::log_perf() {
	std::stringstream ss;
	ss << "Operation stats from " << _descr << ": ";
	ss << "  Num FLOP: " << _ctrs[0];
	ss << "  Num cycles " << _ctrs[1];
	ss << "  Num instructions " << _ctrs[2];
	ss << "  Num cycles stalled " << _ctrs[3];
	ss << "	 GFLOP/S : " << (_ctrs[0]/(std::chrono::duration_cast<std::chrono::microseconds>(_delta_t).count()+1)) / 1000.0;
	ss << "  time total/ms:  " << std::chrono::duration_cast<std::chrono::milliseconds>(_delta_t).count();
	ss << "  time average /us: " << (std::chrono::duration_cast<std::chrono::microseconds>(_delta_t).count()/static_cast<double>(_accu_count));
	PERF_LOG( ss.str() );
}

BranchCounters::BranchCounters(const std::string& descriptor) :
    PapiWrapper({PAPI_BR_CN, PAPI_BR_TKN, PAPI_BR_MSP}, descriptor) {
}

void BranchCounters::log_perf() {
	std::stringstream ss;
	ss << "Branching stats from " << _descr << ": ";
	ss << "  Num cond. branches: " << _ctrs[0];
	ss << "  mispredicted: " << _ctrs[2];
	ss << "  misprediction rate: " << _ctrs[2]/static_cast<double>(_ctrs[0]);
	PERF_LOG( ss.str() );
}
#endif

