#ifndef PAPIWRAPPER_HPP
#define PAPIWRAPPER_HPP

#include <array>
#include <chrono>

namespace matrix_profile {

class IPerfCounter {
    protected:
	    const std::string _descr;
	 //used to be a interface with pure virtual functions. Wanted to avoid and employed some type aliasing with help of the preprocesspor.
	// a kind of nicer alternative might have been a template "ScopedPerfAccumulator" TODO: consider changing...
    public:
		/*virtual */ void start_counters(){};//=0;
		/*virtual */ void stop_and_read_counters(){};//=0;
		/*virtual */ void stop_and_accu_counters(){};//=0;
		IPerfCounter(const std::string& descriptor) : _descr(descriptor) {};
};

class NoCounters{
    public:
	    NoCounters(const std::string& descriptor){};
		void start_counters(){}
		void stop_and_read_counters(){}
		void stop_and_accu_counters(){}
		void log_perf(){}
};

#ifdef PAPI_PERFMON
template<int N> class PapiWrapper : public IPerfCounter
{
    protected:
	    using EvtArray = std::array<int, N>;
    protected:
	    static const int NUM_CTRS = N;
		EvtArray _events;
		long int _accu_count; // number of start-stop_accu invoactions
		std::array<long long, N> _ctrs;
		std::chrono::steady_clock::time_point _start_time;
		std::chrono::high_resolution_clock::duration _delta_t;
    public:
		PapiWrapper(const EvtArray & events, const std::string& descriptor);
		void log_perf();
		virtual ~PapiWrapper() {}

		virtual void start_counters();
		virtual void stop_and_read_counters();
		virtual void stop_and_accu_counters();
};

class CacheCounters : public PapiWrapper<4>
{
    public:
	    CacheCounters(const std::string& descriptor);
		void log_perf();
};

class OperationCounters : public PapiWrapper<4>
{
    public:
	    OperationCounters(const std::string& descriptor);
		void log_perf();
};

class BranchCounters : public PapiWrapper<3>
{
    public:
	    BranchCounters(const std::string& descriptor);
		void log_perf();
};
#endif // ifdef PAPI_PERFMON

#ifndef LIKWID_PERFMON
    #define LIKWID_MARKER_INIT
    #define LIKWID_MARKER_CLOSE
#else
    class LikwidCounters : public IPerfCounter {
	    protected:
		    const std::string _marker_name;

	    public:
			LikwidCounters(std::string descriptor);
			virtual void log_perf();

	    public:
			void start_counters();
			void stop_and_read_counters();
			void stop_and_accu_counters();
	};
#endif


#ifdef LIKWID_PERFMON
	using PerfCounters = LikwidCounters;
#elif PAPI_CACHE_STATS
	using PerfCounters = CacheCounters;
#elif PAPI_INSTR_STATS
	using PerfCounters = OperationCounters;
#elif PAPI_BRANCH_STATS
	using PerfCounters =BranchCounters;
#else
	using PerfCounters = NoCounters;
#endif


class ScopedPerfAccumulator {
    private:
	    PerfCounters& _monitor; // previously used IPapiWrapper
    public:
		ScopedPerfAccumulator(PerfCounters& papi_wrapper) : _monitor(papi_wrapper){_monitor.start_counters();}
		~ScopedPerfAccumulator() {_monitor.stop_and_accu_counters();}
};

} // namespace matrix_profile


#endif // PAPIWRAPPER_HPP
