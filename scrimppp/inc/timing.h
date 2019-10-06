#ifndef TIMING_H
#define TIMING_H

#include <chrono>
#include <iostream>


namespace matrix_profile {
#ifdef USE_CHRONO
    using clock = std::chrono::steady_clock;
    using Timepoint = clock::time_point;
    using Timespan = clock::duration;
#else
    struct Timepoint {double t;};
	struct Timespan {double dt;};
#endif

	Timepoint get_cur_time();
	Timespan  delta_t(Timepoint first, Timepoint last);
	double	  get_seconds(Timepoint t);
	double	  get_seconds(Timespan t);
}

std::ostream& operator<< (std::ostream &out, const matrix_profile::Timepoint& t);
std::ostream& operator<< (std::ostream &out, const matrix_profile::Timespan & dt);

#ifndef USE_CHRONO
    matrix_profile::Timespan operator- (const matrix_profile::Timepoint& t1, const matrix_profile::Timepoint& t2);
	matrix_profile::Timespan operator+ (const matrix_profile::Timespan& t1, const matrix_profile::Timespan& t2);
#endif

#endif // TIMING_H
