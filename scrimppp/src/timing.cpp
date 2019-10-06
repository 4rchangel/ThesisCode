#include <timing.h>

#include <iomanip>
#include <math.h>

using namespace matrix_profile;

//const long double timerprecision = static_cast<long double>(std::chrono::steady_clock::period.den) / static_cast<long double>(std::chrono::steady_clock::period.den);
const int printing_digits = std::numeric_limits<double>::digits10 +2;

#ifdef USE_CHRONO
    using unit_seconds = std::chrono::duration<double>;

    Timepoint matrix_profile::get_cur_time(){
		return clock::now();
	}

	Timespan  matrix_profile::delta_t(Timepoint first, Timepoint last){
		Timespan dt = last-first;
		return dt.count()>0 ? dt: -dt;
	}

	double matrix_profile::get_seconds(Timepoint t) {
		return std::chrono::duration_cast<unit_seconds>(t.time_since_epoch()).count();
	}

	double matrix_profile::get_seconds(Timespan t) {
		return std::chrono::duration_cast<unit_seconds>(t).count();
	}

#else // use MPI_Wtime by default
#include <mpi.h>

Timepoint matrix_profile::get_cur_time(){
	Timepoint tp;
	tp.t = MPI_Wtime();
	return tp;
}

Timespan  matrix_profile::delta_t(Timepoint first, Timepoint last){
	Timespan span;
	span.dt = last.t-first.t;
	if (span.dt < 0) { // timespan is positive...
		span.dt = -span.dt;
	}
	return span;
}

double matrix_profile::get_seconds(Timepoint t) {
	return t.t;
}

double matrix_profile::get_seconds(Timespan t) {
	return t.dt;
}

matrix_profile::Timespan operator- (const matrix_profile::Timepoint& t1, const matrix_profile::Timepoint& t2){
	Timespan span {t1.t - t2.t};
	return span;
}

matrix_profile::Timespan operator+ (const matrix_profile::Timespan& t1, const matrix_profile::Timespan& t2){
	Timespan sum {t1.dt + t2.dt};
	return sum;
}
#endif

std::ostream& operator<< (std::ostream &out, const matrix_profile::Timepoint& t) {
	return out << std::setprecision(printing_digits) << get_seconds(t) << " seconds" ;
}

std::ostream& operator<< (std::ostream &out, const matrix_profile::Timespan & dt) {
	return out << std::setprecision(printing_digits) << get_seconds(dt) << " seconds";
}
