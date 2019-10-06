#ifndef SCRIMPPPORIG_HPP
#define SCRIMPPPORIG_HPP

#include <MatProfAlgo.hpp>

namespace matrix_profile {

    class ScrimpppOrig : public IMatProfAlgo {
		virtual void compute_matrix_profile(const Scrimppp_params& params);
		virtual void initialize(const Scrimppp_params& params);
	};

}

#endif
