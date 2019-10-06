#ifndef MATPROFALGO_H
#define MATPROFALGO_H

#include <string>
#include <map>
#include <memory>
#include <functional>

#include <ScrimpppParams.hpp>

// MatProfImplementation with self-registration in MatProfImplFactory using the Curiously recurring template pattern and architectural inspiration from by https://www.bfilipek.com/2018/02/factory-selfregister.html

namespace matrix_profile {

class IMatProfAlgo
{
    public:
	    virtual void initialize(const Scrimppp_params& params) = 0;
	    virtual void compute_matrix_profile(const Scrimppp_params& params) = 0;
	    virtual void log_info() {};
};

class MatProfImplFactory {
    public:
	using Creator = std::unique_ptr<IMatProfAlgo>(*)();

    public:
	    static MatProfImplFactory& getInstance();
		bool register_impl(const std::string& name, Creator creator_fn);
		std::string get_available_implementations() const;
		std::unique_ptr<IMatProfAlgo> create(const std::string& algo_name) const;
    private:
		MatProfImplFactory(){}

    private:
		std::map<std::string, Creator> _registrations;
};

template<class impl> class FactoryRegistration {
    public:
	    FactoryRegistration(const std::string& name) {MatProfImplFactory::getInstance().register_impl(name, [](){return std::unique_ptr<IMatProfAlgo>(new impl());});}
};

} // namespace matrix_profile
#endif // MATPROFIMPLEMENTATION_H
