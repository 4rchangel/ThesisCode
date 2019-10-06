#include <MatProfAlgo.hpp>
#include <logging.hpp>

#include <map>
#include <iostream>
#include <sstream>
#include <boost/assert.hpp>

using namespace matrix_profile;

MatProfImplFactory& MatProfImplFactory::getInstance()
{
	static std::unique_ptr<MatProfImplFactory> s_factory_inst;
	if (s_factory_inst == nullptr) {
		s_factory_inst = std::unique_ptr<MatProfImplFactory>(new MatProfImplFactory());
	}
	return *s_factory_inst;
}

bool MatProfImplFactory::register_impl(const std::string& name, Creator creator_fn) {
	BOOST_ASSERT_MSG(_registrations.find(name) == _registrations.end(), "Adding algorithm to the factory, which is already existing!");
	EXEC_TRACE( "adding algorithm " << name << " to the factory" );
	_registrations.insert(std::make_pair(name, creator_fn));
	return true;
}

std::string MatProfImplFactory::get_available_implementations() const {
	std::stringstream ss;
	for (auto iter = _registrations.begin(); iter != _registrations.end(); ++iter) {
		ss << iter->first << " ";
	}
	return ss.str();
}

std::unique_ptr<IMatProfAlgo> MatProfImplFactory::create(const std::string& algo_name) const {
	return _registrations.at(algo_name)();
}
