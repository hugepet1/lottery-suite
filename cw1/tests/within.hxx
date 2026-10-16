#ifndef WITHIN_HXX_E2E6104F_84CD_4D99_BB55_7037F18BD585
#define WITHIN_HXX_E2E6104F_84CD_4D99_BB55_7037F18BD585

#include <snitch/snitch.hpp>

#include <cmath>

// Simple helper for floating point tests with Snitch
// CHECK_THAT( 1.f, WithinAbs(1.f) ); // OK
// CHECK_THAT( 1.1f, WithinAbs(1.f, .01f) ); // Error, more than 0.01f apart.
template< typename tFloat >
struct WithinAbs
{
	WithinAbs( tFloat aValue, tFloat aEps = 1e-6f )
		: value(aValue)
		, eps(aEps)
	{}

	constexpr
	bool match( tFloat aInput ) const
	{
		return std::abs(aInput-value) <= eps;
	}

	snitch::small_string<snitch::max_message_length> describe_match( tFloat aInput, snitch::matchers::match_status aStatus ) const
	{
		using namespace snitch;
		small_string<max_message_length> buf;
		append_or_truncate( buf, aInput, 
			matchers::match_status::matched == aStatus ? " within " : " not within ",
			eps,
			" of ",
			value
		);
		return buf;
	}

	tFloat value, eps;
};

#endif // WITHIN_HXX_E2E6104F_84CD_4D99_BB55_7037F18BD585
