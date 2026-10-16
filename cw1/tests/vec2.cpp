#include <snitch/snitch.hpp>

#include <type_traits>

#include <vmlib/small_vector.hpp>

#include "within.hxx"

TEST_CASE( "SmallVec<2,float> accessors", "[SmallVec][Vec2]" )
{
	// For 2D vectors, we want to be able to use the .x and .y accessors
	SmallVec<2,float> f2{ 1.f, 2.f };
	REQUIRE( f2.x == 1.f );
	REQUIRE( f2.y == 2.f );

	// They should be identical to [1]
	REQUIRE( &f2.x == &f2[0] );
	REQUIRE( &f2.y == &f2[1] );
}

TEST_CASE( "SmallVec<2,int> accessors", "[SmallVec][Vec2]" )
{
	// For 2D vectors, we want to be able to use the .x and .y accessors
	SmallVec<2,int> i2{ 1, 2 };
	REQUIRE( i2.x == 1 );
	REQUIRE( i2.y == 2 );

	// They should be identical to [1]
	REQUIRE( &i2.x == &i2[0] );
	REQUIRE( &i2.y == &i2[1] );
}

TEST_CASE( "SmallVec<2,float> operator+", "[SmallVec][Vec2]" )
{
	// Just a few small tests, as an example. Note: we want to avoid just
	// testing some special cases (e.g., just adding zero).
	SmallVec<2,float> a{ 64.303f, -3.476f };
	SmallVec<2,float> b{ 28.227f, 23.097f };
	SmallVec<2,float> z{};

	auto u = a+z;
	REQUIRE_THAT( u.x, WithinAbs(64.303f) );
	REQUIRE_THAT( u.y, WithinAbs(-3.476f) );

	auto v = a+b;
	REQUIRE_THAT( v.x, WithinAbs(64.303f+28.227f) );
	REQUIRE_THAT( v.y, WithinAbs(-3.476f+23.097f) );

	//TODO...
}

TEST_CASE( "SmallVec<2,int> operator+", "[SmallVec][Vec2]" )
{
	// Integer additions are exact.
	SmallVec<2,int> a{ 64, -3 };
	SmallVec<2,int> b{ 28, 23 };
	SmallVec<2,int> z{};

	auto u = a+z;
	REQUIRE( u.x == 64 );
	REQUIRE( u.y == -3 );

	auto v = a+b;
	REQUIRE( v.x == (64+28) );
	REQUIRE( v.y == (-3+23) );

	//TODO...
}
