#include <snitch/snitch.hpp>

#include <type_traits>

#include <vmlib/small_vector.hpp>

TEST_CASE( "SmallVec<float> layout", "[SmallVec][layout]" )
{
	// SmallVec<> should be standard_layout
	REQUIRE( std::is_standard_layout_v<SmallVec<2,float>> );
	REQUIRE( std::is_standard_layout_v<SmallVec<3,float>> );
	REQUIRE( std::is_standard_layout_v<SmallVec<4,float>> );
	REQUIRE( std::is_standard_layout_v<SmallVec<5,float>> );

	// SmallVec<> should be trivially_copyable, as this ensures that we can
	// safely mempcy() it around.
	REQUIRE( std::is_trivially_copyable_v<SmallVec<2,float>> );
	REQUIRE( std::is_trivially_copyable_v<SmallVec<3,float>> );
	REQUIRE( std::is_trivially_copyable_v<SmallVec<4,float>> );
	REQUIRE( std::is_trivially_copyable_v<SmallVec<5,float>> );

	// SmallVec> must not have any padding.
	REQUIRE( sizeof(SmallVec<2,float>) == 2*sizeof(float) );
	REQUIRE( sizeof(SmallVec<3,float>) == 3*sizeof(float) );
	REQUIRE( sizeof(SmallVec<4,float>) == 4*sizeof(float) );
	REQUIRE( sizeof(SmallVec<5,float>) == 5*sizeof(float) );

	// Make sure we can construct things normally
	// Note: this may break at compile time if things are improperly set up.
	SmallVec<2,float> f2{ 1.f, 2.f };
	REQUIRE( f2[0] == 1.f );
	REQUIRE( f2[1] == 2.f );

	SmallVec<3,float> f3{ 1.f, 2.f, 3.f };
	REQUIRE( f3[0] == 1.f );
	REQUIRE( f3[1] == 2.f );
	REQUIRE( f3[2] == 3.f );

	SmallVec<4,float> f4{ 1.f, 2.f, 3.f, 4.f };
	REQUIRE( f4[0] == 1.f );
	REQUIRE( f4[1] == 2.f );
	REQUIRE( f4[2] == 3.f );
	REQUIRE( f4[3] == 4.f );

	SmallVec<5,float> f5{ 1.f, 2.f, 3.f, 4.f, 5.f };
	REQUIRE( f5[0] == 1.f );
	REQUIRE( f5[1] == 2.f );
	REQUIRE( f5[2] == 3.f );
	REQUIRE( f5[3] == 4.f );
	REQUIRE( f5[4] == 5.f );

	// Linear layout in memory
	for( std::size_t i = 1; i < 2; ++i )
		REQUIRE( &f2[i-1] == &f2[i]-1 );
	for( std::size_t i = 1; i < 3; ++i )
		REQUIRE( &f3[i-1] == &f3[i]-1 );
	for( std::size_t i = 1; i < 4; ++i )
		REQUIRE( &f4[i-1] == &f4[i]-1 );
	for( std::size_t i = 1; i < 5; ++i )
		REQUIRE( &f5[i-1] == &f5[i]-1 );

	// data() returns pointer to first element
	REQUIRE( f2.data() == &f2[0] );
	REQUIRE( f3.data() == &f3[0] );
	REQUIRE( f4.data() == &f4[0] );
	REQUIRE( f5.data() == &f5[0] );
}

TEST_CASE( "SmallVec<int> layout", "[SmallVec][layout]" )
{
	// SmallVec<> should be standard_layout
	REQUIRE( std::is_standard_layout_v<SmallVec<2,int>> );
	REQUIRE( std::is_standard_layout_v<SmallVec<3,int>> );
	REQUIRE( std::is_standard_layout_v<SmallVec<4,int>> );
	REQUIRE( std::is_standard_layout_v<SmallVec<5,int>> );

	// SmallVec> must not have any padding.
	REQUIRE( sizeof(SmallVec<2,int>) == 2*sizeof(int) );
	REQUIRE( sizeof(SmallVec<3,int>) == 3*sizeof(int) );
	REQUIRE( sizeof(SmallVec<4,int>) == 4*sizeof(int) );
	REQUIRE( sizeof(SmallVec<5,int>) == 5*sizeof(int) );

	// Make sure we can construct things normally
	// Note: this may break at compile time if things are improperly set up.
	SmallVec<2,int> i2{ 1, 2 };
	REQUIRE( i2[0] == 1 );
	REQUIRE( i2[1] == 2 );

	SmallVec<3,int> i3{ 1, 2, 3 };
	REQUIRE( i3[0] == 1 );
	REQUIRE( i3[1] == 2 );
	REQUIRE( i3[2] == 3 );

	SmallVec<4,int> i4{ 1, 2, 3, 4 };
	REQUIRE( i4[0] == 1 );
	REQUIRE( i4[1] == 2 );
	REQUIRE( i4[2] == 3 );
	REQUIRE( i4[3] == 4 );

	SmallVec<5,int> i5{ 1, 2, 3, 4, 5 };
	REQUIRE( i5[0] == 1 );
	REQUIRE( i5[1] == 2 );
	REQUIRE( i5[2] == 3 );
	REQUIRE( i5[3] == 4 );
	REQUIRE( i5[4] == 5 );
}
