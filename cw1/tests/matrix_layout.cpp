#include <snitch/snitch.hpp>

#include <type_traits>

#include <vmlib/small_matrix.hpp>

TEST_CASE( "SmallMat<float> layout", "[SmallMat][layout]" )
{
	// SmallMat<> should be standard_layout
	REQUIRE( std::is_standard_layout_v<SmallMat<2,2,float>> );
	REQUIRE( std::is_standard_layout_v<SmallMat<3,3,float>> );
	REQUIRE( std::is_standard_layout_v<SmallMat<4,4,float>> );
	REQUIRE( std::is_standard_layout_v<SmallMat<5,5,float>> );
	REQUIRE( std::is_standard_layout_v<SmallMat<4,3,float>> );

	// SmallMat<> should be trivially_copyable, as this ensures that we can
	// safely mempcy() it around.
	REQUIRE( std::is_trivially_copyable_v<SmallMat<2,2,float>> );
	REQUIRE( std::is_trivially_copyable_v<SmallMat<3,3,float>> );
	REQUIRE( std::is_trivially_copyable_v<SmallMat<4,4,float>> );
	REQUIRE( std::is_trivially_copyable_v<SmallMat<5,5,float>> );
	REQUIRE( std::is_trivially_copyable_v<SmallMat<4,3,float>> );

	// SmallMat> must not have any padding.
	REQUIRE( sizeof(SmallMat<2,2,float>) == 2*2*sizeof(float) );
	REQUIRE( sizeof(SmallMat<3,3,float>) == 3*3*sizeof(float) );
	REQUIRE( sizeof(SmallMat<4,4,float>) == 4*4*sizeof(float) );
	REQUIRE( sizeof(SmallMat<5,5,float>) == 5*5*sizeof(float) );
	REQUIRE( sizeof(SmallMat<4,3,float>) == 4*3*sizeof(float) );

	// Row-major layout
	SmallMat<2,2,float> m22{};
	for( std::size_t i = 0; i < 2; ++i )
	{
		if( i != 0 )
			REQUIRE( &m22[i-1,1] == &m22[i,0]-1 );
	
		for( std::size_t j = 1; j < 2; ++j )
			REQUIRE( &m22[i,j-1] == &m22[i,j]-1 );
	}

	SmallMat<3,3,float> m33{};
	for( std::size_t i = 0; i < 3; ++i )
	{
		if( i != 0 )
			REQUIRE( &m33[i-1,2] == &m33[i,0]-1 );
	
		for( std::size_t j = 1; j < 3; ++j )
			REQUIRE( &m33[i,j-1] == &m33[i,j]-1 );
	}

	SmallMat<4,4,float> m44{};
	for( std::size_t i = 0; i < 3; ++i )
	{
		if( i != 0 )
			REQUIRE( &m44[i-1,3] == &m44[i,0]-1 );
	
		for( std::size_t j = 1; j < 3; ++j )
			REQUIRE( &m44[i,j-1] == &m44[i,j]-1 );
	}

	// data() returns pointer to first element
	REQUIRE( m22.data() == &m22[0,0] );
	REQUIRE( m33.data() == &m33[0,0] );
	REQUIRE( m44.data() == &m44[0,0] );
}

