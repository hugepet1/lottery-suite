#ifndef SMALL_VECTOR_HPP_214E3703_A050_411A_834B_4E55AEEC18B8
#define SMALL_VECTOR_HPP_214E3703_A050_411A_834B_4E55AEEC18B8

#include <cassert>
#include <cmath>
#include <cstdlib>

/** SmallVec<N,type>: N-D vector of type elements
 *
 * This is purposefully kept simple. We ultimately want to make sure that the
 * resulting type has a standard layout and is trivially copyable. This gives 
 * a predictable memory layout, and makes it much easier to pass data to e.g.
 * OpenGL.
 *
 * You can construct a SmallVec<> with braced initialization. For example, for
 * a 2D float vector:
 *   SmallVec<2,float> a{ 1.f, 2.f, };
 *   SmallVec<2,float> b = { 1.f, 2.f };
 * Temporary vectors can be constructed similarly:
 *   function_taking_vec2f_argument( SmallVec<2,float>{ 1.f, 2.f } );
 *   function_taking_vec2f_argument( { 1.f, 2.f } );
 *
 * Note that the header stdtypes.hpp provides a few convenient typedefs.
 */
template< std::size_t tSize, typename tType = float >
struct SmallVec
{
	tType v[tSize];

	constexpr
	tType& operator[] (std::size_t aIdx) noexcept
	{
		assert( aIdx < tSize );
		return v[aIdx];
	}

	constexpr
	tType operator[] (std::size_t aIdx) const noexcept
	{
		assert( aIdx < tSize );
		return v[aIdx];
	};

	// Return pointer to the data
	tType const* data() const noexcept
	{
		return v;
	}
};

/* Partial specialization for 2D vectors. This lets us access the components as
 * .x and .y, instead of requiring the use of []. That makes our life as
 * programmers quite a bit easier.
 *
 * Example:
 *   SmallVec<2,float> f2;
 *   f2.x = 1.f;
 *   f2.y = 2.f;
 */
template< typename tType >
struct SmallVec<2,tType>
{
	tType x, y;

	constexpr
	tType& operator[] (std::size_t aIdx) noexcept
	{
		assert( aIdx < 2 );
		return (&x)[aIdx];
	}

	constexpr
	tType operator[] (std::size_t aIdx) const noexcept
	{
		assert( aIdx < 2 );
		return (&x)[aIdx];
	};

	// Return pointer to the data
	tType const* data() const noexcept
	{
		return &x;
	}
};
template< typename tType >    //3D
struct SmallVec<3, tType>
{
    tType x, y, z;
	
};

// Helpers
template< std::size_t tSize, typename tType > constexpr
std::size_t dimension( SmallVec<tSize,tType> ) noexcept
{
	return tSize;
}

// Operators
template< std::size_t tSize, typename tType > constexpr
SmallVec<tSize, tType> operator+ (SmallVec<tSize,tType> const& aVec ) noexcept
{
	return aVec;
}
template< std::size_t tSize, typename tType > constexpr
SmallVec<tSize, tType> operator- (SmallVec<tSize,tType> const& aVec ) noexcept
{
	SmallVec<tSize, tType> ret{};
	for( std::size_t i = 0; i < tSize; ++i )
		ret[i] = -aVec[i];
	return ret;
}

template< std::size_t tSize, typename tType > constexpr
SmallVec<tSize, tType> operator+ (SmallVec<tSize,tType> const& aLhs, SmallVec<tSize,tType> const& aRhs ) noexcept
{
	return {}; //TODO: implement me
}
template< std::size_t tSize, typename tType > constexpr
SmallVec<tSize, tType> operator- (SmallVec<tSize,tType> const& aLhs, SmallVec<tSize,tType> const& aRhs ) noexcept
{
	return {}; //TODO: implement me
}

template< std::size_t tSize, typename tType > constexpr
SmallVec<tSize, tType> operator* (tType aScalar, SmallVec<tSize,tType> const& aRhs ) noexcept
{
	return {}; //TODO: implement me
}
template< std::size_t tSize, typename tType > constexpr
SmallVec<tSize, tType> operator* (SmallVec<tSize,tType> const& aLhs, tType aScalar ) noexcept
{
	return {}; //TODO: implement me
}
template< std::size_t tSize, typename tType > constexpr
SmallVec<tSize, tType> operator/ (SmallVec<tSize,tType> const& aLhs, tType aScalar ) noexcept
{
	return {}; //TODO: implement me
}

template< std::size_t tSize, typename tType > constexpr
SmallVec<tSize, tType>& operator+= (SmallVec<tSize,tType>& aLhs, SmallVec<tSize,tType> const& aRhs ) noexcept
{
	//TODO: implement me
	return aLhs; // Note: operator+= must return a reference to the LHS
}
template< std::size_t tSize, typename tType > constexpr
SmallVec<tSize, tType>& operator-= (SmallVec<tSize,tType>& aLhs, SmallVec<tSize,tType> const& aRhs ) noexcept
{
	//TODO: implement me
	return aLhs; // Note: operator-= must return a reference to the LHS
}
template< std::size_t tSize, typename tType > constexpr
SmallVec<tSize, tType>& operator*= (SmallVec<tSize,tType>& aLhs, tType aScalar ) noexcept
{
	//TODO: implement me
	return aLhs; // Note: operator*= must return a reference to the LHS
}
template< std::size_t tSize, typename tType > constexpr
SmallVec<tSize, tType>& operator/= (SmallVec<tSize,tType>& aLhs, tType aScalar ) noexcept
{
	//TODO: implement me
	return aLhs; // Note: operator/= must return a reference to the LHS
}

// Common operations
template< std::size_t tSize, typename tType > constexpr
tType dot( SmallVec<tSize,tType> const& aLhs, SmallVec<tSize,tType> const& aRhs ) noexcept
{
	return tType(0); //TODO: implement me
}

template< std::size_t tSize, typename tType > inline
tType length( SmallVec<tSize,tType> const& aVec ) noexcept
{
	return tType(0); //TODO: implement me
}

template< std::size_t tSize, typename tType > constexpr
SmallVec<tSize,tType> normalize( SmallVec<tSize,tType> const& aVec ) noexcept
{
	return {}; //TODO: implement me
}

// Only for Vec<3,type>
template< typename tType > constexpr
SmallVec<3,tType> cross( SmallVec<3,tType> const& aLhs, SmallVec<3,tType> const& aRhs ) noexcept
{
	return {}; //TODO: implement me
}

#endif // SMALL_VECTOR_HPP_214E3703_A050_411A_834B_4E55AEEC18B8
