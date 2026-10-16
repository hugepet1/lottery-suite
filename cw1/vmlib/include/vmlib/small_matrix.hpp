#ifndef SMALL_MATRIX_HPP_7266454B_05D7_441E_8775_07E79DDBBE03
#define SMALL_MATRIX_HPP_7266454B_05D7_441E_8775_07E79DDBBE03


#include "small_vector.hpp"

/** SmallMat: NxM matrix
 *
 * See small_vector.hpp for discussion. Similar SmallVec, SmallMat is
 * intentionally kept simple and somewhat bare bones. It is also 
 * standard_layout and trivially_copyable, ensuring that we can easily
 * pass it to relevant C-based APIs such as OpenGL (which mostly just
 * expect a bunch of consecutive floats in memory).
 *
 * The matrix is stored in row-major order. Take care when passing it to 
 * OpenGL!
 *
 * The overloaded operator [] allows access to individual elements. Example:
 *    SmallMat<4,4,float> m = ...;
 *    float m12 = m[1,2];
 *    m[0,3] = 3.f;
 *
 * (Multi-dimensionsal subscripts in operator[] is a C++23 feature!)
 *
 * A 4x4 matrix would be arranged as:
 *
 *   ⎛ 0,0  0,1  0,2  0,3 ⎞
 *   ⎜ 1,0  1,1  1,2  1,3 ⎟
 *   ⎜ 2,0  2,1  2,2  2,3 ⎟
 *   ⎝ 3,0  3,1  3,2  3,3 ⎠
 */
template< std::size_t tM, std::size_t tN, typename tType = float >
struct SmallMat
{
	tType v[tM*tN];

	constexpr
	tType& operator[] (std::size_t aI, std::size_t aJ) noexcept
	{
		assert( aI < tM && aJ < tN );
		return v[aI*tN + aJ];
	}

	constexpr
	tType operator[] (std::size_t aI, std::size_t aJ) const noexcept
	{
		assert( aI < tM && aJ < tN );
		return v[aI*tN + aJ];
	};

	// Return pointer to the data
	// Use this e.g., for glUniformMatrix4fv( location, 1, matrix.data() );
	tType const* data() const noexcept
	{
		return v;
	}
};


// Operators
template< std::size_t tM, std::size_t tN, std::size_t tP, typename tType > constexpr
SmallMat<tM,tP,tType> operator*( SmallMat<tM,tN,tType> const& aLhs, SmallMat<tN,tP,tType> const& aRhs ) noexcept
{
	return {}; //TODO: implement me
}

template< std::size_t tM, std::size_t tN, typename tType > constexpr
SmallVec<tM,tType> operator*( SmallMat<tM,tN,tType> const& aLeft, SmallVec<tN,tType> const& aRight ) noexcept
{
	return {}; //TODO: implement me
}

// Functions
template< std::size_t tM, std::size_t tN, typename tType >
constexpr
SmallMat<tN,tM,tType> transpose( SmallMat<tM,tN,tType> const& aMat ) noexcept
{
	return {}; //TODO: implement me
}

// Pseudo-constructor for identity matrices
// Usage example: auto identity33 = make_identity<Mat33f>();
// Only valid for square matrices.
template< class tMatrix > constexpr
tMatrix make_identity() noexcept
{
	// This uses an immediate invoked lambda with templates to get the matrix
	// size. It also enforces that we call this only on square matrices.
	tMatrix ret{};
	[] <std::size_t tN, typename tType> (SmallMat<tN,tN,tType>& aMat)
	{
		for( std::size_t i = 0; i < tN; ++i )
			aMat[i,i] = 1.f;
	}( ret );
	return ret;
}

// For Task 2.1 only:
template< std::size_t tN, typename tType, std::size_t tM > inline 
void gauss( SmallMat<tN,tN,tType> aMat, SmallMat<tN,tM,tType>& aResult ) noexcept
{
	// TODO-Task 2.1 only: implement me
}

template< std::size_t tN, typename tType > // Must be square!
SmallMat<tN,tN,tType> invert( SmallMat<tN,tN,tType> const& ) noexcept;

template<>
SmallMat<4,4,float> invert( SmallMat<4,4,float> const& ) noexcept;


#endif // SMALL_MATRIX_HPP_7266454B_05D7_441E_8775_07E79DDBBE03
