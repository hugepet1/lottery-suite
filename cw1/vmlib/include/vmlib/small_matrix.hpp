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
	SmallMat<tM, tP, tType> result{};

	for (std::size_t i = 0; i < tM; ++i)
	{
    	for (std::size_t j = 0; j < tP; ++j)
    	{
        	for (std::size_t k = 0; k < tN; ++k)
        	{
            result[i,j] += aLhs[i,k] * aRhs[k,j];
		    }
    	}
	}

	return result;
}

template< std::size_t tM, std::size_t tN, typename tType > constexpr
SmallVec<tM,tType> operator*( SmallMat<tM,tN,tType> const& aLeft, SmallVec<tN,tType> const& aRight ) noexcept
{
	SmallVec<tM, tType> result{};
	for (std::size_t i = 0; i < tM; ++i)
	{
    	for (std::size_t j = 0; j < tN; ++j)
    	{
        result[i] += aLeft[i,j] * aRight[j];
		}
	}
	return result;
}

// Functions
template< std::size_t tM, std::size_t tN, typename tType >
constexpr
SmallMat<tN,tM,tType> transpose( SmallMat<tM,tN,tType> const& aMat ) noexcept
{
	SmallMat<tN, tM, tType> result{};

	for (std::size_t i = 0; i < tM; ++i)
	{
    	for (std::size_t j = 0; j < tN; ++j)
    	{
        	result[j,i] = aMat[i,j];
    	}
	}

	return result;
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
template< std::size_t tN, typename tType, std::size_t tM >inline
void gauss(SmallMat<tN,tN,tType> aMat,SmallMat<tN,tM,tType>& aResult) noexcept
{
    for (std::size_t col = 0; col < tN; ++col)
    {
        std::size_t tRow = col;

        for (std::size_t row = col + 1; row < tN; ++row)
        {
            if (std::abs(aMat[row,col])
                > std::abs(aMat[tRow,col]))
            {
                tRow = row;
            }
        }

        if (tRow != col)
        {
            for (std::size_t j = 0; j < tN; ++j)
            {
                tType temp = aMat[col,j];
                aMat[col,j] = aMat[tRow,j];
                aMat[tRow,j] = temp;
            }

            for (std::size_t j = 0; j < tM; ++j)
            {
                tType temp = aResult[col,j];
                aResult[col,j] = aResult[tRow,j];
                aResult[tRow,j] = temp;
            }
        }

        for (std::size_t row = col + 1; row < tN; ++row)
        {
            tType factor = aMat[row,col] / aMat[col,col];

            for (std::size_t j = col; j < tN; ++j)
            {
                aMat[row,j] -= factor * aMat[col,j];
            }

            for (std::size_t j = 0; j < tM; ++j)
            {
                aResult[row,j] -= factor * aResult[col,j];
            }
        }
    }

    for (std::size_t step = 0; step < tN; ++step)
    {
        std::size_t row = tN - 1 - step;
        tType diagonal = aMat[row,row];

        for (std::size_t j = row; j < tN; ++j)
        {
            aMat[row,j] /= diagonal;
        }

        for (std::size_t j = 0; j < tM; ++j)
        {
            aResult[row,j] /= diagonal;
        }

        for (std::size_t upper = 0; upper < row; ++upper)
        {
            tType factor = aMat[upper,row];

            for (std::size_t j = row; j < tN; ++j)
            {
                aMat[upper,j] -= factor * aMat[row,j];
            }

            for (std::size_t j = 0; j < tM; ++j)
            {
                aResult[upper,j] -= factor * aResult[row,j];
            }
        }
    }
}

template< std::size_t tN, typename tType > // Must be square!
SmallMat<tN,tN,tType> invert( SmallMat<tN,tN,tType> const& aMat) noexcept
{
    auto result = make_identity<SmallMat<tN,tN,tType>>();

    gauss(aMat, result);

    return result;
}
template<>
SmallMat<4,4,float> invert( SmallMat<4,4,float> const& ) noexcept;


#endif // SMALL_MATRIX_HPP_7266454B_05D7_441E_8775_07E79DDBBE03
