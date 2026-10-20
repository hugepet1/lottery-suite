#ifndef TRANSFORM2D_HPP_6A31CA41_7EDF_4BBD_BE8D_02F8ECD4A79D
#define TRANSFORM2D_HPP_6A31CA41_7EDF_4BBD_BE8D_02F8ECD4A79D

#include "stdtypes.hpp"

// For simplicity, these are defined only for Mat22f, i.e., the 2x2 float
// matrix type.
inline 
Mat22f make_scaling_2d( float aX, float aY ) noexcept
{
	Mat22f result{};

	result[0,0] = aX;
	result[1,1] = aY;

	return result;
}
inline 
Mat22f make_scaling_2d( float aUniform ) noexcept
{
	Mat22f result{};
	result[0,0] = aUniform;
	result[1,1] = aUniform;
	return result;
}

inline
Mat22f make_rotation_2d( float aAngleInRadians ) noexcept
{
	float c = std::cos(aAngleInRadians);
	float s = std::sin(aAngleInRadians);

	Mat22f result{};

	result[0,0] = c;
	result[0,1] = -s;
	result[1,0] = s;
	result[1,1] = c;

return result;
}

#endif // TRANSFORM2D_HPP_6A31CA41_7EDF_4BBD_BE8D_02F8ECD4A79D
