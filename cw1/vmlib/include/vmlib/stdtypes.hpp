#ifndef STDTYPES_HPP_35625802_25E3_4A72_804F_BDA259D60611
#define STDTYPES_HPP_35625802_25E3_4A72_804F_BDA259D60611

#include "small_vector.hpp"
#include "small_matrix.hpp"

// Float-based
using Vec2f = SmallVec<2,float>;
using Vec3f = SmallVec<3,float>;
using Vec4f = SmallVec<4,float>;

using Mat22f = SmallMat<2,2,float>;
using Mat33f = SmallMat<3,3,float>;
using Mat44f = SmallMat<4,4,float>;

// Integer
using Vec2i = SmallVec<2,int>;
using Vec3i = SmallVec<3,int>;
using Vec4i = SmallVec<4,int>;

// Integer matrices are less useful for us.

#endif // STDTYPES_HPP_35625802_25E3_4A72_804F_BDA259D60611
