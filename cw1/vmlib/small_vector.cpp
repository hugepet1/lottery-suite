#include <vmlib/small_vector.hpp>

// The following ensures that our core types are instantiated at least once.
// This helps us catch problems with the core definitions of the types early.
namespace
{
	using Vec2f = SmallVec<2,float>;
	using Vec3f = SmallVec<3,float>;
	using Vec4f = SmallVec<4,float>;

	using Vec2i = SmallVec<2,int>;
	using Vec3i = SmallVec<3,int>;
	using Vec4i = SmallVec<4,int>;
}
