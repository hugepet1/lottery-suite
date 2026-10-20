#include <snitch/snitch.hpp>
#include <vmlib/small_matrix.hpp>
#include <vmlib/transform2d.hpp>
#include "within.hxx"

TEST_CASE("Rectangular matrix multiplication", "[matrix]")//test code
{
    SmallMat<2,3,int> a{{1, 2, 3, 4, 5, 6}};
    SmallMat<3,2,int> b{{7, 8, 9, 10, 11, 12}};

    auto result = a * b;

    REQUIRE((result[0,0] == 58));
    REQUIRE((result[0,1] == 64));
    REQUIRE((result[1,0] == 139));
    REQUIRE((result[1,1] == 154));
}

TEST_CASE("Rectangular matrix transpose", "[matrix]")
{
    SmallMat<2,3,int> a{{1, 2, 3, 4, 5, 6}};

    SmallMat<3,2,int> result = transpose(a);

    REQUIRE((result[0,0] == 1));
    REQUIRE((result[0,1] == 4));
    REQUIRE((result[1,0] == 2));
    REQUIRE((result[1,1] == 5));
    REQUIRE((result[2,0] == 3));
    REQUIRE((result[2,1] == 6));
}
TEST_CASE("Non-uniform 2D scaling", "[transform]")
{
    Vec2f point{1.f, 2.f};
    auto result = make_scaling_2d(2.f, 3.f) * point;

    REQUIRE_THAT(result.x, WithinAbs(2.f));
    REQUIRE_THAT(result.y, WithinAbs(6.f));
}

TEST_CASE("Uniform 2D scaling", "[transform]")
{
    Vec2f point{1.f, 2.f};
    auto result = make_scaling_2d(3.f) * point;

    REQUIRE_THAT(result.x, WithinAbs(3.f));
    REQUIRE_THAT(result.y, WithinAbs(6.f));
}

TEST_CASE("Quarter-turn 2D rotation", "[transform]")
{
    Vec2f point{1.f, 2.f};
    auto result = make_rotation_2d(1.5707963f) * point;

    REQUIRE_THAT(result.x, WithinAbs(-2.f));
    REQUIRE_THAT(result.y, WithinAbs(1.f));
}
TEST_CASE("General matrix inverse", "[matrix]")
{
    SmallMat<2,2,float> matrix{{1.f, 2.f, 3.f, 4.f}};

    auto result = invert(matrix);

    REQUIRE_THAT((result[0,0]), WithinAbs(-2.f));
    REQUIRE_THAT((result[0,1]), WithinAbs(1.f));
    REQUIRE_THAT((result[1,0]), WithinAbs(1.5f));
    REQUIRE_THAT((result[1,1]), WithinAbs(-0.5f));
}
TEST_CASE("Gauss with rectangular right side", "[matrix]")
{
    SmallMat<2,2,float> left{{0.f, 2.f, 1.f, 3.f}};

    SmallMat<2,3,float> right{{
        4.f, 8.f, 12.f,
        7.f, 11.f, 15.f
    }};

    gauss(left, right);

    REQUIRE_THAT((right[0,0]), WithinAbs(1.f));
    REQUIRE_THAT((right[0,1]), WithinAbs(-1.f));
    REQUIRE_THAT((right[0,2]), WithinAbs(-3.f));

    REQUIRE_THAT((right[1,0]), WithinAbs(2.f));
    REQUIRE_THAT((right[1,1]), WithinAbs(4.f));
    REQUIRE_THAT((right[1,2]), WithinAbs(6.f));
}