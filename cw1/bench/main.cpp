#include <nanobench.h>
#include <vmlib/stdtypes.hpp>
#include <chrono>
#include <cstddef>
#include <string>

template<std::size_t N>
void benchmark_size(ankerl::nanobench::Bench& bench)
{
    SmallMat<N,N,float> a{};
    SmallMat<N,N,float> b{};

    for (std::size_t i = 0; i < N; ++i)
    {
        for (std::size_t j = 0; j < N; ++j)
        {
            a[i,j] = (i == j)
                ? float(N + 1)
                : float((i + j) % 7 + 1) / 10.f;

            b[i,j] = float((3 * i + 2 * j) % 11 + 1) / 10.f;
        }
    }

    std::string size = std::to_string(N) + "x" + std::to_string(N);

    bench.run("multiply " + size, [&]
    {
        ankerl::nanobench::doNotOptimizeAway(a);
        ankerl::nanobench::doNotOptimizeAway(b);

        auto result = a * b;

        ankerl::nanobench::doNotOptimizeAway(result);
    });

    bench.run("inverse " + size, [&]
    {
        ankerl::nanobench::doNotOptimizeAway(a);

        auto result = invert(a);

        ankerl::nanobench::doNotOptimizeAway(result);
    });
}
int main()
{
    Mat44f a{{
        4.f, 1.f, 3.f, 2.f,
        1.f, 4.f, 1.f, 4.f,
        0.f, 1.f, 4.f, 1.f,
        6.f, 0.f, 1.f, 4.f
    }};

    Mat44f b{{
        3.f, 2.f, 1.f, 0.f,
        0.f, 3.f, 0.f, 1.f,
        1.f, 1.f, 3.f, 0.f,
        3.f, 1.f, 0.f, 6.f
    }};

    ankerl::nanobench::Bench bench;
    bench.performanceCounters(false)
     .epochs(15)
     .minEpochTime(std::chrono::milliseconds(50));

    bench.run("multiply 4x4", [&]
    {
        ankerl::nanobench::doNotOptimizeAway(a);
        ankerl::nanobench::doNotOptimizeAway(b);

        auto result = a * b;

        ankerl::nanobench::doNotOptimizeAway(result);
    });

    bench.run("inverse 4x4", [&]
    {
        ankerl::nanobench::doNotOptimizeAway(a);

        auto result = invert(a);

        ankerl::nanobench::doNotOptimizeAway(result);
    });
	benchmark_size<2>(bench);
	benchmark_size<8>(bench);
	benchmark_size<16>(bench);
	benchmark_size<32>(bench);
	benchmark_size<64>(bench);
	benchmark_size<100>(bench);
    return 0;
}