#include <nanobench.h>

int main()
{
	// This is essentially the example from https://nanobench.ankerl.com/
	float f = 1.f;

	ankerl::nanobench::Bench b;
	// You can configure things here.

	b.run( "some float ops", [&] {
		f += 1.f / f;
		if( f > 5.f )
			f -= 5.f;

		// Without the following, the result in f is never used, and therefore
		// a compiler might optimize away the computations.
		ankerl::nanobench::doNotOptimizeAway( f );
	} );

	
	return 0;
}
