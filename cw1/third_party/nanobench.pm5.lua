project( "x-nanobench" )
	kind "StaticLib"

	location "."

	files( "nanobench/src/*.cpp" )
	files( "nanobench/include/*.h" )

	usage "PUBLIC"
		includedirs "nanobench/include"

	usage "INTERFACE"
		links "x-nanobench"

project ()

