project "x-snitch"
	kind "StaticLib"

	location "."

	files { 
		"snitch/src/snitch.cpp"
	}

	usage "PUBLIC"
		includedirs "snitch/include"

	usage "INTERFACE"
		links "x-snitch"

project ()
