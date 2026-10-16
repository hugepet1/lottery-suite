workspace "COMP3812-vmlib"
	language "C++"
	cppdialect "C++23"

	platforms { "x64" }
	configurations { "debug", "release" }

	startproject "vmlib"

	enablepch "Off"
	multiprocessorcompile "On"
	visibility "Hidden"
	vectorextensions "AVX2"

	debugdir "%{wks.location}"
	objdir "_build_/%{cfg.buildcfg}-%{cfg.platform}-%{cfg.toolset}"
	targetsuffix "-%{cfg.buildcfg}-%{cfg.platform}-%{cfg.toolset}"

	-- Default toolset options
	filter "toolset:gcc or toolset:clang"
		linkoptions { "-pthread" }
		buildoptions { "-march=native", "-Wall", "-pthread" }

		-- Varriable-length arrays (VLAs) are an extension that GCC and clang
		-- have long supported. However, they are not part of the C++ standard.
		-- (MSVC will not compile code with VLAs.)
		buildoptions { "-Werror=vla" }

	filter "toolset:gcc" 
		links { "stdc++exp" }

	filter "toolset:msc-*"
		warnings "extra" -- this enables /W4; default is /W3
		--buildoptions { "/W4" }
		buildoptions { "/utf-8" }
		buildoptions { "/permissive-" }
		defines { "_CRT_SECURE_NO_WARNINGS=1" }
		defines { "_SCL_SECURE_NO_WARNINGS=1" }
	
	filter "*"

	-- default libraries
	filter "system:linux"
		links "dl"
	
	filter "system:windows"
		links "OpenGL32"

	filter "*"

	-- default outputs
	filter "kind:StaticLib"
		targetdir "lib/"

	filter "kind:ConsoleApp"
		targetdir "bin/"
		targetextension ".exe"
	
	filter "*"

	--configurations
	filter "debug"
		symbols "On"
		defines { "_DEBUG=1" }

	filter "release"
		optimize "On"
		defines { "NDEBUG=1" }

	filter "*"


-- Third party code
include "third_party"

-- Projects
project "vmlib"
	kind "StaticLib"

	location "vmlib"
	files {
		"vmlib/**.cpp",
		"vmlib/**.hpp"
	}

	usage "PUBLIC"
		includedirs "vmlib/include"

	usage "INTERFACE"
		links "vmlib"

project "tests"
	kind "ConsoleApp"

	location "tests"
	files {
		"tests/*.cpp"
	}

	uses "vmlib"

	uses "x-snitch"

project "bench"
	kind "ConsoleApp"

	location "bench"
	files {
		"bench/*.cpp"
	}

	uses "vmlib"

	uses "x-nanobench"

--EOF
