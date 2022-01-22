cmake_minimum_required(VERSION 3.3)

set(NAME "bvh8")

set(CMAKE_CONFIGURATION_TYPES "Debug;Release;RelWithDebInfo" CACHE STRING "My multi config types" FORCE)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_USER_MAKE_RULES_OVERRIDE  "${CMAKE_CURRENT_SOURCE_DIR}/cmake_override_cxx.cmake")

project(${NAME})

list(APPEND LLVM_FLAGS
    "-Wall"
    "-Wno-gnu-anonymous-struct"
    "-Wno-nested-anon-types"
    "-Wno-missing-braces"
    "-mavx"
    "-mavx2"
    "-std=c++20"
)

list(APPEND MSVC_FLAGS
    "/W3"
    "/DNOMINMAX"
    "/DWIN32_LEAN_AND_MEAN"
    "/D_CRT_SECURE_NO_WARNINGS"
    "/D_SCL_SECURE_NO_WARNINGS"
    "/D_SILENCE_ALL_CXX17_DEPRECATION_WARNINGS"
	"/std:c++20"
)

if(WIN32)
    set(CPP_FLAGS ${MSVC_FLAGS})
else()
    set(CPP_FLAGS ${LLVM_FLAGS})
endif()

set(CMAKE_CXX_STANDARD 20)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/build/${CMAKE_BUILD_TYPE}/build)

set(SRCS "src/main.cpp"
        "src/bvh8_build.cpp"
        "src/bvh8_build.h"
        "src/bvh8_traverse.h"
        "src/bvh8_traverse.cpp"
        "src/bvh8_render.h"
        "src/bvh8_render.cpp"
        "src/utils.h"
        "src/simple_obj_parser.h"
        "src/allocator.h"
)

add_executable(
    ${NAME}
    ${SRCS}
)

set_target_properties(${NAME}
    PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

if(NOT WIN32)
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -ltbb -pthread")
endif()

target_compile_definitions(${NAME} PUBLIC ${MG_CC_DEFS})
target_compile_options(${NAME} PRIVATE ${CPP_FLAGS})