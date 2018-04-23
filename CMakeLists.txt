cmake_minimum_required(VERSION 2.8)

project(Inpainting)

set(EXECUTABLE_OUTPUT_PATH bin/)

file (GLOB_RECURSE source_files
        src/*.cpp
	    src/*.hpp
)

add_executable(Inpainting ${source_files})

SET(Boost_USE_STATIC_LIBS   ON)
SET(Boost_USE_MULTITHREADED ON)
SET(Boost_USE_STATIC_RUNTIME OFF)
SET(Boost_FOUND FALSE)
FIND_PACKAGE(Boost REQUIRED COMPONENTS program_options)
if ( Boost_FOUND )
  message(STATUS "Boost and boost_program_options found.")
  include_directories( ${Boost_INCLUDE_DIRS} )
endif( Boost_FOUND )

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(${OpenCV_LIBRARY_DIRS})

target_link_libraries(Inpainting ${OpenCV_LIBS} ${Boost_LIBRAIRIES}  ${Boost_PROGRAM_OPTIONS_LIBRARY})

if(CMAKE_COMPILER_IS_GNUCXX)
    add_definitions(-std=c++14 -Wall -Wextra -pedantic)
    add_definitions(-O3 -s -DNDEBUG)
    add_definitions(${GCC_COVERAGE_LINK_FLAGS}$)
endif()
