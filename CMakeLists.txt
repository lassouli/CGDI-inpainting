cmake_minimum_required(VERSION 2.8)

project(Inpainting)

set(EXECUTABLE_OUTPUT_PATH bin/)

file (GLOB_RECURSE source_files
        src/*.cpp
	    src/*.hpp
	    GUI/*.hpp
)

file (GLOB_RECURSE gui_files
	GUI/*.cpp
)

add_executable(Inpainting ${source_files})
add_executable(create_mask ${gui_files})

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(${OpenCV_LIBRARY_DIRS})
target_link_libraries(Inpainting ${OpenCV_LIBS})
target_link_libraries(create_mask ${OpenCV_LIBS})

if(CMAKE_COMPILER_IS_GNUCXX)
    add_definitions(-std=c++14 -Wall -Wextra -pedantic)
    add_definitions(-O3 -s -DNDEBUG)
    add_definitions(${GCC_COVERAGE_LINK_FLAGS}$)
endif()
