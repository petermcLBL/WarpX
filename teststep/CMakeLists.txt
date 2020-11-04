##
## Copyright (c) 2018-2020, Carnegie Mellon University
## All rights reserved.
##
## See LICENSE file for full information
##

## specify this only here, lower levels use CMAKE_MINIMUM_REQUIRED_VERSION that this sets
cmake_minimum_required ( VERSION 3.14 )

set ( CMAKE_BUILD_TYPE Release  CACHE STRING "Debug, Release, RelWithDebInfo, MinSizeRel" )

project ( WARPX_SPIRAL_CUDA
         VERSION 1.0.1
                DESCRIPTION "Test waprx spiral code"
                LANGUAGES C CXX CUDA )

set ( PROJECT cudaspiral )
set ( SOURCES ${PROJECT}.cu )

set ( RUNSPIRAL runspiral )

include_directories ( ${CMAKE_SOURCE_DIR} "Utils"  )

##  Setup compilation flags

if (WIN32)
    set ( CUDA_COMPILE_FLAGS -rdc=false )
    set ( GPU_COMPILE_DEFNS )			## -Xptxas -v
    set ( LIBS_FOR_CUDA cufft )
else ()
    set ( CUDA_COMPILE_FLAGS -m64 -rdc=true )
    set ( GPU_COMPILE_DEFNS -dc )		## -Xptxas -v
    set ( LIBS_FOR_CUDA cufft_static culibos )
endif ()

##  set ( CMAKE_CUDA_ARCHITECTURES 60 61 62 70 72 75 )
set ( CMAKE_CUDA_ARCHITECTURES 70 )
set ( GPU_EXTRAS _CUDAGEN )

add_executable		   ( ${PROJECT} ${SOURCES} )
target_compile_definitions ( ${PROJECT} PRIVATE ${GPU_EXTRAS} )
target_compile_options     ( ${PROJECT} PRIVATE ${CUDA_COMPILE_FLAGS} ${GPU_COMPILE_DEFNS} )
set_property        ( TARGET ${PROJECT} PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON )
target_link_libraries	   ( ${PROJECT} ${LIBS_FOR_CUDA} )

add_executable             ( ${RUNSPIRAL} ${RUNSPIRAL}.cpp )

install ( TARGETS ${PROJECT} ${RUNSPIRAL} DESTINATION ${CMAKE_SOURCE_DIR} )
