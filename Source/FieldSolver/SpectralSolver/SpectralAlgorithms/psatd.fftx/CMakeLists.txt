##
## Copyright (c) 2018-2020, Carnegie Mellon University
## All rights reserved.
##
## See LICENSE file for full information
##

## specify this only here, lower levels use CMAKE_MINIMUM_REQUIRED_VERSION that this sets
cmake_minimum_required (VERSION 3.14)

set (CMAKE_BUILD_TYPE Release  CACHE STRING "Debug, Release, RelWithDebInfo, MinSizeRel")

project (SPIRAL_WARPX_FILES
         VERSION 1.0.1
                DESCRIPTION "Create SPIRAL generated files for WarpX"
                LANGUAGES C CXX)

##  Prevent building directly into the source tree
string (COMPARE EQUAL "${CMAKE_SOURCE_DIR}" "${CMAKE_BINARY_DIR}" DISALLOW_COMPILE_INPLACE)
if (DISALLOW_COMPILE_INPLACE)
    message (FATAL_ERROR, "Building ${PROJECT_NAME} with CMake requires an out-of-source tree.  To proceed:
                 rm -rf CMakeCache.txt CMakeFiles/   # delete files in ${CMAKE_SOURCE_DIR}
                 mkdir <build>
                 cd <build>
                 cmake ..
                 make | make install")
endif()

##  Get SPIRAL home...

if ( DEFINED ENV{SPIRAL_HOME} )
    message ( STATUS "SPIRAL_HOME = $ENV{SPIRAL_HOME}" )
    set ( SPIRAL_SOURCE_DIR $ENV{SPIRAL_HOME} )
else ()
    if ( "x${SPIRAL_HOME}" STREQUAL "x" )
        message ( FATAL "SPIRAL_HOME environment variable undefined and not specified on command line" )
    endif ()
    set ( SPIRAL_SOURCE_DIR ${SPIRAL_HOME} )
endif ()

include ("${SPIRAL_SOURCE_DIR}/config/CMakeIncludes/RunSpiral.cmake")

##  First create the driver program

set ( FIL_PREFIX psatd )
set ( DRIVER_TARG ${FIL_PREFIX}.fftx.driver )
add_custom_command ( OUTPUT ${DRIVER_TARG}.cpp
    COMMAND ${BASH} -c "rm -f ${DRIVER_TARG}.cpp ; cat ${CMAKE_SOURCE_DIR}/${FIL_PREFIX}.fftx.cpp > ${DRIVER_TARG}.cpp ; echo 'int main() { fftx::ID=1; fftx::tracing=true; trace(\"${FIL_PREFIX}\"); return 0; }' >> ${DRIVER_TARG}.cpp"
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    VERBATIM
    COMMENT "Generating ${DRIVER_TARG}.cpp" )

include_directories ( ${CMAKE_SOURCE_DIR} ${CMAKE_BINARY_DIR} ${SPIRAL_SOURCE_DIR}/profiler/targets )
add_executable ( ${DRIVER_TARG} ${DRIVER_TARG}.cpp )
set_property ( TARGET ${DRIVER_TARG} PROPERTY CXX_STANDARD 14 )

set ( SRCS_TO_INSTALL )  ##  build list of files as we go...

##  Run the driver program to create ~.codegen.hpp and ~.plan.g
set ( SPIRAL_PLAN ${FIL_PREFIX}.fftx.plan.g )
set ( CODEGEN_HEADER ${FIL_PREFIX}.fftx.codegen.hpp )
add_custom_command ( OUTPUT ${SPIRAL_PLAN} ${CODEGEN_HEADER}
    COMMAND ${BASH} -c "rm -f ${SPIRAL_PLAN} ; ${CMAKE_BINARY_DIR}/${DRIVER_TARG} > ${SPIRAL_PLAN}"
    DEPENDS ${DRIVER_TARG}
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    VERBATIM
    COMMENT "Generating ${SPIRAL_PLAN}" )
list ( APPEND SRCS_TO_INSTALL ${CMAKE_BINARY_DIR}/${CODEGEN_HEADER} )

##  Create the generator spiral script
set ( SPIRAL_GENERATOR ${FIL_PREFIX}.fftx.generator.g )
set ( SPIRAL_BACKEND_PREAMBLE ${CMAKE_SOURCE_DIR}/preamble.g )
set ( SPIRAL_BACKEND_CODEGEN  ${CMAKE_SOURCE_DIR}/codegen.g  )
add_custom_command ( OUTPUT ${SPIRAL_GENERATOR}
    COMMAND ${BASH} -c "rm -f ${SPIRAL_GENERATOR} ; echo 'prefix := \"${FIL_PREFIX}\";' > ${SPIRAL_GENERATOR} ; cat ${SPIRAL_BACKEND_PREAMBLE} ${SPIRAL_PLAN} ${SPIRAL_BACKEND_CODEGEN} >> ${SPIRAL_GENERATOR}"
    DEPENDS ${SPIRAL_PLAN} ${DRIVER_TARG}
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    VERBATIM
    COMMENT "Generating ${SPIRAL_GENERATOR}" )

##  Create the C source code from the SPIRAL generator script
set ( SPIRAL_C_CODE ${FIL_PREFIX}.fftx.source.cpp )
file (TO_NATIVE_PATH ${CMAKE_BINARY_DIR}/${SPIRAL_GENERATOR} _gfile)
create_source_file ( ${_gfile} ${SPIRAL_C_CODE} ${SPIRAL_GENERATOR} )
list ( APPEND SRCS_TO_INSTALL ${CMAKE_BINARY_DIR}/${SPIRAL_C_CODE} )

##  Create the C source code from remaining SPIRAL scripts
set ( WARPX_FULLSTEP_RHO warpx-fullstep-rho_80 )
file (TO_NATIVE_PATH ${CMAKE_SOURCE_DIR}/${WARPX_FULLSTEP_RHO}.g _gfile)
create_source_file ( ${_gfile} ${WARPX_FULLSTEP_RHO}.c )
list ( APPEND SRCS_TO_INSTALL ${CMAKE_BINARY_DIR}/${WARPX_FULLSTEP_RHO}.c )

set ( WARPX_SYMBOL_NORHO warpx-symbol_norho_80 )
file (TO_NATIVE_PATH ${CMAKE_SOURCE_DIR}/${WARPX_SYMBOL_NORHO}.g _gfile)
create_source_file ( ${_gfile} ${WARPX_SYMBOL_NORHO}.c )
list ( APPEND SRCS_TO_INSTALL ${CMAKE_BINARY_DIR}/${WARPX_SYMBOL_NORHO}.c )

set ( WARPX_SYMBOL_RHO   warpx-symbol_rho_80   )
file (TO_NATIVE_PATH ${CMAKE_SOURCE_DIR}/${WARPX_SYMBOL_RHO}.g _gfile)
create_source_file ( ${_gfile} ${WARPX_SYMBOL_RHO}.c )
list ( APPEND SRCS_TO_INSTALL ${CMAKE_BINARY_DIR}/${WARPX_SYMBOL_RHO}.c )

add_custom_target ( NAME ALL
    DEPENDS ${SPIRAL_PLAN} ${SPIRAL_GENERATOR} ${SPIRAL_C_CODE} ${SRCS_TO_INSTALL}
##    COMMAND ${BASH} -c "cp -f -p ${SRCS_TO_INSTALL} ${INSTALL_DIR_TARGET}"
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    VERBATIM )

set ( INSTALL_DIR_TARGET ${CMAKE_SOURCE_DIR}/.. )

list ( APPEND SRCS_TO_INSTALL ${CMAKE_SOURCE_DIR}/fftx3.hpp )

install ( FILES ${SRCS_TO_INSTALL} DESTINATION ${INSTALL_DIR_TARGET} )

