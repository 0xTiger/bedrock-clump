cmake_minimum_required(VERSION 3.15)
project(cmakeTest2 LANGUAGES C CXX CUDA)


#add_executable("${PROJECT_NAME}" "main.cpp")

#install(TARGETS "${PROJECT_NAME}" DESTINATION bin)
#install(FILES "main.cpp" DESTINATION src)

# the shared library of the cuda should be the static
# provide the cuda compiler: cmake -DCMAKE_CUDA_COMPILER=/usr/local/cuda-10.0/bin/nvcc ..

set_source_files_properties(clumpFinderCUDA.cu PROPERTIES LANGUAGE "CUDA")
add_executable(clumpFinderCUDA clumpFinderCUDA.cu PrecomputedRandAdvance.cpp PrecomputedRandAdvance.h)
set_target_properties(clumpFinderCUDA PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

#------------------------------------------------------------------------------------


find_package( OpenCL REQUIRED )
#include_directories( ${OPENCL_INCLUDE_DIR} )
add_executable(clumpFinderOpenCL clumpFinderOpenCL.cpp PrecomputedRandAdvance.cpp PrecomputedRandAdvance.h)
configure_file(ker.cl ker.cl COPYONLY)


#target_include_directories(clumpFinderOpenCL PRIVATE ${OpenCL_INCLUDE_DIR})
target_link_libraries(clumpFinderOpenCL OpenCL::OpenCL)

#set(SOURCES ${OPENCL_INCLUDE_DIR}/cl.hpp)
