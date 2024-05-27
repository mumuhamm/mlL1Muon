cmake_minimum_required(VERSION 3.0) # setting this is required
project(LutNN)                   # this sets the project name
set(MODULE_NAME "lutNN2")
###############################################################################
## dependencies ###############################################################
###############################################################################
#Boost
find_package(Boost 1.45.0 COMPONENTS timer system serialization)
#if you want to use the newer boost fromt the cmssw
#set(BOOST_ROOT /cvmfs/cms.cern.ch/slc7_amd64_gcc900/external/boost/1.72.0/)
#find_package(Boost 1.72.0 COMPONENTS timer system serialization)

MESSAGE( STATUS "Boost_VERSION_STRING:       " ${Boost_VERSION_STRING})
MESSAGE( STATUS "Boost_INCLUDE_DIRS:         " ${Boost_INCLUDE_DIRS})
MESSAGE( STATUS "Boost_LIBRARIES:            " ${Boost_LIBRARIES})

#ROOT 
list(APPEND CMAKE_PREFIX_PATH $ENV{ROOTSYS})
list(APPEND CMAKE_MODULE_PATH $ENV{ROOTSYS}/etc/cmake)
find_package(ROOT REQUIRED COMPONENTS Physics HistPainter)
include(${ROOT_USE_FILE})

MESSAGE( STATUS "CMAKE_PREFIX_PATH:         " ${CMAKE_PREFIX_PATH} )
MESSAGE( STATUS "CMAKE_MODULE_PATH:         " ${CMAKE_MODULE_PATH} )
MESSAGE( STATUS "ROOTSYS:                   " $ENV{ROOTSYS} )
MESSAGE( STATUS "ROOT_INCLUDE_DIRS:         " ${ROOT_INCLUDE_DIRS} )
MESSAGE( STATUS "ROOT_LIBRARIES:            " ${ROOT_LIBRARIES} )
MESSAGE( STATUS "ROOT_USE_FILE:             " ${ROOT_USE_FILE})

#HLS
set(HLS_INCLUDE_DIRS ${PROJECT_SOURCE_DIR}/hls/include)
MESSAGE( STATUS "HLS_INCLUDE_DIRS:         " ${HLS_INCLUDE_DIRS} )
###############################################################################
## packaging ##################################################################
###############################################################################
set(CMAKE_INSTALL_PREFIX "$ENV{PWD}") # install in the current directory
MESSAGE( STATUS "CMAKE_INSTALL_PREFIX:         " ${CMAKE_INSTALL_PREFIX} )

# This is basically a repeat of the file copy instruction that copies the
# resources in the build directory, but here we tell cmake that we want it
# in the package.
#
set(CMAKE_CXX_FLAGS "${ROOT_CXX_FLAGS} -Wall -MMD -O3 -std=c++14")

#Set RPATH to install dir, so user do not have to set the LD_LIBRARY_PATH
set(CMAKE_INSTALL_RPATH ${CMAKE_INSTALL_PREFIX}/lib)

###############################################################################
## file globbing ##############################################################
###############################################################################
file(GLOB_RECURSE sources       ${PROJECT_SOURCE_DIR}/${MODULE_NAME}/src/*.cpp)

###############################################################################
## dependencies ###############################################################
###############################################################################
include_directories(${ROOTINCDIR})
include_directories(${Boost_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR})
include_directories(${HLS_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/${MODULE_NAME}/include)

find_package(ROOT REQUIRED COMPONENTS RIO GenVector)
###############################################################################
## target definitions #########################################################
###############################################################################
add_library(${MODULE_NAME} SHARED ${sources})

#add_executable(omtfClassifierTest ${MODULE_NAME}/test/omtfClassifierTest.cpp)

###############################################################################
# compiler flags ##############################################################
###############################################################################
target_compile_options(${MODULE_NAME} PUBLIC ${CMAKE_ROOT_CFLAGS})

#target_compile_options(omtfClassifierTest PUBLIC ${CMAKE_ROOT_CFLAGS})
#target_link_libraries(omtfClassifierTest PUBLIC ${MODULE_NAME} ${ROOT_LIBRARIES}  ${Boost_LIBRARIES})

file(GLOB BIN_SOURCE_FILES ${MODULE_NAME}/test/*.cpp)

foreach(SOURCE_PATH ${BIN_SOURCE_FILES})
  get_filename_component(EXECUTABLE_NAME ${SOURCE_PATH} NAME_WE)
  MESSAGE( STATUS "Adding executable:       " ${EXECUTABLE_NAME})
  add_executable(${EXECUTABLE_NAME} ${SOURCE_PATH})
  target_compile_options(${EXECUTABLE_NAME} PUBLIC ${CMAKE_ROOT_CFLAGS})
  target_link_libraries(${EXECUTABLE_NAME} PUBLIC ${MODULE_NAME} ${ROOT_LIBRARIES}  ${Boost_LIBRARIES})
  install(TARGETS ${EXECUTABLE_NAME} RUNTIME DESTINATION bin)
endforeach(SOURCE_PATH ${SOURCE_FILES})
###############################################################################
# install flags ##############################################################
###############################################################################
install(TARGETS ${MODULE_NAME} LIBRARY DESTINATION lib)
#install(TARGETS omtfClassifierTest RUNTIME DESTINATION bin)

# Now comes everything we need, to create a package
# there are a lot more variables you can set, and some
# you need to set for some package types, but we want to
# be minimal here.
# We don't want to split our program up into several incomplete pieces.
set(CPACK_MONOLITHIC_INSTALL 1)

# This must be last
include(CPack)
