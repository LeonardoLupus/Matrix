cmake_minimum_required(VERSION 3.15)

include_guard(GLOBAL)

project(matrix
        VERSION 0.1.0
        DESCRIPTION "Lib to matrix operations"
        LANGUAGES CXX)

option(STANDALONE_matrix "Build dev executable (testing only)" ON)

if(NOT CMAKE_CXX_STANDARD)
    message(STATUS "[${PROJECT_NAME}] setting c++ standard to c++20")
    set(CMAKE_CXX_STANDARD 20)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
    set(CMAKE_CXX_EXTENSIONS OFF)
endif()

if("${CMAKE_GENERATOR}" MATCHES "^Visual Studio")
    set(CMAKE_GENERATOR_PLATFORM "x64" CACHE STRING "" FORCE)
endif()

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
message(STATUS "[${PROJECT_NAME}] setting fPIC: ${CMAKE_POSITION_INDEPENDENT_CODE}")

# set(CMAKE_AUTOMOC ON)
# set(CMAKE_AUTORCC ON)
# set(CMAKE_AUTOUIC ON)

#add_subdirectory(submodules/googletest)

add_library(${PROJECT_NAME})
add_library(matrix::${PROJECT_NAME} ALIAS ${PROJECT_NAME})

target_sources(${PROJECT_NAME}
        PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/matrix/matrix.h> $<INSTALL_INTERFACE:include/matrix/matrix.h>

        PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src/c++/matrix.cc
)

#target_link_libraries(${PROJECT_NAME} gtest_main gmock_main)

target_include_directories(${PROJECT_NAME} PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>$<INSTALL_INTERFACE:include> PRIVATE src/c++
)

set(PROJECT_NAMESPACE matrix::)
message(STATUS "[${PROJECT_NAME}] installing ${PROJECT_NAME} in namespace ${PROJECT_NAMESPACE}")
include(GNUInstallDirs)
include(GenerateExportHeader)
include(CMakePackageConfigHelpers)
message(STATUS "[${PROJECT_NAME}] installing in prefix: ${CMAKE_INSTALL_PREFIX}")

set(PROJECT_CONFIG_FILE_IN "${CMAKE_CURRENT_SOURCE_DIR}/cmake/${PROJECT_NAME}-config.cmake.in")
set(PROJECT_CONFIG_FILE "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}-config.cmake")
set(CONFIG_TARGETS_FILE "${PROJECT_NAME}-config-targets.cmake")
set(CONFIG_VERSION_FILE "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}-config-version.cmake")
set(INSTALL_EXPORT_DESTINATION_DIR "${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}")

install(TARGETS ${PROJECT_NAME}
        EXPORT ${PROJECT_NAME}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(DIRECTORY $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/${PROJECT_NAME}> $<INSTALL_INTERFACE:include/${PROJECT_NAME}> ; DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

configure_package_config_file(${PROJECT_CONFIG_FILE_IN} ${PROJECT_CONFIG_FILE} INSTALL_DESTINATION ${INSTALL_EXPORT_DESTINATION_DIR})
write_basic_package_version_file(${CONFIG_VERSION_FILE} COMPATIBILITY AnyNewerVersion)
install(FILES ${PROJECT_CONFIG_FILE} ${CONFIG_VERSION_FILE} DESTINATION ${INSTALL_EXPORT_DESTINATION_DIR})

message(STATUS "[${PROJECT_NAME}] configuring ${PROJECT_NAME} done!")

if(STANDALONE_matrix)
    message(STATUS "[${PROJECT_NAME}] building test executable for developers")
    add_subdirectory(test)
endif()

unset(STANDALONE_matrix CACHE)
