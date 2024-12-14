cmake_minimum_required(VERSION 3.0)

add_subdirectory(googletest)
add_executable(${PROJECT_NAME}-bench)
target_sources(${PROJECT_NAME}-bench
        PRIVATE
        main.cc
)
target_link_libraries(${PROJECT_NAME}-bench PRIVATE ${PROJECT_NAME} gtest_main gmock_main)