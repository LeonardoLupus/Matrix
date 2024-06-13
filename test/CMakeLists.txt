add_executable(${PROJECT_NAME}-bench)
target_sources(${PROJECT_NAME}-bench
        PRIVATE
        main.cc
)
target_link_libraries(${PROJECT_NAME}-bench PRIVATE ${PROJECT_NAME} )