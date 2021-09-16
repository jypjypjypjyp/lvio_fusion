add_executable(navigation_node
        navigation_node.cpp
        global_planner.cpp
        Dstar.cpp
        # local_planner.cpp
        )

target_link_libraries(navigation_node ${THIRD_PARTY_LIBS})
target_compile_features(navigation_node PRIVATE cxx_std_14)