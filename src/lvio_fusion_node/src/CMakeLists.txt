add_executable(lvio_fusion_node
        lvio_fusion_node.cpp
        parameters.cpp
        visualization.cpp)

add_dependencies(lvio_fusion_node detector_gencpp)
target_link_libraries(lvio_fusion_node ${THIRD_PARTY_LIBS})
target_compile_features(lvio_fusion_node PRIVATE cxx_std_17)