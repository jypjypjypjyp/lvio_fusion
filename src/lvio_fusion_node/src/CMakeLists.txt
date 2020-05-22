add_executable(lvio_fusion_node
        CameraPoseVisualization.cpp
        lvio_fusion_node.cpp
        parameters.cpp
        visualization.cpp)

target_link_libraries(lvio_fusion_node ${THIRD_PARTY_LIBS})