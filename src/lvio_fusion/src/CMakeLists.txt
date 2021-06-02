add_library(lvio_fusion SHARED
        agent.cpp
        association.cpp
        backend.cpp
        config.cpp
        Dstar.cpp
        environment.cpp
        extractor.cpp
        estimator.cpp
        frame.cpp
        frontend.cpp
        gridmap.cpp
        initializer.cpp
        landmark.cpp
        local_map.cpp
        manager.cpp
        map.cpp
        mapping.cpp
        navsat.cpp
        pose_graph.cpp
        preintegration.cpp
        projection.cpp
        relocator.cpp
        tools.cpp
        utility.cpp)

target_link_libraries(lvio_fusion ${THIRD_PARTY_LIBS} blas)
target_compile_features(lvio_fusion PRIVATE cxx_std_14)