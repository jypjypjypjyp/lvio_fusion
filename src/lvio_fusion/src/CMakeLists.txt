add_library(lvio_fusion SHARED
        agent.cpp
        association.cpp
        backend.cpp
        config.cpp
        estimator.cpp
        frame.cpp
        frontend.cpp
        initializer.cpp
        landmark.cpp
        manager.cpp
        map.cpp
        mapping.cpp
        navsat.cpp
        optimizer.cpp
        preintegration.cpp
        projection.cpp
        relocator.cpp)

target_link_libraries(lvio_fusion ${THIRD_PARTY_LIBS})
target_compile_features(lvio_fusion PRIVATE cxx_std_14)