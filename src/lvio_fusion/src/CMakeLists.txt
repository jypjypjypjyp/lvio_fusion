add_library(lvio_fusion SHARED
        atlas.cpp
        backend.cpp
        config.cpp
        estimator.cpp
        frame.cpp
        frontend.cpp
        initializer.cpp
        landmark.cpp
        map.cpp
        mapping.cpp
        navsat.cpp
        preintegration.cpp
        relocation.cpp
        association.cpp)

target_link_libraries(lvio_fusion ${THIRD_PARTY_LIBS})
target_compile_features(lvio_fusion PRIVATE cxx_std_14)