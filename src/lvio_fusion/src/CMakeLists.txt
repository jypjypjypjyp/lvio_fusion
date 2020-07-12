add_library(lvio_fusion SHARED
        frame.cpp
        mappoint.cpp
        map.cpp
        config.cpp
        frontend.cpp
        backend.cpp
        estimator.cpp
        navsat.cpp)

target_link_libraries(lvio_fusion ${THIRD_PARTY_LIBS})
target_compile_features(lvio_fusion PRIVATE cxx_std_17)