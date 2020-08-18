add_library(lvio_fusion SHARED
        backend.cpp
        config.cpp
        estimator.cpp
        frame.cpp
        frontend.cpp
        landmark.cpp
        map.cpp
        mapping.cpp
        navsat.cpp
        scan_registration.cpp)

target_link_libraries(lvio_fusion ${THIRD_PARTY_LIBS})
target_compile_features(lvio_fusion PRIVATE cxx_std_17)