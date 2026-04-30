add_executable(lut-bench-kernel lut-bench-kernel.cpp)
target_compile_features(lut-bench-kernel PRIVATE cxx_std_17)
target_link_libraries(lut-bench-kernel PRIVATE ggml ggml-lut)
target_include_directories(lut-bench-kernel PRIVATE
    ${CMAKE_SOURCE_DIR}/ggml/include
    ${CMAKE_SOURCE_DIR}/ggml/src
)

add_executable(lut-bench-suite lut-bench-suite.cpp)
target_compile_features(lut-bench-suite PRIVATE cxx_std_17)
target_link_libraries(lut-bench-suite PRIVATE ggml ggml-lut)
target_include_directories(lut-bench-suite PRIVATE
    ${CMAKE_SOURCE_DIR}/ggml/include
    ${CMAKE_SOURCE_DIR}/ggml/src
)

add_executable(lut-debug lut-debug.cpp)
target_compile_features(lut-debug PRIVATE cxx_std_17)
target_link_libraries(lut-debug PRIVATE ggml ggml-lut)
target_include_directories(lut-debug PRIVATE
    ${CMAKE_SOURCE_DIR}/ggml/include
    ${CMAKE_SOURCE_DIR}/ggml/src
)