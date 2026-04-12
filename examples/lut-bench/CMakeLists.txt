add_executable(lut-bench-kernel lut-bench-kernel.cpp)
target_link_libraries(lut-bench-kernel PRIVATE ggml ggml-lut)
target_include_directories(lut-bench-kernel PRIVATE
    ${CMAKE_SOURCE_DIR}/ggml/include
    ${CMAKE_SOURCE_DIR}/ggml/src        # block_q4_0, ggml-quants.h, ggml-common.h
)