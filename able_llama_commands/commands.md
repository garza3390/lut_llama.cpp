# DELETE bin folder

rm -rf build

# Configure with LUT support
cmake -B build -DGGML_LUT=ON -DGGML_CUDA=OFF

# Build
cmake --build build --config Release -j $(nproc)

# Run tests
ctest --test-dir build -R lut --output-on-failure

# Run benchmark
./build/bin/lut-bench-kernel
