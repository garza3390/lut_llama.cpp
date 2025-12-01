# llama.cpp - Referencia Completa de Comandos

## Tabla de Contenidos
1. [Comandos CMake (Build)](#comandos-cmake-build)
2. [llama-cli (Inferencia)](#llama-cli-inferencia)
3. [llama-server (Servidor HTTP)](#llama-server-servidor-http)
4. [llama-bench (Benchmarks)](#llama-bench-benchmarks)
5. [llama-quantize (Cuantización)](#llama-quantize-cuantización)
6. [llama-perplexity (Evaluación)](#llama-perplexity-evaluación)
7. [Testing y Validación](#testing-y-validación)

---

## Comandos CMake (Build)

### Build Básico (CPU-only)
```bash
# Configuración
cmake -B build

# Compilación
cmake --build build --config Release -j $(nproc)
```

### Build con Backends Específicos

#### CUDA (NVIDIA GPU)
```bash
cmake -B build-cuda -DGGML_CUDA=ON
cmake --build build-cuda --config Release -j $(nproc)
```

#### Metal (Apple Silicon)
```bash
cmake -B build-metal -DGGML_METAL=ON
cmake --build build-metal --config Release -j $(nproc)
```

#### Vulkan (GPU Cross-platform)
```bash
cmake -B build-vulkan -DGGML_VULKAN=ON
cmake --build build-vulkan --config Release -j $(nproc)
```

#### ROCm (AMD GPU)
```bash
cmake -B build-rocm -DGGML_HIPBLAS=ON
cmake --build build-rocm --config Release -j $(nproc)
```

#### SYCL (Intel GPU)
```bash
cmake -B build-sycl -DGGML_SYCL=ON
cmake --build build-sycl --config Release -j $(nproc)
```

### Opciones de CMake Comunes

| Opción | Descripción | Ejemplo |
|--------|-------------|---------|
| `-DCMAKE_BUILD_TYPE` | Tipo de build (Debug/Release) | `-DCMAKE_BUILD_TYPE=Debug` |
| `-DGGML_CUDA=ON` | Habilitar soporte CUDA | `-DGGML_CUDA=ON` |
| `-DGGML_METAL=ON` | Habilitar Metal (macOS) | `-DGGML_METAL=ON` |
| `-DGGML_VULKAN=ON` | Habilitar Vulkan | `-DGGML_VULKAN=ON` |
| `-DGGML_HIPBLAS=ON` | Habilitar ROCm/HIP | `-DGGML_HIPBLAS=ON` |
| `-DGGML_SYCL=ON` | Habilitar SYCL (Intel) | `-DGGML_SYCL=ON` |
| `-DGGML_BLAS=ON` | Habilitar OpenBLAS | `-DGGML_BLAS=ON` |
| `-DGGML_OPENBLAS=ON` | Usar OpenBLAS específicamente | `-DGGML_OPENBLAS=ON` |
| `-DGGML_AVX=OFF` | Deshabilitar AVX | `-DGGML_AVX=OFF` |
| `-DGGML_AVX2=OFF` | Deshabilitar AVX2 | `-DGGML_AVX2=OFF` |
| `-DGGML_FMA=OFF` | Deshabilitar FMA | `-DGGML_FMA=OFF` |
| `-DBUILD_SHARED_LIBS=ON` | Construir biblioteca compartida | `-DBUILD_SHARED_LIBS=ON` |
| `-DLLAMA_CURL=ON` | Habilitar soporte libcurl | `-DLLAMA_CURL=ON` |

### Build Debug
```bash
# Single-config generators (Unix Makefiles, Ninja)
cmake -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build -j $(nproc)

# Multi-config generators (Visual Studio, Xcode)
cmake -B build -G "Xcode"
cmake --build build --config Debug -j $(nproc)
```

### Limpieza y Rebuild
```bash
# Limpiar build directory
rm -rf build/

# Rebuild completo
cmake -B build
cmake --build build --config Release -j $(nproc)
```

---

## llama-cli (Inferencia)

### Comando Básico
```bash
./build/bin/llama-cli -m <modelo.gguf> -p "<prompt>"
```

### Parámetros Principales

#### Modelo y Entrada
| Parámetro | Descripción | Ejemplo |
|-----------|-------------|---------|
| `-m, --model` | Ruta al modelo GGUF | `-m models/llama-3.2-1b.gguf` |
| `-p, --prompt` | Texto de entrada/prompt | `-p "Hola, ¿cómo estás?"` |
| `-f, --file` | Leer prompt desde archivo | `-f prompt.txt` |
| `--in-prefix` | Prefijo antes del input del usuario | `--in-prefix "User: "` |
| `--in-suffix` | Sufijo después del input | `--in-suffix "\nAssistant: "` |

#### Generación y Muestreo
| Parámetro | Descripción | Valor por Defecto | Ejemplo |
|-----------|-------------|-------------------|---------|
| `-n, --n-predict` | Número de tokens a generar | 128 | `-n 512` |
| `--temp` | Temperatura de muestreo (0.0-2.0) | 0.8 | `--temp 0.7` |
| `--top-k` | Top-k sampling | 40 | `--top-k 50` |
| `--top-p` | Top-p (nucleus) sampling | 0.9 | `--top-p 0.95` |
| `--min-p` | Min-p sampling | 0.05 | `--min-p 0.1` |
| `--tfs` | Tail-free sampling | 1.0 | `--tfs 0.95` |
| `--typical` | Typical sampling | 1.0 | `--typical 0.95` |
| `--repeat-penalty` | Penalización por repetición | 1.1 | `--repeat-penalty 1.15` |
| `--repeat-last-n` | Tokens para penalización | 64 | `--repeat-last-n 128` |
| `--seed` | Semilla aleatoria | -1 (random) | `--seed 42` |

#### Control de Contexto
| Parámetro | Descripción | Valor por Defecto | Ejemplo |
|-----------|-------------|-------------------|---------|
| `-c, --ctx-size` | Tamaño del contexto | 512 | `-c 4096` |
| `-b, --batch-size` | Batch size para procesamiento | 512 | `-b 1024` |
| `--rope-freq-base` | Frecuencia base RoPE | 10000.0 | `--rope-freq-base 10000` |
| `--rope-freq-scale` | Escala de frecuencia RoPE | 1.0 | `--rope-freq-scale 0.5` |

#### GPU y Aceleración
| Parámetro | Descripción | Valor por Defecto | Ejemplo |
|-----------|-------------|-------------------|---------|
| `-ngl, --n-gpu-layers` | Capas a cargar en GPU | 0 | `-ngl 99` |
| `--main-gpu` | GPU principal a usar | 0 | `--main-gpu 1` |
| `--tensor-split` | División de tensores entre GPUs | - | `--tensor-split 3,1` |
| `--no-mmap` | No usar mmap para cargar modelo | false | `--no-mmap` |
| `--mlock` | Bloquear modelo en RAM | false | `--mlock` |
| `--numa` | Habilitar soporte NUMA | false | `--numa` |

#### Formato y Salida
| Parámetro | Descripción | Ejemplo |
|-----------|-------------|---------|
| `--color` | Colorear output | `--color` |
| `--no-display-prompt` | No mostrar prompt | `--no-display-prompt` |
| `-i, --interactive` | Modo interactivo | `-i` |
| `--interactive-first` | Modo interactivo después del prompt | `--interactive-first` |
| `-ins, --instruct` | Modo instrucción | `-ins` |
| `--multiline-input` | Input multilínea | `--multiline-input` |
| `--simple-io` | IO simple sin formateo | `--simple-io` |
| `-cnv, --conversation` | Modo conversación | `-cnv` |

#### Gramática y Restricciones
| Parámetro | Descripción | Ejemplo |
|-----------|-------------|---------|
| `--grammar` | Gramática GBNF inline | `--grammar 'root ::= "yes" | "no"'` |
| `--grammar-file` | Archivo de gramática GBNF | `--grammar-file grammar.gbnf` |
| `-j, --json-schema` | Esquema JSON para output | `--json-schema schema.json` |

#### Performance y Debug
| Parámetro | Descripción | Ejemplo |
|-----------|-------------|---------|
| `-t, --threads` | Número de threads | `-t 8` |
| `--verbose-prompt` | Mostrar prompt procesado | `--verbose-prompt` |
| `--log-disable` | Deshabilitar logging | `--log-disable` |
| `--log-file` | Archivo de log | `--log-file inference.log` |

### Ejemplos Completos

#### Inferencia Básica CPU
```bash
./build/bin/llama-cli \
  -m models/llama-3.2-1b-q4.gguf \
  -p "Explica qué es la inteligencia artificial" \
  -n 200 \
  --temp 0.7 \
  -c 2048
```

#### Inferencia con GPU (CUDA)
```bash
./build/bin/llama-cli \
  -m models/llama-3.2-3b-q4.gguf \
  -p "Escribe un poema sobre la tecnología" \
  -n 300 \
  -ngl 99 \
  --temp 0.8 \
  --top-p 0.95 \
  -c 4096
```

#### Modo Interactivo (Chat)
```bash
./build/bin/llama-cli \
  -m models/llama-3.2-1b-instruct-q4.gguf \
  -i \
  -cnv \
  -ngl 99 \
  --in-prefix "Usuario: " \
  --in-suffix "\nAsistente: " \
  -c 4096
```

#### Con Gramática JSON
```bash
./build/bin/llama-cli \
  -m models/llama-3.2-1b-q4.gguf \
  -p "Generate a person: " \
  -n 100 \
  --grammar 'root ::= "{" ws "\"name\"" ws ":" ws string "," ws "\"age\"" ws ":" ws number "}" ws
string ::= "\"" [^"]* "\""
number ::= [0-9]+
ws ::= [ \t\n]*' \
  -ngl 99
```

#### Con Prompt desde Archivo
```bash
./build/bin/llama-cli \
  -m models/llama-3.2-1b-q4.gguf \
  -f prompts/long-prompt.txt \
  -n 500 \
  -ngl 99 \
  --temp 0.6 \
  -c 8192
```

#### Multiple GPUs
```bash
./build/bin/llama-cli \
  -m models/llama-3.2-7b-q4.gguf \
  -p "Análisis detallado de..." \
  -n 1000 \
  -ngl 99 \
  --main-gpu 0 \
  --tensor-split 3,1 \
  -c 4096
```

---

## llama-server (Servidor HTTP)

### Comando Básico
```bash
./build/bin/llama-server -m <modelo.gguf> --port 8080
```

### Parámetros del Servidor

#### Configuración de Red
| Parámetro | Descripción | Valor por Defecto | Ejemplo |
|-----------|-------------|-------------------|---------|
| `--host` | Dirección de escucha | 127.0.0.1 | `--host 0.0.0.0` |
| `--port` | Puerto de escucha | 8080 | `--port 8000` |
| `--path` | Ruta base de la API | /v1 | `--path /api/v1` |

#### Modelo y Contexto
| Parámetro | Descripción | Ejemplo |
|-----------|-------------|---------|
| `-m, --model` | Ruta al modelo | `-m models/model.gguf` |
| `-a, --alias` | Alias del modelo | `-a "gpt-3.5-turbo"` |
| `-c, --ctx-size` | Tamaño de contexto | `-c 4096` |
| `-ngl, --n-gpu-layers` | Capas en GPU | `-ngl 99` |

#### Configuración de Requests
| Parámetro | Descripción | Valor por Defecto | Ejemplo |
|-----------|-------------|-------------------|---------|
| `-np, --parallel` | Requests paralelos | 1 | `-np 4` |
| `-cb, --cont-batching` | Continuous batching | false | `-cb` |
| `--slots` | Número de slots | 1 | `--slots 8` |
| `-t, --threads` | Threads por request | CPU cores/2 | `-t 8` |
| `-b, --batch-size` | Batch size | 512 | `-b 1024` |

#### API y Endpoints
| Parámetro | Descripción | Ejemplo |
|-----------|-------------|---------|
| `--metrics` | Habilitar endpoint /metrics | `--metrics` |
| `--embeddings` | Habilitar endpoint embeddings | `--embeddings` |
| `--log-format` | Formato de logs (text/json) | `--log-format json` |

### Ejemplos del Servidor

#### Servidor Básico
```bash
./build/bin/llama-server \
  -m models/llama-3.2-1b-q4.gguf \
  --host 0.0.0.0 \
  --port 8080 \
  -ngl 99 \
  -c 4096
```

#### Servidor con Múltiples Slots
```bash
./build/bin/llama-server \
  -m models/llama-3.2-3b-q4.gguf \
  --host 0.0.0.0 \
  --port 8080 \
  -ngl 99 \
  -np 8 \
  --slots 8 \
  -c 4096 \
  -cb
```

#### Con Métricas y Embeddings
```bash
./build/bin/llama-server \
  -m models/llama-3.2-1b-q4.gguf \
  --port 8080 \
  -ngl 99 \
  --metrics \
  --embeddings \
  --log-format json
```

### Uso del API (curl)

#### Completions (OpenAI-compatible)
```bash
curl http://localhost:8080/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Hola, ¿cómo estás?",
    "n_predict": 100,
    "temperature": 0.7,
    "top_p": 0.9
  }'
```

#### Chat Completions
```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "system", "content": "Eres un asistente útil."},
      {"role": "user", "content": "¿Qué es la IA?"}
    ],
    "temperature": 0.7,
    "max_tokens": 200
  }'
```

#### Embeddings
```bash
curl http://localhost:8080/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Texto para generar embedding"
  }'
```

#### Health Check
```bash
curl http://localhost:8080/health
```

#### Slots Info
```bash
curl http://localhost:8080/slots
```

---

## llama-bench (Benchmarks)

### Comando Básico
```bash
./build/bin/llama-bench -m <modelo.gguf>
```

### Parámetros de Benchmark

| Parámetro | Descripción | Ejemplo |
|-----------|-------------|---------|
| `-m, --model` | Modelo(s) a testear | `-m model1.gguf,model2.gguf` |
| `-p, --n-prompt` | Tamaños de prompt | `-p 512,1024,2048` |
| `-n, --n-gen` | Tokens a generar | `-n 128,256,512` |
| `-b, --batch-size` | Batch sizes | `-b 512,1024` |
| `-ngl, --n-gpu-layers` | Capas en GPU | `-ngl 0,32,99` |
| `-t, --threads` | Threads | `-t 1,2,4,8` |
| `-r, --repetitions` | Repeticiones | `-r 5` |
| `-o, --output` | Archivo de salida | `-o results.csv` |
| `--output-format` | Formato (csv/json/md) | `--output-format json` |

### Ejemplos de Benchmarks

#### Benchmark Básico
```bash
./build/bin/llama-bench \
  -m models/llama-3.2-1b-q4.gguf \
  -ngl 99
```

#### Benchmark Completo (CPU vs GPU)
```bash
./build/bin/llama-bench \
  -m models/llama-3.2-1b-q4.gguf \
  -p 512,1024,2048 \
  -n 128,256,512 \
  -ngl 0,99 \
  -r 3 \
  -o benchmark_results.csv
```

#### Comparación de Modelos
```bash
./build/bin/llama-bench \
  -m models/llama-3.2-1b-q4.gguf,models/llama-3.2-3b-q4.gguf \
  -p 1024 \
  -n 256 \
  -ngl 99 \
  --output-format json \
  -o comparison.json
```

#### Test de Threads
```bash
./build/bin/llama-bench \
  -m models/llama-3.2-1b-q4.gguf \
  -t 1,2,4,8,16 \
  -ngl 0 \
  -p 1024 \
  -n 256
```

---

## llama-quantize (Cuantización)

### Comando Básico
```bash
./build/bin/llama-quantize <input.gguf> <output.gguf> <tipo>
```

### Tipos de Cuantización

| Tipo | Bits | Descripción | Uso Recomendado |
|------|------|-------------|-----------------|
| `Q2_K` | 2-3 | Muy pequeño, pérdida alta | Pruebas/desarrollo |
| `Q3_K_S` | 3 | Pequeño, pérdida media | Dispositivos limitados |
| `Q3_K_M` | 3-4 | Balance pequeño/calidad | Uso general móvil |
| `Q3_K_L` | 3-4 | Mejor calidad Q3 | - |
| `Q4_0` | 4 | Legacy, rápido | Compatibilidad |
| `Q4_1` | 4 | Legacy, mejor que Q4_0 | Compatibilidad |
| `Q4_K_S` | 4 | Pequeño, buena velocidad | Recomendado 7B |
| `Q4_K_M` | 4-5 | Balance calidad/tamaño | Recomendado general |
| `Q5_0` | 5 | Legacy, alta calidad | Compatibilidad |
| `Q5_1` | 5 | Legacy, muy alta calidad | Compatibilidad |
| `Q5_K_S` | 5 | Alta calidad | Recomendado 13B |
| `Q5_K_M` | 5-6 | Muy alta calidad | Recomendado 13B+ |
| `Q6_K` | 6 | Calidad casi original | Producción crítica |
| `Q8_0` | 8 | Muy alta precisión | Máxima calidad |
| `F16` | 16 | Media precisión | Base para cuantizar |
| `F32` | 32 | Precisión completa | Original |

### Ejemplos de Cuantización

#### Cuantización Estándar (Q4)
```bash
./build/bin/llama-quantize \
  models/llama-3.2-1b-f16.gguf \
  models/llama-3.2-1b-q4_k_m.gguf \
  Q4_K_M
```

#### Alta Calidad (Q5)
```bash
./build/bin/llama-quantize \
  models/llama-3.2-3b-f16.gguf \
  models/llama-3.2-3b-q5_k_m.gguf \
  Q5_K_M
```

#### Máxima Compresión (Q2)
```bash
./build/bin/llama-quantize \
  models/llama-3.2-1b-f16.gguf \
  models/llama-3.2-1b-q2_k.gguf \
  Q2_K
```

#### Batch de Cuantizaciones
```bash
for quant in Q4_K_M Q5_K_M Q6_K; do
  ./build/bin/llama-quantize \
    models/source.gguf \
    models/output-${quant}.gguf \
    ${quant}
done
```

---

## llama-perplexity (Evaluación)

### Comando Básico
```bash
./build/bin/llama-perplexity -m <modelo.gguf> -f <dataset.txt>
```

### Parámetros

| Parámetro | Descripción | Ejemplo |
|-----------|-------------|---------|
| `-m, --model` | Modelo a evaluar | `-m model.gguf` |
| `-f, --file` | Dataset para evaluación | `-f wikitext.txt` |
| `-c, --ctx-size` | Tamaño de contexto | `-c 2048` |
| `-ngl, --n-gpu-layers` | Capas en GPU | `-ngl 99` |
| `--chunks` | Número de chunks | `--chunks 100` |

### Ejemplos de Evaluación

#### Perplexity Básico
```bash
./build/bin/llama-perplexity \
  -m models/llama-3.2-1b-q4.gguf \
  -f datasets/wikitext-2.txt \
  -ngl 99
```

#### Con Contexto Grande
```bash
./build/bin/llama-perplexity \
  -m models/llama-3.2-3b-q4.gguf \
  -f datasets/wikitext-2.txt \
  -c 4096 \
  -ngl 99
```

#### Comparación de Cuantizaciones
```bash
for model in models/*q*.gguf; do
  echo "Testing $model"
  ./build/bin/llama-perplexity \
    -m "$model" \
    -f datasets/wikitext-2.txt \
    -ngl 99
done
```

---

## Testing y Validación

### CTest (Test Suite)

#### Ejecutar Todos los Tests
```bash
ctest --test-dir build --output-on-failure -j $(nproc)
```

#### Ejecutar Test Específico
```bash
ctest --test-dir build -R test-tokenizer-0 --output-on-failure
```

#### Ver Lista de Tests
```bash
ctest --test-dir build --show-only
```

#### Tests con Verbose
```bash
ctest --test-dir build --verbose
```

### Tests del Servidor

#### Activar Entorno
```bash
source .venv/bin/activate
```

#### Ejecutar Tests del Servidor
```bash
cd tools/server/tests
./tests.sh
```

#### Test Específico del Servidor
```bash
cd tools/server/tests
pytest test_chat_completion.py -v
```

### Validación Local CI

#### CI Completo
```bash
bash ./ci/run.sh ./tmp/results ./tmp/mnt
```

### Pre-commit Hooks

#### Ejecutar Todos los Hooks
```bash
pre-commit run --all-files
```

#### Instalar Hooks
```bash
pre-commit install
```

---

## Ejemplos de Flujos Completos

### Workflow 1: Build y Test Rápido
```bash
# 1. Build
cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release -j $(nproc)

# 2. Test
ctest --test-dir build --output-on-failure

# 3. Inferencia rápida
./build/bin/llama-cli -m models/test.gguf -p "Hello" -n 10 -ngl 99
```

### Workflow 2: Comparación CPU vs GPU
```bash
# CPU benchmark
./build/bin/llama-bench \
  -m models/llama-3.2-1b-q4.gguf \
  -ngl 0 \
  -p 1024 \
  -n 256 \
  -r 3

# GPU benchmark
./build/bin/llama-bench \
  -m models/llama-3.2-1b-q4.gguf \
  -ngl 99 \
  -p 1024 \
  -n 256 \
  -r 3
```

### Workflow 3: Cuantización y Evaluación
```bash
# 1. Cuantizar
./build/bin/llama-quantize \
  models/source-f16.gguf \
  models/quantized-q4.gguf \
  Q4_K_M

# 2. Evaluar perplexity
./build/bin/llama-perplexity \
  -m models/quantized-q4.gguf \
  -f datasets/wikitext-2.txt \
  -ngl 99

# 3. Benchmark performance
./build/bin/llama-bench \
  -m models/quantized-q4.gguf \
  -ngl 99
```

### Workflow 4: Servidor en Producción
```bash
# Iniciar servidor
./build/bin/llama-server \
  -m models/llama-3.2-3b-q4.gguf \
  --host 0.0.0.0 \
  --port 8080 \
  -ngl 99 \
  -np 8 \
  --slots 8 \
  -c 4096 \
  -cb \
  --metrics \
  --log-format json > server.log 2>&1 &

# Guardar PID
echo $! > server.pid

# Test del servidor
curl http://localhost:8080/health

# Detener servidor
kill $(cat server.pid)
```

---

## Tips y Mejores Prácticas

### Performance
- Usa `-ngl 99` para cargar todo el modelo en GPU
- Ajusta `-t` según tus CPU cores (típicamente cores/2)
- Usa `-c` según tu VRAM disponible
- Habilita `--mlock` para evitar swapping en RAM

### Muestreo
- `--temp 0.7`: Balance creatividad/coherencia
- `--temp 0.1-0.3`: Respuestas más determinísticas
- `--temp 1.0-1.5`: Mayor creatividad
- `--top-p 0.9`: Buen default
- `--repeat-penalty 1.1`: Evita repeticiones

### GPU Multi-GPU
- `--tensor-split 3,1`: División 75%/25% entre 2 GPUs
- `--main-gpu 0`: GPU principal para procesamiento
- Divide según VRAM disponible en cada GPU

### Debugging
- `--verbose-prompt`: Ver cómo se procesa el prompt
- `--log-file`: Guardar logs para análisis
- `--simple-io`: Output más limpio para scripting

---

## Recursos Adicionales

- **Documentación**: `docs/` en el repositorio
- **Ejemplos**: `examples/` con 30+ casos de uso
- **Build docs**: `docs/build.md`
- **Backend docs**: `docs/backend/`
- **API docs**: `docs/api/`

---

## Notas de tu Sistema

**Hardware:**
- CPU: Intel i5-14400 (10 cores, 16 threads)
- GPU: NVIDIA RTX 4060 (8GB VRAM)
- RAM: 32GB
- OS: Ubuntu 24.04.1 LTS

**Recomendaciones para tu setup:**
```bash
# Build óptimo
cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release -j 16

# Threads recomendados
-t 8  # Mitad de threads disponibles

# GPU layers
-ngl 99  # Todo en GPU (8GB VRAM suficiente para modelos hasta ~7B Q4)

# Batch size óptimo
-b 1024  # Buen balance para tu VRAM
```