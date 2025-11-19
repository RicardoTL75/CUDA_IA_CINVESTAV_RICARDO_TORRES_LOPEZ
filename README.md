# CUDA_IA_CINVESTAV_RICARDO_TORRES_LOPEZ
CUDA_1, CUDA_2 AND CUDA_3

📊 Reporte Técnico: Programación CUDA con Numba
📋 Introducción
Este reporte documenta el trabajo realizado en tres ejercicios de programación paralela utilizando CUDA con Numba para ejecución en GPUs NVIDIA. Los códigos implementan desde operaciones básicas hasta algoritmos complejos de procesamiento de imágenes.

https://via.placeholder.com/800x200/4A90E2/FFFFFF?text=CUDA+Architecture+Grids+Blocks+Threads

🚀 ECU1 - Fundamentos de CUDA y Transferencia de Datos
🎯 Objetivo
Implementar un kernel CUDA básico para comprender el flujo de trabajo CPU-GPU y medir los tiempos de transferencia.

🔧 Implementación
@cuda.jit
def first_kernel(a, result):
    idx = cuda.grid(1)
    if idx < a.size:
        result[idx] = a[idx]

📊 Arquitectura de Ejecución
CPU Data > Transfer GPU > Kernel Execution > Transfer CPU > CPU Result

⏱️ Resultados de Performance
Operación	Tiempo	Porcentaje
CPU Computation	1.43 μs	-
GPU Transfer to Device	101.48 ms	63%
GPU Kernel Execution	43.72 ms	27%
GPU Transfer to Host	14.70 ms	9%
Total GPU Time	159.90 ms	100%

📈 Análisis
🔄 Proceso GPU vs CPU:
├── ⚡ CPU: Procesamiento inmediato (1.43μs)
└── 🎯 GPU: Overhead significativo por transferencias
    ├── 📤 Entrada: 101.48ms (63%)
    ├── ⚙️ Procesamiento: 43.72ms (27%)
    └── 📥 Salida: 14.70ms (9%)
Conclusión: Las transferencias de datos representan el mayor costo temporal, destacando la importancia de minimizar comunicaciones CPU-GPU.

🧮 ECU2 - Modelo de Hilos y Dimensiones
🎯 Objetivo
Explorar la organización jerárquica de hilos en CUDA (Grids, Blocks, Threads).

🔧 Conceptos Clave
Ejemplo 1: 1 Bloque × 8 Threads
Grid: [1 bloque]
Block: [8 threads]
Total: 8 hilos

Ejemplo 2: 2 Bloques × 4 Threads
Grid: [2 bloques]
Block: [4 threads cada uno]
Total: 8 hilos

🏗️ Estructura 2D/3D
# Configuración 2D
blocks_per_grid = (2, 2)      # 4 bloques total
threads_per_block = (4, 1)    # 4 hilos por bloque
# Total: 16 hilos

📐 Fórmulas de Indexación
Global ID = Block ID × Threads per Block + Thread ID
Block ID = blockIdx.x + blockIdx.y × gridDim.x
Thread Offset = threadIdx.x + threadIdx.y × blockDim.x

🎪 Visualización de Ejecución 2D
Ejemplo 2D (2×2 bloques, 4×1 threads):
┌─────────────┬─────────────┐
│ Bloque (0,0)│ Bloque (1,0)│
│ T0 T1 T2 T3 │ T0 T1 T2 T3 │
│ G0 G1 G2 G3 │ G4 G5 G6 G7 │
├─────────────┼─────────────┤
│ Bloque (0,1)│ Bloque (1,1)│
│ T0 T1 T2 T3 │ T0 T1 T2 T3 │
│ G8 G9 G10 G11│ G12 G13 G14 G15│
└─────────────┴─────────────┘

🔢 Salida del Kernel Whoami
020 | Block[x,y](0 0) = 4 | Thread[x,y](0 0) = 4
021 | Block[x,y](0 0) = 4 | Thread[x,y](1 0) = 5
...
035 | Block[x,y](1 1) = 7 | Thread[x,y](3 0) = 7

Observación: Se evidencia el cálculo correcto de IDs globales a partir de las coordenadas 2D.

⚡ ECU3 - Algoritmos Paralelos Avanzados
🎯 Objetivo
Implementar algoritmos computacionalmente intensivos y comparar performance CPU vs GPU.

📊 Benchmark de Algoritmos
1. 🧮 Vector Addition
@cuda.jit
def vector_add_kernel(a, b, c):
    idx = cuda.grid(1)
    if idx < c.size:
        c[idx] = a[idx] + b[idx]

⚡ Resultados:
🎯 GPU: ~2.5ms
💻 CPU NumPy: ~15ms
🚀 Speedup: 6x

2. 📐 Matrix Scaling (2D)
@cuda.jit
def matrix_scale_kernel(mat, scalar, out):
    row, col = cuda.grid(2)
    if row < out.shape[0] and col < out.shape[1]:
        out[row, col] = mat[row, col] * scalar

Configuración:

Matriz: 4096×4096 (16.7M elementos)
Threads: (32, 32) por bloque
Blocks: (128, 128) en grid

⚡ Resultados:
🎯 GPU: ~15ms
💻 CPU NumPy: ~45ms
🚀 Speedup: 3x

3. 🔢 Matrix Multiplication
@cuda.jit
def matmul_naive_kernel(A, B, C):
    row, col = cuda.grid(2)
    if row < M and col < N:
        total = 0.0
        for k in range(K):
            total += A[row, k] * B[k, col]
        C[row, col] = total

⚡ Resultados:
🎯 GPU: ~250ms
💻 CPU NumPy: ~500ms
🚀 Speedup: 2x

4. 🖼️ Sobel Edge Detection
@cuda.jit
def sobel_kernel(img, out):
    row, col = cuda.grid(2)
    if 0 < row < H-1 and 0 < col < W-1:
        # Cálculo de gradientes Gx y Gy
        gx = (-img[row-1,col-1] + img[row-1,col+1] 
              -2*img[row,col-1] + 2*img[row,col+1]
              -img[row+1,col-1] + img[row+1,col+1])
        gy = (-img[row-1,col-1] - 2*img[row-1,col] - img[row-1,col+1]
              + img[row+1,col-1] + 2*img[row+1,col] + img[row+1,col+1])
        out[row, col] = (gx*gx + gy*gy)**0.5

⚡ Resultados (Imagen 4K):
🎯 GPU: ~8ms
💻 CPU OpenCV: ~25ms
🚀 Speedup: 3.1x

📈 Resumen Comparativo de Performance
graph TD
    A[Operaciones CUDA] --> B[Vector Add]
    A --> C[Matrix Scale]
    A --> D[Matrix Multiply]
    A --> E[Sobel Filter]
    
    B --> F[Speedup: 6x]
    C --> G[Speedup: 3x]
    D --> H[Speedup: 2x]
    E --> I[Speedup: 3.1x]

🎯 Análisis de Patrones de Acceso
Algoritmo	Patrón Acceso	Eficiencia	Bottleneck
Vector Add	Coalescido	Alta	Ancho de banda
Matrix Scale	Coalescido	Alta	Ancho de banda
Matrix Mult	Estríado	Media	Latencia memoria
Sobel	Local	Alta	Cálculos

🏆 Conclusiones Generales
✅ Logros Alcanzados
🎯 Dominio Conceptual: Comprensión profunda del modelo de programación CUDA
⚡ Optimización: Implementación eficiente de kernels para diferentes cargas de trabajo
📊 Análisis: Capacidad para identificar cuellos de botella y oportunidades de optimización
🛠️ Versatilidad: Aplicación en múltiples dominios (álgebra lineal, procesamiento de imágenes)

🔧 Lecciones Aprendidas
Las transferencias CPU-GPU son costosas → Minimizar comunicaciones
La organización de hilos afecta performance → Elegir grid/block size apropiado
Patrones de acceso a memoria son cruciales → Buscar coalescencia
Kernels simples pueden superar a CPU para operaciones paralelizables

🚀 Recomendaciones para Futuros Trabajos
Usar memoria compartida para algoritmos como matrix multiplication
Implementar tiling para mejor utilización de caché
Experimentar con diferentes configuraciones de blocks/threads
Considerar uso de streams para operaciones concurrentes

📚 Recursos Técnicos
🔗 Librerías Utilizadas
numba-cuda==0.4.0
numpy
opencv-python
pynvjitlink-cu12

🖥️ Hardware
GPU: NVIDIA T4
Entorno: Google Colab

📖 Referencias
Documentación oficial de Numba CUDA
NVIDIA CUDA Programming Guide
Best Practices for CUDA C++ Programming
-------------------------------------------------------------
🎓 Elaborado por: Ricardo Torres
📅 Fecha: Noviembre 2024
🏷️ Tecnologías: CUDA, Numba, Python, NVIDIA GPU