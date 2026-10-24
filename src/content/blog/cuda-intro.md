---
title: "CUDA 入门：从零手敲 6 关实战笔记"
subtitle: "在 NVIDIA H20 (Hopper sm_90, CUDA 12.9) 上从第一个核函数出发，实测验证 SIMT 线程束硬件调度、显存隔离与通信、Grid-Stride Loop 合并访存、Thrust 算子传递与原子锁实现"
description: "围绕 6 个递进的 CUDA 实验程序（01_hello.cu ～ 06_atomic.cu），完整记录 Host/Device 分离编译、Block/Warp/Thread 三级硬件执行顺序 1000 次统计验证、cudaMemcpy 与 Unified Memory 对比、Grid-Stride Loop 合并访问原理、Functor/Lambda 算子传递以及 atomicAdd 与手搓 atomicCAS 乐观锁。"
pubDate: 2026-10-23
dateRange: "2026.10"
tags: ["CUDA", "GPU 体系结构", "H20 (sm_90)", "C++17", "并行计算", "Thrust"]
---

# 第一个CUDA程序

## 准备工作

### 查看硬件，驱动，nvcc，c++版本

```bash
# 看显卡型号和驱动支持的最高CUDA版本
nvidia-smi
```

![nvidia-smi 显卡与驱动信息](/cuda_tutorial/images/nvidia-smi.png)

```bash
# 看 CUDA 编译器（nvcc）版本
nvcc --version
```

```text
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2025 NVIDIA Corporation
Built on Tue_May_27_02:21:03_PDT_2025
Cuda compilation tools, release 12.9, V12.9.86
Build cuda_12.9.r12.9/compiler.36037853_0
```

```bash
# 看 CPU C++ 编译器版本
g++ --version
```

```text
g++ (GCC) 13.3.1 20240611 (Red Hat 13.3.1-2)
Copyright (C) 2023 Free Software Foundation, Inc.
This is free software; see the source for copying conditions.  There is NO
warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
```

**背后逻辑**：一份 cuda 代码（后缀 `.cu`）里面同时混合 CPU 代码（Host）和 GPU 代码（Device）。

`nvcc` 不会从头编译所有东西，而是把 `.cu` 拆成两半：GPU 部分由 `nvcc` 编译成 GPU 指令，CPU 部分转交 `g++` 编译，最后链接成一个可执行文件。因此必须同时有 `nvcc` 和 `g++`。

## 第一个程序，区分 Host 和 Device

| 修饰符 | 在哪跑 | 谁能调用它 | 备注 |
| --- | --- | --- | --- |
| **`__host__`** | CPU（主机） | 只能 CPU 调 | 不写修饰符时的默认值（比如 `main`） |
| **`__device__`** | GPU（设备） | 只能 GPU 调 | GPU 内部辅助函数 |
| **`__global__`** | GPU（设备） | CPU 调，GPU 跑 | 也叫 核函数（Kernel），是 CPU 启动 GPU 干活的入口，返回值必须是 `void` |

```cpp
#include <cstdio>

__device__ void gpu_hello() {
    printf("gpu hello!\n");
}

__host__ void cpu_hello() {
    printf("cpu hello!\n");
}

__global__ void kernel() {
    gpu_hello();
}

int main() {
    // 核函数的<<<...>>>用于向 GPU 申请计算资源
    kernel<<<1, 2>>>();
    // 等待
    cudaDeviceSynchronize();
    cpu_hello();
    return 0;
}
```

```bash
# nvcc 编译运行
nvcc 01_hello.cu -o 01_hello
./01_hello

## 输出内容
gpu hello!
gpu hello!
cpu hello!
```

### 实验A：验证 CPU 和 GPU 异步执行

把 `cudaDeviceSynchronize()` 去掉，重新 `nvcc 01_hello.cu -o 01_hello && ./01_hello`

现象：只打印 `cpu hello!` ，因为 GPU 还没来得及输出，CPU 就跑完了 `return 0`，整个进程退出了。

### 实验B：查看 GPU 算力架构号（Compute Capability）和 CMake 配置

每代 NVIDIA 显卡都有一个算力版本号（比如老卡 GTX 1080 是 `6.1` 即 `610`，我们这台 H20 是 Hopper 架构 `9.0` 即 `900`）。如果不指定架构，编译器会按默认的老架构编译，用不上新卡的硬件特性。

1. `vim 01_hello.cu`，把 `gpu_hello()` 改成打印内置宏 **`__CUDA_ARCH__`**

```cpp
__device__ void gpu_hello() {
#ifdef __CUDA_ARCH__
    printf("gpu hello! arch = %d\n", __CUDA_ARCH__);
#endif
}
```

2. 分别用不带参数和指定 `-arch=sm_90` 编译运行，对比输出的 `arch` 数字

```bash
nvcc 01_hello.cu -o 01_hello && ./01_hello
# 输出
# 520 代表算力版本 5.2（2014 年发布的 GTX 980 Maxwell 架构）。CUDA 12 的 nvcc 在你不指定架构时，默认按它所支持的最老硬件架构 sm_52 生成代码，保证十年前的显卡也能跑，但 H20（算力 9.0，即 900）的新硬件指令就全浪费了。
gpu hello! arch = 520
gpu hello! arch = 520
cpu hello!


nvcc -arch=sm_90 01_hello.cu -o 01_hello && ./01_hello
# 输出
gpu hello! arch = 900
gpu hello! arch = 900
cpu hello!
```

3. 工程一般用 CMake 管理编译。用 `vim CMakeLists.txt` 写一个最小配置

如果只写一个文件，直接敲 `nvcc -std=c++17 -arch=sm_90 01_hello.cu -o 01_hello` 最省事。但工程一旦变大（有几十个 `.cu` / `.cpp`、要引很多外部头文件目录和库），每次手敲一长串编译命令很容易漏参数，而且改了一个文件就要把所有文件重新编一遍。

```cmake
# 要求机器上的 cmake 版本至少是 3.18
# 因为从 3.18 开始，CMake 才原生支持下面 CMAKE_CUDA_ARCHITECTURES 这个配置项
cmake_minimum_required(VERSION 3.18)

# set(变量名 值) 是给 CMake 内置的开关赋值：
set(CMAKE_CXX_STANDARD 17)                 # 相当于给 g++ 自动加上 -std=c++17
set(CMAKE_CUDA_STANDARD 17)                # 相当于给 nvcc 自动加上 -std=c++17
set(CMAKE_CUDA_ARCHITECTURES 90)           # 相当于给 nvcc 自动加上 -arch=sm_90（不写就会退回 520）

# 定义项目名（名字随便起），LANGUAGES 声明本项目同时用到 C++ (CXX) 和 CUDA
project(cuda_tutorial LANGUAGES CXX CUDA)

# add_executable(生成的可执行文件名 源码文件1 源码文件2 ...)
add_executable(01_hello 01_hello.cu)
```

然后用 `cmake` 编译：

- `-S .`：Source（源码目录）是当前目录 `.`，让 CMake 在 `.` 下面找 `CMakeLists.txt`。
- `-B build`：Build（构建目录）指定为 `build` 文件夹（不存在会自动创建）。把 CMake 生成的一大堆中间文件全隔离在 `build/` 里，不弄脏你的源码目录；万一配置乱了，直接 `rm -rf build` 就能重来。

```bash
cmake -S . -B build # 生成编译规则
cmake --build build # 去build/目录里读取规则，调用nvcc和g++把可执行文件编译出来，产物放在./build/01_hello
./build/01_hello
```

---

# 第二关：Block 和 Thread

拆解一下 `kernel<<<1, 2>>>` 的参数

## 为什么要分 Block 和 Thread 两层

![GPU SM / Block / Thread 硬件层次结构](/cuda_tutorial/images/gpu-sm-block-thread.png)

- 一张 GPU 芯片内部由很多个 **SM（Streaming Multiprocessor，流多处理器）** 组成。
- 每个 SM 内部又包含很多个 **SP（也就是常说的 CUDA Core）** 和一块共享缓存。
- `<<<参数1, 参数2>>>`：
  - **参数 1（Block 数量）**：把大任务切成多少个「线程块（Block）」，丢给各个 SM 去领。在核函数里用内置变量 `gridDim.x` 读取总共有几个 Block，用 `blockIdx.x` 读取当前是第几个 Block（从 0 开始）。
  - **参数 2（每个 Block 里的 Thread 数量）**：每个 Block 内部开多少个「线程（Thread）」，交给 SM 里的 CUDA Core 并行跑。在核函数里用 `blockDim.x` 读取每块有几个线程，用 `threadIdx.x` 读取当前是块内的第几个线程（从 0 开始）。

如果共有 4 个 Block，每个 Block 有 3 个 Thread（共 `4 * 3 = 12` 个线程），要算出当前线程在全局 12 个线程里排第几（Flatten Mode 压平编号），公式就像「几单元几零几室」换算成总户号一样：

$$
\text{全局线程号} = \texttt{blockIdx.x} \times \texttt{blockDim.x} + \texttt{threadIdx.x}
$$

## 动手写 `02_grid.cu`：一维与三维网格

1. 在机器上 `vim 02_grid.cu`，敲入下面的代码：

```cpp
#include <cstdio>

__global__ void kernel_1d() {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = gridDim.x * blockDim.x;
    printf("Block %d/%d, Thread %d/%d -> Global Thread %d/%d\n",
           blockIdx.x, gridDim.x,
           threadIdx.x, blockDim.x,
           tid, total);
}

int main() {
    kernel_1d<<<4, 3>>>();
    cudaDeviceSynchronize();
    return 0;
}
```

打开 `vim CMakeLists.txt`，在文件最后加一行：`add_executable(02_grid 02_grid.cu)`

编译并连续运行两三次：

```bash
cmake --build build
./build/02_grid
```

```text
# 手动执行多次发现顺序一致
Block 3/4, Thread 0/3 -> Global Thread 9/12
Block 3/4, Thread 1/3 -> Global Thread 10/12
Block 3/4, Thread 2/3 -> Global Thread 11/12
Block 2/4, Thread 0/3 -> Global Thread 6/12
Block 2/4, Thread 1/3 -> Global Thread 7/12
Block 2/4, Thread 2/3 -> Global Thread 8/12
Block 0/4, Thread 0/3 -> Global Thread 0/12
Block 0/4, Thread 1/3 -> Global Thread 1/12
Block 0/4, Thread 2/3 -> Global Thread 2/12
Block 1/4, Thread 0/3 -> Global Thread 3/12
Block 1/4, Thread 1/3 -> Global Thread 4/12
Block 1/4, Thread 2/3 -> Global Thread 5/12
```

## 三维网格 `dim3`（Dimension Mode）

GPU 最初是画图（2D 像素）和算三维空间（3D 网格）的，所以 `gridDim`、`blockIdx`、`blockDim`、`threadIdx` 天生都是带 `.x, .y, .z` 的三维结构体（类型叫 `dim3`）。
刚才写的 `<<<4, 3>>>` 其实是 `<<<dim3(4, 1, 1), dim3(3, 1, 1)>>>` 的简写。

把 `02_grid.cu` 的 `main` 和核函数改成三维（比如 `2x3` 个 Block，每个 Block 里 `2x1` 个线程）：

```cpp
#include <cstdio>

__global__ void kernel_3d() {
    printf("Block (%d,%d,%d) of (%d,%d,%d), Thread (%d,%d,%d) of (%d,%d,%d)\n",
           blockIdx.x, blockIdx.y, blockIdx.z,
           gridDim.x, gridDim.y, gridDim.z,
           threadIdx.x, threadIdx.y, threadIdx.z,
           blockDim.x, blockDim.y, blockDim.z);
}

int main() {
    kernel_3d<<<dim3(2, 3, 1), dim3(2, 1, 1)>>>();
    cudaDeviceSynchronize();
    return 0;
}
```

重新 `cmake --build build && ./build/02_grid` 看一眼输出：

```text
# 手动执行多次，第一次和第二次不同，第二次之后每次顺序一致
Block (1, 1, 0) of (2, 3, 1), Thread (0, 0, 0) of (2, 1, 1)
Block (1, 1, 0) of (2, 3, 1), Thread (1, 0, 0) of (2, 1, 1)
Block (1, 2, 0) of (2, 3, 1), Thread (0, 0, 0) of (2, 1, 1)
Block (1, 2, 0) of (2, 3, 1), Thread (1, 0, 0) of (2, 1, 1)
Block (1, 0, 0) of (2, 3, 1), Thread (0, 0, 0) of (2, 1, 1)
Block (1, 0, 0) of (2, 3, 1), Thread (1, 0, 0) of (2, 1, 1)
Block (0, 1, 0) of (2, 3, 1), Thread (0, 0, 0) of (2, 1, 1)
Block (0, 1, 0) of (2, 3, 1), Thread (1, 0, 0) of (2, 1, 1)
Block (0, 0, 0) of (2, 3, 1), Thread (0, 0, 0) of (2, 1, 1)
Block (0, 0, 0) of (2, 3, 1), Thread (1, 0, 0) of (2, 1, 1)
Block (0, 2, 0) of (2, 3, 1), Thread (0, 0, 0) of (2, 1, 1)
Block (0, 2, 0) of (2, 3, 1), Thread (1, 0, 0) of (2, 1, 1)
```

### <mark class="hl-red">执行顺序问题</mark>：

- **SIMT（单指令多线程）：为什么每个 Block 内部的 Thread 永远是 `0 -> 1 -> 2`**
  - GPU 硬件调度的最小单位叫 **Warp（线程束）**，固定包含 32 个线程（`threadIdx` `0~31`）。
  - 上面实验每个 Block 只有 2 或 3 个线程，全在同一个 Warp 里。同一个 Warp 的所有线程在同一纳秒执行同一条 `printf` 指令，硬件会严格按线程号 `0 -> 1 -> 2` 打包写入缓冲区，100% 不会乱序。
- **会乱序的是不同 Warp 之间，和不同 Block 之间**
  - **同一个 Block（同一个 SM）内部的不同 Warp**：
    一个 SM 内部有 4 个独立的 Warp 调度器（Warp Scheduler）。当你启动 `<<<1, 256>>>`（共 $256 / 32 = 8$ 个 Warp）时：
    - SM 的 4 个调度器会先同时抓起前 4 个 Warp（`Warp 0, 1, 2, 3`）一起跑，这 4 个 Warp 互相争抢 `printf` 缓冲区；
    - 然后再抓起后 4 个 Warp（`Warp 4, 5, 6, 7`）跑。
  - **不同 Block（不同 SM）之间的 Warp**：
    当你启动多个 Block（比如 `<<<4, 64>>>`）时，4 个 Block 分散在 4 个不同的物理 SM 上，它们各自的 Warp 完全独立运行，谁先抢到缓冲区谁先打。

| 场景 | 谁和谁比？ | 顺序是确定的还是随机的？ | 为什么？ |
| --- | --- | --- | --- |
| **实验 1** | 同一个 Warp 内的 32 个线程<br/>（走同一条指令，无分支） | **100% 确定**<br/>（永远 `0->1->...->31`） | 32 个线程在同一纳秒同时执行同一条指令（时间差 `0 ns`），硬件按通道号 `0~31` 顺次编号 |
| **实验 2** | 同一个 Warp 内的 32 个线程<br/>（奇偶线程走 `if/else` 不同分支） | **100% 确定，但打破了 `0~31`**<br/>（永远先跑全体奇数，再跑全体偶数） | 同一个 Warp 同一时刻只能走一条分支，硬件只能先把偶数线程挂起、跑完奇数分支，再回来跑偶数分支 |
| **实验 3** | 同一个 Block 内的不同 Warp<br/>（比如 1 个 Block 里有 16 个 Warp） | **随机 / 不确定**<br/>（跑 1000 次会出现 40+ 种顺序） | 同一个 SM 里有 4 个 Warp 调度器并发抢资源，谁先抢到内存总线谁先走 |
| **实验 4** | 不同 Block 之间<br/>（比如 6 个 Block 分散在 6 个 SM） | **随机 / 不确定**<br/>（跑 1000 次会出现 70+ 种顺序） | 各个物理 SM 完全独立并行，没有任何先后顺序保证 |

```cpp
#include <cstdio>
#include <cstdint>
#include <string>
#include <map>
#include <vector>
#include <algorithm>

// 读取 GPU 硬件纳秒时钟
__device__ uint64_t get_globaltimer() {
    uint64_t t;
    asm volatile("mov.u64 %0, %globaltimer;" : "=l"(t));
    return t;
}

// 显存中的全局排队叫号机：谁先执行 atomicAdd(&g_counter, 1)，谁就领到更小的名次 pos (0, 1, 2...)
__device__ int g_counter;

// 【实验 1】同一个 Warp 内部（32 个线程，无分支）
__global__ void k1_intra_warp_same(int *order_out, uint64_t *time_out) {
    uint64_t t = get_globaltimer();
    int pos = atomicAdd(&g_counter, 1);
    order_out[pos] = threadIdx.x;
    time_out[threadIdx.x] = t;
}

// 【实验 2】同一个 Warp 内部（32 个线程，if/else 奇偶分支）
__global__ void k2_intra_warp_diverged(int *order_out, uint64_t *time_out) {
    if (threadIdx.x % 2 == 1) {
        uint64_t t = get_globaltimer();
        int pos = atomicAdd(&g_counter, 1);
        order_out[pos] = threadIdx.x;
        time_out[threadIdx.x] = t;
    } else {
        uint64_t t = get_globaltimer();
        int pos = atomicAdd(&g_counter, 1);
        order_out[pos] = threadIdx.x;
        time_out[threadIdx.x] = t;
    }
}

// 【实验 3】同一个 Block 内的 16 个不同 Warp（<<<1, 512>>>，每个 Warp 派 0 号代表领号）
__global__ void k3_inter_warp(int *order_out) {
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    if (lane_id == 0) {
        int pos = atomicAdd(&g_counter, 1);
        order_out[pos] = warp_id;
    }
}

// 【实验 4】6 个不同 Block 之间（<<<6, 32>>>，每个 Block 派 0 号代表领号）
__global__ void k4_inter_block(int *order_out) {
    if (threadIdx.x == 0) {
        int pos = atomicAdd(&g_counter, 1);
        order_out[pos] = blockIdx.x;
    }
}

// 统计打印辅助函数：按出现次数从高到低排，打印前 top_k 种
void print_stats(const std::map<std::string, int> &counts, int total_runs, int top_k = 5) {
    std::vector<std::pair<std::string, int>> vec(counts.begin(), counts.end());
    std::sort(vec.begin(), vec.end(), [](const auto &a, const auto &b) {
        return a.second > b.second;
    });
    const char *verdict = (vec.size() == 1) ? "【完全确定】" : "【随机/不确定】";
    printf("  => 结论: %s 跑 %d 次共出现 【%zu 种】 不同顺序\n", verdict, total_runs, vec.size());
    int shown = 0;
    for (const auto &kv : vec) {
        if (shown++ >= top_k) {
            printf("     ... (其余 %zu 种低频顺序省略)\n", vec.size() - top_k);
            break;
        }
        printf("     第 %d 多 (%4d/%d, %5.1f%%): %s\n",
               shown, kv.second, total_runs, 100.0 * kv.second / total_runs, kv.first.c_str());
    }
}

int main() {
    const int N_RUNS = 1000;
    int *order;
    uint64_t *timestamps;
    cudaMallocManaged(&order, 64 * sizeof(int));
    cudaMallocManaged(&timestamps, 64 * sizeof(uint64_t));
    int zero = 0;

    // ================== 1. 同一个 Warp 内（无分支） ==================
    printf("======================================================================\n");
    printf("实验 1：同一个 Warp 内的 32 个线程（走同一条指令，连跑 %d 次）\n", N_RUNS);
    std::map<std::string, int> stat1;
    uint64_t max_time_diff_1 = 0;
    for (int r = 0; r < N_RUNS; ++r) {
        cudaMemcpyToSymbol(g_counter, &zero, sizeof(int));
        k1_intra_warp_same<<<1, 32>>>(order, timestamps);
        cudaDeviceSynchronize();

        std::string s;
        for (int i = 0; i < 6; ++i) s += std::to_string(order[i]) + "->";
        s += "...->" + std::to_string(order[31]);
        stat1[s]++;

        for (int i = 1; i < 32; ++i) {
            uint64_t diff = (timestamps[i] > timestamps[0])
                          ? (timestamps[i] - timestamps[0])
                          : (timestamps[0] - timestamps[i]);
            if (diff > max_time_diff_1) max_time_diff_1 = diff;
        }
    }
    print_stats(stat1, N_RUNS);
    printf("  => 硬件证据: 32 个线程同一轮的时间戳最大差值 = %llu ns (同一纳秒锁步同时执行)\n\n",
           (unsigned long long)max_time_diff_1);

    // ================== 2. 同一个 Warp 内（if/else 分支） ==================
    printf("======================================================================\n");
    printf("实验 2：同一个 Warp 内的 32 个线程（奇偶走 if/else 不同分支，连跑 %d 次）\n", N_RUNS);
    std::map<std::string, int> stat2;
    uint64_t branch_gap_ns = 0;
    for (int r = 0; r < N_RUNS; ++r) {
        cudaMemcpyToSymbol(g_counter, &zero, sizeof(int));
        k2_intra_warp_diverged<<<1, 32>>>(order, timestamps);
        cudaDeviceSynchronize();

        std::string s = "先跑奇数[";
        for (int i = 0; i < 3; ++i) s += std::to_string(order[i]) + ",";
        s += ".." + std::to_string(order[15]) + "] -> 后跑偶数[";
        for (int i = 16; i < 19; ++i) s += std::to_string(order[i]) + ",";
        s += ".." + std::to_string(order[31]) + "]";
        stat2[s]++;

        branch_gap_ns = (timestamps[0] > timestamps[1])
                      ? (timestamps[0] - timestamps[1])
                      : (timestamps[1] - timestamps[0]);
    }
    print_stats(stat2, N_RUNS);
    printf("  => 硬件证据: 奇数分支与偶数分支时间戳相差 = %llu ns (分支内部有序，但两个分支被硬件串行拆开跑)\n\n",
           (unsigned long long)branch_gap_ns);

    // ================== 3. 同一个 Block 内的不同 Warp ==================
    printf("======================================================================\n");
    printf("实验 3：同一个 Block 内的 16 个不同 Warp（<<<1, 512>>> 同一个 SM，连跑 %d 次）\n", N_RUNS);
    std::map<std::string, int> stat3;
    for (int r = 0; r < N_RUNS; ++r) {
        cudaMemcpyToSymbol(g_counter, &zero, sizeof(int));
        k3_inter_warp<<<1, 512>>>(order);
        cudaDeviceSynchronize();

        std::string s;
        for (int i = 0; i < 16; ++i) {
            s += "W" + std::to_string(order[i]) + (i == 15 ? "" : "->");
        }
        stat3[s]++;
    }
    print_stats(stat3, N_RUNS);
    printf("\n");

    // ================== 4. 不同 Block 之间 ==================
    printf("======================================================================\n");
    printf("实验 4：6 个不同 Block 之间（<<<6, 32>>> 分散在 6 个不同 SM，连跑 %d 次）\n", N_RUNS);
    std::map<std::string, int> stat4;
    for (int r = 0; r < N_RUNS; ++r) {
        cudaMemcpyToSymbol(g_counter, &zero, sizeof(int));
        k4_inter_block<<<6, 32>>>(order);
        cudaDeviceSynchronize();

        std::string s;
        for (int i = 0; i < 6; ++i) {
            s += "B" + std::to_string(order[i]) + (i == 5 ? "" : "->");
        }
        stat4[s]++;
    }
    print_stats(stat4, N_RUNS);

    cudaFree(order);
    cudaFree(timestamps);
    return 0;
}
```

```text
======================================================================
实验 1：同一个 Warp 内的 32 个线程（走同一条指令，连跑 1000 次）
  => 结论: 【完全确定】 跑 1000 次共出现 【1 种】 不同顺序
     第 1 多 (1000/1000, 100.0%): 0->1->2->3->4->5->...->31
  => 硬件证据: 32 个线程同一轮的时间戳最大差值 = 0 ns (同一纳秒锁步同时执行)

======================================================================
实验 2：同一个 Warp 内的 32 个线程（奇偶走 if/else 不同分支，连跑 1000 次）
  => 结论: 【完全确定】 跑 1000 次共出现 【1 种】 不同顺序
     第 1 多 (1000/1000, 100.0%): 先跑奇数[1,3,5,..31] -> 后跑偶数[0,2,4,..30]
  => 硬件证据: 奇数分支与偶数分支时间戳相差 = 800 ns (分支内部有序，但两个分支被硬件串行拆开跑)

======================================================================
实验 3：同一个 Block 内的 16 个不同 Warp（<<<1, 512>>> 同一个 SM，连跑 1000 次）
  => 结论: 【随机/不确定】 跑 1000 次共出现 【44 种】 不同顺序
     第 1 多 ( 345/1000,  34.5%): W12->W13->W14->W15->W8->W9->W10->W11->W4->W5->W6->W0->W7->W1->W2->W3
     第 2 多 ( 232/1000,  23.2%): W0->W1->W2->W3->W4->W5->W6->W7->W8->W9->W10->W11->W12->W13->W14->W15
     第 3 多 (  97/1000,   9.7%): W12->W13->W14->W15->W8->W9->W10->W11->W5->W4->W6->W7->W1->W0->W2->W3
     第 4 多 (  63/1000,   6.3%): W4->W5->W6->W7->W8->W9->W10->W11->W0->W2->W1->W3->W12->W13->W14->W15
     第 5 多 (  39/1000,   3.9%): W14->W15->W0->W1->W2->W3->W4->W5->W6->W7->W8->W9->W10->W11->W12->W13
     ... (其余 39 种低频顺序省略)

======================================================================
实验 4：6 个不同 Block 之间（<<<6, 32>>> 分散在 6 个不同 SM，连跑 1000 次）
  => 结论: 【随机/不确定】 跑 1000 次共出现 【73 种】 不同顺序
     第 1 多 ( 150/1000,  15.0%): B0->B4->B3->B5->B1->B2
     第 2 多 ( 101/1000,  10.1%): B3->B5->B0->B4->B1->B2
     第 3 多 (  98/1000,   9.8%): B2->B4->B5->B3->B0->B1
     第 4 多 (  85/1000,   8.5%): B5->B3->B2->B4->B0->B1
     第 5 多 (  71/1000,   7.1%): B5->B3->B2->B0->B1->B4
     ... (其余 68 种低频顺序省略)
```

**为什么跨 SM 比同一个 SM 执行更容易乱序**

实验 3 中最常出现的顺序：`W12->W13->W14->W15 -> W8->W9->W10->W11 -> ...` 或者 `W0->W1->W2->W3 -> W4->W5->W6->W7 -> ...`
几乎全是以 `(W0,W1,W2,W3)`、`(W4,W5,W6,W7)`、`(W8,W9,W10,W11)`、`(W12,W13,W14,W15)` 四人小组抱团出现的。

实验 4（6 个不同 SM 上的 6 个 Block）：只有 6 个参与者（比实验 3 的 16 个还少得多），却跑出了 73 种顺序，而且 `B0 ~ B5` 完全随机洗牌。

1. **为什么总是 4 个 Warp 一起出现**：一个 SM 内部被平分成 4 个独立子分区（Sub-core / SMSP, SM Sub-Partition），当一个 Block 的 16 个 Warp 被装进同一个 SM 时，硬件会按照 `Warp ID % 4` 的取模轮询规则给每个 Warp 分配调度器 `Scheduler 0~3`：

| 硬件调度器编号 | 第 1 排 | 第 2 排 | 第 3 排 | 第 4 排 |
| --- | --- | --- | --- | --- |
| **Scheduler 0** | `w0` | `w4` | `w8` | `w12` |
| **Scheduler 1** | `w1` | `w5` | `w9` | `w13` |
| **Scheduler 2** | `w2` | `w6` | `w10` | `w14` |
| **Scheduler 3** | `w3` | `w7` | `w11` | `w15` |

四个 Scheduler 共用一个 SM 时钟信号，在第一个时钟周期，4 个 Scheduler 在同一纳秒各自挑选自己队列里的第一个 Warp 发送指令。

2. **为什么一组 Warp 内部永远是 `0->1->2->3`**：
涉及到 **SM 的内存出口管线（L1 数据缓存管线）**。

4 个 Scheduler 共用同一个 L1 Data Cache 和访存单元（LSU，Load-Store Unit），`g_counter` 存放在全局显存（L2 Cache）中，4 个调度器发出的 `atomicAdd` 请求想要离开当前 SM 去往 L2，必须先经过这个共用的 L1 缓存管线。

当 4 个 Scheduler 在**同一个时钟周期**把请求怼到 L1 管线入口时，它内部有一个硬编码的**端口优先级仲裁器（Port 0 > Port 1 > Port 2 > Port 3）**。它永远固定先放行 Scheduler 0 的请求（如 `W0`），紧接着放行 Scheduler 1（`W1`）、Scheduler 2（`W2`）、Scheduler 3（`W3`）。

**SM 出口仲裁与 L1 管线排队**：更关键的是，虽然全局显存的 `atomicAdd` 最终要在 L2 Cache 的原子单元里执行，但同一个 SM 内的 4 个 Scheduler 发出的请求必须先经过共用的 SM 访存出口管线（MIO / L1 出口仲裁器），在出 SM 大门发往片上网络之前，就已经被固定端口优先级按 `0 -> 1 -> 2 -> 3` 排好队依次送往 L2 了。

**因此相比之下跨 SM 更容易乱序**：

| 对比维度 | 实验 3：同一个 SM 内部（16 个 Warp） | 实验 4：跨 6 个不同的 SM（6 个 Block） |
| --- | --- | --- |
| **出发时间** | **同步发车**：受同一个 SM 时钟控制，4 个调度器严格同周期发射指令。 | **异步发车**：全局调度器把 Block 派发给 6 个不同 SM 时存在微小延迟差，且各 SM 时钟存在相位差。 |
| **排队地点** | **家门口（L1 Cache）**：共用同一个 L1 出口，出大门前就被固定端口优先级（`0->1->2->3`）排好队了。 | **跨城高速路口（L2 Cache）**：6 个 SM 各自独立出大门，通过片上互联网络（Crossbar/NoC）长途跋涉去 L2 抢锁。 |
| **最终表现** | 严格按 **`(W0~W3)`、4人抱团** 的固定队形输出。 | 每个 SM 到 L2 的物理走线距离不同、网络拥堵抖动不同，`B0 ~ B5` 谁先到全看当时的微秒级路况，跑出 73 种随机排列。 |

## 查看硬件上限

终端直接运行从官方 `cuda-samples` 编译好的硬件查询工具 `deviceQuery`：

```text
# 部分输出
# H20 有 78 个 SM
(078) Multiprocessors, (128) CUDA Cores/MP:    9984 CUDA Cores
Maximum number of threads per block:           1024
Max dimension size of a thread block (x,y,z): (1024, 1024, 64)
Max dimension size of a grid size    (x,y,z): (2147483647, 65535, 65535)
```

一个 Block 内的总线程数（`x * y * z` 的乘积）最高不能超过 `1024`，且三维各自的上限是 `(1024, 1024, 64)`。比如 `dim3(32, 32, 1)` 乘积是 `1024` 合法，而 `dim3(33, 32, 1)` 乘积 `1056 > 1024` 就非法。

改成 `kernel_3d<<<1, 1025>>>();` 后再编译并运行，直接没有输出——因为单 Block 线程数超过了 1024 的硬件上限，核函数启动直接报 `cudaErrorInvalidConfiguration` 并被硬件静默跳过。至于怎么在代码里抓住这种静默报错，就是下一关要解决的问题。

---

# 第三关：内存显存隔离，抓取GPU报错，通信方式

## 为什么要分显存和内存

物理主板上，CPU 有内存条（Host Memory，主机内存），显卡有 HBM 显存（Device Memory，设备显存），两块芯片之间隔着 PCIe 总线，默认不互通：

- **GPU 读 CPU 内存**：用 C 语言 `malloc` 在 CPU 内存条上开了一块空间，把指针直接传给 GPU 核函数 $\rightarrow$ GPU 去自己的显存里找这个地址，触发 **非法内存地址错误（`cudaErrorIllegalAddress`）**。
- **CPU 读 GPU 显存**：用 `cudaMalloc` 在 GPU 显存上开了一块空间，却在 CPU 的 `main` 函数里直接赋值 `cuda_a[i] = i` $\rightarrow$ Linux 操作系统发现 CPU 访问了不属于它的地址，当场击毙进程，报 **段错误（`Segmentation fault`）**。

## 语法细节

- **怎么抓出 GPU 的静默报错？（`checkCudaErrors`）**：GPU 核函数就算在里面崩了，默认也不会弹窗报错，只会静默跳过。
  其实每一个 CUDA API（如 `cudaDeviceSynchronize()`、`cudaMalloc()`）都会返回一个错误码 `cudaError_t`（正常是 `0`）。
  官方在 `<helper_cuda.h>` 头文件里提供了一个宏 `checkCudaErrors(...)`：只要把 CUDA 函数包在里面，一旦返回码不是 `0`，它就会立刻打印出文件名、行号、错误码英文名并终止程序。
- **为什么 `cudaMalloc` 要传二级指针 `&cuda_a`**：
  - C 语言在 CPU 开内存：`int *a = (int *)malloc(bytes);` —— 地址直接通过函数返回值给你。
  - CUDA 在 GPU 开显存：`cudaError_t err = cudaMalloc(&cuda_a, bytes);` —— 因为函数的返回值位置已经被错误码 `cudaError_t` 占了！所以你必须把指针变量 `cuda_a` 自己的地址 `&cuda_a` 传进去，让 `cudaMalloc` 顺着这个地址把申请到的显存门牌号填进 `cuda_a` 里。

## 动手写 `03_memory.cu`

1. 修改 `CMakeLists.txt`
打开 `vim CMakeLists.txt`，加上 `/usr/local/cuda/samples/common/inc` 头文件目录（让编译器能找到 `<helper_cuda.h>`）和 `03_memory` 目标：

```cmake
cmake_minimum_required(VERSION 3.18)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 90)

project(cuda_tutorial LANGUAGES CXX CUDA)

# 告诉编译器去哪里找 <helper_cuda.h>
include_directories(/usr/local/cuda/samples/common/inc)

add_executable(01_hello 01_hello.cu)
add_executable(02_grid 02_grid.cu)
add_executable(03_memory 03_memory.cu)
```

2. 写 `03_memory.cu`

```cpp
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <helper_cuda.h>

// GPU 核函数：把 arr[1..10] 累加到 arr[0]（遇到 0 停止）
__global__ void sum_kernel(int *arr) {
    arr[0] = 0;
    int index = 1;
    while (arr[index] != 0) {
        arr[0] += arr[index];
        index++;
    }
}

// 【模式 1 - 踩坑 A】把 CPU malloc 出来的内存指针直接塞给 GPU
void mode1_gpu_read_cpu_mem() {
    printf("=== 模式 1：GPU 强读 CPU 内存 (malloc) ===\n");
    int *a = (int *)malloc(sizeof(int) * 12);
    a[0] = 0;
    for (int i = 1; i <= 10; ++i) a[i] = i;
    a[11] = 0;

    sum_kernel<<<1, 1>>>(a);

    // 如果不包 checkCudaErrors，这里不会报错，只会打出错误的 a[0] = 0！
    checkCudaErrors(cudaDeviceSynchronize());
    printf("Result: %d\n", a[0]);
    free(a);
}

// 【模式 2 - 踩坑 B】在CPU 上直接写 cudaMalloc 出来的 GPU 显存指针
void mode2_cpu_write_gpu_mem() {
    printf("=== 模式 2：CPU 强写 GPU 显存 (cudaMalloc) ===\n");
    int *cuda_a;
    checkCudaErrors(cudaMalloc(&cuda_a, sizeof(int) * 12));

    // CPU 直接往 GPU 显存地址写数据，操作系统会当场报 Segmentation fault
    for (int i = 1; i <= 10; ++i) cuda_a[i] = i;

    cudaFree(cuda_a);
}

// 【模式 3 - 正解 A】标准做法：CPU 一份、GPU 一份，用 cudaMemcpy 来回搬运
void mode3_explicit_memcpy() {
    printf("=== 模式 3：使用 cudaMemcpy 显式搬运数据 ===\n");
    size_t bytes = sizeof(int) * 12;

    // 1. CPU 申请内存并填数
    int *host_a = (int *)malloc(bytes);
    host_a[0] = 0;
    for (int i = 1; i <= 10; ++i) host_a[i] = i;
    host_a[11] = 0;

    // 2. GPU 申请显存
    int *dev_a;
    checkCudaErrors(cudaMalloc(&dev_a, bytes));

    // 3. 从 CPU 拷到 GPU (Host -> Device)
    checkCudaErrors(cudaMemcpy(dev_a, host_a, bytes, cudaMemcpyHostToDevice));

    // 4. GPU 计算
    sum_kernel<<<1, 1>>>(dev_a);

    // 5. 从 GPU 拷回 CPU (Device -> Host)，cudaMemcpy 自带同步等待功能
    checkCudaErrors(cudaMemcpy(host_a, dev_a, bytes, cudaMemcpyDeviceToHost));

    printf("Result (期望 55): %d\n", host_a[0]);

    free(host_a);
    checkCudaErrors(cudaFree(dev_a));
}

// 【模式 4 - 正解 B】懒人做法：Unified Memory 统一内存 (cudaMallocManaged)
void mode4_unified_memory() {
    printf("=== 模式 4：使用 cudaMallocManaged 统一内存 ===\n");
    int *a;
    // 申请一块 CPU 和 GPU 都能直接读写的托管内存（底层靠缺页中断自动搬运）
    checkCudaErrors(cudaMallocManaged(&a, sizeof(int) * 12));

    for (int i = 1; i <= 10; ++i) a[i] = i; // CPU 直接写
    a[11] = 0;

    sum_kernel<<<1, 1>>>(a);                // GPU 直接算
    // 注意：统一内存没有 cudaMemcpy 帮你同步，CPU 读结果前必须显式同步！
    checkCudaErrors(cudaDeviceSynchronize());

    printf("Result (期望 55): %d\n", a[0]); // CPU 直接读
    checkCudaErrors(cudaFree(a));
}

int main(int argc, char **argv) {
    int mode = (argc > 1) ? atoi(argv[1]) : 1;
    if (mode == 1) mode1_gpu_read_cpu_mem();
    else if (mode == 2) mode2_cpu_write_gpu_mem();
    else if (mode == 3) mode3_explicit_memcpy();
    else if (mode == 4) mode4_unified_memory();
    return 0;
}
```

```text
(torch-base) [root@TENCENT64 ~]# ./build/03_memory 1
=== 模式 1：GPU 强读 CPU 内存 (malloc) ===
CUDA error at /root/03_memory.cu:26 code=700(cudaErrorIllegalAddress) "cudaDeviceSynchronize()" 

(torch-base) [root@TENCENT64 ~]# ./build/03_memory 2
=== 模式 2：CPU 强写 GPU 显存 (cudaMalloc) ===
Segmentation fault (core dumped)

(torch-base) [root@TENCENT64 ~]# ./build/03_memory 3
=== 模式 3：使用 cudaMemcpy 显式搬运数据 ===
Result (期望 55): 55

(torch-base) [root@TENCENT64 ~]# ./build/03_memory 4
=== 模式 4：使用 cudaMallocManaged 统一内存 ===
Result (期望 55): 55
```

模式 3 和模式 4 的区别：

- **`cudaMallocManaged`（统一内存）慢在哪？「边吃边买菜（缺页中断 Page Fault）」**
  用 `cudaMallocManaged` 申请的 100MB 数组在 CPU 上赋好值后依然还在 CPU，等启动 `kernel<<<...>>>(a)` 时：
  - GPU 线程去读 `a[0]`，发现显存里根本没有这一页数据；
  - GPU 硬件被迫踩急刹车暂停当前线程，向驱动程序发一个「缺页中断（Page Fault）」信号：「报告！`a[0]` 所在的 4KB 内存页不在显存里，快从 CPU 给我搬过来！」
  - 驱动走 PCIe 总线把这几 KB 搬过来，GPU 恢复运行；跑了两步去读下一页，又触发一次缺页中断、又踩一次刹车……
- **`cudaMemcpy`（显式拷贝）强在哪？「大卡车整车直达」**
  用 `cudaMemcpy` 时，你在核函数启动前就明确告诉硬件：「把这 100 MB 一次性全搬过去」：
  - 硬件直接启动 **DMA 引擎（直接内存访问）**，把 100 MB 打包成一整条连续的高速数据流，把 PCIe 总线带宽吃满一次性拉到 GPU 显存；
  - 等 GPU 核函数真正开跑时，所有数据已经整整齐齐躺在 H20 的超高速 HBM3 显存里了，全程 0 中断、0 刹车。

| 场景 | 选哪个？ | 原因 |
| --- | --- | --- |
| 写教程、快速写原型、调试算法逻辑 | `cudaMallocManaged`（模式 4） | 省掉一半模板代码，不用维护两套指针，专注写核函数逻辑 |
| 超大稀疏图 / 树结构，GPU 只会随机访问其中 1% 的节点 | `cudaMallocManaged`（模式 4） | 不知道 GPU 会访问哪一小块，按需缺页搬运反而省了搬另外 99% 无用数据的开销 |
| 生产环境、深度学习算子、性能敏感代码 | `cudaMalloc` + `cudaMemcpy`（模式 3） | 整块 DMA 搬运带宽最高，核函数运行期间无缺页中断卡顿，数据在不在显存完全可控 |

---

# 第四关：Grid-Stride Loop（网格跨步循环）—— 怎么用有限的线程处理任意长度的数组？

前面 3 关里，我们的 `sum_kernel<<<1, 1>>>` 实际上只用了 1 个线程在 GPU 上跑循环，跟在 CPU 上跑没区别。现在我们要真正把几百万个元素分给成百上千个 GPU 线程并行算。

数组长度 $N$ 不是 `blockDim` 整数倍时按照向上取整方法切分，多余的线程要在 `kernel` 函数中加 `if (i < n)` 的越界判断。

## Grid-Stride Loop（网格跨步循环）—— 为什么核函数里要用 `for` 而不是 `if`？

如果只写 `if (i < n)`，一旦数组长度 $N$ 特别大（比如 10 亿），或者你想固定只开一小批线程（比如只开 `<<<2, 4>>>` 共 8 个线程去处理 20 个元素），`if` 就只能处理前 8 个元素。

NVIDIA 官方最推崇的标准写法叫 **Grid-Stride Loop（网格跨步循环）**：
假设总共有 `total_threads = gridDim.x * blockDim.x` 个线程（比如 8 个线程，编号 `0 ~ 7`）：

- 第 1 轮：线程 `0 ~ 7` 分别处理下标 `0 ~ 7`；
- 第 2 轮：每个线程都往前跳一步，步长正好是总线程数 `8`！于是线程 `0` 去处理 `0 + 8 = 8`，线程 `1` 去处理 `1 + 8 = 9`……
- 第 3 轮：再跳 `8`，线程 `0` 处理 `16`……直到下标 $\ge n$ 退出循环。

```text
总线程数 stride = 8 (Thread 0 ~ Thread 7)，处理 n = 20 个元素：
第 1 轮: [T0] [T1] [T2] [T3] [T4] [T5] [T6] [T7]   (处理 arr[0..7])
           │    │                  (每个线程下标 +8)
           ▼    ▼
第 2 轮: [T0] [T1] [T2] [T3] [T4] [T5] [T6] [T7]   (处理 arr[8..15])
           │    │                  (每个线程下标 +8)
           ▼    ▼
第 3 轮: [T0] [T1] [T2] [T3]                       (处理 arr[16..19]，T4~T7 发现 >=20 自动退出)
```

```cpp
__global__ void kernel(int *arr, int n) {
    int tid    = blockIdx.x * blockDim.x + threadIdx.x; // 我的起点：全局线程号
    int stride = gridDim.x * blockDim.x;                // 我的步长：网格总线程数

    for (int i = tid; i < n; i += stride) {
        arr[i] = ...;
    }
}
```

1. **任意解耦**：不管 $N$ 是 10 还是 10 亿，不管外面传 `<<<1, 1>>>`（退化成单线程串行，极度方便 Debug！）还是 `<<<256, 256>>>`，核函数内部一行代码都不用改，永远正确。
2. **跨步 `stride = gridDim.x * blockDim.x` 能保证显存合并访问（Coalesced Memory Access）**：

**显存读取方法**：每一轮里，同一个 Warp 的 32 个相邻线程访问的永远是紧挨着的 32 个连续内存地址（`arr[0..31]`），当这 32 个线程**同时**执行 `arr[i]` 去读取全局显存（Global Memory）时，GPU 的显存控制器有一个非常霸道的特性：**它不按单个 `int`（4 字节）去读，而是按“内存事务块（Memory Transaction，通常是 32 字节或 128 字节的连续内存段）”去打包读取。**

- **a.** 如果 32 个线程要读的 32 个 `int`（刚好 $32 \times 4 = 128$ 字节）在内存里是紧挨着的连续地址，显存控制器只需要发起 1 次 128 字节的内存读取事务，就能一次性把数据全部取回来分给 32 个线程。这叫做**合并访问（Coalesced Access）**，带宽利用率 100%。
- **b.** 反之，如果这 32 个线程要读的地址东一个、西一个（跨步很大），显存控制器就不得不发起 32 次独立的内存读取，每次读 128 字节却只用其中 4 字节，带宽利用率暴跌到 3%（$1/32$）。

假设我们一共只有 4 个线程（`stride = 4`），要处理长度 $n = 16$ 的数组，每个线程要干 4 个元素的活。如果你设计每个线程怎么分这 16 个元素，通常有两种分法：

- **a. 错误分法：连续切块法**（像 CPU 多线程 `openmp` 的习惯）让线程 0 负责前 4 个，线程 1 负责接下来 4 个……代码写成：

```cpp
// 灾难后果：在循环的第 1 轮，线程 0、1、2、3 在同一时刻分别去读 arr[0]、arr[4]、arr[8]、arr[12]！地址全部是隔开的（Strided Access），导致显存合并访问彻底失效，性能断崖式下跌。
int start = tid * 4;
for (int i = start; i < start + 4; i++) { arr[i] = ...; }
```

- **b. 正确分法：Grid-Stride Loop（每次跳过 `stride = gridDim.x * blockDim.x`）**
  - **第 1 轮**：线程 0、1、2、3 **同一时刻**读 `arr[0]`、`arr[1]`、`arr[2]`、`arr[3]` —— **完全连续，1 次合并读取！**
  - **第 2 轮**：大家一起加上 `stride (4)`，**同一时刻**读 `arr[4]`、`arr[5]`、`arr[6]`、`arr[7]` —— **依然完全连续，1 次合并读取！**

## 动手写 `04_loop.cu`

先在 `vim CMakeLists.txt` 末尾加一行：

```cmake
add_executable(04_loop 04_loop.cu)
```

打开 `vim 04_loop.cu`，敲入下面的代码（包含两个实验：实验 1 用 `<<<2, 4>>>` 共 8 个线程跨步处理 20 个元素，打印出每个元素是谁算的；实验 2 跑通经典的 SAXPY 向量乘加 $y[i] = a \cdot x[i] + y[i]$）：

```cpp
#include <cstdio>
#include <cuda_runtime.h>
#include <helper_cuda.h>

// 实验 1：观察 Grid-Stride Loop 是怎么「跨步」分配任务的
__global__ void stride_demo_kernel(int *arr, int n) {
    int tid    = blockIdx.x * blockDim.x + threadIdx.x; // 起点：全局线程号
    int stride = gridDim.x * blockDim.x;                // 步长：总线程数

    for (int i = tid; i < n; i += stride) {
        arr[i] = tid; // 记录下第 i 个元素是由几号全局线程处理的
    }
}

// 实验 2：经典并行算子 SAXPY (y[i] = a * x[i] + y[i])
__global__ void saxpy_kernel(int n, float a, const float *x, float *y) {
    int tid    = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    // 如果写成 int i = 0, 8 个线程都去写 0，8，16，剩下的下标没有线程动
    for (int i = tid; i < n; i += stride) {
        y[i] = a * x[i] + y[i];
    }
}

int main() {
    // ================= 实验 1：8 个线程处理 20 个元素 =================
    printf("=== 实验 1：<<<2, 4>>> (共 8 个线程) 跨步处理 n=20 个元素 ===\n");
    int n1 = 20;
    int *arr;
    checkCudaErrors(cudaMallocManaged(&arr, n1 * sizeof(int)));

    // 故意只开 2 个 Block × 4 个线程 = 8 个总线程（步长 stride = 8）
    stride_demo_kernel<<<2, 4>>>(arr, n1);
    checkCudaErrors(cudaDeviceSynchronize());

    for (int i = 0; i < n1; ++i) {
        printf("arr[%2d] 由 Global Thread %d 处理\n", i, arr[i]);
    }
    checkCudaErrors(cudaFree(arr));

    // ================= 实验 2：向上取整公式 + 10000 元素 SAXPY =================
    printf("\n=== 实验 2：用向上取整公式处理 n=10000 的 SAXPY ===\n");
    int n2 = 10000;
    float *x, *y;
    checkCudaErrors(cudaMallocManaged(&x, n2 * sizeof(float)));
    checkCudaErrors(cudaMallocManaged(&y, n2 * sizeof(float)));

    for (int i = 0; i < n2; ++i) {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }

    int block_dim = 128;
    int grid_dim  = (n2 - 1) / block_dim + 1; // 10000 -> 79 个 Block (10112 线程)
    printf("n=%d, block_dim=%d -> 向上取整 grid_dim=%d (总线程数 %d)\n",
           n2, block_dim, grid_dim, grid_dim * block_dim);

    // 计算 y[i] = 3.0 * 1.0 + 2.0 = 5.0
    saxpy_kernel<<<grid_dim, block_dim>>>(n2, 3.0f, x, y);
    checkCudaErrors(cudaDeviceSynchronize());

    printf("y[0] = %.1f, y[9999] = %.1f (期望均为 5.0)\n", y[0], y[n2 - 1]);

    checkCudaErrors(cudaFree(x));
    checkCudaErrors(cudaFree(y));
    return 0;
}
```

```text
=== 实验 1：<<<2, 4>>> (共 8 个线程) 跨步处理 n=20 个元素 ===
arr[ 0] 由 Global Thread 0 处理
arr[ 1] 由 Global Thread 1 处理
arr[ 2] 由 Global Thread 2 处理
arr[ 3] 由 Global Thread 3 处理
arr[ 4] 由 Global Thread 4 处理
arr[ 5] 由 Global Thread 5 处理
arr[ 6] 由 Global Thread 6 处理
arr[ 7] 由 Global Thread 7 处理
arr[ 8] 由 Global Thread 0 处理
arr[ 9] 由 Global Thread 1 处理
arr[10] 由 Global Thread 2 处理
arr[11] 由 Global Thread 3 处理
arr[12] 由 Global Thread 4 处理
arr[13] 由 Global Thread 5 处理
arr[14] 由 Global Thread 6 处理
arr[15] 由 Global Thread 7 处理
arr[16] 由 Global Thread 0 处理
arr[17] 由 Global Thread 1 处理
arr[18] 由 Global Thread 2 处理
arr[19] 由 Global Thread 3 处理

=== 实验 2：用向上取整公式处理 n=10000 的 SAXPY ===
n=10000, block_dim=128 -> 向上取整 grid_dim=79 (总线程数 10112)
y[0] = 5.0, y[9999] = 5.0 (期望均为 5.0)
```

---

# 第五关：算子传递 & Thrust 容器

第 4 关的 `04_loop.cu` 里有两个核函数：`stride_demo_kernel` 和 `saxpy_kernel`。把它们放一起对比，前 3 行代码一模一样：

```cpp
int tid = blockIdx.x * blockDim.x + threadIdx.x;
int stride = blockDim.x * gridDim.x;
for (int i = tid; i < n; i += stride) {
    // 只有这里的一行计算逻辑不同
}
```

如果工程里有几十种逐元素运算（加法、`sin`、`cos`、ReLU……），每次都把 `tid`、`stride`、`for` 循环抄一遍，加上每个数组都要写 `cudaMalloc`、`cudaMemcpy`、`cudaFree`，代码里全是样板戏。

因此第五关需要解决 2 个工程问题：

- **算子传递（通用 `parallel_for` 核函数）**：把 Grid-Stride Loop 抽成一个只写一次的模板核函数，把具体的计算公式当成参数传进去。
- **Thrust 容器（`thrust::host_vector` / `thrust::device_vector`）**：用 CUDA 自带的 C++ STL 风格容器接管内存/显存分配、跨端 `cudaMemcpy` 和自动释放。

## 动手写 `05_thrust.cu`

在 `CMakeLists.txt` 加上 `--extended-lambda` 和 `05_thrust`：

```cmake
cmake_minimum_required(VERSION 3.18)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 90)

project(cuda_tutorial LANGUAGES CXX CUDA)

# 第 5 关新增：允许向 __global__ 核函数传递 [=] __device__ (...) lambda
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --extended-lambda")

# 告诉编译器去哪里找 <helper_cuda.h>
include_directories(/usr/local/cuda/samples/common/inc)

add_executable(01_hello 01_hello.cu)
add_executable(02_grid 02_grid.cu)
add_executable(03_memory 03_memory.cu)
add_executable(04_loop 04_loop.cu)
add_executable(05_thrust 05_thrust.cu)
```

在 `05_thrust.cu` 编写第 5 关代码。我们在同一个文件里对比 **方式 A（Functor 仿函数结构体）** 和 **方式 B（Thrust 容器 + `__device__` Lambda）**：

```cpp
#include <cstdio>
#include <cmath>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>

// 通用 Grid-Stride Loop 模板核函数：整个工程写这一次就够了
template <typename Func>
__global__ void parallel_for(int n, Func func) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = tid; i < n; i += stride) {
        func(i);
    }
}

// 方式 A：传统的仿函数结构体（Functor）
struct SaxpyFunctor {
    float a;
    const float *x;
    float *y;

    __device__ void operator()(int i) const {
        y[i] = a * x[i] + y[i];
    }
};

int main() {
    int n = 65536;
    float a = 3.14f;
    int block_dim = 128;
    int grid_dim = (n - 1) / block_dim + 1;

    // 1. 在 CPU 上用 thrust::host_vector 初始化测试数据
    thrust::host_vector<float> x_host(n);
    thrust::host_vector<float> y_host(n);

    auto rand_float = []() { return std::rand() * (1.0f / RAND_MAX); };
    thrust::generate(x_host.begin(), x_host.end(), rand_float);
    thrust::generate(y_host.begin(), y_host.end(), rand_float);

    // 先算出一份第 0 个和最后一个元素的 CPU 标准答案，用来核对
    float expected_0 = a * x_host[0] + y_host[0];
    float expected_last = a * x_host[n - 1] + y_host[n - 1];

    // ================= 实验 1：使用 Functor 结构体传递算子 =================
    printf("=== 实验 1：Thrust 跨端拷贝 + Functor 仿函数结构体 ===\n");
    // 一行 '=' 自动完成 cudaMalloc + cudaMemcpyHostToDevice
    thrust::device_vector<float> x_dev1 = x_host;
    thrust::device_vector<float> y_dev1 = y_host;

    // .data().get() 取出底层 float* 裸指针传给结构体
    SaxpyFunctor functor{a, x_dev1.data().get(), y_dev1.data().get()};
    parallel_for<<<grid_dim, block_dim>>>(n, functor);
    checkCudaErrors(cudaDeviceSynchronize());

    // 一行 '=' 自动完成 cudaMemcpyDeviceToHost
    thrust::host_vector<float> res1 = y_dev1;
    printf("Functor 结果: res1[0] = %.6f (期望 %.6f), res1[%d] = %.6f (期望 %.6f)\n",
           res1[0], expected_0, n - 1, res1[n - 1], expected_last);

    // ================= 实验 2：使用 [=] __device__ Lambda 传递算子 =================
    printf("\n=== 实验 2：Thrust 跨端拷贝 + [=] __device__ Lambda ===\n");
    thrust::device_vector<float> x_dev2 = x_host;
    thrust::device_vector<float> y_dev2 = y_host;

    // 注意：Lambda 不能直接捕获 device_vector，必须先 .data() 取出 device_ptr 再按值 [=] 捕获
    parallel_for<<<grid_dim, block_dim>>>(n, [a, x = x_dev2.data(), y = y_dev2.data()] __device__ (int i) {
        y[i] = a * x[i] + y[i];
    });
    checkCudaErrors(cudaDeviceSynchronize());

    thrust::host_vector<float> res2 = y_dev2;
    printf("Lambda  结果: res2[0] = %.6f (期望 %.6f), res2[%d] = %.6f (期望 %.6f)\n",
           res2[0], expected_0, n - 1, res2[n - 1], expected_last);

    return 0;
}
```

```text
=== 实验 1：Thrust 跨端拷贝 + Functor 仿函数结构体 ===
Functor 结果: res1[0] = 3.413810 (期望 3.413810), res1[65535] = 3.584183 (期望 3.584182)

=== 实验 2：Thrust 跨端拷贝 + [=] __device__ Lambda ===
Lambda  结果: res2[0] = 3.413810 (期望 3.413810), res2[65535] = 3.584183 (期望 3.584182)
```

## 核心原理与避坑点

### 1. 怎么把「计算公式」传给 GPU 核函数？

写一个模板核函数 `template <typename Func> __global__ void parallel_for(int n, Func func)`，在循环里调用 `func(i)`。向它传 `func` 有两种方式：

- **方式 A：仿函数结构体（Functor，传统 C++ 写法）** 定义一个结构体，把数组指针存成成员变量，重载 `__device__ void operator()(int i) const`。核函数收到这个结构体副本后，在 GPU 上调它的 `operator()(i)`。
- **方式 B：`__device__` Lambda 表达式（现代 CUDA 写法）** 直接在调用处写 `[=] __device__ (int i) { y[i] = a * x[i] + y[i]; }`。
  - **为什么必须写 `[=]`（按值捕获）而不能写 `[&]`（按引用捕获）？** Lambda 是在 CPU（`main` 函数）里创建的，但要在 GPU 线程里执行。如果按引用 `[&]` 捕获，GPU 拿到的是 CPU 栈变量的内存地址，一解引用就会像第 3 关那样触发 `cudaErrorIllegalAddress`。按值 `[=]` 捕获则会把指针的值（显存地址）和标量 `a` 直接打包拷进核函数参数里。
  - **为什么必须在 `CMakeLists.txt` 开启 `--extended-lambda`？** 标准 C++ 的 Lambda 默认只生成 CPU（`__host__`）代码。只有给 `nvcc` 加上 `--extended-lambda` 编译参数，并在 Lambda 上显式标出 `__device__`，编译器才会把这个匿名闭包编译成能跨端传递并在 GPU 上执行的指令。

### 2. Thrust 容器怎么和自定义核函数配合？

Thrust 是随 CUDA Toolkit 自带的官方模板库（不需要额外装包，直接 `#include <thrust/host_vector.h>` 和 `#include <thrust/device_vector.h>`）：

- `thrust::host_vector<float> h_vec(n);`：在 **CPU 内存** 分配 `n` 个元素。
- `thrust::device_vector<float> d_vec(n);`：在 **GPU 显存** 分配 `n` 个元素。
- **一行 `=` 替代 `cudaMemcpy`**：
  - `d_vec = h_vec;` $\rightarrow$ 自动执行 `cudaMemcpyHostToDevice`
  - `h_vec = d_vec;` $\rightarrow$ 自动执行 `cudaMemcpyDeviceToHost`
  - 函数结束离开作用域时，析构函数自动调用 `cudaFree`。
- **关键坑点：`device_vector` 不能直接传进核函数或被 `__device__` Lambda 捕获！** `thrust::device_vector` 本身是一个驻留在主机侧、带复杂构造/析构逻辑的管理类对象。传给 GPU 前，必须先用 `.data()` 取出它的设备指针（`thrust::device_ptr<float>`），或者用 `.data().get()` 取出原始裸指针 `float*`，再让 Lambda 按值捕获这个指针。
- **顺手修掉博客 §4.4.1 的一个笔误**：
  博客在算 `grid_dim` 时写成了 `int grid_dim = (n - 1) / block_dim;`，漏了末尾的 `+ 1`（虽然内部是 Grid-Stride Loop，少开 1 个 Block 不会漏算数据，但我们沿用第 4 关验证过的正确向上取整公式 `(n - 1) / block_dim + 1`）。

---

# 第六关：并行归约、数据竞争与原子操作

这一关我们要把数组里 $N = 65536$ 个元素的 $\sin(x_i)$ 全部加到一个 GPU 全局变量 `dev_sum` 里（即 **并行归约 Parallel Reduction** 的最简形式）。

## 动手写 `06_atomic.cu`

```cpp
#include <cstdio>
#include <cmath>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>

// 定义在 GPU 显存上的全局变量
__device__ float dev_sum;

// 通用 Grid-Stride Loop 模板核函数
template <typename Func>
__global__ void parallel_for(int n, Func func) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = tid; i < n; i += stride) {
        func(i);
    }
}

// 用 atomicCAS 手搓浮点数原子加法
__device__ float my_atomic_add(float *addr, float val) {
    int *addr_as_int = (int *)addr;
    int old_int = *addr_as_int;
    int assumed_int;

    do {
        assumed_int = old_int;
        // 1. 把旧比特还原成 float，加上 val，再把新结果打包成 int 比特
        float new_float = __int_as_float(assumed_int) + val;
        int new_int = __float_as_int(new_float);

        // 2. 如果 *addr_as_int 依然等于 assumed_int，就把 new_int 写进去；
        //    否则不写入，返回当前显存里的最新值赋给 old_int，继续下一轮循环重试
        old_int = atomicCAS(addr_as_int, assumed_int, new_int);
    } while (assumed_int != old_int);

    return __int_as_float(old_int);
}

// 辅助函数：把 dev_sum 清零
void reset_dev_sum() {
    float zero = 0.0f;
    checkCudaErrors(cudaMemcpyToSymbol(dev_sum, &zero, sizeof(float)));
}

// 辅助函数：把 dev_sum 的值拷回 CPU
float read_dev_sum() {
    float result = 0.0f;
    checkCudaErrors(cudaMemcpyFromSymbol(&result, dev_sum, sizeof(float)));
    return result;
}

int main() {
    int n = 65536;
    int block_dim = 128;
    int grid_dim = (n - 1) / block_dim + 1;

    // 1. 生成随机数组并拷贝到 GPU
    thrust::host_vector<float> x_host(n);
    auto rand_float = []() { return std::rand() * (1.0f / RAND_MAX); };
    thrust::generate(x_host.begin(), x_host.end(), rand_float);
    thrust::device_vector<float> x_dev = x_host;

    // 2. 先在 CPU 上顺序算出标准答案
    float cpu_sum = 0.0f;
    for (int i = 0; i < n; i++) {
        cpu_sum += std::sin(x_host[i]);
    }
    printf("CPU 顺序累加标准答案: %.6f\n\n", cpu_sum);

    // ================= 阶段 1：直接裸写 dev_sum += ...（数据竞争） =================
    printf("=== 阶段 1：无原子保护直接 dev_sum += sinf(x[i])（连跑 3 次） ===\n");
    for (int run = 1; run <= 3; run++) {
        reset_dev_sum();
        parallel_for<<<grid_dim, block_dim>>>(n, [x = x_dev.data()] __device__ (int i) {
            dev_sum += sinf(x[i]); // 65536 个线程同时读-改-写，严重互相覆盖！
        });
        checkCudaErrors(cudaDeviceSynchronize());
        printf("  第 %d 次运行结果: %.6f (丢失了 %.1f%% 的累加!)\n",
               run, read_dev_sum(), (1.0f - read_dev_sum() / cpu_sum) * 100.0f);
    }

    // ================= 阶段 2：使用 CUDA 官方 atomicAdd =================
    printf("\n=== 阶段 2：使用硬件 atomicAdd(&dev_sum, ...)（连跑 3 次） ===\n");
    for (int run = 1; run <= 3; run++) {
        reset_dev_sum();
        parallel_for<<<grid_dim, block_dim>>>(n, [x = x_dev.data()] __device__ (int i) {
            atomicAdd(&dev_sum, sinf(x[i]));
        });
        checkCudaErrors(cudaDeviceSynchronize());
        printf("  第 %d 次运行结果: %.6f\n", run, read_dev_sum());
    }

    // ================= 阶段 3：使用手搓的 atomicCAS 版 my_atomic_add =================
    printf("\n=== 阶段 3：使用手搓 atomicCAS 版 my_atomic_add（连跑 3 次） ===\n");
    for (int run = 1; run <= 3; run++) {
        reset_dev_sum();
        parallel_for<<<grid_dim, block_dim>>>(n, [x = x_dev.data()] __device__ (int i) {
            my_atomic_add(&dev_sum, sinf(x[i]));
        });
        checkCudaErrors(cudaDeviceSynchronize());
        printf("  第 %d 次运行结果: %.6f\n", run, read_dev_sum());
    }

    return 0;
}
```

```text
CPU 顺序累加标准答案: 30075.328125

=== 阶段 1：无原子保护直接 dev_sum += sinf(x[i])（连跑 3 次） ===
  第 1 次运行结果: 0.787899 (丢失了 100.0% 的累加!)
  第 2 次运行结果: 0.376831 (丢失了 100.0% 的累加!)
  第 3 次运行结果: 0.376831 (丢失了 100.0% 的累加!)

=== 阶段 2：使用硬件 atomicAdd(&dev_sum, ...)（连跑 3 次） ===
  第 1 次运行结果: 30075.044922
  第 2 次运行结果: 30075.126953
  第 3 次运行结果: 30075.113281

=== 阶段 3：使用手搓 atomicCAS 版 my_atomic_add（连跑 3 次） ===
  第 1 次运行结果: 30075.087891
  第 2 次运行结果: 30075.228516
  第 3 次运行结果: 30075.185547
```

1. **C++ 里的一行 `dev_sum += val`，编译成 GPU 机器码后是 3 条分开的指令**：
   1. **Load（读）**：从显存把 `dev_sum` 当前的值读进线程自己的寄存器；
   2. **Add（改）**：在线程自己的寄存器里加上 `val`；
   3. **Store（写）**：把寄存器算出的新值写回显存 `dev_sum`。
   当成千上万个线程同时跑这三步时：假设 `dev_sum` 当前是 `0`，100 个线程在同一时刻 `Load` 读到了 `0`，各自在寄存器里算出了 `0 + val_i`，然后先后 `Store` 写回显存——前 99 个线程写进去的值全被第 100 个线程盖掉了。

2. **`__device__` 全局变量怎么在 CPU 和 GPU 之间读写？**
   在函数外面写 `__device__ float dev_sum;` 会在 GPU 显存里开辟一个全局标量。
   - 注意：CPU 代码不能直接写 `dev_sum = 0.0f;`，也不能写 `&dev_sum` 传给普通 `cudaMemcpy`（因为在 CPU 看来 `dev_sum` 只是一个符号 Symbol，不是主机指针）。
   - 必须用专门操作设备符号的 API：
     - CPU $\rightarrow$ GPU（清零）：`checkCudaErrors(cudaMemcpyToSymbol(dev_sum, &zero, sizeof(float)));`
     - GPU $\rightarrow$ CPU（读结果）：`checkCudaErrors(cudaMemcpyFromSymbol(&result, dev_sum, sizeof(float)));`

3. **硬件原子加法 `atomicAdd(&dev_sum, val)`**
   `atomicAdd` 会通知 GPU 的 L2 Cache 原子运算单元：把「读旧值 $\rightarrow$ 加新值 $\rightarrow$ 写回」锁成一个不可打断的硬件事务。65536 个线程在硬件层面排队累加，一个都不会丢。
   - **细节观察**：你会发现 `atomicAdd` 算出的总和跟 CPU `for` 循环顺序累加的总和，在小数点后第 1～2 位（相差约 `0.1～0.28`）有一点点差异。
   - **原因**：单精度浮点数 `float` 只有 24 位有效尾数，浮点加法不满足结合律（$(a + b) + c \neq a + (b + c)$，大数加小数会有舍入误差）。第 2 关我们测过，跨 Warp、跨 SM 的到达顺序是随机的，求和顺序变了，末尾舍入误差就会略有不同。

4. **用 `atomicCAS` 手搓自己的 `my_atomic_add`**
   如果遇到硬件原生没提供原子指令的运算（比如你想写个浮点数的 `atomicMul` 或自定义的复杂更新），就必须用原子操作的「万能积木」——`atomicCAS`（Compare-And-Swap，比较并交换）。

   它的思路叫 **乐观锁（Optimistic Lock）**：
   1. 先偷看一眼显存地址 `addr` 里的旧值，记为 `old_val`；
   2. 算出如果加上 `val` 之后的新值 `new_val = old_val + val`；
   3. 调用硬件指令 `atomicCAS(addr, old_val, new_val)`，意思是告诉硬件：
      - 「请帮我看一眼现在 `addr` 里的数是不是还等于我刚才看到的 `old_val`？」
      - 如果是（说明这期间没人插队改过它）：把 `new_val` 写进去，大功告成退出循环！
      - 如果不是（说明别的线程刚才抢先改过了）：绝不写入，把别人刚写进去的最新值返回给我，我拿着最新值再算一遍重试（`while` 循环）！

   **为什么代码里要用 `__float_as_int` 和 `__int_as_float`？** 因为 CUDA 的 `atomicCAS` 硬件接口只按整数比特位做比较（只收 `int*` / `unsigned int*`），不直接收 `float*`。
   - 不能用 C 语言强转 `(int)x`（那会把 `3.14f` 截断成整数 `3`）；
   - 必须用 CUDA 内置的比特位重解释函数：`__float_as_int(x)`（把 `float` 的 32 个 0/1 比特原封不动当成 `int` 传给 `atomicCAS`）和 `__int_as_float(k)`（把读回来的 32 位 `int` 比特还原回 `float` 做加法）。

---

# 第七关：RAII 显存管理与 CUDA Stream 异步流

回想一下第 3 关：当时我们在 `03_memory.cu` 里每次申请显存都要手写 `cudaMalloc(&dev_a, bytes)`，用完还要手写 `cudaFree(dev_a)`。如果一个函数里有七八个数组，或者中途报错提前退出了，一旦漏写 `cudaFree` 就会显存泄漏；反过来如果对同一个指针不小心调用了两次 `cudaFree`，程序又会当场崩溃。

另外，前 6 关我们每次启动核函数都只写 `<<<grid, block>>>`，每跑一步 CPU 都要停下来等 GPU。这一关我们写一个程序 `07_raii_stream.cu`，完成两件具体的事：

1. **给显存配一个「自动管家类」 `GpuBuffer`**：创建对象时自动申请显存，离开大括号 `{}` 时自动调 `cudaFree` 释放；并且规定一块显存同时只能归一个对象管（只允许转交所有权，不允许复制指针）。
2. **用一条「异步流水线」 `cudaStream_t` 连续跑多个任务**：把「清零状态 $\rightarrow$ 核函数 1 填数 $\rightarrow$ 核函数 2 乘加 $\rightarrow$ 拷回 CPU」全部塞进同一条流水线里排队自动执行，CPU 只在最后读取结果前等一次。

## 动手写 `07_raii_stream.cu`

1. 先在 GPU 机器上打开 `vim /root/cuda_tutorial/CMakeLists.txt`，在文件末尾加一行：

```cmake
add_executable(07_raii_stream 07_raii_stream.cu)
```

2. 打开 `vim /root/cuda_tutorial/07_raii_stream.cu`，敲入下面的代码（所有新语法的解释都直接写在了对应代码行的上方注释里，不用上下翻页对照）：

```cpp
#include <cuda_runtime.h>
#include <cstdio>
#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

// 自己写一个 5 行的错误检查函数（作用和第 3 关的 checkCudaErrors 一样）：
// 只要 CUDA 函数返回的状态码不是 cudaSuccess (0)，
// 就用 cudaGetErrorString 把错误码翻译成英文报错，并用 throw 抛出异常停下程序
inline void cuda_check(cudaError_t status) {
    if (status != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(status));
    }
}

// ==================== 第一部分：显存自动管家类 GpuBuffer ====================
// template <class T> 叫「模板」：把元素类型做成填空题 T。
// 当你写 GpuBuffer<float> 时，编译器自动把下面所有的 T 替换成 float；
// 当你写 GpuBuffer<int>   时，编译器自动把下面所有的 T 替换成 int。
template <class T>
class GpuBuffer {
    T* ptr_ = nullptr;        // 成员变量 1：保存 GPU 显存地址的指针（初始为空指针 nullptr）
    std::size_t count_ = 0;   // 成员变量 2：保存数组的元素个数（size_t 就是无符号整数）

public:
    // 【1. 构造函数】函数名与类名完全一样，创建对象 GpuBuffer<float> a(4); 时自动执行。
    // 语法拆解：
    //   - explicit：禁止编译器做隐式转换（比如防止写出 GpuBuffer<float> a = 4; 这种怪代码，强制只能写 a(4)）。
    //   - : count_(n) 叫「成员初始化列表」，意思就是在进入大括号 { 之前，先把参数 n 赋值给成员变量 count_
    //     （完全等价于在大括号第一行写 count_ = n;）。
    //   - reinterpret_cast<void**>(&ptr_)：就是 C 语言里的指针强制类型转换 (void**)&ptr_。
    //     因为 cudaMalloc 底层 C 接口规定第一个参数类型是 void**（通用二级指针），
    //     而 &ptr_ 的类型是 T**（如 float**），所以用 reinterpret_cast 把指针类型强转成 void**。
    explicit GpuBuffer(std::size_t n) : count_(n) {
        if (n > 0) {
            cuda_check(cudaMalloc(reinterpret_cast<void**>(&ptr_), n * sizeof(T)));
        }
    }

    // 【2. 析构函数】类名前面加波浪号 ~，当对象离开大括号 {} 寿命结束时，由编译器自动调用。
    // 语法拆解：
    //   - noexcept：向编译器保证「这个函数里绝不会 throw 抛出异常」（C++ 规定析构函数不能抛异常，否则程序直接强退）。
    //   - 因此里面只写 if (ptr_) cudaFree(ptr_);，千万不要在析构函数外面包会抛异常的 cuda_check！
    ~GpuBuffer() noexcept {
        if (ptr_) {
            cudaFree(ptr_);
        }
    }

    // 【3. 禁止拷贝（2 个函数）】
    // 语法拆解：
    //   - const GpuBuffer& 是普通的拷贝引用；后面写 = delete 意思是「废除这个函数」。
    //   - 一旦有人在代码里写 GpuBuffer b = a;（拷贝构造）或 b = a;（拷贝赋值），编译器直接报错拦住！
    GpuBuffer(const GpuBuffer&) = delete;
    GpuBuffer& operator=(const GpuBuffer&) = delete;

    // 【4. 移动构造函数】当写 GpuBuffer b(std::move(a)); 把旧对象 a 转交给新对象 b 时自动执行。
    // 语法拆解：
    //   - GpuBuffer&&（两个 &）叫「右值引用」，专门用来接收 std::move(a) 传过来的「即将废弃的对象」。
    //   - 里面就干最直白的 4 步：把 other 的指针和长度拿过来，再把 other 的指针置空、长度清零。
    //   - 这样 other 后面析构时看到 other.ptr_ 是 nullptr 就不会重复调 cudaFree 了！
    GpuBuffer(GpuBuffer&& other) noexcept {
        ptr_ = other.ptr_;       // 1. 拿走 other 的显存地址
        count_ = other.count_;   // 2. 拿走 other 的元素个数
        other.ptr_ = nullptr;    // 3. 把 other 的指针清空
        other.count_ = 0;        // 4. 把 other 的长度清零
    }

    // 【5. 移动赋值函数】当 b 和 c 都已经创建好了，写 c = std::move(b); 时自动执行。
    // 语法拆解：
    //   - this 是当前对象 c 自己的地址，&other 是右边对象 b 的地址。
    //   - 第一步必须先判断 if (this != &other)：防止有人写出 c = std::move(c);（自己转给自己）时误把自己的显存释放掉。
    //   - 第二步必须先释放 c 自己原本占用的旧显存 if (ptr_) cudaFree(ptr_);，然后再接管 other 的指针并清空 other。
    GpuBuffer& operator=(GpuBuffer&& other) noexcept {
        if (this != &other) {
            if (ptr_) {
                cudaFree(ptr_);  // 1. 先释放当前对象原本持有的旧显存（防内存泄漏）
            }
            ptr_ = other.ptr_;   // 2. 接管 other 的显存地址
            count_ = other.count_;
            other.ptr_ = nullptr;// 3. 把 other 清空
            other.count_ = 0;
        }
        return *this;            // 返回当前对象自身的引用
    }

    // 普通成员函数：取出底层显存裸指针和元素个数
    T* data() const noexcept { return ptr_; }
    std::size_t size() const noexcept { return count_; }
};

// ==================== 第二部分：在流水线 (Stream) 里跑核函数 ====================
// 核函数 1：用第 4 关学过的网格跨步循环，填入 a[i] = 0, 1, 2, ...
__global__ void fill_seq_kernel(float* a, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = tid; i < n; i += stride) {
        a[i] = float(i);
    }
}

// 核函数 2：用网格跨步循环，把每个元素乘上 scale 再加 bias
__global__ void scale_bias_kernel(float* a, int n, float scale, float bias) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = tid; i < n; i += stride) {
        a[i] = a[i] * scale + bias;
    }
}

// 把整套 GPU 操作放进指定的 stream 流水线中
void run_pipeline_async(float* d_a, int* d_status, int n, cudaStream_t stream) {
    // 1. cudaMemsetAsync：在指定的 stream 流水线里把 d_status 的 4 个字节（sizeof(int)）清零。
    //    注意：哪怕 n == 0，状态清零也要执行。
    cuda_check(cudaMemsetAsync(d_status, 0, sizeof(int), stream));

    // 2. 边界保护：如果数组长度 n <= 0，直接 return，千万不能往下走启动核函数！
    //    （因为 n == 0 时算出的 grid 是 0，启动 <<<0, 256>>> 会报错 cudaErrorInvalidConfiguration）
    if (n <= 0) return;

    int block = 256;
    int grid = (n - 1) / block + 1;

    // 3. 核函数完整四参数启动语法：<<<grid, block, 动态共享内存字节数, stream>>>
    //    这里不用动态共享内存，所以第 3 个参数写 0，第 4 个参数传入流水线 stream。
    //    把两个核函数先后塞进同一个 stream，GPU 硬件会自动排队：等 fill_seq_kernel 全部算完，才跑 scale_bias_kernel。
    fill_seq_kernel<<<grid, block, 0, stream>>>(d_a, n);
    //    核函数启动没有返回值，必须紧跟一句 cudaGetLastError() 抓取启动参数错误（比如 block > 1024）
    cuda_check(cudaGetLastError());

    scale_bias_kernel<<<grid, block, 0, stream>>>(d_a, n, 2.0f, 10.0f);
    cuda_check(cudaGetLastError());
}

int main() {
    // ================= 实验 1：测试 GpuBuffer 的转交与防呆能力 =================
    printf("=== 实验 1：测试 GpuBuffer 自动管家 ===\n");
    GpuBuffer<float> empty_buf(0);
    printf("1) 空对象 empty_buf(0): data = %p, size = %zu\n",
           (void*)empty_buf.data(), empty_buf.size());

    GpuBuffer<float> a(4);
    float* orig_ptr = a.data();
    printf("2) 创建 a(4): a.data = %p, a.size = %zu\n", (void*)a.data(), a.size());

    // 移动构造：把 a 的显存转交给新对象 b（转交后 a 变成空指针）
    GpuBuffer<float> b(std::move(a));
    printf("3) 转交给 b 后: a.data = %p, a.size = %zu | b.data = %p, b.size = %zu (地址没变: %s)\n",
           (void*)a.data(), a.size(), (void*)b.data(), b.size(),
           (b.data() == orig_ptr && a.data() == nullptr) ? "是" : "否");

    // 移动赋值：把 b 的显存转交给已有对象 c（c 原本的 2 个元素显存会被自动先释放）
    GpuBuffer<float> c(2);
    c = std::move(b);
    printf("4) 转交给 c 后: b.data = %p, b.size = %zu | c.data = %p, c.size = %zu\n",
           (void*)b.data(), b.size(), (void*)c.data(), c.size());

    // 自移动测试：自己转给自己 c = std::move(c)，指针依然完好
    c = std::move(c);
    printf("5) 自己转给自己后: c.data = %p, c.size = %zu (依然完好: %s)\n",
           (void*)c.data(), c.size(), (c.data() == orig_ptr && c.size() == 4) ? "是" : "否");

    // ================= 实验 2：测试 CUDA Stream 异步流水线 =================
    printf("\n=== 实验 2：测试 CUDA Stream 异步流水线 ===\n");
    cudaStream_t stream;
    cuda_check(cudaStreamCreate(&stream)); // 创建一条异步流

    const int N = 1025; // 1024 + 1，故意留 1 个余数测尾部边界
    GpuBuffer<float> d_arr(N);
    GpuBuffer<int> d_status(1);
    std::vector<float> h_arr(N, 0.0f);

    // 先测一次 n = 0 空数组，确认不会报错崩溃
    run_pipeline_async(d_arr.data(), d_status.data(), 0, stream);

    // 再测正式的 N = 1025：依次在 stream 中执行 a[i] = i，再执行 a[i] = a[i] * 2 + 10
    run_pipeline_async(d_arr.data(), d_status.data(), N, stream);

    // 把算好的结果放进同一个 stream 排队异步拷回 CPU
    cuda_check(cudaMemcpyAsync(h_arr.data(), d_arr.data(), N * sizeof(float),
                               cudaMemcpyDeviceToHost, stream));

    // 关键一步：CPU 去读 h_arr 之前，必须等这条 stream 流水线里的活全部干完！
    cuda_check(cudaStreamSynchronize(stream));

    printf("流水线计算结果: h_arr[0] = %.1f (期望 10.0), h_arr[1024] = %.1f (期望 2058.0)\n",
           h_arr[0], h_arr[1024]);

    cuda_check(cudaStreamDestroy(stream)); // 销毁异步流
    return 0; // main 函数结束，d_arr、d_status、c 自动触发 ~GpuBuffer() 释放显存！
}
```

3. 编译并运行：

```bash
cmake -S /root/cuda_tutorial -B /root/cuda_tutorial/build
cmake --build /root/cuda_tutorial/build
/root/cuda_tutorial/build/07_raii_stream
```

## 深入原理与避坑剖析（跑通代码后再看这 4 个问题）

1. **直接复制这个显存指针会出什么问题？移动时转交了什么？**
   - 如果允许直接复制对象（`GpuBuffer b = a;`），C++ 默认只会把指针地址 `ptr_` 的数字抄一份给 `b`（浅拷贝），不会去 GPU 新开显存。此时 `a` 和 `b` 指向同一块显存：一方面通过其中一个修改数据会污染另一个；更致命的是当其中一个对象先析构调 `cudaFree` 后，另一个对象的指针就变成了悬垂指针（Use-After-Free），等第二个对象析构时又会对同一地址调第二次 `cudaFree`（重复释放 Double-Free）导致崩溃。
   - 移动（`std::move`）时，只是在 CPU 上把指针地址 `ptr_` 和长度 `count_` 的所有权转交给新对象，同时把原对象的 `ptr_` 置为 `nullptr`、`count_` 置为 `0`，不需要搬动 GPU 显存里的任何数据，始终保证一块显存只有一个对象负责释放。
2. **同一条 `cudaStream_t` 如何安排先后顺序？CPU 读取结果前需要什么同步？**
   - 同一条 `cudaStream_t` 内部是严格按提交顺序排队执行的（FIFO）：前一个操作（如 `cudaMemsetAsync` 或前一个核函数）在 GPU 上彻底完成后，后一个核函数或 `cudaMemcpyAsync` 才会开始，中间不需要 CPU 同步。
   - 但核函数启动与异步拷贝对 CPU 都是立即返回的，所以 CPU 在读取主机内存结果前，必须调用 `cudaStreamSynchronize(stream)` 或 `cudaDeviceSynchronize()`（或者使用同步版 `cudaMemcpy`）等待 GPU 完成。
3. **如果异步任务引用了 `GpuBuffer` 对象，什么时候才能安全地销毁它？**
   - 必须等**引用该缓冲区的所有异步核函数和异步传输全部执行完毕（即完成流同步或设备同步）之后**，才能销毁 `GpuBuffer` 对象。如果异步任务还在流中排队或正在 GPU 上执行时就提前析构 `GpuBuffer` 调用了 `cudaFree`，GPU 将会读写已释放的显存，触发非法内存访问。
4. **为什么 `n == 0` 时既不申请显存，也不启动核函数？**
   - 空对象 `n == 0` 不需要占用显存，直接保持 `ptr_ = nullptr` 最安全；而在启动核函数时，若 `n == 0` 算出的 `grid` 为 `0`，启动 `<<<0, block>>>` 会触发 `cudaErrorInvalidConfiguration` 错误，因此遇到 `n == 0` 应直接返回，不启动核函数。

---

# 第八关：网格跨步填充与 `atomicAdd` 并行条件筛选

在第 6 关里，我们用 `atomicAdd(&dev_sum, val)` 把所有线程的数加进同一个变量，当时我们只关心它能防数据竞争，**忽略了 `atomicAdd` 的返回值**。

如果现在换一个场景：成千上万个线程同时在数组 `a` 里找大于等于 `0` 的数，找到后要紧挨着塞进输出数组 `out` 里（中间不能留空洞）。因为事先不知道哪些位置满足条件，没法提前给每个线程固定分配输出下标。这一关我们写一个程序 `08_atomic_compact.cu`，完成两件具体的事：

1. **用网格跨步循环填满任意长度的 GPU 数组**：无论数组长度 `n` 是多少（哪怕尾部不足一个 Block，或者 `n == 0`），都能安全填充且不漏掉尾部元素。
2. **把 `int pos = atomicAdd(count, 1)` 当成「硬件排队取号机」做并行筛选**：每个满足条件的线程去原子计数器领一个独一无二的写入位置 `pos`，把保留下来的元素紧凑地写进 `out[pos]`，同时统计出总个数。

## 动手写 `08_atomic_compact.cu`

1. 先在 GPU 机器上打开 `vim /root/cuda_tutorial/CMakeLists.txt`，在文件末尾加一行：

```cmake
add_executable(08_atomic_compact 08_atomic_compact.cu)
```

2. 打开 `vim /root/cuda_tutorial/08_atomic_compact.cu`，敲入下面的代码（所有新语法的解释都直接写在了对应代码行的上方注释里，不用上下翻页对照）：

```cpp
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <algorithm>
#include <stdexcept>
#include <vector>

inline void cuda_check(cudaError_t status) {
    if (status != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(status));
    }
}

// ==================== 1. 填充核函数：用网格跨步循环填入数据 ====================
// 语法拆解：
//   - tid 是当前线程的全局起点，stride 是整个网格的总线程数。
//   - 用 for (int i = tid; i < n; i += stride) 让每个线程跨步处理多个元素，
//     哪怕数组长度 n 比总线程数大得多，或者尾部不足一个 Block，也绝不会漏掉任何元素。
//   - cosf(float(i))：先用 float(i) 把整数下标转成单精度浮点数，再调 GPU 单精度余弦函数 cosf。
__global__ void fill_wave_kernel(float* a, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = tid; i < n; i += stride) {
        a[i] = cosf(float(i));
    }
}

// ==================== 2. 筛选核函数：用 atomicAdd「排队取号」挑出非负数 ====================
// 语法拆解：int pos = atomicAdd(count, 1);
//   - atomicAdd(count, 1) 会在 GPU 硬件层面一口气做完两件事（中间绝不会被别的线程插队打断）：
//     1) 先读取显存里 *count 当前的【旧值】（比如刚清零时是 0），并把这个【加之前的旧值】作为返回值赋给 pos；
//     2) 紧接着把显存里的 *count 加上 1（变成 1）。
//   - 就像银行大厅的排队叫号机：第 1 个按按钮的线程领到号码 pos = 0（同时计数器变成 1），
//     第 2 个按按钮的线程领到号码 pos = 1（同时计数器变成 2）……
//   - 这样每个满足 a[i] >= 0.0f 的线程都能领到一个独一无二、绝不撞车的槽位下标 pos，
//     然后把当前元素 a[i] 本身写进 out[pos]！
__global__ void filter_atomic_kernel(const float* a, float* out, int* count, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = tid; i < n; i += stride) {
        if (a[i] >= 0.0f) {
            int pos = atomicAdd(count, 1); // 1. 原子领号：拿到专属写入下标 pos，同时总个数 +1
            out[pos] = a[i];               // 2. 把当前元素本身 a[i] 写入专属槽位 out[pos]
        }
    }
}

// ==================== 3. 异步填充包装函数 ====================
void fill_wave_async(float* d_a, int n, cudaStream_t stream) {
    // 边界保护：n <= 0 时直接返回，千万不能往下走启动 <<<0, 256>>> 核函数！
    if (n <= 0) return;

    int block = 256;
    int grid = (n - 1) / block + 1; // 向上取整切分 Block

    fill_wave_kernel<<<grid, block, 0, stream>>>(d_a, n);
    cuda_check(cudaGetLastError()); // 检查核函数启动参数是否合法
}

// ==================== 4. 异步原子筛选包装函数 ====================
void filter_atomic_async(const float* d_a, float* d_out, int* d_count,
                         int n, cudaStream_t stream) {
    // 【关键一步】：每次筛选前，必须在同一个 stream 里用 cudaMemsetAsync 把计数器 d_count 清零！
    // 为什么要把这句放在 if (n <= 0) return; 的【前面】？
    //   1) 哪怕传入的是空数组 n == 0，也必须保证把 *d_count 清零成 0；
    //   2) 如果外部拿着同一个 d_count 连续调用多次筛选函数，不清零就会在上一次的计数值上继续累加，导致结果翻倍越界！
    cuda_check(cudaMemsetAsync(d_count, 0, sizeof(int), stream));

    if (n <= 0) return;

    int block = 256;
    int grid = (n - 1) / block + 1;

    filter_atomic_kernel<<<grid, block, 0, stream>>>(d_a, d_out, d_count, n);
    cuda_check(cudaGetLastError());
}

int main() {
    const int N = 1025; // 256 * 4 + 1，故意留 1 个余数验证尾部边界
    cudaStream_t stream;
    cuda_check(cudaStreamCreate(&stream));

    float* d_a = nullptr;
    float* d_out = nullptr;
    int* d_count = nullptr;
    cuda_check(cudaMalloc(&d_a, N * sizeof(float)));
    cuda_check(cudaMalloc(&d_out, N * sizeof(float)));
    cuda_check(cudaMalloc(&d_count, sizeof(int)));

    // 在同一条 stream 流水线中：先填数据 -> 再连续跑两次筛选（验证第二次 count 不会翻倍）
    fill_wave_async(d_a, N, stream);
    filter_atomic_async(d_a, d_out, d_count, N, stream);
    filter_atomic_async(d_a, d_out, d_count, N, stream);

    // 异步拷回计数器和筛选结果，等待 stream 全部执行完毕
    int h_count = -1;
    std::vector<float> h_out(N);
    cuda_check(cudaMemcpyAsync(&h_count, d_count, sizeof(int), cudaMemcpyDeviceToHost, stream));
    cuda_check(cudaStreamSynchronize(stream)); // 先等 h_count 拷回 CPU
    cuda_check(cudaMemcpyAsync(h_out.data(), d_out, h_count * sizeof(float), cudaMemcpyDeviceToHost, stream));
    cuda_check(cudaStreamSynchronize(stream)); // 再等 h_out 拷回 CPU

    // 在 CPU 上按同样规则算一份标准答案做核对
    std::vector<float> expected;
    for (int i = 0; i < N; ++i) {
        float v = std::cos(float(i));
        if (v >= 0.0f) expected.push_back(v);
    }

    printf("GPU 筛出非负数个数: %d (CPU 期望个数: %zu)\n", h_count, expected.size());
    printf("排序前前 5 个元素对比（观察原子筛选的乱序现象）:\n");
    printf("  GPU 实际前 5 个: %.4f, %.4f, %.4f, %.4f, %.4f\n",
           h_out[0], h_out[1], h_out[2], h_out[3], h_out[4]);
    printf("  CPU 顺序前 5 个: %.4f, %.4f, %.4f, %.4f, %.4f\n",
           expected[0], expected[1], expected[2], expected[3], expected[4]);

    // 排序后比对元素集合与重复次数是否完全一致
    std::sort(h_out.begin(), h_out.begin() + h_count);
    std::sort(expected.begin(), expected.end());
    bool match = (h_count == (int)expected.size());
    for (int i = 0; match && i < h_count; ++i) {
        if (std::abs(h_out[i] - expected[i]) > 1e-5f) match = false;
    }
    printf("排序后全部元素核对: %s\n", match ? "完全一致 (PASS)" : "不一致 (FAIL)");

    cudaFree(d_a);
    cudaFree(d_out);
    cudaFree(d_count);
    cudaStreamDestroy(stream);
    return 0;
}
```

3. 编译并运行：

```bash
cmake -S /root/cuda_tutorial -B /root/cuda_tutorial/build
cmake --build /root/cuda_tutorial/build
/root/cuda_tutorial/build/08_atomic_compact
```

## 深入原理与避坑剖析（跑通代码后再看这 3 个问题）

1. **初学者写并行填充与筛选时，为什么很容易写出下面这段「四处全错」的代码？**
   看下面这段典型的错误核函数：
   ```cpp
   __global__ void broken(int* a, int* out, int* count, int n) {
       int i = blockIdx.x * blockDim.x + threadIdx.x;
       if (i < n) return;
       a[i] = sinf(i);
       if (a[i] >= 0) {
           int loc = *count;
           *count += 1;
           out[loc] = n;
       }
   }
   ```
   - **坑 1（下标边界条件写反，且缺跨步循环）**：`if (i < n) return;` 把所有合法下标 `0 <= i < n` 的线程全部直接退出了，反而让 `i >= n` 的越界线程继续往下跑并写显存，触发越界非法访问；并且没有写网格跨步循环，当总线程数小于 `n` 时会漏掉后面的元素。单步应写 `if (i >= n) return;`，通用写法应为 `for (int i = tid; i < n; i += stride)`。
   - **坑 2（数组数值类型写错，精度截断）**：参数 `a` 和 `out` 被声明成了整型指针 `int*`，而 `sinf(i)` 返回的是 `[-1.0f, 1.0f]` 的单精度浮点数 `float`，存入 `int` 会被直接向零截断成整数 `0`，丢失所有小数。应改为 `float*`。
   - **坑 3（非原子计数引发严重数据竞争 Data Race）**：`int loc = *count; *count += 1;` 是无锁的普通显存读写。成百上千个满足条件的线程会同时读到同一个旧的 `*count` 值作为 `loc`，不仅把结果写进同一个 `out[loc]` 互相覆盖，还会导致最终 `*count` 远小于真实保留个数。应改为硬件原子指令 `int loc = atomicAdd(count, 1);`。
   - **坑 4（写入输出数组的值写错）**：`out[loc] = n;` 把数组总长度 `n` 写进了输出数组，导致输出里装的全是常数 `n`。应改为写入当前筛出来的元素本身 `out[loc] = a[i];`。
2. **为什么 `atomicAdd` 计数筛选能保证「个数和元素不丢」，却不能保证「输出顺序与原数组一致」？**
   - **不丢不重的原因**：`atomicAdd(count, 1)` 在 GPU 硬件 L2 Cache 原子单元上是严格排队执行的，每次调用必定返回互不相同的旧整数 `0, 1, 2, ...`，因此每个满足条件的元素拿到的写入下标 `pos` 绝对唯一，绝不会撞车覆盖。
   - **顺序打乱的原因**：正如第 2 关实测过的，不同 Block（跨 SM）以及同一 Block 内不同 Warp 之间的硬件调度顺序是不确定的。比如第 500 号线程所在的 Warp 可能比第 0 号线程所在的 Warp 更早冲到 `atomicAdd` 领走 `0` 号槽位，导致 `a[500]` 被写进 `out[0]`，而 `a[0]` 反而排到了后面。
3. **把「填数组 $\rightarrow$ 清零计数器 $\rightarrow$ 原子筛选」放进同一条 `cudaStream_t` 时，先后依赖如何保证？CPU 读取结果前要等什么？**
   - **同一 `stream` 内部**：把填充核函数、`cudaMemsetAsync(count, 0, ...)` 和筛选核函数都放进同一个 `cudaStream_t stream`，CUDA 硬件会保证同一条流内的操作严格按提交顺序（FIFO）依次执行——必须等前一个核函数或清零操作在 GPU 上彻底完成后，后一个核函数才会开始执行，中间无需 CPU 同步。
   - **CPU 读取前**：因为核函数启动和异步操作对 CPU 都是立即返回的，CPU 在读取主机侧的 `count` 或 `out` 结果之前，必须调用 `cudaStreamSynchronize(stream)` 或 `cudaDeviceSynchronize()`（或使用同步版 `cudaMemcpy`）等待流中任务全部完成。

---

# 第九关：用排他前缀和（Exclusive Scan）实现保序并行筛选

在第 8 关里，我们用 `atomicAdd(count, 1)` 实现了并行条件筛选，但它有一个明显的副作用：**谁先抢到原子计数器，谁就先写进输出数组，导致输出元素的先后顺序被彻底打乱了**。

如果业务要求**筛选出来的非负数必须严格保持它们在原数组里的先后顺序**（例如输入 `{-2, 0, 3, -1, 3}`，输出必须是 `{0, 3, 3}`，且不能靠事后排序糊弄），GPU 该怎么在成千上万个线程同时干活的前提下，让每个线程精确算出自己应该写到 `out` 的第几个格子？答案就是 GPU 并行算法中最经典的「三步流压缩法（Mark $\rightarrow$ Scan $\rightarrow$ Scatter）」：

1. **第一步 Mark（打标记）**：每个线程检查 `a[i]`，需要保留的在 `flags[i]` 写 `1`，丢弃的写 `0`。
2. **第二步 Exclusive Scan（排他前缀和）**：对 `flags` 做排他前缀和得到 `offsets`，让 `offsets[i]` 等于「下标 `i` 前面一共有多少个 `1`」——既然我前面有 `offsets[i]` 个元素被保留，那我在输出数组 `out` 里的座位号就必然是 `offsets[i]`！
3. **第三步 Scatter（按址搬运）**：所有 `flags[i] == 1` 的线程直接把 `a[i]` 写进 `out[offsets[i]]`，并由最后一个元素的 `offsets[n - 1] + flags[n - 1]` 直接得出总个数 `*count`。

```text
下标 i       :    0     1     2     3     4
输入 a[i]    : [ -2,    0,    3,   -1,    3 ]
                  │     │     │     │     │
1. Mark      : [  0,    1,    1,    0,    1 ]  <-- flags[i]：非负标 1，负数标 0
                  │     │     │     │     │
2. Excl Scan : [  0,    0,    1,    2,    2 ]  <-- offsets[i] = flags[0] + ... + flags[i-1]
                        │     │           │        （即“我前面有几个 1”，正好是写入下标！）
3. Scatter   :          ▼     ▼           ▼
   out[...]  :       out[0]=0 out[1]=3  out[2]=3  --> out = { 0, 3, 3 }
   *count    : offsets[4] + flags[4] = 2 + 1 = 3
```

## 动手写 `09_scan_compact.cu`

1. 先在 GPU 机器上打开 `vim /root/cuda_tutorial/CMakeLists.txt`，在文件末尾加一行：

```cmake
add_executable(09_scan_compact 09_scan_compact.cu)
```

2. 打开 `vim /root/cuda_tutorial/09_scan_compact.cu`，敲入下面的代码（所有新语法的解释都直接写在了对应代码行的上方注释里，不用上下翻页对照）：

```cpp
#include <cuda_runtime.h>
#include <thrust/scan.h>
#include <thrust/system/cuda/execution_policy.h>
#include <cstdio>
#include <stdexcept>
#include <vector>

inline void cuda_check(cudaError_t status) {
    if (status != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(status));
    }
}

// ==================== 第一步：标记核函数 (Mark) ====================
// 每个线程检查自己负责的元素 a[i]：
//   - 如果 a[i] >= 0.0f（需要保留），就在 flags[i] 写 1；
//   - 如果 a[i] <  0.0f（需要丢弃），就在 flags[i] 写 0。
// 例如输入 a     = {-2, 0, 3, -1, 3}
// 跑完得到 flags = { 0, 1, 1,  0, 1}
__global__ void mark_nonneg_kernel(const float* a, int* flags, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = tid; i < n; i += stride) {
        if (a[i] >= 0.0f) {
            flags[i] = 1;
        } else {
            flags[i] = 0;
        }
    }
}

// ==================== 第三步：按址搬运核函数 (Scatter) ====================
// 经过第二步的排他前缀和后，offsets[i] 里存的正好是「下标 i 前面一共有多少个 1」，
// 也就是当前元素 a[i] 在输出数组 out 里的专属写入下标！
//   - 如果 flags[i] == 1，直接把 a[i] 写进 out[offsets[i]]，无需任何原子操作，天然保序且绝不撞车！
//   - 总共保留了多少个元素怎么算？
//     看最后一个位置 n - 1：它前面有 offsets[n - 1] 个保留元素，再加上它自己是否保留 flags[n - 1]，
//     所以总个数就是 offsets[n - 1] + flags[n - 1]！我们让处理 i == n - 1 的那个线程顺手写进 *count 即可。
__global__ void scatter_keep_order_kernel(const float* a, float* out,
                                          const int* flags, const int* offsets,
                                          int* count, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = tid; i < n; i += stride) {
        if (flags[i] == 1) {
            int pos = offsets[i];
            out[pos] = a[i];
        }
        if (i == n - 1) {
            *count = offsets[n - 1] + flags[n - 1];
        }
    }
}

// ==================== 主机端流水线包装函数：三步保序筛选 ====================
void filter_stable_async(const float* d_a, float* d_out, int* d_count,
                         int* d_flags, int* d_offsets, int n,
                         cudaStream_t stream) {
    // 边界保护：如果输入为空数组 (n <= 0)，后面不会启动 scatter_kernel，
    // 所以必须在这里用 cudaMemsetAsync 把计数器 d_count 在 stream 里清零为 0，然后直接返回！
    if (n <= 0) {
        cuda_check(cudaMemsetAsync(d_count, 0, sizeof(int), stream));
        return;
    }

    int block = 256;
    int grid = (n - 1) / block + 1;

    // 第 1 步：在 stream 中启动标记核函数，算出 0/1 标记数组 d_flags
    mark_nonneg_kernel<<<grid, block, 0, stream>>>(d_a, d_flags, n);
    cuda_check(cudaGetLastError());

    // 第 2 步：调用 Thrust 的排他前缀和 exclusive_scan，根据 d_flags 算出写入下标数组 d_offsets。
    // 语法拆解：
    //   - thrust::cuda::par.on(stream) 叫「CUDA 流执行策略」：
    //     告诉 Thrust 必须在 GPU 上并行计算，并且把任务塞进我们指定的这条 stream 流水线里排队
    //     （等第 1 步 mark_nonneg_kernel 算完 flags 后再扫，扫完再跑第 3 步）。
    //   - d_flags, d_flags + n：输入范围的【起点指针】和【终点指针】（共 n 个元素）。
    //   - d_offsets：输出数组的【起点指针】。
    //   - 什么是「排他前缀和（exclusive_scan）」？
    //     就是第 i 项只累加它【前面】的 flags[0 .. i-1]，不包含 flags[i] 自己（第 0 项前面没元素，所以 offsets[0] 恒为 0）。
    thrust::exclusive_scan(thrust::cuda::par.on(stream),
                           d_flags, d_flags + n, d_offsets);

    // 第 3 步：在同一个 stream 中启动搬运核函数，把保留元素按序写进 d_out，并由末尾线程算出 *d_count
    scatter_keep_order_kernel<<<grid, block, 0, stream>>>(
        d_a, d_out, d_flags, d_offsets, d_count, n);
    cuda_check(cudaGetLastError());
}

int main() {
    cudaStream_t stream;
    cuda_check(cudaStreamCreate(&stream));

    // 用一个一目了然的 5 元素数组验证三步全过程
    std::vector<float> h_a = {-2.0f, 0.0f, 3.0f, -1.0f, 3.0f};
    const int N = (int)h_a.size();

    float *d_a = nullptr, *d_out = nullptr;
    int *d_count = nullptr, *d_flags = nullptr, *d_offsets = nullptr;
    cuda_check(cudaMalloc(&d_a, N * sizeof(float)));
    cuda_check(cudaMalloc(&d_out, N * sizeof(float)));
    cuda_check(cudaMalloc(&d_count, sizeof(int)));
    cuda_check(cudaMalloc(&d_flags, N * sizeof(int)));
    cuda_check(cudaMalloc(&d_offsets, N * sizeof(int)));

    cuda_check(cudaMemcpyAsync(d_a, h_a.data(), N * sizeof(float), cudaMemcpyHostToDevice, stream));

    // 连续调用两次（中间还穿插一次 n = 0 测试），验证状态完全覆盖、无需每次手动清零 count
    filter_stable_async(d_a, d_out, d_count, d_flags, d_offsets, 0, stream);
    filter_stable_async(d_a, d_out, d_count, d_flags, d_offsets, N, stream);
    filter_stable_async(d_a, d_out, d_count, d_flags, d_offsets, N, stream);

    // 拷回中间数组 flags、offsets 以及最终的 count 和 out，直观打印每一步的结果
    std::vector<int> h_flags(N), h_offsets(N);
    std::vector<float> h_out(N);
    int h_count = -1;

    cuda_check(cudaMemcpyAsync(h_flags.data(), d_flags, N * sizeof(int), cudaMemcpyDeviceToHost, stream));
    cuda_check(cudaMemcpyAsync(h_offsets.data(), d_offsets, N * sizeof(int), cudaMemcpyDeviceToHost, stream));
    cuda_check(cudaMemcpyAsync(&h_count, d_count, sizeof(int), cudaMemcpyDeviceToHost, stream));
    cuda_check(cudaMemcpyAsync(h_out.data(), d_out, N * sizeof(float), cudaMemcpyDeviceToHost, stream));
    cuda_check(cudaStreamSynchronize(stream));

    printf("输入数组 a      : { %.0f, %.0f, %.0f, %.0f, %.0f }\n",
           h_a[0], h_a[1], h_a[2], h_a[3], h_a[4]);
    printf("第 1 步 flags   : { %2d, %2d, %2d, %2d, %2d } (1 代表保留，0 代表丢弃)\n",
           h_flags[0], h_flags[1], h_flags[2], h_flags[3], h_flags[4]);
    printf("第 2 步 offsets : { %2d, %2d, %2d, %2d, %2d } (我前面有几个保留元素 = 我的写入下标)\n",
           h_offsets[0], h_offsets[1], h_offsets[2], h_offsets[3], h_offsets[4]);
    printf("第 3 步 count   : offsets[4](%d) + flags[4](%d) = %d\n",
           h_offsets[N - 1], h_flags[N - 1], h_count);
    printf("保序输出 out    : { %.0f, %.0f, %.0f } (期望 { 0, 3, 3 })\n",
           h_out[0], h_out[1], h_out[2]);

    cudaFree(d_a);
    cudaFree(d_out);
    cudaFree(d_count);
    cudaFree(d_flags);
    cudaFree(d_offsets);
    cudaStreamDestroy(stream);
    return 0;
}
```

3. 编译并运行：

```bash
cmake -S /root/cuda_tutorial -B /root/cuda_tutorial/build
cmake --build /root/cuda_tutorial/build
/root/cuda_tutorial/build/09_scan_compact
```

## 深入原理与避坑剖析（跑通代码后再看这 3 个问题）

1. **手算一遍 `a = {-2, 0, 3, -1, 3}` 的每一步数组：为什么排他前缀和算出来的正好是写入下标？**
   - **第 1 步（打标记 `flags`）**：把大于等于 `0` 的位置标为 `1`，负数标为 `0`，得到 `flags = {0, 1, 1, 0, 1}`。
   - **第 2 步（排他前缀和 `offsets`）**：`offsets[i]` 只加 `i` 左边的 `flags[0 .. i-1]`（不含自己）：
     - `offsets[0] = 0`（左边没有元素）
     - `offsets[1] = flags[0] = 0`
     - `offsets[2] = flags[0] + flags[1] = 0 + 1 = 1`
     - `offsets[3] = flags[0] + flags[1] + flags[2] = 0 + 1 + 1 = 2`
     - `offsets[4] = flags[0] + flags[1] + flags[2] + flags[3] = 0 + 1 + 1 + 0 = 2`
     所以 `offsets = {0, 0, 1, 2, 2}`。
   - **第 3 步（写出 `out` 与总个数 `count`）**：只看 `flags[i] == 1` 的三个位置（下标 `1, 2, 4`）：
     - `i = 1`（值为 `0`）：写入 `out[offsets[1]] = out[0] = 0`；
     - `i = 2`（值为 `3`）：写入 `out[offsets[2]] = out[1] = 3`；
     - `i = 4`（值为 `3`）：写入 `out[offsets[4]] = out[2] = 3`；
     最终 `out = {0, 3, 3}`，总个数 `count = offsets[4] + flags[4] = 2 + 1 = 3`。
2. **为什么两个保留的元素绝不会写到同一个位置？它和第 8 关的原子筛选有什么区别？**
   - **为什么绝不重叠写入**：设任意两个保留元素的下标满足 `i < j`（即 `flags[i] == 1` 且 `flags[j] == 1`）。因为 `offsets[j]` 是前 `j` 项的非负累加，里面比 `offsets[i]` 至少多加了一项 `flags[i] = 1`，所以必有 `offsets[j] >= offsets[i] + 1 > offsets[i]`。也就是说，所有保留位置的 `offsets` 是严格单调递增的 `0, 1, 2, ...`，每个保留元素都有唯一对应的写入槽位，绝不会写到同一位置。
   - **与第 8 关原子筛选的区别**：第 8 关用 `atomicAdd` 领号时，槽位顺序取决于硬件 Warp/Block 谁先抢到原子指令，因此输出顺序是随机打乱的；而本关通过排他前缀和预先算好了确定的递增偏移量 `offsets[i]`，即使核函数里第 4 号线程比第 1 号线程先执行写入，第 4 号线程写的也是 `out[2]`、第 1 号线程写的也是 `out[0]`，最终输出顺序与输入顺序 100% 一致（稳定筛选）。
3. **为什么总个数等于 `offsets[n - 1] + flags[n - 1]`，且不用在每次筛选前先把 `count` 清零？**
   - 因为排他前缀和的定义是 `offsets[n - 1] = flags[0] + flags[1] + ... + flags[n - 2]`（前 `n - 1` 个元素里保留的个数），此时唯一没算进去的就是最后一个元素自己的标记 `flags[n - 1]`。把两者相加，正好等于全部 `n` 个元素里 `1` 的总和。
   - 当 `n > 0` 时，最后一个线程 `i == n - 1` 会用赋值语句 `*count = offsets[n - 1] + flags[n - 1];` 直接覆盖掉 `*count` 的旧值（不是 `+=` 累加），所以哪怕全负数（算出 `0 + 0 = 0`）、全零（算出 `(n - 1) + 1 = n`）或连续重复调用，结果都天然正确；唯独当 `n == 0` 时核函数不启动，才需要单独用 `cudaMemsetAsync(count, 0, sizeof(int), stream)` 把 `*count` 设为 `0`。

---

# 第十关：共享内存（Shared Memory）与分块矩阵乘法

前面 9 关处理的都是一维数组，每个数据基本上只读一次。但到了二维矩阵乘法 `C = A × B`（其中 `A` 是 `M × K`，`B` 是 `K × N`，`C` 是 `M × N`，在一维显存里按行存放为 `A[row * K + k]`、`B[k * N + col]`、`C[row * N + col]`），如果让每个线程各自去全局显存里读 `A` 的一整行和 `B` 的一整列，同一个 `16 × 16` 线程块里的 256 个线程会把相同的 `A` 行和 `B` 列从慢速全局显存里反反复复重复读取 **16 遍**！

为了把这 16 倍的重复读取省下来，这一关我们写一个程序 `10_tiled_2d.cu`，利用焊在 SM 芯片内部的超高速共享内存（`__shared__`），完成三件具体的事：

1. **用 `16 × 16` 线程块协作搬运小块（Tile）**：在块内开辟两块共享内存黑板 `As[16][16]` 和 `Bs[16][16]`，沿着 `K` 维度每次切一段宽为 16 的小块（`for (int base = 0; base < K; base += 16)`），让 256 个线程一人从全局显存搬 1 个数贴到黑板上，然后大家在黑板上高速共享读取 16 次。
2. **处理非 16 倍数的矩阵边缘（越界填 `0.0f`）**：当矩阵长宽不是 16 的整数倍（例如 `17 × 19 × 23`）时，超出 `A` 或 `B` 边界的载入位置统一填 `0.0f`（加 0 不会改变乘加结果），且**严禁边缘线程提前 `return`**。
3. **用两道 `__syncthreads()` 同步栅栏卡住块内节奏**：第 1 道栅栏等全块 256 个线程都把当前小块写进 `As`、`Bs` 后再开始算乘加；第 2 道栅栏等所有人都算完本轮小块后，才允许进入下一轮覆盖黑板。

## 图解 1：为什么二维线程坐标永远写成 `(ty, tx)` / `As[ty][tx]`，而不是 `(tx, ty)`？

- **原因 1（坐标系方向相反）**：在数学和图像的直角坐标系里，`x` 轴是**水平向右**的（对应矩阵里的**列号 `col`**），`y` 轴是**垂直向下**的（对应矩阵里的**行号 `row`**）。而在 C++ 二维数组和线性代数里，索引顺序永远是**「先行后列」**（`Matrix[row][col]`），把 `row -> ty`、`col -> tx` 代入进去，自然就是 `As[ty][tx]`！
- **原因 2（硬件合并访存铁律）**：在二维线程块中，GPU 硬件是按 `threadIdx.x (tx)` 连续编号来切分 Warp 的（即同一个 `ty` 下，`tx = 0, 1, 2, ..., 15` 的线程属于同一个 Warp）。因为矩阵按行主序存储，同一行相邻列的元素在显存里紧挨着，只有让变化最快的 `tx` 对应列号 `col`，同一个 Warp 的线程在读 `A[row * K + (base + tx)]`、读 `B[(base + ty) * N + col]`、写 `C[row * N + col]` 时访问的才是连续显存地址（1 次合并读取）；如果反过来把 `tx` 当行号，相邻线程就会跨行跳着读显存，带宽暴跌 16 倍！

```text
       ───► 水平方向 x (threadIdx.x = tx)  ==  矩阵的【列 col】
     │      tx=0     tx=1     tx=2     tx=3
  垂 │  ┌────────┬────────┬────────┬────────┐
  直 │  │(ty=0,  │(ty=0,  │(ty=0,  │(ty=0,  │  <-- 第 0 行 (ty = 0)：
  方 ▼  │ tx=0)  │ tx=1)  │ tx=2)  │ tx=3)  │      同一个 Warp 里 tx=0,1,2,3 紧挨着，
  向 y  ├────────┼────────┼────────┼────────┤      同时读 A[row][0..3] 正好是连续显存地址（合并访问！）
 (ty)   │(ty=1,  │(ty=1,  │(ty=1,  │(ty=1,  │  <-- 第 1 行 (ty = 1)
  ==    │ tx=0)  │ tx=1)  │ tx=2)  │ tx=3)  │
【行】  └────────┴────────┴────────┴────────┘

一句话口诀：
  - CUDA 网格按「图像坐标系 (x, y)」命名：x 是水平向右（列 col），y 是垂直向下（行 row）。
  - C++ 二维数组按「矩阵习惯 [行][列]」命名：第一个括号是行（y），第二个括号是列（x）。
  - 所以写二维下标永远是 As[行][列] = As[ty][tx]（绝不能写成 As[tx][ty]）！
```

## 图解 2：分块矩阵乘在数学上为什么完全等价于普通矩阵乘？

很多人第一次看分块矩阵乘法会犯迷糊：为什么切成一个个小方块乘起来再相加，就能等于整行乘整列？其实它在数学上只用到了最朴素的**加法结合律**——原本算 `C[row, col]` 需要把 `K` 个乘积一口气加完，现在只是把这 `K` 个乘积按每段 16 个切开，分轮次累加到同一个寄存器 `sum` 里。下面我们用一个能用肉眼算清的 **`4 × 4` 矩阵（切成 `2 × 2` 小块）** 把两轮全过程画出来：

```text
目标：用 1 个 2×2 线程块（4 个线程），算出 C 左上角的 2×2 小块：
  线程 (ty=0, tx=0) 算 C00      线程 (ty=0, tx=1) 算 C01
  线程 (ty=1, tx=0) 算 C10      线程 (ty=1, tx=1) 算 C11

【普通矩阵乘】：拿 A 的前 2 行（长 4） ×  B 的前 2 列（高 4），一口气连加 4 项：
  C00 = a00*b00 + a01*b10 + a02*b20 + a03*b30
  C01 = a00*b01 + a01*b11 + a02*b21 + a03*b31   <-- 痛点：a00..a03 被 C00 和 C01 从全局显存重复读了 2 遍！
  C10 = a10*b00 + a11*b10 + a12*b20 + a13*b30   <-- 痛点：b00..b30 被 C00 和 C10 从全局显存重复读了 2 遍！
  C11 = a10*b01 + a11*b11 + a12*b21 + a13*b31

=====================================================================================
【分块矩阵乘】：利用加法结合律，把 4 项连加拆成「前 2 项 + 后 2 项」两轮小块乘加：
  C00 = (a00*b00 + a01*b10)  +  (a02*b20 + a03*b30)
        └─ 第 1 轮 base=0 ─┘     └─ 第 2 轮 base=2 ─┘

                  矩阵 B (4×4)
               col 0..1   col 2..3
              ┌─────────┬─────────┐
      row 0..1│ b00 b01 │   ...   │ ◄── 第 1 轮 (base=0) 搬这块进 Bs[2][2]
              │ b10 b11 │         │
              ├─────────┼─────────┤
      row 2..3│ b20 b21 │   ...   │ ◄── 第 2 轮 (base=2) 搬这块进 Bs[2][2]
              │ b30 b31 │         │
              └─────────┴─────────┘
                  ▲
                  │ 逐轮小块相乘，结果累加到同一寄存器 sum
  矩阵 A (4×4)    ▼
  col 0..1  col 2..3          输出矩阵 C (4×4)
┌─────────┬─────────┐       ┌─────────────────┬─────┐
│ a00 a01 │ a02 a03 │       │ C00   C01       │ ... │
│ a10 a11 │ a12 a13 │ ────► │ C10   C11       │     │
├─────────┼─────────┤       ├─────────────────┼─────┤
│   ...   │   ...   │       │       ...       │ ... │
└─────────┴─────────┘       └─────────────────┴─────┘
    ▲         ▲
    │         └── 第 2 轮 (base=2) 搬这块进 As[2][2]
    └──────────── 第 1 轮 (base=0) 搬这块进 As[2][2]

-------------------------------------------------------------------------------------
▶ 第 1 轮 (base = 0)：4 个线程一人从全局显存搬 1 个数贴到黑板 As、Bs，算【前半截 (k = 0, 1)】
  As = ┌ a00  a01 ┐     Bs = ┌ b00  b01 ┐
       └ a10  a11 ┘          └ b10  b11 ┘
  每个线程在黑板上算: sum += As[ty][0]*Bs[0][tx] + As[ty][1]*Bs[1][tx]
    - 线程(ty=0, tx=0) 算 C00: sum = a00*b00 + a01*b10
    - 线程(ty=0, tx=1) 算 C01: sum = a00*b01 + a01*b11  (直接从黑板 As 读同伴搬来的 a00, a01！)
    - 线程(ty=1, tx=0) 算 C10: sum = a10*b00 + a11*b10  (直接从黑板 Bs 读同伴搬来的 b00, b10！)
    - 线程(ty=1, tx=1) 算 C11: sum = a10*b01 + a11*b11

▶ 第 2 轮 (base = 2)：擦掉黑板，4 个线程搬下一块进 As、Bs，往同一个 sum 里继续加【后半截 (k = 2, 3)】
  As = ┌ a02  a03 ┐     Bs = ┌ b20  b21 ┐
       └ a12  a13 ┘          └ b30  b31 ┘
  每个线程继续累加:   sum += As[ty][0]*Bs[0][tx] + As[ty][1]*Bs[1][tx]
    - 线程(ty=0, tx=0) 算 C00: sum = (a00*b00 + a01*b10) + (a02*b20 + a03*b30)  <-- 与普通矩阵乘分毫不差！
    - 线程(ty=0, tx=1) 算 C01: sum = (a00*b01 + a01*b11) + (a02*b21 + a03*b31)  <-- 与普通矩阵乘分毫不差！
    - 线程(ty=1, tx=0) 算 C10: sum = (a10*b00 + a11*b10) + (a12*b20 + a13*b30)  <-- 与普通矩阵乘分毫不差！
    - 线程(ty=1, tx=1) 算 C11: sum = (a10*b01 + a11*b11) + (a12*b21 + a13*b31)  <-- 与普通矩阵乘分毫不差！

（代码里只是把这里的 2×2 小块放大成了 16×16 小块，每轮算 16 项乘积，数学原理 100% 一模一样！）
```

## 动手写 `10_tiled_2d.cu`

1. 先在 GPU 机器上打开 `vim /root/cuda_tutorial/CMakeLists.txt`，在文件末尾加一行：

```cmake
add_executable(10_tiled_2d 10_tiled_2d.cu)
```

2. 打开 `vim /root/cuda_tutorial/10_tiled_2d.cu`，敲入下面的代码（所有新语法的解释都直接写在了对应代码行的上方注释里，不用上下翻页对照）：

```cpp
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <vector>

inline void cuda_check(cudaError_t status) {
    if (status != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(status));
    }
}

// ==================== 对照组：普通版矩阵乘法 (Naive GEMM) ====================
// 每个线程直接从慢速全局显存读取 A 的一整行 (K 个数) 和 B 的一整列 (K 个数)。
// 缺点：同一个 16x16 线程块里的 16 行线程，会把 B 的同一列从全局显存重复读 16 遍！
__global__ void gemm_naive_kernel(const float* A, const float* B,
                                  float* C, int M, int N, int K) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= M || col >= N) return;

    float sum = 0.0f;
    for (int k = 0; k < K; ++k) {
        sum += A[row * K + k] * B[k * N + col];
    }
    C[row * N + col] = sum;
}

// ==================== 核心主角：共享内存分块矩阵乘法 (Tiled GEMM) ====================
__global__ void gemm_tiled_kernel(const float* A, const float* B,
                                  float* C, int M, int N, int K) {
    // 【1. 声明片上共享内存 (__shared__)】
    // 语法拆解：
    //   - __shared__ 表示这两个 16x16 的二维数组分配在 SM 芯片内部的超高速共享内存（Shared Memory）上。
    //   - 同一个 Block 里的 256 个线程共享同一份 As 和 Bs（就像教室墙上的公共黑板）；
    //     不同 Block 之间各自有独立的黑板，互不干扰。
    __shared__ float As[16][16];
    __shared__ float Bs[16][16];

    // 【2. 计算当前线程在块内的局部坐标 (ty, tx) 和在矩阵 C 中的全局行列号 (row, col)】
    //   - 注意对应关系：x 维度对应列号 col（水平方向），y 维度对应行号 row（垂直方向）。
    //   - 这样同一个 Warp 里相邻的 threadIdx.x (0, 1, 2...) 对应的 col 也是连续的，读写内存时能触发合并访问！
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * 16 + ty;
    int col = blockIdx.x * 16 + tx;

    // 每个线程用自己的私有寄存器 sum 累加 C[row, col] 的点积结果
    float sum = 0.0f;

    // 【3. 沿着公共维度 K，每次切宽为 16 的一小块（Tile）推进】
    //   - 注意：前面千万【不能】写 if (row >= M || col >= N) return; 提前退出！
    //   - 因为哪怕某个边缘线程自己的输出坐标 (row, col) 超出了 C 的边界，
    //     它依然要帮同块的伙伴搬运 A 或 B 里的合法元素，并且必须一起参加下面的 __syncthreads() 同步！
    for (int base = 0; base < K; base += 16) {
        // ---- 步骤 A：256 个线程协作，把 A 和 B 的当前 16x16 小块搬进共享内存 As 和 Bs ----
        // 1) 搬 A 的元素：当前小块在 A 中的行号是 row，列号是 a_col = base + tx。
        //    如果行号 row < M 且列号 a_col < K，就从全局显存读 A[row * K + a_col]；
        //    否则说明超出了矩阵 A 的边界，填 0.0f（因为 0 乘任何数都是 0，不会污染后面的累加 sum）。
        int a_col = base + tx;
        if (row < M && a_col < K) {
            As[ty][tx] = A[row * K + a_col];
        } else {
            As[ty][tx] = 0.0f;
        }

        // 2) 搬 B 的元素：当前小块在 B 中的行号是 b_row = base + ty，列号是 col。
        //    如果行号 b_row < K 且列号 col < N，就从全局显存读 B[b_row * N + col]；
        //    否则超出了矩阵 B 的边界，同样填 0.0f。
        int b_row = base + ty;
        if (b_row < K && col < N) {
            Bs[ty][tx] = B[b_row * N + col];
        } else {
            Bs[ty][tx] = 0.0f;
        }

        // ---- 步骤 B：第 1 次块内同步屏障 (__syncthreads) ----
        // 为什么要等？因为计算 sum 时，当前线程 (ty, tx) 需要读取 As[ty][0..15] 和 Bs[0..15][tx]，
        // 这些格子是由同一个 Block 里的【其他线程】负责搬运的！
        // 必须等全块 256 个线程全部把数据写进 As 和 Bs 后，大家才能开始读黑板（防止“写后读 RAW”竞争）。
        __syncthreads();

        // ---- 步骤 C：在超高速共享内存上计算当前 16x16 小块的乘积累加 ----
        for (int t = 0; t < 16; ++t) {
            sum += As[ty][t] * Bs[t][tx];
        }

        // ---- 步骤 D：第 2 次块内同步屏障 (__syncthreads) ----
        // 为什么要等？如果跑得快的线程先算完了这 16 次乘加，直接冲进下一轮 for 循环（base += 16），
        // 它就会把下一轮的新数据写进 As[ty][tx] 和 Bs[ty][tx]，把黑板上旧数据擦掉！
        // 而此时跑得慢的线程可能还在用本轮的旧数据算 sum！所以必须等大家都用完本轮黑板，才能一起进下一轮（防止“读后写 WAR”竞争）。
        __syncthreads();
    }

    // 【4. 全部小块累加完毕后，只让落在矩阵 C 合法范围内的线程把结果写回全局显存】
    //   - 注意：当 K == 0 时，上面的 for 循环一次都不会执行，sum 保持初值 0.0f，
    //     这里依然会把 C 的所有合法位置正确写成 0.0f！
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// ==================== 主机端异步启动包装函数 ====================
void gemm_tiled_async(const float* d_A, const float* d_B, float* d_C,
                      int M, int N, int K, cudaStream_t stream) {
    // 边界保护：如果输出矩阵 C 的行数 M 或列数 N 为 0（输出为空），直接返回不启动核函数。
    // （注意：如果 M > 0 且 N > 0，哪怕 K == 0 也要正常启动核函数，把输出矩阵 C 清零！）
    if (M <= 0 || N <= 0) return;

    // 固定使用 16 x 16 的二维线程块（共 256 个线程）
    dim3 block(16, 16);
    // 二维网格向上取整：x 方向覆盖 N 列，y 方向覆盖 M 行
    dim3 grid((N - 1) / 16 + 1, (M - 1) / 16 + 1);

    gemm_tiled_kernel<<<grid, block, 0, stream>>>(d_A, d_B, d_C, M, N, K);
    cuda_check(cudaGetLastError());
}

int main() {
    cudaStream_t stream;
    cuda_check(cudaStreamCreate(&stream));

    // 故意选一个全不是 16 倍数的非方阵尺寸：M = 17, N = 19, K = 23
    const int M = 17, N = 19, K = 23;
    std::vector<float> h_A(M * K), h_B(K * N), h_C_ref(M * N, 0.0f), h_C_gpu(M * N, 0.0f);

    for (int i = 0; i < M * K; ++i) h_A[i] = float(i % 11 - 5) / 8.0f;
    for (int i = 0; i < K * N; ++i) h_B[i] = float(i % 7 - 3) / 4.0f;

    // 在 CPU 上算出标准答案 h_C_ref
    for (int r = 0; r < M; ++r) {
        for (int c = 0; c < N; ++c) {
            double s = 0.0;
            for (int k = 0; k < K; ++k) {
                s += double(h_A[r * K + k]) * double(h_B[k * N + c]);
            }
            h_C_ref[r * N + c] = float(s);
        }
    }

    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    cuda_check(cudaMalloc(&d_A, M * K * sizeof(float)));
    cuda_check(cudaMalloc(&d_B, K * N * sizeof(float)));
    cuda_check(cudaMalloc(&d_C, M * N * sizeof(float)));

    cuda_check(cudaMemcpyAsync(d_A, h_A.data(), M * K * sizeof(float), cudaMemcpyHostToDevice, stream));
    cuda_check(cudaMemcpyAsync(d_B, h_B.data(), K * N * sizeof(float), cudaMemcpyHostToDevice, stream));

    // 启动共享内存分块矩阵乘法
    gemm_tiled_async(d_A, d_B, d_C, M, N, K, stream);

    cuda_check(cudaMemcpyAsync(h_C_gpu.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost, stream));
    cuda_check(cudaStreamSynchronize(stream));

    // 计算最大绝对误差
    float max_err = 0.0f;
    for (int i = 0; i < M * N; ++i) {
        max_err = std::max(max_err, std::abs(h_C_gpu[i] - h_C_ref[i]));
    }
    printf("非 16 倍数矩阵乘法 (M=%d, N=%d, K=%d) 验证:\n", M, N, K);
    printf("  C[0, 0]   : GPU = %.6f, CPU = %.6f\n", h_C_gpu[0], h_C_ref[0]);
    printf("  C[16, 18] : GPU = %.6f, CPU = %.6f\n", h_C_gpu[M * N - 1], h_C_ref[M * N - 1]);
    printf("  最大绝对误差: %.2e (%s)\n", max_err, max_err < 1e-4f ? "PASS" : "FAIL");

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaStreamDestroy(stream);
    return 0;
}
```

3. 编译并运行：

```bash
cmake -S /root/cuda_tutorial -B /root/cuda_tutorial/build
cmake --build /root/cuda_tutorial/build
/root/cuda_tutorial/build/10_tiled_2d
```

## 深入原理与避坑剖析（跑通代码后再看这 5 个问题）

1. **每个线程负责哪个位置？搬运小块时为什么 `A` 的列号是 `base + tx`，而 `B` 的行号是 `base + ty`？**
   - **输出位置映射**：二维线程块大小为 `16 × 16`，块内局部坐标为 `tx = threadIdx.x`、`ty = threadIdx.y`。每个线程唯一负责输出矩阵 `C` 中的一个元素 `C[row, col]`，其中行号 `row = blockIdx.y * 16 + ty`，列号 `col = blockIdx.x * 16 + tx`。
   - **小块读取地址推导**：在第 `base / 16` 轮循环中，当前线程块需要把 `A` 的子块 `A[blockIdx.y*16 .. +15][base .. base+15]` 搬进 `As[16][16]`，因此线程 `(ty, tx)` 对应的行号就是 `row`，列号就是 `base + tx`（一维下标 `row * K + (base + tx)`）；同理，需要把 `B` 的子块 `B[base .. base+15][blockIdx.x*16 .. +15]` 搬进 `Bs[16][16]`，线程 `(ty, tx)` 对应的行号是 `base + ty`，列号是 `col`（一维下标 `(base + ty) * N + col`）。超出 `row < M && base + tx < K` 或 `base + ty < K && col < N` 范围的位置统一填入 `0.0f`。
2. **循环里的两处 `__syncthreads()` 分别在等什么？为什么边缘越界线程绝不能提前 `return`？**
   - **第 1 处 `__syncthreads()`（加载后、计算前）**：等待同一个线程块内的全部 256 个线程都把当前小块的数据写入 `As` 和 `Bs`，防止某些跑得快的线程在别的线程还没写完时就去读取未就绪的共享内存（防 Read-After-Write / RAW 数据竞争）。
   - **第 2 处 `__syncthreads()`（计算后、下一轮加载前）**：等待全块所有线程都算完当前小块的 16 次乘加，防止跑得快的线程提前进入下一轮循环（`base += 16`）覆盖掉慢线程还在读取的 `As` 和 `Bs`（防 Write-After-Read / WAR 数据竞争）。
   - **为什么边缘线程不能提前 `return`**：第一，某线程虽然输出坐标 `col >= N`（超出 `C` 的右边界），但它的 `row < M` 且 `base + tx < K` 可能是合法的，同块其他合法线程算内积时指望着它把 `As[ty][tx]` 搬进来；第二，`__syncthreads()` 要求同一个 Block 内的所有线程必须全员到达，如果有部分线程提前 `return`，将导致未定义行为或死锁。
3. **共享内存（Shared Memory）究竟是怎样减少全局显存重复读取的？**
   - 在普通版（Naive）中，每个线程算 1 个输出元素要从全局显存读 `K` 次 `A` 和 `K` 次 `B`，一个 `16 × 16` 线程块总共发起 `256 × 2K` 次全局显存读取。
   - 而在共享内存分块版（Tiled）中，每个元素只由 1 个线程从慢速全局显存读入共享内存 **1 次**，随后被同一行（或同一列）的 **16 个线程在片上高速共享内存中重复读取 16 次**，从而将全局显存的访问流量直接降低到了原来的 **`1 / 16`**，大幅缓解了显存带宽瓶颈。
4. **相邻线程连续访问（合并访存 Coalesced Access）有什么用？**
   - 因为矩阵按行主序存储，同一行相邻列的元素在显存地址上是紧挨着的。我们将列偏移绑定到 `tx = threadIdx.x`（加载 `A[row * K + (base + tx)]`、`B[(base + ty) * N + col]` 以及写回 `C[row * N + col]`），使得同一个 Warp 内相邻的 `tx` 线程在同一时刻访问连续的全局显存地址。GPU 硬件可以将这些连续访问打包合并为极少数几次 128 字节内存事务（Coalesced Memory Access），最大化显存带宽利用率。
5. **既然分块能减少重复读取，那把小块越调越大（比如从 `16 × 16` 改成 `32 × 32` 甚至更大）为什么未必更快？**
   - **硬件资源受限与 SM 占用率（Occupancy）下降**：每个 SM 上的共享内存容量和寄存器总数是固定的，且单 Block 线程数上限为 1024（`32 × 32 = 1024` 已达硬上限，再大直接无法启动）。当分块变大时，单 Block 占用的共享内存（翻 4 倍）和线程数激增，导致每个 SM 能同时驻留并发执行的活跃 Block 数量减少，SM 占用率（Occupancy）下降，反而不利于隐藏访存与指令延迟。
   - **同步与小矩阵尾部浪费增加**：更大的线程块意味着每次 `__syncthreads()` 需要等待多达 1024 个线程（32 个 Warp）全部到齐，块内木桶效应更明显；且当矩阵尺寸较小时（如 `128 × 128`），总 Block 数过少无法铺满 GPU 的全部 SM，或者边缘非整倍数块的空闲线程浪费比例更高。

---

# 第十一关（收官之战）：GPU 精准计时、性能对比与内存消毒器（`compute-sanitizer`）

在前 10 关里，我们已经写出了两条截然不同的实现路线：筛选问题有「第 8 关的 `atomicAdd` 原子筛选」和「第 9 关的 `exclusive_scan` 保序筛选」；矩阵乘法有「普通全局显存版 `naive`」和「第 10 关的共享内存分块版 `tiled`」。那到底哪个更快？快多少？在小数据量下“高级优化版”会不会反而比“普通版”更慢？

要科学地回答这个问题，不能靠感觉，更不能直接拿 CPU 秒表包住一行 `kernel<<<...>>>()`（因为核函数启动是异步的，那样测出来的只是 CPU 花了几微秒把任务单塞进流里的时间，根本不是 GPU 算完的时间！）。这一关作为全套教程的收官之战，我们写一个完整的评测程序 `11_bench_sanitizer.cu`，掌握三项核心技能：

1. **用两把不同的尺子精准测速（预热 1 次 + 正式测 5 次取中位数）**：
   - **尺子 1（纯 GPU 操作耗时）**：在同一条 `stream` 内部用硬件事件 `cudaEvent_t` 打点，等结束事件 `cudaEventSynchronize(stop)` 完成后用 `cudaEventElapsedTime` 读取毫秒数；
   - **尺子 2（含传输的端到端总耗时 E2E）**：用 CPU 的 `std::chrono::steady_clock`，从 CPU 向 GPU 拷入数据开始，到有效结果拷回 CPU 并 `cudaStreamSynchronize` 同步完成为止。
2. **先验对错、再测性能**：在计时前先严格校验每个算子的输出元素、个数、顺序与浮点误差，打印出清晰的对账表格。
3. **用官方内存消毒器 `compute-sanitizer --tool memcheck` 做全面体检**：哪怕程序表面输出了 `PASS`，只要核函数里有哪怕 1 个字节的越界读写或未对齐访问，都会被它当场揪出来。

```text
为什么不能直接用 CPU 秒表包住 kernel<<<...>>>() 来测核函数耗时？
  CPU 时间线:  [t0] ──► 提交 kernel (仅花 3 微秒派单) ──► [t1]  <-- 测出来的只是“CPU 贴任务单”的时间！
  GPU 流水线:                └──► [ 真正跑核函数 (比如 5 毫秒) ]

正确的两把计时尺子：
====================================================================================================
【尺子 1：cudaEvent_t（GPU 流水线内部打点器 —— 只测纯 GPU 操作耗时）】
  GPU stream 传送带:
    ... ──► [ 打点 start ] ──► [ GPU 操作 run() ] ──► [ 打点 stop ] ──► (CPU 调用 cudaEventSynchronize(stop) 等待)
            └────────────────── cudaEventElapsedTime(&ms, start, stop) ──────────────────┘
            （由 GPU 硬件时钟直接记录，精准包含 stream 内提交的所有核函数与清零操作，不含 PCIe 拷数据）

====================================================================================================
【尺子 2：std::chrono::steady_clock（CPU 墙上挂钟 —— 测包含数据传输与等待的端到端 E2E 总耗时）】
  CPU + GPU 端到端全流程:
    [ CPU 记录 t0 ]
          │
          ├──► 1. H2D 传输：CPU 向 GPU 拷入输入数据 (cudaMemcpyAsync HostToDevice)
          ├──► 2. GPU 计算：在 stream 中启动核函数 (清零 / 筛选 / 矩阵乘)
          ├──► 3. D2H 传输：把 count 和有效结果拷回 CPU (cudaMemcpyAsync DeviceToHost)
          ├──► 4. 同步等待：cudaStreamSynchronize(stream)  <-- 【铁律】停表前必须等 GPU 全部干完！
          ▼
    [ CPU 记录 t1 ]  ──► 端到端总耗时 ms = (t1 - t0)
```

## 动手写 `11_bench_sanitizer.cu`

1. 先在 GPU 机器上打开 `vim /root/cuda_tutorial/CMakeLists.txt`，在文件末尾加一行：

```cmake
add_executable(11_bench_sanitizer 11_bench_sanitizer.cu)
```

2. 打开 `vim /root/cuda_tutorial/11_bench_sanitizer.cu`，敲入下面的代码（所有新语法的解释都直接写在了对应代码行的上方注释里，不用上下翻页对照）：

```cpp
#include <cuda_runtime.h>
#include <thrust/scan.h>
#include <thrust/system/cuda/execution_policy.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <stdexcept>
#include <vector>

inline void cuda_check(cudaError_t status) {
    if (status != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(status));
    }
}

// ==================== 1. RAII 封装：显存管家与 CUDA Event 打点器 ====================
template <class T>
class DeviceBuffer {
    T* ptr_ = nullptr;
    std::size_t count_ = 0;
public:
    explicit DeviceBuffer(std::size_t n) : count_(n) {
        if (n > 0) cuda_check(cudaMalloc(reinterpret_cast<void**>(&ptr_), n * sizeof(T)));
    }
    ~DeviceBuffer() noexcept { if (ptr_) cudaFree(ptr_); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    DeviceBuffer(DeviceBuffer&& other) noexcept : ptr_(other.ptr_), count_(other.count_) {
        other.ptr_ = nullptr;
        other.count_ = 0;
    }
    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
        if (this != &other) {
            if (ptr_) cudaFree(ptr_);
            ptr_ = other.ptr_;
            count_ = other.count_;
            other.ptr_ = nullptr;
            other.count_ = 0;
        }
        return *this;
    }
    T* data() const noexcept { return ptr_; }
    std::size_t size() const noexcept { return count_; }
};

// 用第 7 关同样的 RAII 思路封装 cudaEvent_t（GPU 流水线时间戳打点器）：
// 构造时自动 cudaEventCreate，离开作用域析构时自动 cudaEventDestroy，且禁止拷贝。
struct Event {
    cudaEvent_t value{};
    Event() { cuda_check(cudaEventCreate(&value)); }
    ~Event() { cudaEventDestroy(value); }
    Event(const Event&) = delete;
    Event& operator=(const Event&) = delete;
};

// ==================== 2. 核心计时工具 1：用 CUDA Event 测纯 GPU 耗时的中位数 ====================
// 语法拆解：
//   - template <class F>：接收任意可调用的打包任务 run（比如我们在 main 里传入的 [&]() { ... } Lambda）。
//   - 为什么要【预热 (Warm-up) 1 次】？
//     因为 GPU 第一次跑某个核函数时，驱动需要做上下文初始化、PTX 即时编译或把指令加载进指令缓存，耗时虚高。
//   - 为什么要【正式跑 5 次并排序取 times[2] 中位数】？
//     因为硬件调度偶有抖动，把 5 次耗时从小到大 std::sort 排序后，正中间的第 3 个数（下标 2）就是中位数，完全不怕单次毛刺干扰！
template <class F>
float median_gpu_ms(F run, cudaStream_t stream) {
    Event start, stop;
    std::vector<float> times;

    // 【第 1 步：预热 1 次，并等待预热彻底完成】
    run();
    cuda_check(cudaStreamSynchronize(stream));

    // 【第 2 步：正式测量 5 次】
    for (int r = 0; r < 5; ++r) {
        // 1) 在 stream 流水线里插下起点旗子 start
        cuda_check(cudaEventRecord(start.value, stream));
        // 2) 调用 run()，往同一个 stream 里提交待测的 GPU 操作
        run();
        // 3) 在同一个 stream 流水线里插下终点旗子 stop
        cuda_check(cudaEventRecord(stop.value, stream));

        // 4) 【关键一步】必须等终点旗子 stop 在 GPU 上真正被跨过，才能读取耗时！
        cuda_check(cudaEventSynchronize(stop.value));

        // 5) 计算 start 到 stop 之间经过的 GPU 硬件毫秒数，存入 times
        float ms = 0.0f;
        cuda_check(cudaEventElapsedTime(&ms, start.value, stop.value));
        times.push_back(ms);
    }

    // 【第 3 步：从小到大排序，返回正中间的中位数 times[2]】
    std::sort(times.begin(), times.end());
    return times[2];
}

// ==================== 3. 核心计时工具 2：用 steady_clock 测含传输的端到端总耗时 ====================
// 语法拆解：
//   - std::chrono::steady_clock::now()：C++17 标准库的单调递增高精度时钟（不会因为系统校准时间而倒退）。
//   - std::chrono::duration<float, std::milli>(t1 - t0).count()：把两个时间点的差值换算成 float 类型的毫秒数 (ms)。
//   - 端到端（End-to-End, E2E）包含：CPU->GPU 拷入 + GPU 计算 + GPU->CPU 拷出 + 同步等待。
template <class F>
float median_e2e_ms(F run_e2e) {
    // 预热 1 次（run_e2e 内部末尾已包含同步等待）
    run_e2e();

    std::vector<float> times;
    for (int r = 0; r < 5; ++r) {
        auto t0 = std::chrono::steady_clock::now();
        run_e2e(); // 执行完整的 H2D -> Kernel -> D2H -> Synchronize
        auto t1 = std::chrono::steady_clock::now();

        float ms = std::chrono::duration<float, std::milli>(t1 - t0).count();
        times.push_back(ms);
    }
    std::sort(times.begin(), times.end());
    return times[2];
}

// ==================== 4. 把前几关的算子汇总进来做对比实验 ====================
// (A) 第 8 关的原子筛选
__global__ void atomic_filter_kernel(const float* a, float* out, int* count, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = tid; i < n; i += stride) {
        if (a[i] >= 0.0f) {
            int pos = atomicAdd(count, 1);
            out[pos] = a[i];
        }
    }
}
void filter_atomic(const float* a, float* out, int* count, int n, cudaStream_t stream) {
    cuda_check(cudaMemsetAsync(count, 0, sizeof(int), stream));
    if (n <= 0) return;
    int block = 256, grid = (n - 1) / block + 1;
    atomic_filter_kernel<<<grid, block, 0, stream>>>(a, out, count, n);
    cuda_check(cudaGetLastError());
}

// (B) 第 9 关的保序前缀和筛选
__global__ void mark_kernel(const float* a, int* flags, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = tid; i < n; i += stride) flags[i] = (a[i] >= 0.0f) ? 1 : 0;
}
__global__ void scatter_kernel(const float* a, float* out, const int* flags,
                               const int* offsets, int* count, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = tid; i < n; i += stride) {
        if (flags[i] == 1) out[offsets[i]] = a[i];
        if (i == n - 1) *count = offsets[n - 1] + flags[n - 1];
    }
}
void filter_stable(const float* a, float* out, int* count, int* flags, int* offsets,
                   int n, cudaStream_t stream) {
    if (n <= 0) {
        cuda_check(cudaMemsetAsync(count, 0, sizeof(int), stream));
        return;
    }
    int block = 256, grid = (n - 1) / block + 1;
    mark_kernel<<<grid, block, 0, stream>>>(a, flags, n);
    cuda_check(cudaGetLastError());
    thrust::exclusive_scan(thrust::cuda::par.on(stream), flags, flags + n, offsets);
    scatter_kernel<<<grid, block, 0, stream>>>(a, out, flags, offsets, count, n);
    cuda_check(cudaGetLastError());
}

// (C) 第 10 关的普通矩阵乘法 vs 共享内存分块矩阵乘法
__global__ void naive_gemm_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= M || col >= N) return;
    float sum = 0.0f;
    for (int k = 0; k < K; ++k) sum += A[row * K + k] * B[k * N + col];
    C[row * N + col] = sum;
}
void gemm_naive(const float* A, const float* B, float* C, int M, int N, int K, cudaStream_t stream) {
    if (M <= 0 || N <= 0) return;
    dim3 grid((N - 1) / 16 + 1, (M - 1) / 16 + 1);
    naive_gemm_kernel<<<grid, dim3(16, 16), 0, stream>>>(A, B, C, M, N, K);
    cuda_check(cudaGetLastError());
}

__global__ void tiled_gemm_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    __shared__ float As[16][16], Bs[16][16];
    int tx = threadIdx.x, ty = threadIdx.y;
    int row = blockIdx.y * 16 + ty, col = blockIdx.x * 16 + tx;
    float sum = 0.0f;
    for (int base = 0; base < K; base += 16) {
        As[ty][tx] = (row < M && base + tx < K) ? A[row * K + base + tx] : 0.0f;
        Bs[ty][tx] = (base + ty < K && col < N) ? B[(base + ty) * N + col] : 0.0f;
        __syncthreads();
        for (int t = 0; t < 16; ++t) sum += As[ty][t] * Bs[t][tx];
        __syncthreads();
    }
    if (row < M && col < N) C[row * N + col] = sum;
}
void gemm_tiled(const float* A, const float* B, float* C, int M, int N, int K, cudaStream_t stream) {
    if (M <= 0 || N <= 0) return;
    dim3 grid((N - 1) / 16 + 1, (M - 1) / 16 + 1);
    tiled_gemm_kernel<<<grid, dim3(16, 16), 0, stream>>>(A, B, C, M, N, K);
    cuda_check(cudaGetLastError());
}

int main() {
    // 1. 打印当前 GPU 硬件与运行环境信息
    cudaDeviceProp prop{};
    cuda_check(cudaGetDeviceProperties(&prop, 0));
    int driver_ver = 0, runtime_ver = 0;
    cuda_check(cudaDriverGetVersion(&driver_ver));
    cuda_check(cudaRuntimeGetVersion(&runtime_ver));
    printf("================ 环境信息 ================\n");
    printf("GPU 型号: %s (SM 数量: %d, Compute Capability: %d.%d)\n",
           prop.name, prop.multiProcessorCount, prop.major, prop.minor);
    printf("Driver Version: %d, CUDA Runtime Version: %d\n\n", driver_ver, runtime_ver);

    cudaStream_t stream;
    cuda_check(cudaStreamCreate(&stream));

    // ================ 实验组 1：对比 Atomic 筛选 vs Stable 筛选 ================
    // 测量规范：先核对答案正确性，再分别测量【纯 GPU 耗时】与【含传输 E2E 总耗时】（缓冲区提前分配并复用，分配本身不计入时间）
    printf("================ 实验 1：筛选算子对比 (Atomic vs Stable) ================\n");
    printf("%-10s | %-13s | %-13s | %-13s | %-13s | %s\n",
           "n", "atomic GPU ms", "stable GPU ms", "atomic E2E ms", "stable E2E ms", "count/正确性");
    printf("------------------------------------------------------------------------------------------\n");

    for (int n : {1025, 65537, 1048576}) {
        // 统一生成输入数据 float(i % 7 - 3)，同时算出 CPU 标准答案 expected
        std::vector<float> h_in(n), expected;
        for (int i = 0; i < n; ++i) {
            h_in[i] = float(i % 7 - 3);
            if (h_in[i] >= 0.0f) expected.push_back(h_in[i]);
        }

        // 预先分配并复用 GPU 缓冲区（显存分配本身不计入计时）
        DeviceBuffer<float> d_in(n), d_out(n);
        DeviceBuffer<int> d_count(1), d_flags(n), d_offsets(n);
        cuda_check(cudaMemcpyAsync(d_in.data(), h_in.data(), n * sizeof(float), cudaMemcpyHostToDevice, stream));
        cuda_check(cudaStreamSynchronize(stream));

        // 1) 先核对 Atomic 筛选结果（比较个数与排序后的多重集）
        filter_atomic(d_in.data(), d_out.data(), d_count.data(), n, stream);
        int c_atomic = 0;
        cuda_check(cudaMemcpyAsync(&c_atomic, d_count.data(), sizeof(int), cudaMemcpyDeviceToHost, stream));
        cuda_check(cudaStreamSynchronize(stream));
        std::vector<float> h_out_atomic(c_atomic);
        cuda_check(cudaMemcpyAsync(h_out_atomic.data(), d_out.data(), c_atomic * sizeof(float), cudaMemcpyDeviceToHost, stream));
        cuda_check(cudaStreamSynchronize(stream));

        std::vector<float> exp_sorted = expected;
        std::sort(exp_sorted.begin(), exp_sorted.end());
        std::sort(h_out_atomic.begin(), h_out_atomic.end());
        bool atomic_ok = (c_atomic == (int)expected.size()) && (h_out_atomic == exp_sorted);

        // 2) 再核对 Stable 筛选结果（直接逐项比对顺序与元素）
        filter_stable(d_in.data(), d_out.data(), d_count.data(), d_flags.data(), d_offsets.data(), n, stream);
        int c_stable = 0;
        cuda_check(cudaMemcpyAsync(&c_stable, d_count.data(), sizeof(int), cudaMemcpyDeviceToHost, stream));
        cuda_check(cudaStreamSynchronize(stream));
        std::vector<float> h_out_stable(c_stable);
        cuda_check(cudaMemcpyAsync(h_out_stable.data(), d_out.data(), c_stable * sizeof(float), cudaMemcpyDeviceToHost, stream));
        cuda_check(cudaStreamSynchronize(stream));
        bool stable_ok = (c_stable == (int)expected.size()) && (h_out_stable == expected);

        if (!atomic_ok || !stable_ok) {
            throw std::runtime_error("Filter verification failed!");
        }

        // 3) 用 median_gpu_ms 测量纯 GPU 操作时间（包含 count 清零 / mark+scan+scatter 全部步骤）
        //    注意：[&] 表示 Lambda 按引用捕获外面的缓冲区别名，只在 CPU 侧把调用语句打包传给 median_gpu_ms
        float atomic_gpu = median_gpu_ms([&]() {
            filter_atomic(d_in.data(), d_out.data(), d_count.data(), n, stream);
        }, stream);

        float stable_gpu = median_gpu_ms([&]() {
            filter_stable(d_in.data(), d_out.data(), d_count.data(), d_flags.data(), d_offsets.data(), n, stream);
        }, stream);

        // 4) 用 steady_clock 测量包含传输与同步的端到端总耗时：
        //    从 CPU 向 GPU 传入输入数据开始，到 count 和有效输出回到 CPU 并同步完成为止
        std::vector<float> h_out_buf(n);
        float atomic_e2e = median_e2e_ms([&]() {
            cuda_check(cudaMemcpyAsync(d_in.data(), h_in.data(), n * sizeof(float), cudaMemcpyHostToDevice, stream));
            filter_atomic(d_in.data(), d_out.data(), d_count.data(), n, stream);
            int cnt = 0;
            cuda_check(cudaMemcpyAsync(&cnt, d_count.data(), sizeof(int), cudaMemcpyDeviceToHost, stream));
            cuda_check(cudaStreamSynchronize(stream));
            if (cnt > 0) {
                cuda_check(cudaMemcpyAsync(h_out_buf.data(), d_out.data(), cnt * sizeof(float), cudaMemcpyDeviceToHost, stream));
                cuda_check(cudaStreamSynchronize(stream));
            }
        });

        float stable_e2e = median_e2e_ms([&]() {
            cuda_check(cudaMemcpyAsync(d_in.data(), h_in.data(), n * sizeof(float), cudaMemcpyHostToDevice, stream));
            filter_stable(d_in.data(), d_out.data(), d_count.data(), d_flags.data(), d_offsets.data(), n, stream);
            int cnt = 0;
            cuda_check(cudaMemcpyAsync(&cnt, d_count.data(), sizeof(int), cudaMemcpyDeviceToHost, stream));
            cuda_check(cudaStreamSynchronize(stream));
            if (cnt > 0) {
                cuda_check(cudaMemcpyAsync(h_out_buf.data(), d_out.data(), cnt * sizeof(float), cudaMemcpyDeviceToHost, stream));
                cuda_check(cudaStreamSynchronize(stream));
            }
        });

        printf("%-10d | %-13.4f | %-13.4f | %-13.4f | %-13.4f | count=%d (PASS)\n",
               n, atomic_gpu, stable_gpu, atomic_e2e, stable_e2e, c_stable);
    }

    // ================ 实验组 2：对比 Naive 矩阵乘 vs Tiled 共享内存矩阵乘 ================
    printf("\n================ 实验 2：矩阵乘法对比 (Naive vs Tiled) ================\n");
    printf("%-13s | %-12s | %-12s | %-12s | %s\n",
           "M,N,K", "naive ms", "tiled ms", "naive/tiled", "最大绝对误差");
    printf("--------------------------------------------------------------------------------\n");

    for (int sz : {128, 256}) {
        int M = sz, N = sz, K = sz;
        std::vector<float> h_A(M * K), h_B(K * N), h_C_ref(M * N), h_C_naive(M * N), h_C_tiled(M * N);
        for (int i = 0; i < M * K; ++i) h_A[i] = float(i % 11 - 5) / 8.0f;
        for (int i = 0; i < K * N; ++i) h_B[i] = float(i % 7 - 3) / 4.0f;

        // CPU 双精度累加求参考真值
        for (int r = 0; r < M; ++r) {
            for (int c = 0; c < N; ++c) {
                double s = 0.0;
                for (int k = 0; k < K; ++k) s += double(h_A[r * K + k]) * double(h_B[k * N + c]);
                h_C_ref[r * N + c] = float(s);
            }
        }

        DeviceBuffer<float> d_A(M * K), d_B(K * N), d_C(M * N);
        cuda_check(cudaMemcpyAsync(d_A.data(), h_A.data(), M * K * sizeof(float), cudaMemcpyHostToDevice, stream));
        cuda_check(cudaMemcpyAsync(d_B.data(), h_B.data(), K * N * sizeof(float), cudaMemcpyHostToDevice, stream));

        // 1) 先核对 Naive 和 Tiled 的计算误差是否满足 <= 1e-4 + 1e-4 * abs(ref)
        gemm_naive(d_A.data(), d_B.data(), d_C.data(), M, N, K, stream);
        cuda_check(cudaMemcpyAsync(h_C_naive.data(), d_C.data(), M * N * sizeof(float), cudaMemcpyDeviceToHost, stream));
        gemm_tiled(d_A.data(), d_B.data(), d_C.data(), M, N, K, stream);
        cuda_check(cudaMemcpyAsync(h_C_tiled.data(), d_C.data(), M * N * sizeof(float), cudaMemcpyDeviceToHost, stream));
        cuda_check(cudaStreamSynchronize(stream));

        float max_err = 0.0f;
        for (int i = 0; i < M * N; ++i) {
            float err_n = std::abs(h_C_naive[i] - h_C_ref[i]);
            float err_t = std::abs(h_C_tiled[i] - h_C_ref[i]);
            float tol = 1e-4f + 1e-4f * std::abs(h_C_ref[i]);
            if (err_n > tol || err_t > tol) throw std::runtime_error("GEMM verification failed!");
            max_err = std::max(max_err, std::max(err_n, err_t));
        }

        // 2) 测量两个 Kernel 的纯 GPU 时间并计算加速比 naive / tiled
        float naive_ms = median_gpu_ms([&]() {
            gemm_naive(d_A.data(), d_B.data(), d_C.data(), M, N, K, stream);
        }, stream);

        float tiled_ms = median_gpu_ms([&]() {
            gemm_tiled(d_A.data(), d_B.data(), d_C.data(), M, N, K, stream);
        }, stream);

        char dim_str[32];
        snprintf(dim_str, sizeof(dim_str), "%d,%d,%d", M, N, K);
        printf("%-13s | %-12.4f | %-12.4f | %-11.2fx | %.2e (PASS)\n",
               dim_str, naive_ms, tiled_ms, naive_ms / tiled_ms, max_err);
    }

    cuda_check(cudaStreamDestroy(stream));
    return 0;
}
```

3. 以 `Release` 优化模式编译、运行基准测试，并用 `compute-sanitizer` 检查内存错误：

```bash
cmake -S /root/cuda_tutorial -B /root/cuda_tutorial/build -DCMAKE_BUILD_TYPE=Release
cmake --build /root/cuda_tutorial/build -j
/root/cuda_tutorial/build/11_bench_sanitizer
compute-sanitizer --tool memcheck /root/cuda_tutorial/build/11_bench_sanitizer
```

## 深入原理与避坑剖析（跑通代码后再看这 4 个问题）

1. **为什么测 GPU 性能必须「预热 1 次 + 重复测 5 次取中位数」，而且读取时间前必须先等结束事件同步？**
   - **为什么要预热（Warm-up）**：第 1 次调用核函数（尤其是像 Thrust 的 `exclusive_scan` 这种内部包含临时状态初始化或 JIT 缓存加载的操作）时，GPU 硬件和 CUDA Runtime 会有一次性的冷启动开销，耗时往往比正常状态慢几倍到几十倍。预热 1 次并同步等待完成后，后续测量才能反映真实的稳态算子性能。
   - **为什么要取中位数（Median）而不是平均值（Mean）**：系统偶发的后台中断或调度抖动可能会让某一次测量出现极端高值（Outlier）。如果取平均值会被这一个极值拉偏，而将 5 次结果排序后取正中间的 `times[2]` 中位数，能够天然过滤掉偶发尖峰，结果最稳定。
   - **为什么必须先 `cudaEventSynchronize(stop.value)`**：因为 `cudaEventRecord` 和核函数一样都是异步塞进 `stream` 里的，CPU 执行完 `cudaEventRecord(stop.value, stream)` 返回时，GPU 可能还没跑完核函数。如果立刻调 `cudaEventElapsedTime`，就会报错 `cudaErrorNotReady`。同样地，用 CPU 的 `steady_clock` 测端到端总耗时时，在记录结束时间点 `t1` 之前也必须先调用 `cudaStreamSynchronize(stream)` 等待 GPU 彻底完成。
2. **对比实测表格里的两组时间：每项时间从哪里开始、到哪里结束？显存分配算进去了吗？**
   - **显存分配（`DeviceBuffer` / `cudaMalloc`）是否计时**：输入与输出缓冲区在循环体外提前分配好并复用，**显存分配与释放本身不计入任何一项测量时间**。
   - **筛选纯 GPU 时间（`atomic GPU ms` / `stable GPU ms`）**：从在 `stream` 中记录 `start` 事件开始，到记录 `stop` 事件结束。其中 `atomic` 包含 `cudaMemsetAsync` 清零 `count` 与 `atomic_kernel` 执行；`stable` 包含 `mark_kernel` 标记、`thrust::exclusive_scan` 扫描和 `scatter_kernel` 写出全部三步，不包含主机与设备间的数据拷贝。
   - **筛选端到端总耗时（`atomic E2E ms` / `stable E2E ms`）**：用 `steady_clock` 测量，从 CPU 调用 `cudaMemcpyAsync(HostToDevice)` 向 GPU 传入输入数据开始，经过 GPU 筛选计算，再到把 `count` 和 `count` 个有效输出元素通过 `cudaMemcpyAsync(DeviceToHost)` 拷回 CPU 并完成 `cudaStreamSynchronize` 同步等待为止。
   - **矩阵乘时间（`naive ms` / `tiled ms`）**：用 `cudaEvent_t` 测量单个矩阵乘法核函数（`naive_kernel` 或 `tiled_kernel`）在 `stream` 中的纯 GPU 执行时间，不含 H2D/D2H 传输。
3. **为什么在实测中，「看似更高级的算法或优化版本」有时反而比普通版本更慢？**
   - **现象 A：保序筛选 `stable` 比原子筛选 `atomic` 慢**：
     1. **核函数启动次数与显存往返流量差距**：`atomic` 只需要 1 次 `cudaMemsetAsync` 加 1 个核函数，对输入数组只读 1 遍；而 `stable` 必须分三步启动多个核函数（`mark_kernel` + `thrust::exclusive_scan` 内部的多次扫描核函数 + `scatter_kernel`），不仅多了好几次 Kernel Launch 调度开销，还要反复读写额外的中间数组 `flags` 和 `offsets`。
     2. **硬件原子指令加速**：现代架构（如 Hopper H20）在 L2 Cache 和硬件层面高度优化了原子加指令，因此如果不要求保序，单趟 `atomicAdd` 往往比多趟扫描快得多（`stable` 是用额外的计算和访存换取了「确定性保序」）。
   - **现象 B：在小矩阵（如 `128×128`、`256×256`）下，共享内存 `tiled` 未必比 `naive` 快多少（甚至在部分卡上相近或略慢）**：
     1. **大容量硬件 L2 Cache 已经帮 `naive` 兜底了**：当矩阵只有 `128×128`（仅 64 KB）或 `256×256`（仅 256 KB）时，整张矩阵 `A` 和 `B` 轻轻松松就能完全塞进 GPU 的片上 L1/L2 Cache（H20 的 L2 Cache 高达数十 MB）！这意味着 `naive` 核函数的后续重复读取其实全部命中了高速缓存，并没有真正去慢速 HBM 显存里拉数据。
     2. **分块同步与指令开销**：与此同时，`tiled` 版本每走 16 步就要额外执行地址边界分支判断、往共享内存做一次显式写入，并强制全块 256 个线程停在两道 `__syncthreads()` 屏障上互相等候。当矩阵很小时，硬件缓存已经消除了显存瓶颈，这些额外的指令和同步屏障开销就会抵消掉共享内存带来的收益；只有当矩阵尺寸增大到超出缓存容量、真正受限于全局显存带宽时，分块优化的威力才会彻底爆发。
4. **如何设计严密的单元测试来抓出 GPU 代码里的隐蔽 Bug？（挑选典型测试案例拆解）**
   - **测试案例 1：专门测非整倍数尾部长度（如 `n = 1, 255, 257, 1025, 65537`）**
     - **能抓出什么错误**：专门抓「网格数量计算错误（比如写成 `n / block` 忘了向上取整加 1，导致最后不足一个 Block 的尾部元素全没算）」或者「核函数里没写 `if (i < n)` / `for (int i = tid; i < n; i += stride)` 边界检查，导致最后一个 Block 里多余的线程越界读写显存」。配合 `compute-sanitizer --tool memcheck` 可以当场抓出越界非法访问。
   - **测试案例 2：拿着同一个 `count` 缓冲区连续调用两次筛选函数（`repeat = 0..1`）**
     - **能抓出什么错误**：专门抓「在 `filter_atomic` 里忘了每次用 `cudaMemsetAsync` 把 `count` 清零」的 Bug。如果只在外部分配显存时初始化一次 0，第 1 次调用虽然能蒙混过关，但第 2 次调用会在第 1 次的结果上继续累加，导致返回的 `count` 翻倍（变成 `2 * c`）甚至写穿 `out` 数组边界。
   - **测试案例 3：全零输入模式（`pattern = 1`，所有 `a[i] == 0.0f`）与全负数模式（`pattern = 0`，所有 `a[i] == -1.0f`）**
     - **能抓出什么错误**：全零模式专门抓「把非负条件 `a[i] >= 0` 误写成正数条件 `a[i] > 0`（漏掉 0）」，同时结合 `count` 校验防止把未写入的默认全零内存当成正确输出；全负数模式则专门验证当没有任何元素保留时，`count` 能否正确归零为 `0`。
   - **测试案例 4：非 16 倍数矩阵边缘小块（如 `{17, 19, 23}`、`{33, 31, 65}`）与退化维度（`{3, 5, 0}`）**
     - **能抓出什么错误**：`{17, 19, 23}` 专门抓分块矩阵乘在边缘小块加载共享内存时「越界没填 `0.0f`（读进垃圾数据污染内积）」、「把 `A` 的行数边界 `M` 和 `B` 的列数边界 `N` 写混」、或者「让 `row >= M || col >= N` 的边缘线程提前 `return` 导致少搬数据或 `__syncthreads()` 死锁」；而 `{3, 5, 0}` 则专门检验当公共维度 `K == 0` 时核函数是否依然启动并把输出矩阵 `C` 正确写满 `0.0f`。








