# DerelictCLBlast

A [dynamic loading](https://en.wikipedia.org/wiki/Dynamic_loading) binding to
[CLBlast](https://github.com/CNugteren/CLBlast) for the D Programming Language.

Please see the pages [Building and Linking Derelict](http://derelictorg.github.io/compiling.html)
and [Using Derelict](http://derelictorg.github.io/using.html), or information on how to build
DerelictCLBlast and load the OpenCL library at run time.

## What is CLBlast?

The name "CLBlast" is a library that uses [OpenCL](https://www.khronos.org/opencl/) to execute
[BLAS](https://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms) functions using a parallel
computing device, such as a GPU. It is tuned to have better performance on most types of GPUs. Thus
the name "CLBlast" = "OpenCL" + "BLAS" + "Tuned".

## What is it for?

In many practical applications, for example in scientific computing or machine learning, very large
numbers of very large matrices need to have large numbers of mathematical operations applied to
them. For example, most [Deep Learning](https://en.wikipedia.org/wiki/Deep_learning) systems are
based on [Neural Networks](https://en.wikipedia.org/wiki/Neural_network), which need to train
against large amounts of data using a
[backpropagation](https://en.wikipedia.org/wiki/Backpropagation) algorithm. In a nutshell, this
means that it must do a LOT of linear algebra operations on very large
[Tensors](https://en.wikipedia.org/wiki/Tensor). Tensors are essentially the generalization of
vectors and matrices, having an arbitrary number of dimensions, and can also be called
"N-Dimensional Arrays".

### BLAS Functions and Performance

Attempting to execute such algorithms directly on a comptuer's CPU can result in suboptimal
performance, which limits how much data can be processed in a given cost and amount of time. Highly
optimized algorithms that take advantage of a particular architecture's capabilities, for example,
CPUs that offer [Single-Instruction,
Multiple-Data](https://en.wikipedia.org/wiki/Single_instruction,_multiple_data) (SIMD) instructions,
are often created and bundled under a standard set of functions called [Basic Linear Algebra
Subprograms](https://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms) (BLAS).

BLAS funtions will follow a slightly odd, but consistent, naming convention. Most function names are
abbreviations for what they do and what type of data they operate on.

Consider the oddly named function "Saxpy". This name is short for "Single-Precision (float), A*X
Plus Y". The function "Gemm" stands for "Generalized Matrix Multiplication". If one can convert
one's problems to use these high-optimized functions, it can lead to large boosts in performance.

### GPUs and other Compute Devices

Thus far, we've only spoken about executing algorithms on a CPU. However, there are other hardware
devices that do large volumes of linear algebra operations in parallel. Chief among them are
GPUs. GPUs excel at highly parallized computation devices that specialize in linear algebra. Those
familiar with computer games may recognize terms such as a
[shaders](https://en.wikipedia.org/wiki/Shader), which are essentially small programs known as
[kernels](https://en.wikipedia.org/wiki/Compute_kernel), that perform operations in parallel on
large matrixes. Converting an image (a matrix) into black-and-white, projecting the vectors of a 3D
object onto a surface in order to create a shadow, rotating the vectors of a 3D object, these are
all examples of problems which ultimately boil down in to linear algebra operations on 2D and 3D
matrices. This is why GPUs can be harnessed for scientific and machine learning purposes.

### BLAS and GPUs

One complication with running code on GPUs, however, is that they vary considerably in their
capabilities and even their core architecture. Different vendors, such as
[Intel](https://www.intel.com/content/www/us/en/products/details/discrete-gpus.html),
[NVIDIA](https://www.nvidia.com/en-us/), and
[AMD](https://www.amd.com/en/products/specifications/graphics) all make powerful GPUs.

An abstraction layer that is used to access GPUs in a consistent manner for scientific and machine
learning computing purposes is the [Compute Unified Device
Architecture](https://en.wikipedia.org/wiki/CUDA) (CUDA). However, the libraries and SDKs for CUDA
are specific to NVIDIA and are proprietary. This means that, if you do not have NVIDIA GPUs
available, then you are simply out of luck.

An alternative compute device interface is the [Open Computing
Language](https://en.wikipedia.org/wiki/OpenCL) (OpenCL). In addition to working on NVIDIA GPUs,
OpenCL can be used to execute on a much wider variety of GPUs and other computing devices. In the
absence of a GPU, it can also execute directly on the CPU using CPU-based implementations such as
[Portable Computing Language](http://portablecl.org/) (PoCL). CPU-based implementations of OpenCL
can simplify work efforts, allowing testing and development even on weaker personal computers, while
still being able to execute the finished product on a powerful multi-GPU architecture.

### Joining GPUs and BLAS

This is where the [CLBlast](https://github.com/CNugteren/CLBlast) library comes into play. It is an
implementation of the same BLAS algorithms that are familiar in scientific and machine learning
computing, but it is built on top of OpenCL, permitting these operations to be run on a GPU. CLBlast
is written originally for C and C++.

CLBLast contains a number of kernels written for different GPUs and tuned for performance. CLBlast
uses OpenCL to compile these kernels if necessary and load them onto the GPU. This allows the user
to benefit from BLAS-like methods, but using the parallel computing power of a GPU.

### A D Binding

For those wishing to pursue scientific computing and machine learning problems using the [D
Programming Language](https://dlang.org/), this binding library was created.

Why D? It is built with three main ideas in mind:
- **Read Fast**: Clear, readable, documented code familiar to those using C, Java, or any related
  language.
- **Write Fast**: A succinct syntax that avoids excess boilerplate, a powerful multi-paradigm
  language letting you pick the right tool for the right job.
- **Run Fast**: D is a compiled language that runs directly on your CPU without an interpretter.

D combines the simplicity of Python with the performance of C++.

## Setup (Ubuntu Linux)

This binding performs [dynamic loading](https://en.wikipedia.org/wiki/Dynamic_loading), this means
that the C libraries are loaded at runtime rather than during linking. Thus, it is important to make
sure the proper dependencies are installed and available on your computer.

1. Install a tool used to check OpenCL support on your system:
   ```
   $ sudo apt install clinfo
   ```

2. Check to see what OpenCL platforms are supported:
   ```
   $ clinfo
   Number of platforms                               0
   ```

3. Install an OpenCL platform, which will vary with the GPU available on your system. For a quick
   setup, try using PoCL, which will let your CPU act as an OpenCL computing device:
   ```
   $ sudo apt install libpocl2 libpocl-dev  # Run on CPU
   $ sudo apt install nvidia-opencl-icd-384  # An NVidia GPU Platform
   $ # Other platforms as per their official instructions...
   ```

4. Validate that a compute platform is available, e.g.:
   ```
   $ clinfo
   Number of platforms                               1
     Platform Name                                   Portable Computing Language
     Platform Vendor                                 The pocl project
     Platform Version                                OpenCL 2.0 pocl 1.8  Linux, None+Asserts, RELOC, LLVM 11.1.0, SLEEF, DISTRO, POCL_DEBUG
     Platform Profile                                FULL_PROFILE
     Platform Extensions                             cl_khr_icd cl_pocl_content_size
     Platform Extensions function suffix             POCL
   ```

5. Finally install CLBlast:
   ```
   $ sudo apt install libclblast1
   ```

## Usage

The usage of OpenCL and CLBlast isn't something that fits into a 5-line example program, but the
[test program](test/source/app.d) serves as a good example.

This program demonstrates the following:
1. Initializing OpenCL, viewing Compute Platforms and Compute Devices.
2. Initializing CLBlast.
3. Creating buffers on the GPU, loading data into them.
4. Executing a function (GEMM in this case).
5. Pulling the results out of the GPU and printing them on the host.
