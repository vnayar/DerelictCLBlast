/**
 * Most methods names in the CLBlast API are of the form: `CLBlast<T><name>`
 * The <name> indicates what the BLAS method, e.g. SWAP, SCAL, AXPY, etc.
 * The <T> indicates the data-type of the data being operated on.
 *
 * The different data-types supported by the library are:
 * __S:__ Single-precision 32-bit floating-point (`float`).
 * __D:__ Double-precision 64-bit floating-point (`double`).
 * __C:__ Complex single-precision 2x32-bit floating-point (`std::complex<float>`).
 * __Z:__ Complex double-precision 2x64-bit floating-point (`std::complex<double>`).
 * __H:__ Half-precision 16-bit floating-point (`cl_half`).
 *
 * See_Also:
 *   https://github.com/CNugteren/CLBlast/blob/master/doc/api.md
 *   https://github.com/CNugteren/CLBlast/blob/master/doc/routines.md
 */
module derelict.clblast.functions;

import derelict.opencl.types : cl_mem, cl_command_queue, cl_event, cl_float, cl_double, cl_half,
  cl_device_id;

private import derelict.clblast.constants : CLBlastStatusCode, CLBlastLayout, CLBlastTranspose,
  CLBlastTriangle, CLBlastDiagonal, CLBlastSide, CLBlastKernelMode, CLBlastPrecision;
private import derelict.clblast.types : cl_float2, cl_double2;

// =================================================================================================
// BLAS level-1 (vector-vector) routines
// =================================================================================================

/**
 * xSWAP: Swap two vectors
 *
 * Interchanges _n_ elements of vectors _x_ and _y_.
 *
 * Arguments to SWAP:
 * - `const size_t n`: Integer size argument. This value must be positive.
 * - `cl_mem x_buffer`: OpenCL buffer to store the output x vector.
 * - `const size_t x_offset`: The offset in elements from the start of the output x vector.
 * - `const size_t x_inc`: Stride/increment of the output x vector. This value must be greater than 0.
 * - `cl_mem y_buffer`: OpenCL buffer to store the output y vector.
 * - `const size_t y_offset`: The offset in elements from the start of the output y vector.
 * - `const size_t y_inc`: Stride/increment of the output y vector. This value must be greater than 0.
 * - `cl_command_queue* queue`: Pointer to an OpenCL command queue associated with a context and device to execute the routine on.
 * - `cl_event* event`: Pointer to an OpenCL event to be able to wait for completion of the routine's OpenCL kernel(s). This is an optional argument.
 */
extern(System) @nogc nothrow {
  alias da_CLBlastSswap = CLBlastStatusCode function(const size_t n,
      cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
      cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
      cl_command_queue* queue, cl_event* event);
  alias da_CLBlastDswap = CLBlastStatusCode function(const size_t n,
      cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
      cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
      cl_command_queue* queue, cl_event* event);
  alias da_CLBlastCswap = CLBlastStatusCode function(const size_t n,
      cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
      cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
      cl_command_queue* queue, cl_event* event);
  alias da_CLBlastZswap = CLBlastStatusCode function(const size_t n,
      cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
      cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
      cl_command_queue* queue, cl_event* event);
  alias da_CLBlastHswap = CLBlastStatusCode function(const size_t n,
      cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
      cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
      cl_command_queue* queue, cl_event* event);
}

/**
 * xSCAL: Vector scaling
 *
 * Multiplies _n_ elements of vector _x_ by a scalar constant _alpha_.
 *
 * Arguments to SCAL:
 * - `const size_t n`: Integer size argument. This value must be positive.
 * - `const T alpha`: Input scalar constant.
 * - `cl_mem x_buffer`: OpenCL buffer to store the output x vector.
 * - `const size_t x_offset`: The offset in elements from the start of the output x vector.
 * - `const size_t x_inc`: Stride/increment of the output x vector. This value must be greater than 0.
 * - `cl_command_queue* queue`: Pointer to an OpenCL command queue associated with a context and device to execute the routine on.
 * - `cl_event* event`: Pointer to an OpenCL event to be able to wait for completion of the routine's OpenCL kernel(s). This is an optional argument.
 */
extern(System) @nogc nothrow {
  alias da_CLBlastSscal = CLBlastStatusCode function(const size_t n,
      const float alpha,
      cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
      cl_command_queue* queue, cl_event* event);
  alias da_CLBlastDscal = CLBlastStatusCode function(const size_t n,
      const double alpha,
      cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
      cl_command_queue* queue, cl_event* event);
  alias da_CLBlastCscal = CLBlastStatusCode function(const size_t n,
      const cl_float2 alpha,
      cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
      cl_command_queue* queue, cl_event* event);
  alias da_CLBlastZscal = CLBlastStatusCode function(const size_t n,
      const cl_double2 alpha,
      cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
      cl_command_queue* queue, cl_event* event);
  alias da_CLBlastHscal = CLBlastStatusCode function(const size_t n,
      const cl_half alpha,
      cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
      cl_command_queue* queue, cl_event* event);
}

/**
 * xCOPY: Vector copy
 *
 * Copies the contents of vector _x_ into vector _y_.
 *
 * Arguments to COPY:
 * - `const size_t n`: Integer size argument. This value must be positive.
 * - `const cl_mem x_buffer`: OpenCL buffer to store the input x vector.
 * - `const size_t x_offset`: The offset in elements from the start of the input x vector.
 * - `const size_t x_inc`: Stride/increment of the input x vector. This value must be greater than 0.
 * - `cl_mem y_buffer`: OpenCL buffer to store the output y vector.
 * - `const size_t y_offset`: The offset in elements from the start of the output y vector.
 * - `const size_t y_inc`: Stride/increment of the output y vector. This value must be greater than 0.
 * - `cl_command_queue* queue`: Pointer to an OpenCL command queue associated with a context and device to execute the routine on.
 * - `cl_event* event`: Pointer to an OpenCL event to be able to wait for completion of the routine's OpenCL kernel(s). This is an optional argument.
 */
extern(System) @nogc nothrow {
  alias da_CLBlastScopy = CLBlastStatusCode function(const size_t n,
      const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
      cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
      cl_command_queue* queue, cl_event* event);
  alias da_CLBlastDcopy = CLBlastStatusCode function(const size_t n,
      const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
      cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
      cl_command_queue* queue, cl_event* event);
  alias da_CLBlastCcopy = CLBlastStatusCode function(const size_t n,
      const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
      cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
      cl_command_queue* queue, cl_event* event);
  alias da_CLBlastZcopy = CLBlastStatusCode function(const size_t n,
      const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
      cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
      cl_command_queue* queue, cl_event* event);
  alias da_CLBlastHcopy = CLBlastStatusCode function(const size_t n,
      const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
      cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
      cl_command_queue* queue, cl_event* event);
}


/**
 * xAXPY: Vector-times-constant plus vector
 *
 * Performs the operation _y = alpha * x + y_, in which _x_ and _y_ are vectors and _alpha_ is a scalar constant.
 *
 * Arguments to AXPY:
 * - `const size_t n`: Integer size argument. This value must be positive.
 * - `const T alpha`: Input scalar constant.
 * - `const cl_mem x_buffer`: OpenCL buffer to store the input x vector.
 * - `const size_t x_offset`: The offset in elements from the start of the input x vector.
 * - `const size_t x_inc`: Stride/increment of the input x vector. This value must be greater than 0.
 * - `cl_mem y_buffer`: OpenCL buffer to store the output y vector.
 * - `const size_t y_offset`: The offset in elements from the start of the output y vector.
 * - `const size_t y_inc`: Stride/increment of the output y vector. This value must be greater than 0.
 * - `cl_command_queue* queue`: Pointer to an OpenCL command queue associated with a context and device to execute the routine on.
 * - `cl_event* event`: Pointer to an OpenCL event to be able to wait for completion of the routine's OpenCL kernel(s). This is an optional argument.
 */
extern(System) @nogc nothrow {
  alias da_CLBlastSaxpy = CLBlastStatusCode function(const size_t n,
      const float alpha,
      const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
      cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
      cl_command_queue* queue, cl_event* event);
  alias da_CLBlastDaxpy = CLBlastStatusCode function(const size_t n,
      const double alpha,
      const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
      cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
      cl_command_queue* queue, cl_event* event);
  alias da_CLBlastCaxpy = CLBlastStatusCode function(const size_t n,
      const cl_float2 alpha,
      const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
      cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
      cl_command_queue* queue, cl_event* event);
  alias da_CLBlastZaxpy = CLBlastStatusCode function(const size_t n,
      const cl_double2 alpha,
      const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
      cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
      cl_command_queue* queue, cl_event* event);
  alias da_CLBlastHaxpy = CLBlastStatusCode function(const size_t n,
      const cl_half alpha,
      const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
      cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
      cl_command_queue* queue, cl_event* event);
}

/**
 * xDOT: Dot product of two vectors
 *
 * Multiplies _n_ elements of the vectors _x_ and _y_ element-wise and accumulates the results. The sum is stored in the _dot_ buffer.
 *
 * Arguments to DOT:
 * - `const size_t n`: Integer size argument. This value must be positive.
 * - `cl_mem dot_buffer`: OpenCL buffer to store the output dot vector.
 * - `const size_t dot_offset`: The offset in elements from the start of the output dot vector.
 * - `const cl_mem x_buffer`: OpenCL buffer to store the input x vector.
 * - `const size_t x_offset`: The offset in elements from the start of the input x vector.
 * - `const size_t x_inc`: Stride/increment of the input x vector. This value must be greater than 0.
 * - `const cl_mem y_buffer`: OpenCL buffer to store the input y vector.
 * - `const size_t y_offset`: The offset in elements from the start of the input y vector.
 * - `const size_t y_inc`: Stride/increment of the input y vector. This value must be greater than 0.
 * - `cl_command_queue* queue`: Pointer to an OpenCL command queue associated with a context and device to execute the routine on.
 * - `cl_event* event`: Pointer to an OpenCL event to be able to wait for completion of the routine's OpenCL kernel(s). This is an optional argument.
 */
extern(System) @nogc nothrow {
  alias da_CLBlastSdot = CLBlastStatusCode function(const size_t n,
      cl_mem dot_buffer, const size_t dot_offset,
      const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
      const cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
      cl_command_queue* queue, cl_event* event);
  alias da_CLBlastDdot = CLBlastStatusCode function(const size_t n,
      cl_mem dot_buffer, const size_t dot_offset,
      const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
      const cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
      cl_command_queue* queue, cl_event* event);
  alias da_CLBlastHdot = CLBlastStatusCode function(const size_t n,
      cl_mem dot_buffer, const size_t dot_offset,
      const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
      const cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
      cl_command_queue* queue, cl_event* event);
}

/**
 * xDOTU: Dot product of two complex vectors
 *
 * See the regular xDOT routine.
 *
 * Arguments to DOTU:
 * - `const size_t n`: Integer size argument. This value must be positive.
 * - `cl_mem dot_buffer`: OpenCL buffer to store the output dot vector.
 * - `const size_t dot_offset`: The offset in elements from the start of the output dot vector.
 * - `const cl_mem x_buffer`: OpenCL buffer to store the input x vector.
 * - `const size_t x_offset`: The offset in elements from the start of the input x vector.
 * - `const size_t x_inc`: Stride/increment of the input x vector. This value must be greater than 0.
 * - `const cl_mem y_buffer`: OpenCL buffer to store the input y vector.
 * - `const size_t y_offset`: The offset in elements from the start of the input y vector.
 * - `const size_t y_inc`: Stride/increment of the input y vector. This value must be greater than 0.
 * - `cl_command_queue* queue`: Pointer to an OpenCL command queue associated with a context and device to execute the routine on.
 * - `cl_event* event`: Pointer to an OpenCL event to be able to wait for completion of the routine's OpenCL kernel(s). This is an optional argument.
 */
extern(System) @nogc nothrow {
  alias da_CLBlastCdotu = CLBlastStatusCode function(const size_t n,
      cl_mem dot_buffer, const size_t dot_offset,
      const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
      const cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
      cl_command_queue* queue, cl_event* event);
  alias da_CLBlastZdotu = CLBlastStatusCode function(const size_t n,
      cl_mem dot_buffer, const size_t dot_offset,
      const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
      const cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
      cl_command_queue* queue, cl_event* event);
}

/**
 * xDOTC: Dot product of two complex vectors, one conjugated
 *
 * See the regular xDOT routine.
 *
 * Arguments to DOTC:
 * - `const size_t n`: Integer size argument. This value must be positive.
 * - `cl_mem dot_buffer`: OpenCL buffer to store the output dot vector.
 * - `const size_t dot_offset`: The offset in elements from the start of the output dot vector.
 * - `const cl_mem x_buffer`: OpenCL buffer to store the input x vector.
 * - `const size_t x_offset`: The offset in elements from the start of the input x vector.
 * - `const size_t x_inc`: Stride/increment of the input x vector. This value must be greater than 0.
 * - `const cl_mem y_buffer`: OpenCL buffer to store the input y vector.
 * - `const size_t y_offset`: The offset in elements from the start of the input y vector.
 * - `const size_t y_inc`: Stride/increment of the input y vector. This value must be greater than 0.
 * - `cl_command_queue* queue`: Pointer to an OpenCL command queue associated with a context and device to execute the routine on.
 * - `cl_event* event`: Pointer to an OpenCL event to be able to wait for completion of the routine's OpenCL kernel(s). This is an optional argument.
 */
extern(System) @nogc nothrow {
  alias da_CLBlastCdotc = CLBlastStatusCode function(const size_t n,
      cl_mem dot_buffer, const size_t dot_offset,
      const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
      const cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
      cl_command_queue* queue, cl_event* event);
  alias da_CLBlastZdotc = CLBlastStatusCode function(const size_t n,
      cl_mem dot_buffer, const size_t dot_offset,
      const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
      const cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
      cl_command_queue* queue, cl_event* event);
}

/**
 * xNRM2: Euclidian norm of a vector
 *
 * Accumulates the square of _n_ elements in the _x_ vector and takes the square root. The resulting L2 norm is stored in the _nrm2_ buffer.
 *
 * Arguments to NRM2:
 * - `const size_t n`: Integer size argument. This value must be positive.
 * - `cl_mem nrm2_buffer`: OpenCL buffer to store the output nrm2 vector.
 * - `const size_t nrm2_offset`: The offset in elements from the start of the output nrm2 vector.
 * - `const cl_mem x_buffer`: OpenCL buffer to store the input x vector.
 * - `const size_t x_offset`: The offset in elements from the start of the input x vector.
 * - `const size_t x_inc`: Stride/increment of the input x vector. This value must be greater than 0.
 * - `cl_command_queue* queue`: Pointer to an OpenCL command queue associated with a context and device to execute the routine on.
 * - `cl_event* event`: Pointer to an OpenCL event to be able to wait for completion of the routine's OpenCL kernel(s). This is an optional argument.
 */
extern(System) @nogc nothrow {
  alias da_CLBlastSnrm2 = CLBlastStatusCode function(const size_t n,
      cl_mem nrm2_buffer, const size_t nrm2_offset,
      const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
      cl_command_queue* queue, cl_event* event);
  alias da_CLBlastDnrm2 = CLBlastStatusCode function(const size_t n,
      cl_mem nrm2_buffer, const size_t nrm2_offset,
      const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
      cl_command_queue* queue, cl_event* event);
  alias da_CLBlastScnrm2 = CLBlastStatusCode function(const size_t n,
      cl_mem nrm2_buffer, const size_t nrm2_offset,
      const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
      cl_command_queue* queue, cl_event* event);
  alias da_CLBlastDznrm2 = CLBlastStatusCode function(const size_t n,
      cl_mem nrm2_buffer, const size_t nrm2_offset,
      const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
      cl_command_queue* queue, cl_event* event);
  alias da_CLBlastHnrm2 = CLBlastStatusCode function(const size_t n,
      cl_mem nrm2_buffer, const size_t nrm2_offset,
      const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
      cl_command_queue* queue, cl_event* event);
}

/**
 * xASUM: Absolute sum of values in a vector
 *
 * Accumulates the absolute value of _n_ elements in the _x_ vector. The results are stored in the _asum_ buffer.
 *
 * Arguments to ASUM:
 * - `const size_t n`: Integer size argument. This value must be positive.
 * - `cl_mem asum_buffer`: OpenCL buffer to store the output asum vector.
 * - `const size_t asum_offset`: The offset in elements from the start of the output asum vector.
 * - `const cl_mem x_buffer`: OpenCL buffer to store the input x vector.
 * - `const size_t x_offset`: The offset in elements from the start of the input x vector.
 * - `const size_t x_inc`: Stride/increment of the input x vector. This value must be greater than 0.
 * - `cl_command_queue* queue`: Pointer to an OpenCL command queue associated with a context and device to execute the routine on.
 * - `cl_event* event`: Pointer to an OpenCL event to be able to wait for completion of the routine's OpenCL kernel(s). This is an optional argument.
 */
extern(System) @nogc nothrow {
  alias da_CLBlastSasum = CLBlastStatusCode function(const size_t n,
      cl_mem asum_buffer, const size_t asum_offset,
      const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
      cl_command_queue* queue, cl_event* event);
  alias da_CLBlastDasum = CLBlastStatusCode function(const size_t n,
      cl_mem asum_buffer, const size_t asum_offset,
      const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
      cl_command_queue* queue, cl_event* event);
  alias da_CLBlastScasum = CLBlastStatusCode function(const size_t n,
      cl_mem asum_buffer, const size_t asum_offset,
      const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
      cl_command_queue* queue, cl_event* event);
  alias da_CLBlastDzasum = CLBlastStatusCode function(const size_t n,
      cl_mem asum_buffer, const size_t asum_offset,
      const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
      cl_command_queue* queue, cl_event* event);
  alias da_CLBlastHasum = CLBlastStatusCode function(const size_t n,
      cl_mem asum_buffer, const size_t asum_offset,
      const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
      cl_command_queue* queue, cl_event* event);
}

/**
 * xSUM: Sum of values in a vector (non-BLAS function)
 *
 * Accumulates the values of _n_ elements in the _x_ vector. The results are stored in the _sum_ buffer. This routine is the non-absolute version of the xASUM BLAS routine.
 *
 * Arguments to SUM:
 * - `const size_t n`: Integer size argument. This value must be positive.
 * - `cl_mem sum_buffer`: OpenCL buffer to store the output sum vector.
 * - `const size_t sum_offset`: The offset in elements from the start of the output sum vector.
 * - `const cl_mem x_buffer`: OpenCL buffer to store the input x vector.
 * - `const size_t x_offset`: The offset in elements from the start of the input x vector.
 * - `const size_t x_inc`: Stride/increment of the input x vector. This value must be greater than 0.
 * - `cl_command_queue* queue`: Pointer to an OpenCL command queue associated with a context and device to execute the routine on.
 * - `cl_event* event`: Pointer to an OpenCL event to be able to wait for completion of the routine's OpenCL kernel(s). This is an optional argument.
 */
extern(System) @nogc nothrow {
  alias da_CLBlastSsum = CLBlastStatusCode function(const size_t n,
      cl_mem sum_buffer, const size_t sum_offset,
      const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
      cl_command_queue* queue, cl_event* event);
  alias da_CLBlastDsum = CLBlastStatusCode function(const size_t n,
      cl_mem sum_buffer, const size_t sum_offset,
      const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
      cl_command_queue* queue, cl_event* event);
  alias da_CLBlastScsum = CLBlastStatusCode function(const size_t n,
      cl_mem sum_buffer, const size_t sum_offset,
      const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
      cl_command_queue* queue, cl_event* event);
  alias da_CLBlastDzsum = CLBlastStatusCode function(const size_t n,
      cl_mem sum_buffer, const size_t sum_offset,
      const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
      cl_command_queue* queue, cl_event* event);
  alias da_CLBlastHsum = CLBlastStatusCode function(const size_t n,
      cl_mem sum_buffer, const size_t sum_offset,
      const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
      cl_command_queue* queue, cl_event* event);
}

/**
 * xAMAX: Index of absolute maximum value in a vector
 *
 * Finds the index of a maximum (not necessarily the first if there are multiple) of the absolute values in the _x_ vector. The resulting integer index is stored in the _imax_ buffer.
 *
 * Arguments to AMAX:
 * - `const size_t n`: Integer size argument. This value must be positive.
 * - `cl_mem imax_buffer`: OpenCL buffer to store the output imax vector.
 * - `const size_t imax_offset`: The offset in elements from the start of the output imax vector.
 * - `const cl_mem x_buffer`: OpenCL buffer to store the input x vector.
 * - `const size_t x_offset`: The offset in elements from the start of the input x vector.
 * - `const size_t x_inc`: Stride/increment of the input x vector. This value must be greater than 0.
 * - `cl_command_queue* queue`: Pointer to an OpenCL command queue associated with a context and device to execute the routine on.
 * - `cl_event* event`: Pointer to an OpenCL event to be able to wait for completion of the routine's OpenCL kernel(s). This is an optional argument.
 */
extern(System) @nogc nothrow {
  alias da_CLBlastiSamax = CLBlastStatusCode function(const size_t n,
      cl_mem imax_buffer, const size_t imax_offset,
      const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
      cl_command_queue* queue, cl_event* event);
  alias da_CLBlastiDamax = CLBlastStatusCode function(const size_t n,
      cl_mem imax_buffer, const size_t imax_offset,
      const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
      cl_command_queue* queue, cl_event* event);
  alias da_CLBlastiCamax = CLBlastStatusCode function(const size_t n,
      cl_mem imax_buffer, const size_t imax_offset,
      const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
      cl_command_queue* queue, cl_event* event);
  alias da_CLBlastiZamax = CLBlastStatusCode function(const size_t n,
      cl_mem imax_buffer, const size_t imax_offset,
      const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
      cl_command_queue* queue, cl_event* event);
  alias da_CLBlastiHamax = CLBlastStatusCode function(const size_t n,
      cl_mem imax_buffer, const size_t imax_offset,
      const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
      cl_command_queue* queue, cl_event* event);
}

/**
 * xAMIN: Index of absolute minimum value in a vector (non-BLAS function)
 *
 * Finds the index of a minimum (not necessarily the first if there are multiple) of the absolute values in the _x_ vector. The resulting integer index is stored in the _imin_ buffer.
 *
 * Arguments to AMIN:
 * - `const size_t n`: Integer size argument. This value must be positive.
 * - `cl_mem imin_buffer`: OpenCL buffer to store the output imin vector.
 * - `const size_t imin_offset`: The offset in elements from the start of the output imin vector.
 * - `const cl_mem x_buffer`: OpenCL buffer to store the input x vector.
 * - `const size_t x_offset`: The offset in elements from the start of the input x vector.
 * - `const size_t x_inc`: Stride/increment of the input x vector. This value must be greater than 0.
 * - `cl_command_queue* queue`: Pointer to an OpenCL command queue associated with a context and device to execute the routine on.
 * - `cl_event* event`: Pointer to an OpenCL event to be able to wait for completion of the routine's OpenCL kernel(s). This is an optional argument.
 */
extern(System) @nogc nothrow {
  alias da_CLBlastiSamin = CLBlastStatusCode function(const size_t n,
      cl_mem imin_buffer, const size_t imin_offset,
      const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
      cl_command_queue* queue, cl_event* event);
  alias da_CLBlastiDamin = CLBlastStatusCode function(const size_t n,
      cl_mem imin_buffer, const size_t imin_offset,
      const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
      cl_command_queue* queue, cl_event* event);
  alias da_CLBlastiCamin = CLBlastStatusCode function(const size_t n,
      cl_mem imin_buffer, const size_t imin_offset,
      const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
      cl_command_queue* queue, cl_event* event);
  alias da_CLBlastiZamin = CLBlastStatusCode function(const size_t n,
      cl_mem imin_buffer, const size_t imin_offset,
      const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
      cl_command_queue* queue, cl_event* event);
  alias da_CLBlastiHamin = CLBlastStatusCode function(const size_t n,
      cl_mem imin_buffer, const size_t imin_offset,
      const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
      cl_command_queue* queue, cl_event* event);
}

/**
 * xMAX: Index of maximum value in a vector (non-BLAS function)
 *
 * Finds the index of a maximum (not necessarily the first if there are multiple) of the values in the _x_ vector. The resulting integer index is stored in the _imax_ buffer. This routine is the non-absolute version of the IxAMAX BLAS routine.
 *
 * Arguments to MAX:
 * - `const size_t n`: Integer size argument. This value must be positive.
 * - `cl_mem imax_buffer`: OpenCL buffer to store the output imax vector.
 * - `const size_t imax_offset`: The offset in elements from the start of the output imax vector.
 * - `const cl_mem x_buffer`: OpenCL buffer to store the input x vector.
 * - `const size_t x_offset`: The offset in elements from the start of the input x vector.
 * - `const size_t x_inc`: Stride/increment of the input x vector. This value must be greater than 0.
 * - `cl_command_queue* queue`: Pointer to an OpenCL command queue associated with a context and device to execute the routine on.
 * - `cl_event* event`: Pointer to an OpenCL event to be able to wait for completion of the routine's OpenCL kernel(s). This is an optional argument.
 */
extern(System) @nogc nothrow {
  alias da_CLBlastiSmax = CLBlastStatusCode function(const size_t n,
      cl_mem imax_buffer, const size_t imax_offset,
      const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
      cl_command_queue* queue, cl_event* event);
  alias da_CLBlastiDmax = CLBlastStatusCode function(const size_t n,
      cl_mem imax_buffer, const size_t imax_offset,
      const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
      cl_command_queue* queue, cl_event* event);
  alias da_CLBlastiCmax = CLBlastStatusCode function(const size_t n,
      cl_mem imax_buffer, const size_t imax_offset,
      const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
      cl_command_queue* queue, cl_event* event);
  alias da_CLBlastiZmax = CLBlastStatusCode function(const size_t n,
      cl_mem imax_buffer, const size_t imax_offset,
      const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
      cl_command_queue* queue, cl_event* event);
  alias da_CLBlastiHmax = CLBlastStatusCode function(const size_t n,
      cl_mem imax_buffer, const size_t imax_offset,
      const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
      cl_command_queue* queue, cl_event* event);
}

/**
 * xMIN: Index of minimum value in a vector (non-BLAS function)
 *
 * Finds the index of a minimum (not necessarily the first if there are multiple) of the values in the _x_ vector. The resulting integer index is stored in the _imin_ buffer. This routine is the non-absolute minimum version of the IxAMAX BLAS routine.
 *
 * Arguments to MIN:
 * - `const size_t n`: Integer size argument. This value must be positive.
 * - `cl_mem imin_buffer`: OpenCL buffer to store the output imin vector.
 * - `const size_t imin_offset`: The offset in elements from the start of the output imin vector.
 * - `const cl_mem x_buffer`: OpenCL buffer to store the input x vector.
 * - `const size_t x_offset`: The offset in elements from the start of the input x vector.
 * - `const size_t x_inc`: Stride/increment of the input x vector. This value must be greater than 0.
 * - `cl_command_queue* queue`: Pointer to an OpenCL command queue associated with a context and device to execute the routine on.
 * - `cl_event* event`: Pointer to an OpenCL event to be able to wait for completion of the routine's OpenCL kernel(s). This is an optional argument.
 */
extern(System) @nogc nothrow {
  alias da_CLBlastiSmin = CLBlastStatusCode function(const size_t n,
      cl_mem imin_buffer, const size_t imin_offset,
      const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
      cl_command_queue* queue, cl_event* event);
  alias da_CLBlastiDmin = CLBlastStatusCode function(const size_t n,
      cl_mem imin_buffer, const size_t imin_offset,
      const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
      cl_command_queue* queue, cl_event* event);
  alias da_CLBlastiCmin = CLBlastStatusCode function(const size_t n,
      cl_mem imin_buffer, const size_t imin_offset,
      const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
      cl_command_queue* queue, cl_event* event);
  alias da_CLBlastiZmin = CLBlastStatusCode function(const size_t n,
      cl_mem imin_buffer, const size_t imin_offset,
      const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
      cl_command_queue* queue, cl_event* event);
  alias da_CLBlastiHmin = CLBlastStatusCode function(const size_t n,
      cl_mem imin_buffer, const size_t imin_offset,
      const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
      cl_command_queue* queue, cl_event* event);
}

// =================================================================================================
// BLAS level-2 (matrix-vector) routines
// =================================================================================================

/**
 * xGEMV: General matrix-vector multiplication
 *
 * Performs the operation _y = alpha * A * x + beta * y_, in which _x_ is an input vector, _y_ is an input and output vector, _A_ is an input matrix, and _alpha_ and _beta_ are scalars. The matrix _A_ can optionally be transposed before performing the operation.
 *
 * Arguments to GEMV:
 * - `const Layout layout`: Data-layout of the matrices, either `Layout::kRowMajor` (101) for row-major layout or `Layout::kColMajor` (102) for column-major data-layout.
 * - `const Transpose a_transpose`: Transposing the input matrix A, either `Transpose::kNo` (111), `Transpose::kYes` (112), or `Transpose::kConjugate` (113) for a complex-conjugate transpose.
 * - `const size_t m`: Integer size argument. This value must be positive.
 * - `const size_t n`: Integer size argument. This value must be positive.
 * - `const T alpha`: Input scalar constant.
 * - `const cl_mem a_buffer`: OpenCL buffer to store the input A matrix.
 * - `const size_t a_offset`: The offset in elements from the start of the input A matrix.
 * - `const size_t a_ld`: Leading dimension of the input A matrix. This value must be greater than 0.
 * - `const cl_mem x_buffer`: OpenCL buffer to store the input x vector.
 * - `const size_t x_offset`: The offset in elements from the start of the input x vector.
 * - `const size_t x_inc`: Stride/increment of the input x vector. This value must be greater than 0.
 * - `const T beta`: Input scalar constant.
 * - `cl_mem y_buffer`: OpenCL buffer to store the output y vector.
 * - `const size_t y_offset`: The offset in elements from the start of the output y vector.
 * - `const size_t y_inc`: Stride/increment of the output y vector. This value must be greater than 0.
 * - `cl_command_queue* queue`: Pointer to an OpenCL command queue associated with a context and device to execute the routine on.
 * - `cl_event* event`: Pointer to an OpenCL event to be able to wait for completion of the routine's OpenCL kernel(s). This is an optional argument.
 */
extern(System) @nogc nothrow {
  alias da_CLBlastSgemv = CLBlastStatusCode function(const CLBlastLayout layout, const CLBlastTranspose a_transpose,
      const size_t m, const size_t n,
      const float alpha,
      const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
      const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
      const float beta,
      cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
      cl_command_queue* queue, cl_event* event);
  alias da_CLBlastDgemv = CLBlastStatusCode function(const CLBlastLayout layout, const CLBlastTranspose a_transpose,
      const size_t m, const size_t n,
      const double alpha,
      const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
      const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
      const double beta,
      cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
      cl_command_queue* queue, cl_event* event);
  alias da_CLBlastCgemv = CLBlastStatusCode function(const CLBlastLayout layout, const CLBlastTranspose a_transpose,
      const size_t m, const size_t n,
      const cl_float2 alpha,
      const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
      const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
      const cl_float2 beta,
      cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
      cl_command_queue* queue, cl_event* event);
  alias da_CLBlastZgemv = CLBlastStatusCode function(const CLBlastLayout layout, const CLBlastTranspose a_transpose,
      const size_t m, const size_t n,
      const cl_double2 alpha,
      const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
      const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
      const cl_double2 beta,
      cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
      cl_command_queue* queue, cl_event* event);
  alias da_CLBlastHgemv = CLBlastStatusCode function(const CLBlastLayout layout, const CLBlastTranspose a_transpose,
      const size_t m, const size_t n,
      const cl_half alpha,
      const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
      const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
      const cl_half beta,
      cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
      cl_command_queue* queue, cl_event* event);
}

/**
 * xGBMV: General banded matrix-vector multiplication
 *
 * Same operation as xGEMV, but matrix _A_ is banded instead.
 *
 * Arguments to GBMV:
 * - `const Layout layout`: Data-layout of the matrices, either `Layout::kRowMajor` (101) for row-major layout or `Layout::kColMajor` (102) for column-major data-layout.
 * - `const Transpose a_transpose`: Transposing the input matrix A, either `Transpose::kNo` (111), `Transpose::kYes` (112), or `Transpose::kConjugate` (113) for a complex-conjugate transpose.
 * - `const size_t m`: Integer size argument. This value must be positive.
 * - `const size_t n`: Integer size argument. This value must be positive.
 * - `const size_t kl`: Integer size argument. This value must be positive.
 * - `const size_t ku`: Integer size argument. This value must be positive.
 * - `const T alpha`: Input scalar constant.
 * - `const cl_mem a_buffer`: OpenCL buffer to store the input A matrix.
 * - `const size_t a_offset`: The offset in elements from the start of the input A matrix.
 * - `const size_t a_ld`: Leading dimension of the input A matrix. This value must be greater than 0.
 * - `const cl_mem x_buffer`: OpenCL buffer to store the input x vector.
 * - `const size_t x_offset`: The offset in elements from the start of the input x vector.
 * - `const size_t x_inc`: Stride/increment of the input x vector. This value must be greater than 0.
 * - `const T beta`: Input scalar constant.
 * - `cl_mem y_buffer`: OpenCL buffer to store the output y vector.
 * - `const size_t y_offset`: The offset in elements from the start of the output y vector.
 * - `const size_t y_inc`: Stride/increment of the output y vector. This value must be greater than 0.
 * - `cl_command_queue* queue`: Pointer to an OpenCL command queue associated with a context and device to execute the routine on.
 * - `cl_event* event`: Pointer to an OpenCL event to be able to wait for completion of the routine's OpenCL kernel(s). This is an optional argument.
 */
extern(System) @nogc nothrow {
  alias da_CLBlastSgbmv = CLBlastStatusCode function(const CLBlastLayout layout, const CLBlastTranspose a_transpose,
      const size_t m, const size_t n, const size_t kl, const size_t ku,
      const float alpha,
      const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
      const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
      const float beta,
      cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
      cl_command_queue* queue, cl_event* event);
  alias da_CLBlastDgbmv = CLBlastStatusCode function(const CLBlastLayout layout, const CLBlastTranspose a_transpose,
      const size_t m, const size_t n, const size_t kl, const size_t ku,
      const double alpha,
      const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
      const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
      const double beta,
      cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
      cl_command_queue* queue, cl_event* event);
  alias da_CLBlastCgbmv = CLBlastStatusCode function(const CLBlastLayout layout, const CLBlastTranspose a_transpose,
      const size_t m, const size_t n, const size_t kl, const size_t ku,
      const cl_float2 alpha,
      const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
      const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
      const cl_float2 beta,
      cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
      cl_command_queue* queue, cl_event* event);
  alias da_CLBlastZgbmv = CLBlastStatusCode function(const CLBlastLayout layout, const CLBlastTranspose a_transpose,
      const size_t m, const size_t n, const size_t kl, const size_t ku,
      const cl_double2 alpha,
      const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
      const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
      const cl_double2 beta,
      cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
      cl_command_queue* queue, cl_event* event);
  alias da_CLBlastHgbmv = CLBlastStatusCode function(const CLBlastLayout layout, const CLBlastTranspose a_transpose,
      const size_t m, const size_t n, const size_t kl, const size_t ku,
      const cl_half alpha,
      const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
      const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
      const cl_half beta,
      cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
      cl_command_queue* queue, cl_event* event);
}

/**
 * xHEMV: Hermitian matrix-vector multiplication
 *
 * Same operation as xGEMV, but matrix _A_ is an Hermitian matrix instead.
 *
 * Arguments to HEMV:
 * - `const Layout layout`: Data-layout of the matrices, either `Layout::kRowMajor` (101) for row-major layout or `Layout::kColMajor` (102) for column-major data-layout.
 * - `const Triangle triangle`: The part of the array of the triangular matrix to be used, either `Triangle::kUpper` (121) or `Triangle::kLower` (122).
 * - `const size_t n`: Integer size argument. This value must be positive.
 * - `const T alpha`: Input scalar constant.
 * - `const cl_mem a_buffer`: OpenCL buffer to store the input A matrix.
 * - `const size_t a_offset`: The offset in elements from the start of the input A matrix.
 * - `const size_t a_ld`: Leading dimension of the input A matrix. This value must be greater than 0.
 * - `const cl_mem x_buffer`: OpenCL buffer to store the input x vector.
 * - `const size_t x_offset`: The offset in elements from the start of the input x vector.
 * - `const size_t x_inc`: Stride/increment of the input x vector. This value must be greater than 0.
 * - `const T beta`: Input scalar constant.
 * - `cl_mem y_buffer`: OpenCL buffer to store the output y vector.
 * - `const size_t y_offset`: The offset in elements from the start of the output y vector.
 * - `const size_t y_inc`: Stride/increment of the output y vector. This value must be greater than 0.
 * - `cl_command_queue* queue`: Pointer to an OpenCL command queue associated with a context and device to execute the routine on.
 * - `cl_event* event`: Pointer to an OpenCL event to be able to wait for completion of the routine's OpenCL kernel(s). This is an optional argument.
 *
 * Requirements for HEMV:
 * The value of `a_ld` must be at least `n`.
 */
extern(System) @nogc nothrow {
  alias da_CLBlastChemv = CLBlastStatusCode function(const CLBlastLayout layout, const CLBlastTriangle triangle,
      const size_t n,
      const cl_float2 alpha,
      const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
      const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
      const cl_float2 beta,
      cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
      cl_command_queue* queue, cl_event* event);
  alias da_CLBlastZhemv = CLBlastStatusCode function(const CLBlastLayout layout, const CLBlastTriangle triangle,
      const size_t n,
      const cl_double2 alpha,
      const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
      const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
      const cl_double2 beta,
      cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
      cl_command_queue* queue, cl_event* event);
}

/**
 * xHBMV: Hermitian banded matrix-vector multiplication
 *
 * Same operation as xGEMV, but matrix _A_ is an Hermitian banded matrix instead.
 *
 * Arguments to HBMV:
 * - `const Layout layout`: Data-layout of the matrices, either `Layout::kRowMajor` (101) for row-major layout or `Layout::kColMajor` (102) for column-major data-layout.
 * - `const Triangle triangle`: The part of the array of the triangular matrix to be used, either `Triangle::kUpper` (121) or `Triangle::kLower` (122).
 * - `const size_t n`: Integer size argument. This value must be positive.
 * - `const size_t k`: Integer size argument. This value must be positive.
 * - `const T alpha`: Input scalar constant.
 * - `const cl_mem a_buffer`: OpenCL buffer to store the input A matrix.
 * - `const size_t a_offset`: The offset in elements from the start of the input A matrix.
 * - `const size_t a_ld`: Leading dimension of the input A matrix. This value must be greater than 0.
 * - `const cl_mem x_buffer`: OpenCL buffer to store the input x vector.
 * - `const size_t x_offset`: The offset in elements from the start of the input x vector.
 * - `const size_t x_inc`: Stride/increment of the input x vector. This value must be greater than 0.
 * - `const T beta`: Input scalar constant.
 * - `cl_mem y_buffer`: OpenCL buffer to store the output y vector.
 * - `const size_t y_offset`: The offset in elements from the start of the output y vector.
 * - `const size_t y_inc`: Stride/increment of the output y vector. This value must be greater than 0.
 * - `cl_command_queue* queue`: Pointer to an OpenCL command queue associated with a context and device to execute the routine on.
 * - `cl_event* event`: Pointer to an OpenCL event to be able to wait for completion of the routine's OpenCL kernel(s). This is an optional argument.
 *
 * Requirements for HBMV:
 * The value of `a_ld` must be at least `k + 1`.
 */
extern(System) @nogc nothrow {
  alias da_CLBlastChbmv = CLBlastStatusCode function(const CLBlastLayout layout, const CLBlastTriangle triangle,
      const size_t n, const size_t k,
      const cl_float2 alpha,
      const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
      const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
      const cl_float2 beta,
      cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
      cl_command_queue* queue, cl_event* event);
  alias da_CLBlastZhbmv = CLBlastStatusCode function(const CLBlastLayout layout, const CLBlastTriangle triangle,
      const size_t n, const size_t k,
      const cl_double2 alpha,
      const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
      const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
      const cl_double2 beta,
      cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
      cl_command_queue* queue, cl_event* event);
}

/**
 * xSYMV: Symmetric matrix-vector multiplication
 *
 * Same operation as xGEMV, but matrix _A_ is symmetric instead.
 *
 * Arguments to SYMV:

 * - `const Layout layout`: Data-layout of the matrices, either `Layout::kRowMajor` (101) for row-major layout or `Layout::kColMajor` (102) for column-major data-layout.
 * - `const Triangle triangle`: The part of the array of the triangular matrix to be used, either `Triangle::kUpper` (121) or `Triangle::kLower` (122).
 * - `const size_t n`: Integer size argument. This value must be positive.
 * - `const T alpha`: Input scalar constant.
 * - `const cl_mem a_buffer`: OpenCL buffer to store the input A matrix.
 * - `const size_t a_offset`: The offset in elements from the start of the input A matrix.
 * - `const size_t a_ld`: Leading dimension of the input A matrix. This value must be greater than 0.
 * - `const cl_mem x_buffer`: OpenCL buffer to store the input x vector.
 * - `const size_t x_offset`: The offset in elements from the start of the input x vector.
 * - `const size_t x_inc`: Stride/increment of the input x vector. This value must be greater than 0.
 * - `const T beta`: Input scalar constant.
 * - `cl_mem y_buffer`: OpenCL buffer to store the output y vector.
 * - `const size_t y_offset`: The offset in elements from the start of the output y vector.
 * - `const size_t y_inc`: Stride/increment of the output y vector. This value must be greater than 0.
 * - `cl_command_queue* queue`: Pointer to an OpenCL command queue associated with a context and device to execute the routine on.
 * - `cl_event* event`: Pointer to an OpenCL event to be able to wait for completion of the routine's OpenCL kernel(s). This is an optional argument.
 *
 * Requirements for SYMV:
 * - The value of `a_ld` must be at least `n`.
 */
extern(System) @nogc nothrow {
  alias da_CLBlastSsymv = CLBlastStatusCode function(const CLBlastLayout layout, const CLBlastTriangle triangle,
      const size_t n,
      const float alpha,
      const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
      const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
      const float beta,
      cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
      cl_command_queue* queue, cl_event* event);
  alias da_CLBlastDsymv = CLBlastStatusCode function(const CLBlastLayout layout, const CLBlastTriangle triangle,
      const size_t n,
      const double alpha,
      const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
      const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
      const double beta,
      cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
      cl_command_queue* queue, cl_event* event);
  alias da_CLBlastHsymv = CLBlastStatusCode function(const CLBlastLayout layout, const CLBlastTriangle triangle,
      const size_t n,
      const cl_half alpha,
      const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
      const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
      const cl_half beta,
      cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
      cl_command_queue* queue, cl_event* event);
}

/**
 * xSBMV: Symmetric banded matrix-vector multiplication
 *
 * Same operation as xGEMV, but matrix _A_ is symmetric and banded instead.
 *
 * Arguments to SBMV:
 * - `const Layout layout`: Data-layout of the matrices, either `Layout::kRowMajor` (101) for row-major layout or `Layout::kColMajor` (102) for column-major data-layout.
 * - `const Triangle triangle`: The part of the array of the triangular matrix to be used, either `Triangle::kUpper` (121) or `Triangle::kLower` (122).
 * - `const size_t n`: Integer size argument. This value must be positive.
 * - `const size_t k`: Integer size argument. This value must be positive.
 * - `const T alpha`: Input scalar constant.
 * - `const cl_mem a_buffer`: OpenCL buffer to store the input A matrix.
 * - `const size_t a_offset`: The offset in elements from the start of the input A matrix.
 * - `const size_t a_ld`: Leading dimension of the input A matrix. This value must be greater than 0.
 * - `const cl_mem x_buffer`: OpenCL buffer to store the input x vector.
 * - `const size_t x_offset`: The offset in elements from the start of the input x vector.
 * - `const size_t x_inc`: Stride/increment of the input x vector. This value must be greater than 0.
 * - `const T beta`: Input scalar constant.
 * - `cl_mem y_buffer`: OpenCL buffer to store the output y vector.
 * - `const size_t y_offset`: The offset in elements from the start of the output y vector.
 * - `const size_t y_inc`: Stride/increment of the output y vector. This value must be greater than 0.
 * - `cl_command_queue* queue`: Pointer to an OpenCL command queue associated with a context and device to execute the routine on.
 * - `cl_event* event`: Pointer to an OpenCL event to be able to wait for completion of the routine's OpenCL kernel(s). This is an optional argument.
 *
 * Requirements for SBMV:
 * - The value of `a_ld` must be at least `k + 1`.
 */
extern(System) @nogc nothrow {
  alias da_CLBlastSsbmv = CLBlastStatusCode function(const CLBlastLayout layout, const CLBlastTriangle triangle,
      const size_t n, const size_t k,
      const float alpha,
      const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
      const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
      const float beta,
      cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
      cl_command_queue* queue, cl_event* event);
  alias da_CLBlastDsbmv = CLBlastStatusCode function(const CLBlastLayout layout, const CLBlastTriangle triangle,
      const size_t n, const size_t k,
      const double alpha,
      const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
      const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
      const double beta,
      cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
      cl_command_queue* queue, cl_event* event);
  alias da_CLBlastHsbmv = CLBlastStatusCode function(const CLBlastLayout layout, const CLBlastTriangle triangle,
      const size_t n, const size_t k,
      const cl_half alpha,
      const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
      const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
      const cl_half beta,
      cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
      cl_command_queue* queue, cl_event* event);
}

/**
 * xSPMV: Symmetric packed matrix-vector multiplication
 *
 * Same operation as xGEMV, but matrix _A_ is a symmetric packed matrix instead and represented as _AP_.
 *
 * Arguments to SPMV:
 * - `const Layout layout`: Data-layout of the matrices, either `Layout::kRowMajor` (101) for row-major layout or `Layout::kColMajor` (102) for column-major data-layout.
 * - `const Triangle triangle`: The part of the array of the triangular matrix to be used, either `Triangle::kUpper` (121) or `Triangle::kLower` (122).
 * - `const size_t n`: Integer size argument. This value must be positive.
 * - `const T alpha`: Input scalar constant.
 * - `const cl_mem ap_buffer`: OpenCL buffer to store the input AP matrix.
 * - `const size_t ap_offset`: The offset in elements from the start of the input AP matrix.
 * - `const cl_mem x_buffer`: OpenCL buffer to store the input x vector.
 * - `const size_t x_offset`: The offset in elements from the start of the input x vector.
 * - `const size_t x_inc`: Stride/increment of the input x vector. This value must be greater than 0.
 * - `const T beta`: Input scalar constant.
 * - `cl_mem y_buffer`: OpenCL buffer to store the output y vector.
 * - `const size_t y_offset`: The offset in elements from the start of the output y vector.
 * - `const size_t y_inc`: Stride/increment of the output y vector. This value must be greater than 0.
 * - `cl_command_queue* queue`: Pointer to an OpenCL command queue associated with a context and device to execute the routine on.
 * - `cl_event* event`: Pointer to an OpenCL event to be able to wait for completion of the routine's OpenCL kernel(s). This is an optional argument.
 */
extern(System) @nogc nothrow {
  alias da_CLBlastSspmv = CLBlastStatusCode function(const CLBlastLayout layout, const CLBlastTriangle triangle,
      const size_t n,
      const float alpha,
      const cl_mem ap_buffer, const size_t ap_offset,
      const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
      const float beta,
      cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
      cl_command_queue* queue, cl_event* event);
  alias da_CLBlastDspmv = CLBlastStatusCode function(const CLBlastLayout layout, const CLBlastTriangle triangle,
      const size_t n,
      const double alpha,
      const cl_mem ap_buffer, const size_t ap_offset,
      const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
      const double beta,
      cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
      cl_command_queue* queue, cl_event* event);
  alias da_CLBlastHspmv = CLBlastStatusCode function(const CLBlastLayout layout, const CLBlastTriangle triangle,
      const size_t n,
      const cl_half alpha,
      const cl_mem ap_buffer, const size_t ap_offset,
      const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
      const cl_half beta,
      cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
      cl_command_queue* queue, cl_event* event);
}

/**
 * xTRMV: Triangular matrix-vector multiplication
 *
 * Same operation as xGEMV, but matrix _A_ is triangular instead.
 *
 * Arguments to TRMV:
 * - `const Layout layout`: Data-layout of the matrices, either `Layout::kRowMajor` (101) for row-major layout or `Layout::kColMajor` (102) for column-major data-layout.
 * - `const Triangle triangle`: The part of the array of the triangular matrix to be used, either `Triangle::kUpper` (121) or `Triangle::kLower` (122).
 * - `const Transpose a_transpose`: Transposing the input matrix A, either `Transpose::kNo` (111), `Transpose::kYes` (112), or `Transpose::kConjugate` (113) for a complex-conjugate transpose.
 * - `const Diagonal diagonal`: The property of the diagonal matrix, either `Diagonal::kNonUnit` (131) for non-unit values on the diagonal or `Diagonal::kUnit` (132) for unit values on the diagonal.
 * - `const size_t n`: Integer size argument. This value must be positive.
 * - `const cl_mem a_buffer`: OpenCL buffer to store the input A matrix.
 * - `const size_t a_offset`: The offset in elements from the start of the input A matrix.
 * - `const size_t a_ld`: Leading dimension of the input A matrix. This value must be greater than 0.
 * - `cl_mem x_buffer`: OpenCL buffer to store the output x vector.
 * - `const size_t x_offset`: The offset in elements from the start of the output x vector.
 * - `const size_t x_inc`: Stride/increment of the output x vector. This value must be greater than 0.
 * - `cl_command_queue* queue`: Pointer to an OpenCL command queue associated with a context and device to execute the routine on.
 * - `cl_event* event`: Pointer to an OpenCL event to be able to wait for completion of the routine's OpenCL kernel(s). This is an optional argument.
 *
 * Requirements for TRMV:
 * - The value of `a_ld` must be at least `n`.
 */
extern(System) @nogc nothrow {
  alias da_CLBlastStrmv = CLBlastStatusCode function(const CLBlastLayout layout, const CLBlastTriangle triangle, const CLBlastTranspose a_transpose, const CLBlastDiagonal diagonal,
      const size_t n,
      const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
      cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
      cl_command_queue* queue, cl_event* event);
  alias da_CLBlastDtrmv = CLBlastStatusCode function(const CLBlastLayout layout, const CLBlastTriangle triangle, const CLBlastTranspose a_transpose, const CLBlastDiagonal diagonal,
      const size_t n,
      const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
      cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
      cl_command_queue* queue, cl_event* event);
  alias da_CLBlastCtrmv = CLBlastStatusCode function(const CLBlastLayout layout, const CLBlastTriangle triangle, const CLBlastTranspose a_transpose, const CLBlastDiagonal diagonal,
      const size_t n,
      const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
      cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
      cl_command_queue* queue, cl_event* event);
  alias da_CLBlastZtrmv = CLBlastStatusCode function(const CLBlastLayout layout, const CLBlastTriangle triangle, const CLBlastTranspose a_transpose, const CLBlastDiagonal diagonal,
      const size_t n,
      const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
      cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
      cl_command_queue* queue, cl_event* event);
  alias da_CLBlastHtrmv = CLBlastStatusCode function(const CLBlastLayout layout, const CLBlastTriangle triangle, const CLBlastTranspose a_transpose, const CLBlastDiagonal diagonal,
      const size_t n,
      const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
      cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
      cl_command_queue* queue, cl_event* event);
}

/**
 * xTBMV: Triangular banded matrix-vector multiplication
 *
 * Same operation as xGEMV, but matrix _A_ is triangular and banded instead.
 *
 * Arguments to TBMV:
 * - `const Layout layout`: Data-layout of the matrices, either `Layout::kRowMajor` (101) for row-major layout or `Layout::kColMajor` (102) for column-major data-layout.
 * - `const Triangle triangle`: The part of the array of the triangular matrix to be used, either `Triangle::kUpper` (121) or `Triangle::kLower` (122).
 * - `const Transpose a_transpose`: Transposing the input matrix A, either `Transpose::kNo` (111), `Transpose::kYes` (112), or `Transpose::kConjugate` (113) for a complex-conjugate transpose.
 * - `const Diagonal diagonal`: The property of the diagonal matrix, either `Diagonal::kNonUnit` (131) for non-unit values on the diagonal or `Diagonal::kUnit` (132) for unit values on the diagonal.
 * - `const size_t n`: Integer size argument. This value must be positive.
 * - `const size_t k`: Integer size argument. This value must be positive.
 * - `const cl_mem a_buffer`: OpenCL buffer to store the input A matrix.
 * - `const size_t a_offset`: The offset in elements from the start of the input A matrix.
 * - `const size_t a_ld`: Leading dimension of the input A matrix. This value must be greater than 0.
 * - `cl_mem x_buffer`: OpenCL buffer to store the output x vector.
 * - `const size_t x_offset`: The offset in elements from the start of the output x vector.
 * - `const size_t x_inc`: Stride/increment of the output x vector. This value must be greater than 0.
 * - `cl_command_queue* queue`: Pointer to an OpenCL command queue associated with a context and device to execute the routine on.
 * - `cl_event* event`: Pointer to an OpenCL event to be able to wait for completion of the routine's OpenCL kernel(s). This is an optional argument.
 *
 * Requirements for TBMV:
 * - The value of `a_ld` must be at least `k + 1`.
 */
extern(System) @nogc nothrow {
  alias da_CLBlastStbmv = CLBlastStatusCode function(const CLBlastLayout layout, const CLBlastTriangle triangle, const CLBlastTranspose a_transpose, const CLBlastDiagonal diagonal,
      const size_t n, const size_t k,
      const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
      cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
      cl_command_queue* queue, cl_event* event);
  alias da_CLBlastDtbmv = CLBlastStatusCode function(const CLBlastLayout layout, const CLBlastTriangle triangle, const CLBlastTranspose a_transpose, const CLBlastDiagonal diagonal,
      const size_t n, const size_t k,
      const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
      cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
      cl_command_queue* queue, cl_event* event);
  alias da_CLBlastCtbmv = CLBlastStatusCode function(const CLBlastLayout layout, const CLBlastTriangle triangle, const CLBlastTranspose a_transpose, const CLBlastDiagonal diagonal,
      const size_t n, const size_t k,
      const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
      cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
      cl_command_queue* queue, cl_event* event);
  alias da_CLBlastZtbmv = CLBlastStatusCode function(const CLBlastLayout layout, const CLBlastTriangle triangle, const CLBlastTranspose a_transpose, const CLBlastDiagonal diagonal,
      const size_t n, const size_t k,
      const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
      cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
      cl_command_queue* queue, cl_event* event);
  alias da_CLBlastHtbmv = CLBlastStatusCode function(const CLBlastLayout layout, const CLBlastTriangle triangle, const CLBlastTranspose a_transpose, const CLBlastDiagonal diagonal,
      const size_t n, const size_t k,
      const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
      cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
      cl_command_queue* queue, cl_event* event);
}

/**
 * xTPMV: Triangular packed matrix-vector multiplication
 *
 * Same operation as xGEMV, but matrix _A_ is a triangular packed matrix instead and repreented as _AP_.
 *
 * Arguments to TPMV:
 * - `const Layout layout`: Data-layout of the matrices, either `Layout::kRowMajor` (101) for row-major layout or `Layout::kColMajor` (102) for column-major data-layout.
 * - `const Triangle triangle`: The part of the array of the triangular matrix to be used, either `Triangle::kUpper` (121) or `Triangle::kLower` (122).
 * - `const Transpose a_transpose`: Transposing the input matrix A, either `Transpose::kNo` (111), `Transpose::kYes` (112), or `Transpose::kConjugate` (113) for a complex-conjugate transpose.
 * - `const Diagonal diagonal`: The property of the diagonal matrix, either `Diagonal::kNonUnit` (131) for non-unit values on the diagonal or `Diagonal::kUnit` (132) for unit values on the diagonal.
 * - `const size_t n`: Integer size argument. This value must be positive.
 * - `const cl_mem ap_buffer`: OpenCL buffer to store the input AP matrix.
 * - `const size_t ap_offset`: The offset in elements from the start of the input AP matrix.
 * - `cl_mem x_buffer`: OpenCL buffer to store the output x vector.
 * - `const size_t x_offset`: The offset in elements from the start of the output x vector.
 * - `const size_t x_inc`: Stride/increment of the output x vector. This value must be greater than 0.
 * - `cl_command_queue* queue`: Pointer to an OpenCL command queue associated with a context and device to execute the routine on.
 * - `cl_event* event`: Pointer to an OpenCL event to be able to wait for completion of the routine's OpenCL kernel(s). This is an optional argument.
 */
extern(System) @nogc nothrow {
  alias da_CLBlastStpmv = CLBlastStatusCode function(const CLBlastLayout layout, const CLBlastTriangle triangle, const CLBlastTranspose a_transpose, const CLBlastDiagonal diagonal,
      const size_t n,
      const cl_mem ap_buffer, const size_t ap_offset,
      cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
      cl_command_queue* queue, cl_event* event);
  alias da_CLBlastDtpmv = CLBlastStatusCode function(const CLBlastLayout layout, const CLBlastTriangle triangle, const CLBlastTranspose a_transpose, const CLBlastDiagonal diagonal,
      const size_t n,
      const cl_mem ap_buffer, const size_t ap_offset,
      cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
      cl_command_queue* queue, cl_event* event);
  alias da_CLBlastCtpmv = CLBlastStatusCode function(const CLBlastLayout layout, const CLBlastTriangle triangle, const CLBlastTranspose a_transpose, const CLBlastDiagonal diagonal,
      const size_t n,
      const cl_mem ap_buffer, const size_t ap_offset,
      cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
      cl_command_queue* queue, cl_event* event);
  alias da_CLBlastZtpmv = CLBlastStatusCode function(const CLBlastLayout layout, const CLBlastTriangle triangle, const CLBlastTranspose a_transpose, const CLBlastDiagonal diagonal,
      const size_t n,
      const cl_mem ap_buffer, const size_t ap_offset,
      cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
      cl_command_queue* queue, cl_event* event);
  alias da_CLBlastHtpmv = CLBlastStatusCode function(const CLBlastLayout layout, const CLBlastTriangle triangle, const CLBlastTranspose a_transpose, const CLBlastDiagonal diagonal,
      const size_t n,
      const cl_mem ap_buffer, const size_t ap_offset,
      cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
      cl_command_queue* queue, cl_event* event);
}

/**
 * xTRSV: Solves a triangular system of equations
 *
 * Arguments to TRSV:
 * - `const Layout layout`: Data-layout of the matrices, either `Layout::kRowMajor` (101) for row-major layout or `Layout::kColMajor` (102) for column-major data-layout.
 * - `const Triangle triangle`: The part of the array of the triangular matrix to be used, either `Triangle::kUpper` (121) or `Triangle::kLower` (122).
 * - `const Transpose a_transpose`: Transposing the input matrix A, either `Transpose::kNo` (111), `Transpose::kYes` (112), or `Transpose::kConjugate` (113) for a complex-conjugate transpose.
 * - `const Diagonal diagonal`: The property of the diagonal matrix, either `Diagonal::kNonUnit` (131) for non-unit values on the diagonal or `Diagonal::kUnit` (132) for unit values on the diagonal.
 * - `const size_t n`: Integer size argument. This value must be positive.
 * - `const cl_mem a_buffer`: OpenCL buffer to store the input A matrix.
 * - `const size_t a_offset`: The offset in elements from the start of the input A matrix.
 * - `const size_t a_ld`: Leading dimension of the input A matrix. This value must be greater than 0.
 * - `cl_mem x_buffer`: OpenCL buffer to store the output x vector.
 * - `const size_t x_offset`: The offset in elements from the start of the output x vector.
 * - `const size_t x_inc`: Stride/increment of the output x vector. This value must be greater than 0.
 * - `cl_command_queue* queue`: Pointer to an OpenCL command queue associated with a context and device to execute the routine on.
 * - `cl_event* event`: Pointer to an OpenCL event to be able to wait for completion of the routine's OpenCL kernel(s). This is an optional argument.
 */
extern(System) @nogc nothrow {
  alias da_CLBlastStrsv = CLBlastStatusCode function(const CLBlastLayout layout, const CLBlastTriangle triangle, const CLBlastTranspose a_transpose, const CLBlastDiagonal diagonal,
      const size_t n,
      const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
      cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
      cl_command_queue* queue, cl_event* event);
  alias da_CLBlastDtrsv = CLBlastStatusCode function(const CLBlastLayout layout, const CLBlastTriangle triangle, const CLBlastTranspose a_transpose, const CLBlastDiagonal diagonal,
      const size_t n,
      const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
      cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
      cl_command_queue* queue, cl_event* event);
  alias da_CLBlastCtrsv = CLBlastStatusCode function(const CLBlastLayout layout, const CLBlastTriangle triangle, const CLBlastTranspose a_transpose, const CLBlastDiagonal diagonal,
      const size_t n,
      const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
      cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
      cl_command_queue* queue, cl_event* event);
  alias da_CLBlastZtrsv = CLBlastStatusCode function(const CLBlastLayout layout, const CLBlastTriangle triangle, const CLBlastTranspose a_transpose, const CLBlastDiagonal diagonal,
      const size_t n,
      const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
      cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
      cl_command_queue* queue, cl_event* event);
}

/**
 * xGER: General rank-1 matrix update
 *
 * Performs the operation _A = alpha * x * y^T + A_, in which _x_ is an input vector, _y^T_ is the transpose of the input vector _y_, _A_ is the matrix to be updated, and _alpha_ is a scalar value.
 *
 * Arguments to GER:
 * - `const Layout layout`: Data-layout of the matrices, either `Layout::kRowMajor` (101) for row-major layout or `Layout::kColMajor` (102) for column-major data-layout.
 * - `const size_t m`: Integer size argument. This value must be positive.
 * - `const size_t n`: Integer size argument. This value must be positive.
 * - `const T alpha`: Input scalar constant.
 * - `const cl_mem x_buffer`: OpenCL buffer to store the input x vector.
 * - `const size_t x_offset`: The offset in elements from the start of the input x vector.
 * - `const size_t x_inc`: Stride/increment of the input x vector. This value must be greater than 0.
 * - `const cl_mem y_buffer`: OpenCL buffer to store the input y vector.
 * - `const size_t y_offset`: The offset in elements from the start of the input y vector.
 * - `const size_t y_inc`: Stride/increment of the input y vector. This value must be greater than 0.
 * - `cl_mem a_buffer`: OpenCL buffer to store the output A matrix.
 * - `const size_t a_offset`: The offset in elements from the start of the output A matrix.
 * - `const size_t a_ld`: Leading dimension of the output A matrix. This value must be greater than 0.
 * - `cl_command_queue* queue`: Pointer to an OpenCL command queue associated with a context and device to execute the routine on.
 * - `cl_event* event`: Pointer to an OpenCL event to be able to wait for completion of the routine's OpenCL kernel(s). This is an optional argument.
 *
 * Requirements for GER:
 * - The value of `a_ld` must be at least `m`.
 */
extern(System) @nogc nothrow {
  alias da_CLBlastSger = CLBlastStatusCode function(const CLBlastLayout layout,
      const size_t m, const size_t n,
      const float alpha,
      const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
      const cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
      cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
      cl_command_queue* queue, cl_event* event);
  alias da_CLBlastDger = CLBlastStatusCode function(const CLBlastLayout layout,
      const size_t m, const size_t n,
      const double alpha,
      const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
      const cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
      cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
      cl_command_queue* queue, cl_event* event);
  alias da_CLBlastHger = CLBlastStatusCode function(const CLBlastLayout layout,
      const size_t m, const size_t n,
      const cl_half alpha,
      const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
      const cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
      cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
      cl_command_queue* queue, cl_event* event);
}

/**
 * xGERU: General rank-1 complex matrix update
 *
 * Same operation as xGER, but with complex data-types.
 *
 * Arguments to GERU:
 * - `const Layout layout`: Data-layout of the matrices, either `Layout::kRowMajor` (101) for row-major layout or `Layout::kColMajor` (102) for column-major data-layout.
 * - `const size_t m`: Integer size argument. This value must be positive.
 * - `const size_t n`: Integer size argument. This value must be positive.
 * - `const T alpha`: Input scalar constant.
 * - `const cl_mem x_buffer`: OpenCL buffer to store the input x vector.
 * - `const size_t x_offset`: The offset in elements from the start of the input x vector.
 * - `const size_t x_inc`: Stride/increment of the input x vector. This value must be greater than 0.
 * - `const cl_mem y_buffer`: OpenCL buffer to store the input y vector.
 * - `const size_t y_offset`: The offset in elements from the start of the input y vector.
 * - `const size_t y_inc`: Stride/increment of the input y vector. This value must be greater than 0.
 * - `cl_mem a_buffer`: OpenCL buffer to store the output A matrix.
 * - `const size_t a_offset`: The offset in elements from the start of the output A matrix.
 * - `const size_t a_ld`: Leading dimension of the output A matrix. This value must be greater than 0.
 * - `cl_command_queue* queue`: Pointer to an OpenCL command queue associated with a context and device to execute the routine on.
 * - `cl_event* event`: Pointer to an OpenCL event to be able to wait for completion of the routine's OpenCL kernel(s). This is an optional argument.
 *
 * Requirements for GERU:
 * - The value of `a_ld` must be at least `m`.
 */
extern(System) @nogc nothrow {
  alias da_CLBlastCgeru = CLBlastStatusCode function(const CLBlastLayout layout,
      const size_t m, const size_t n,
      const cl_float2 alpha,
      const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
      const cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
      cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
      cl_command_queue* queue, cl_event* event);
  alias da_CLBlastZgeru = CLBlastStatusCode function(const CLBlastLayout layout,
      const size_t m, const size_t n,
      const cl_double2 alpha,
      const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
      const cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
      cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
      cl_command_queue* queue, cl_event* event);
}

/**
 * xGERC: General rank-1 complex conjugated matrix update
 *
 * Same operation as xGERU, but the update is done based on the complex conjugate of the input vectors.
 *
 * Arguments to GERC:
 * - `const Layout layout`: Data-layout of the matrices, either `Layout::kRowMajor` (101) for row-major layout or `Layout::kColMajor` (102) for column-major data-layout.
 * - `const size_t m`: Integer size argument. This value must be positive.
 * - `const size_t n`: Integer size argument. This value must be positive.
 * - `const T alpha`: Input scalar constant.
 * - `const cl_mem x_buffer`: OpenCL buffer to store the input x vector.
 * - `const size_t x_offset`: The offset in elements from the start of the input x vector.
 * - `const size_t x_inc`: Stride/increment of the input x vector. This value must be greater than 0.
 * - `const cl_mem y_buffer`: OpenCL buffer to store the input y vector.
 * - `const size_t y_offset`: The offset in elements from the start of the input y vector.
 * - `const size_t y_inc`: Stride/increment of the input y vector. This value must be greater than 0.
 * - `cl_mem a_buffer`: OpenCL buffer to store the output A matrix.
 * - `const size_t a_offset`: The offset in elements from the start of the output A matrix.
 * - `const size_t a_ld`: Leading dimension of the output A matrix. This value must be greater than 0.
 * - `cl_command_queue* queue`: Pointer to an OpenCL command queue associated with a context and device to execute the routine on.
 * - `cl_event* event`: Pointer to an OpenCL event to be able to wait for completion of the routine's OpenCL kernel(s). This is an optional argument.
 *
 * Requirements for GERC:
 * - The value of `a_ld` must be at least `m`.
 */
extern(System) @nogc nothrow {
  alias da_CLBlastCgerc = CLBlastStatusCode function(const CLBlastLayout layout,
      const size_t m, const size_t n,
      const cl_float2 alpha,
      const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
      const cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
      cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
      cl_command_queue* queue, cl_event* event);
  alias da_CLBlastZgerc = CLBlastStatusCode function(const CLBlastLayout layout,
      const size_t m, const size_t n,
      const cl_double2 alpha,
      const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
      const cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
      cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
      cl_command_queue* queue, cl_event* event);
}

/**
 * xHER: Hermitian rank-1 matrix update
 *
 * Performs the operation _A = alpha * x * x^T + A_, in which x is an input vector, x^T is the transpose of this vector, _A_ is the triangular Hermetian matrix to be updated, and alpha is a scalar value.
 *
 * Arguments to HER:
 * - `const Layout layout`: Data-layout of the matrices, either `Layout::kRowMajor` (101) for row-major layout or `Layout::kColMajor` (102) for column-major data-layout.
 * - `const Triangle triangle`: The part of the array of the triangular matrix to be used, either `Triangle::kUpper` (121) or `Triangle::kLower` (122).
 * - `const size_t n`: Integer size argument. This value must be positive.
 * - `const T alpha`: Input scalar constant.
 * - `const cl_mem x_buffer`: OpenCL buffer to store the input x vector.
 * - `const size_t x_offset`: The offset in elements from the start of the input x vector.
 * - `const size_t x_inc`: Stride/increment of the input x vector. This value must be greater than 0.
 * - `cl_mem a_buffer`: OpenCL buffer to store the output A matrix.
 * - `const size_t a_offset`: The offset in elements from the start of the output A matrix.
 * - `const size_t a_ld`: Leading dimension of the output A matrix. This value must be greater than 0.
 * - `cl_command_queue* queue`: Pointer to an OpenCL command queue associated with a context and device to execute the routine on.
 * - `cl_event* event`: Pointer to an OpenCL event to be able to wait for completion of the routine's OpenCL kernel(s). This is an optional argument.
 *
 * Requirements for HER:
 * - The value of `a_ld` must be at least `n`.
 */
extern(System) @nogc nothrow {
  alias da_CLBlastCher = CLBlastStatusCode function(const CLBlastLayout layout, const CLBlastTriangle triangle,
      const size_t n,
      const float alpha,
      const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
      cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
      cl_command_queue* queue, cl_event* event);
  alias da_CLBlastZher = CLBlastStatusCode function(const CLBlastLayout layout, const CLBlastTriangle triangle,
      const size_t n,
      const double alpha,
      const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
      cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
      cl_command_queue* queue, cl_event* event);
}

/**
 * xHPR: Hermitian packed rank-1 matrix update
 *
 * Same operation as xHER, but matrix _A_ is an Hermitian packed matrix instead and represented as _AP_.
 *
 * Arguments to HPR:
 * - `const Layout layout`: Data-layout of the matrices, either `Layout::kRowMajor` (101) for row-major layout or `Layout::kColMajor` (102) for column-major data-layout.
 * - `const Triangle triangle`: The part of the array of the triangular matrix to be used, either `Triangle::kUpper` (121) or `Triangle::kLower` (122).
 * - `const size_t n`: Integer size argument. This value must be positive.
 * - `const T alpha`: Input scalar constant.
 * - `const cl_mem x_buffer`: OpenCL buffer to store the input x vector.
 * - `const size_t x_offset`: The offset in elements from the start of the input x vector.
 * - `const size_t x_inc`: Stride/increment of the input x vector. This value must be greater than 0.
 * - `cl_mem ap_buffer`: OpenCL buffer to store the output AP matrix.
 * - `const size_t ap_offset`: The offset in elements from the start of the output AP matrix.
 * - `cl_command_queue* queue`: Pointer to an OpenCL command queue associated with a context and device to execute the routine on.
 * - `cl_event* event`: Pointer to an OpenCL event to be able to wait for completion of the routine's OpenCL kernel(s). This is an optional argument.
 */
extern(System) @nogc nothrow {
  alias da_CLBlastChpr = CLBlastStatusCode function(const CLBlastLayout layout, const CLBlastTriangle triangle,
      const size_t n,
      const float alpha,
      const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
      cl_mem ap_buffer, const size_t ap_offset,
      cl_command_queue* queue, cl_event* event);
  alias da_CLBlastZhpr = CLBlastStatusCode function(const CLBlastLayout layout, const CLBlastTriangle triangle,
      const size_t n,
      const double alpha,
      const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
      cl_mem ap_buffer, const size_t ap_offset,
      cl_command_queue* queue, cl_event* event);
}

/**
 * xHER2: Hermitian rank-2 matrix update
 *
 * Performs the operation _A = alpha * x * y^T + conj(alpha) * y * x^T + A_, in which _x_ is an input vector and _x^T_ its transpose, _y_ is an input vector and _y^T_ its transpose, _A_ is the triangular Hermetian matrix to be updated, _alpha_ is a scalar value and _conj(alpha)_ its complex conjugate.
 *
 * Arguments to HER2:
 * - `const Layout layout`: Data-layout of the matrices, either `Layout::kRowMajor` (101) for row-major layout or `Layout::kColMajor` (102) for column-major data-layout.
 * - `const Triangle triangle`: The part of the array of the triangular matrix to be used, either `Triangle::kUpper` (121) or `Triangle::kLower` (122).
 * - `const size_t n`: Integer size argument. This value must be positive.
 * - `const T alpha`: Input scalar constant.
 * - `const cl_mem x_buffer`: OpenCL buffer to store the input x vector.
 * - `const size_t x_offset`: The offset in elements from the start of the input x vector.
 * - `const size_t x_inc`: Stride/increment of the input x vector. This value must be greater than 0.
 * - `const cl_mem y_buffer`: OpenCL buffer to store the input y vector.
 * - `const size_t y_offset`: The offset in elements from the start of the input y vector.
 * - `const size_t y_inc`: Stride/increment of the input y vector. This value must be greater than 0.
 * - `cl_mem a_buffer`: OpenCL buffer to store the output A matrix.
 * - `const size_t a_offset`: The offset in elements from the start of the output A matrix.
 * - `const size_t a_ld`: Leading dimension of the output A matrix. This value must be greater than 0.
 * - `cl_command_queue* queue`: Pointer to an OpenCL command queue associated with a context and device to execute the routine on.
 * - `cl_event* event`: Pointer to an OpenCL event to be able to wait for completion of the routine's OpenCL kernel(s). This is an optional argument.
 *
 * Requirements for HER2:
 * - The value of `a_ld` must be at least `n`.
 */
extern(System) @nogc nothrow {
  alias da_CLBlastCher2 = CLBlastStatusCode function(const CLBlastLayout layout, const CLBlastTriangle triangle,
      const size_t n,
      const cl_float2 alpha,
      const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
      const cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
      cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
      cl_command_queue* queue, cl_event* event);
  alias da_CLBlastZher2 = CLBlastStatusCode function(const CLBlastLayout layout, const CLBlastTriangle triangle,
      const size_t n,
      const cl_double2 alpha,
      const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
      const cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
      cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
      cl_command_queue* queue, cl_event* event);
}

/**
 * xHPR2: Hermitian packed rank-2 matrix update
 *
 * Same operation as xHER2, but matrix _A_ is an Hermitian packed matrix instead and represented as _AP_.
 *
 * Arguments to HPR2:
 * - `const Layout layout`: Data-layout of the matrices, either `Layout::kRowMajor` (101) for row-major layout or `Layout::kColMajor` (102) for column-major data-layout.
 * - `const Triangle triangle`: The part of the array of the triangular matrix to be used, either `Triangle::kUpper` (121) or `Triangle::kLower` (122).
 * - `const size_t n`: Integer size argument. This value must be positive.
 * - `const T alpha`: Input scalar constant.
 * - `const cl_mem x_buffer`: OpenCL buffer to store the input x vector.
 * - `const size_t x_offset`: The offset in elements from the start of the input x vector.
 * - `const size_t x_inc`: Stride/increment of the input x vector. This value must be greater than 0.
 * - `const cl_mem y_buffer`: OpenCL buffer to store the input y vector.
 * - `const size_t y_offset`: The offset in elements from the start of the input y vector.
 * - `const size_t y_inc`: Stride/increment of the input y vector. This value must be greater than 0.
 * - `cl_mem ap_buffer`: OpenCL buffer to store the output AP matrix.
 * - `const size_t ap_offset`: The offset in elements from the start of the output AP matrix.
 * - `cl_command_queue* queue`: Pointer to an OpenCL command queue associated with a context and device to execute the routine on.
 * - `cl_event* event`: Pointer to an OpenCL event to be able to wait for completion of the routine's OpenCL kernel(s). This is an optional argument.
 */
extern(System) @nogc nothrow {
  alias da_CLBlastChpr2 = CLBlastStatusCode function(const CLBlastLayout layout, const CLBlastTriangle triangle,
      const size_t n,
      const cl_float2 alpha,
      const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
      const cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
      cl_mem ap_buffer, const size_t ap_offset,
      cl_command_queue* queue, cl_event* event);
  alias da_CLBlastZhpr2 = CLBlastStatusCode function(const CLBlastLayout layout, const CLBlastTriangle triangle,
      const size_t n,
      const cl_double2 alpha,
      const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
      const cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
      cl_mem ap_buffer, const size_t ap_offset,
      cl_command_queue* queue, cl_event* event);
}

/**
 * xSYR: Symmetric rank-1 matrix update
 *
 * Same operation as xHER, but matrix A is a symmetric matrix instead.
 *
 * Arguments to SYR:
 * - `const Layout layout`: Data-layout of the matrices, either `Layout::kRowMajor` (101) for row-major layout or `Layout::kColMajor` (102) for column-major data-layout.
 * - `const Triangle triangle`: The part of the array of the triangular matrix to be used, either `Triangle::kUpper` (121) or `Triangle::kLower` (122).
 * - `const size_t n`: Integer size argument. This value must be positive.
 * - `const T alpha`: Input scalar constant.
 * - `const cl_mem x_buffer`: OpenCL buffer to store the input x vector.
 * - `const size_t x_offset`: The offset in elements from the start of the input x vector.
 * - `const size_t x_inc`: Stride/increment of the input x vector. This value must be greater than 0.
 * - `cl_mem a_buffer`: OpenCL buffer to store the output A matrix.
 * - `const size_t a_offset`: The offset in elements from the start of the output A matrix.
 * - `const size_t a_ld`: Leading dimension of the output A matrix. This value must be greater than 0.
 * - `cl_command_queue* queue`: Pointer to an OpenCL command queue associated with a context and device to execute the routine on.
 * - `cl_event* event`: Pointer to an OpenCL event to be able to wait for completion of the routine's OpenCL kernel(s). This is an optional argument.
 *
 * Requirements for SYR:
 * - The value of `a_ld` must be at least `n`.
 */
extern(System) @nogc nothrow {
  alias da_CLBlastSsyr = CLBlastStatusCode function(const CLBlastLayout layout, const CLBlastTriangle triangle,
      const size_t n,
      const float alpha,
      const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
      cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
      cl_command_queue* queue, cl_event* event);
  alias da_CLBlastDsyr = CLBlastStatusCode function(const CLBlastLayout layout, const CLBlastTriangle triangle,
      const size_t n,
      const double alpha,
      const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
      cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
      cl_command_queue* queue, cl_event* event);
  alias da_CLBlastHsyr = CLBlastStatusCode function(const CLBlastLayout layout, const CLBlastTriangle triangle,
      const size_t n,
      const cl_half alpha,
      const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
      cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
      cl_command_queue* queue, cl_event* event);
}

/**
 * xSPR: Symmetric packed rank-1 matrix update
 *
 * Same operation as xSPR, but matrix _A_ is a symmetric packed matrix instead and represented as _AP_.
 *
 * Arguments to SPR:
 * - `const Layout layout`: Data-layout of the matrices, either `Layout::kRowMajor` (101) for row-major layout or `Layout::kColMajor` (102) for column-major data-layout.
 * - `const Triangle triangle`: The part of the array of the triangular matrix to be used, either `Triangle::kUpper` (121) or `Triangle::kLower` (122).
 * - `const size_t n`: Integer size argument. This value must be positive.
 * - `const T alpha`: Input scalar constant.
 * - `const cl_mem x_buffer`: OpenCL buffer to store the input x vector.
 * - `const size_t x_offset`: The offset in elements from the start of the input x vector.
 * - `const size_t x_inc`: Stride/increment of the input x vector. This value must be greater than 0.
 * - `cl_mem ap_buffer`: OpenCL buffer to store the output AP matrix.
 * - `const size_t ap_offset`: The offset in elements from the start of the output AP matrix.
 * - `cl_command_queue* queue`: Pointer to an OpenCL command queue associated with a context and device to execute the routine on.
 * - `cl_event* event`: Pointer to an OpenCL event to be able to wait for completion of the routine's OpenCL kernel(s). This is an optional argument.
 */
extern(System) @nogc nothrow {
  alias da_CLBlastSspr = CLBlastStatusCode function(const CLBlastLayout layout, const CLBlastTriangle triangle,
      const size_t n,
      const float alpha,
      const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
      cl_mem ap_buffer, const size_t ap_offset,
      cl_command_queue* queue, cl_event* event);
  alias da_CLBlastDspr = CLBlastStatusCode function(const CLBlastLayout layout, const CLBlastTriangle triangle,
      const size_t n,
      const double alpha,
      const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
      cl_mem ap_buffer, const size_t ap_offset,
      cl_command_queue* queue, cl_event* event);
  alias da_CLBlastHspr = CLBlastStatusCode function(const CLBlastLayout layout, const CLBlastTriangle triangle,
      const size_t n,
      const cl_half alpha,
      const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
      cl_mem ap_buffer, const size_t ap_offset,
      cl_command_queue* queue, cl_event* event);
}

/**
 * xSYR2: Symmetric rank-2 matrix update
 *
 * Same operation as xHER2, but matrix _A_ is a symmetric matrix instead.
 *
 * Arguments to SYR2:
 * - `const Layout layout`: Data-layout of the matrices, either `Layout::kRowMajor` (101) for row-major layout or `Layout::kColMajor` (102) for column-major data-layout.
 * - `const Triangle triangle`: The part of the array of the triangular matrix to be used, either `Triangle::kUpper` (121) or `Triangle::kLower` (122).
 * - `const size_t n`: Integer size argument. This value must be positive.
 * - `const T alpha`: Input scalar constant.
 * - `const cl_mem x_buffer`: OpenCL buffer to store the input x vector.
 * - `const size_t x_offset`: The offset in elements from the start of the input x vector.
 * - `const size_t x_inc`: Stride/increment of the input x vector. This value must be greater than 0.
 * - `const cl_mem y_buffer`: OpenCL buffer to store the input y vector.
 * - `const size_t y_offset`: The offset in elements from the start of the input y vector.
 * - `const size_t y_inc`: Stride/increment of the input y vector. This value must be greater than 0.
 * - `cl_mem a_buffer`: OpenCL buffer to store the output A matrix.
 * - `const size_t a_offset`: The offset in elements from the start of the output A matrix.
 * - `const size_t a_ld`: Leading dimension of the output A matrix. This value must be greater than 0.
 * - `cl_command_queue* queue`: Pointer to an OpenCL command queue associated with a context and device to execute the routine on.
 * - `cl_event* event`: Pointer to an OpenCL event to be able to wait for completion of the routine's OpenCL kernel(s). This is an optional argument.
 *
 * Requirements for SYR2:
 * - The value of `a_ld` must be at least `n`.
 */
extern(System) @nogc nothrow {
  alias da_CLBlastSsyr2 = CLBlastStatusCode function(const CLBlastLayout layout, const CLBlastTriangle triangle,
      const size_t n,
      const float alpha,
      const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
      const cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
      cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
      cl_command_queue* queue, cl_event* event);
  alias da_CLBlastDsyr2 = CLBlastStatusCode function(const CLBlastLayout layout, const CLBlastTriangle triangle,
      const size_t n,
      const double alpha,
      const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
      const cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
      cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
      cl_command_queue* queue, cl_event* event);
  alias da_CLBlastHsyr2 = CLBlastStatusCode function(const CLBlastLayout layout, const CLBlastTriangle triangle,
      const size_t n,
      const cl_half alpha,
      const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
      const cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
      cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
      cl_command_queue* queue, cl_event* event);
}

/**
 * xSPR2: Symmetric packed rank-2 matrix update
 *
 * Same operation as xSPR2, but matrix _A_ is a symmetric packed matrix instead and represented as _AP_.
 *
 * Arguments to SPR2:
 * - `const Layout layout`: Data-layout of the matrices, either `Layout::kRowMajor` (101) for row-major layout or `Layout::kColMajor` (102) for column-major data-layout.
 * - `const Triangle triangle`: The part of the array of the triangular matrix to be used, either `Triangle::kUpper` (121) or `Triangle::kLower` (122).
 * - `const size_t n`: Integer size argument. This value must be positive.
 * - `const T alpha`: Input scalar constant.
 * - `const cl_mem x_buffer`: OpenCL buffer to store the input x vector.
 * - `const size_t x_offset`: The offset in elements from the start of the input x vector.
 * - `const size_t x_inc`: Stride/increment of the input x vector. This value must be greater than 0.
 * - `const cl_mem y_buffer`: OpenCL buffer to store the input y vector.
 * - `const size_t y_offset`: The offset in elements from the start of the input y vector.
 * - `const size_t y_inc`: Stride/increment of the input y vector. This value must be greater than 0.
 * - `cl_mem ap_buffer`: OpenCL buffer to store the output AP matrix.
 * - `const size_t ap_offset`: The offset in elements from the start of the output AP matrix.
 * - `cl_command_queue* queue`: Pointer to an OpenCL command queue associated with a context and device to execute the routine on.
 * - `cl_event* event`: Pointer to an OpenCL event to be able to wait for completion of the routine's OpenCL kernel(s). This is an optional argument.
 */
extern(System) @nogc nothrow {
  alias da_CLBlastSspr2 = CLBlastStatusCode function(const CLBlastLayout layout, const CLBlastTriangle triangle,
      const size_t n,
      const float alpha,
      const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
      const cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
      cl_mem ap_buffer, const size_t ap_offset,
      cl_command_queue* queue, cl_event* event);
  alias da_CLBlastDspr2 = CLBlastStatusCode function(const CLBlastLayout layout, const CLBlastTriangle triangle,
      const size_t n,
      const double alpha,
      const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
      const cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
      cl_mem ap_buffer, const size_t ap_offset,
      cl_command_queue* queue, cl_event* event);
  alias da_CLBlastHspr2 = CLBlastStatusCode function(const CLBlastLayout layout, const CLBlastTriangle triangle,
      const size_t n,
      const cl_half alpha,
      const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
      const cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
      cl_mem ap_buffer, const size_t ap_offset,
      cl_command_queue* queue, cl_event* event);
}

/**
 * xGEMM: General matrix-matrix multiplication
 *
 * Performs the matrix product _C = alpha * A * B + beta * C_, in which _A_ (_m_ by _k_) and _B_ (_k_ by _n_) are two general rectangular input matrices, _C_ (_m_ by _n_) is the matrix to be updated, and _alpha_ and _beta_ are scalar values. The matrices _A_ and/or _B_ can optionally be transposed before performing the operation.
 *
 * Arguments to GEMM:
 * - `const Layout layout`: Data-layout of the matrices, either `Layout::kRowMajor` (101) for row-major layout or `Layout::kColMajor` (102) for column-major data-layout.
 * - `const Transpose a_transpose`: Transposing the input matrix A, either `Transpose::kNo` (111), `Transpose::kYes` (112), or `Transpose::kConjugate` (113) for a complex-conjugate transpose.
 * - `const Transpose b_transpose`: Transposing the input matrix B, either `Transpose::kNo` (111), `Transpose::kYes` (112), or `Transpose::kConjugate` (113) for a complex-conjugate transpose.
 * - `const size_t m`: Integer size argument. This value must be positive.
 * - `const size_t n`: Integer size argument. This value must be positive.
 * - `const size_t k`: Integer size argument. This value must be positive.
 * - `const T alpha`: Input scalar constant.
 * - `const cl_mem a_buffer`: OpenCL buffer to store the input A matrix.
 * - `const size_t a_offset`: The offset in elements from the start of the input A matrix.
 * - `const size_t a_ld`: Leading dimension of the input A matrix. This value must be greater than 0.
 * - `const cl_mem b_buffer`: OpenCL buffer to store the input B matrix.
 * - `const size_t b_offset`: The offset in elements from the start of the input B matrix.
 * - `const size_t b_ld`: Leading dimension of the input B matrix. This value must be greater than 0.
 * - `const T beta`: Input scalar constant.
 * - `cl_mem c_buffer`: OpenCL buffer to store the output C matrix.
 * - `const size_t c_offset`: The offset in elements from the start of the output C matrix.
 * - `const size_t c_ld`: Leading dimension of the output C matrix. This value must be greater than 0.
 * - `cl_command_queue* queue`: Pointer to an OpenCL command queue associated with a context and device to execute the routine on.
 * - `cl_event* event`: Pointer to an OpenCL event to be able to wait for completion of the routine's OpenCL kernel(s). This is an optional argument.
 *
 * Requirements for GEMM:
 * - When `(transpose_a == Transpose::kNo && layout == Layout::kColMajor) || (transpose_a == Transpose::kYes && layout == Layout::kRowMajor)`, then `a_ld` must be at least `m`, otherwise `a_ld` must be at least `k`.
 * - When `(transpose_b == Transpose::kNo && layout == Layout::kColMajor) || (transpose_b == Transpose::kYes && layout == Layout::kRowMajor)`, then `b_ld` must be at least `k`, otherwise `b_ld` must be at least `n`.
 * - The value of `c_ld` must be at least `m`.
 */
extern(System) @nogc nothrow {
  alias da_CLBlastSgemm = CLBlastStatusCode function(const CLBlastLayout layout, const CLBlastTranspose a_transpose, const CLBlastTranspose b_transpose,
      const size_t m, const size_t n, const size_t k,
      const float alpha,
      const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
      const cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
      const float beta,
      cl_mem c_buffer, const size_t c_offset, const size_t c_ld,
      cl_command_queue* queue, cl_event* event);
  alias da_CLBlastDgemm = CLBlastStatusCode function(const CLBlastLayout layout, const CLBlastTranspose a_transpose, const CLBlastTranspose b_transpose,
      const size_t m, const size_t n, const size_t k,
      const double alpha,
      const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
      const cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
      const double beta,
      cl_mem c_buffer, const size_t c_offset, const size_t c_ld,
      cl_command_queue* queue, cl_event* event);
  alias da_CLBlastCgemm = CLBlastStatusCode function(const CLBlastLayout layout, const CLBlastTranspose a_transpose, const CLBlastTranspose b_transpose,
      const size_t m, const size_t n, const size_t k,
      const cl_float2 alpha,
      const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
      const cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
      const cl_float2 beta,
      cl_mem c_buffer, const size_t c_offset, const size_t c_ld,
      cl_command_queue* queue, cl_event* event);
  alias da_CLBlastZgemm = CLBlastStatusCode function(const CLBlastLayout layout, const CLBlastTranspose a_transpose, const CLBlastTranspose b_transpose,
      const size_t m, const size_t n, const size_t k,
      const cl_double2 alpha,
      const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
      const cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
      const cl_double2 beta,
      cl_mem c_buffer, const size_t c_offset, const size_t c_ld,
      cl_command_queue* queue, cl_event* event);
  alias da_CLBlastHgemm = CLBlastStatusCode function(const CLBlastLayout layout, const CLBlastTranspose a_transpose, const CLBlastTranspose b_transpose,
      const size_t m, const size_t n, const size_t k,
      const cl_half alpha,
      const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
      const cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
      const cl_half beta,
      cl_mem c_buffer, const size_t c_offset, const size_t c_ld,
      cl_command_queue* queue, cl_event* event);
}

/**
 * xSYMM: Symmetric matrix-matrix multiplication
 *
 * Same operation as xGEMM, but _A_ is symmetric instead. In case of `side == kLeft`, _A_ is a symmetric _m_ by _m_ matrix and _C = alpha * A * B + beta * C_ is performed. Otherwise, in case of `side == kRight`, _A_ is a symmtric _n_ by _n_ matrix and _C = alpha * B * A + beta * C_ is performed.
 *
 * Arguments to SYMM:
 * - `const Layout layout`: Data-layout of the matrices, either `Layout::kRowMajor` (101) for row-major layout or `Layout::kColMajor` (102) for column-major data-layout.
 * - `const Side side`: The position of the triangular matrix in the operation, either on the `Side::kLeft` (141) or `Side::kRight` (142).
 * - `const Triangle triangle`: The part of the array of the triangular matrix to be used, either `Triangle::kUpper` (121) or `Triangle::kLower` (122).
 * - `const size_t m`: Integer size argument. This value must be positive.
 * - `const size_t n`: Integer size argument. This value must be positive.
 * - `const T alpha`: Input scalar constant.
 * - `const cl_mem a_buffer`: OpenCL buffer to store the input A matrix.
 * - `const size_t a_offset`: The offset in elements from the start of the input A matrix.
 * - `const size_t a_ld`: Leading dimension of the input A matrix. This value must be greater than 0.
 * - `const cl_mem b_buffer`: OpenCL buffer to store the input B matrix.
 * - `const size_t b_offset`: The offset in elements from the start of the input B matrix.
 * - `const size_t b_ld`: Leading dimension of the input B matrix. This value must be greater than 0.
 * - `const T beta`: Input scalar constant.
 * - `cl_mem c_buffer`: OpenCL buffer to store the output C matrix.
 * - `const size_t c_offset`: The offset in elements from the start of the output C matrix.
 * - `const size_t c_ld`: Leading dimension of the output C matrix. This value must be greater than 0.
 * - `cl_command_queue* queue`: Pointer to an OpenCL command queue associated with a context and device to execute the routine on.
 * - `cl_event* event`: Pointer to an OpenCL event to be able to wait for completion of the routine's OpenCL kernel(s). This is an optional argument.
 *
 * Requirements for SYMM:
 * - When `side = Side::kLeft` then `a_ld` must be at least `m`, otherwise `a_ld` must be at least `n`.
 * - The value of `b_ld` must be at least `m`.
 * - The value of `c_ld` must be at least `m`.
 */
extern(System) @nogc nothrow {
  alias da_CLBlastSsymm = CLBlastStatusCode function(const CLBlastLayout layout, const CLBlastSide side, const CLBlastTriangle triangle,
      const size_t m, const size_t n,
      const float alpha,
      const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
      const cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
      const float beta,
      cl_mem c_buffer, const size_t c_offset, const size_t c_ld,
      cl_command_queue* queue, cl_event* event);
  alias da_CLBlastDsymm = CLBlastStatusCode function(const CLBlastLayout layout, const CLBlastSide side, const CLBlastTriangle triangle,
      const size_t m, const size_t n,
      const double alpha,
      const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
      const cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
      const double beta,
      cl_mem c_buffer, const size_t c_offset, const size_t c_ld,
      cl_command_queue* queue, cl_event* event);
  alias da_CLBlastCsymm = CLBlastStatusCode function(const CLBlastLayout layout, const CLBlastSide side, const CLBlastTriangle triangle,
      const size_t m, const size_t n,
      const cl_float2 alpha,
      const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
      const cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
      const cl_float2 beta,
      cl_mem c_buffer, const size_t c_offset, const size_t c_ld,
      cl_command_queue* queue, cl_event* event);
  alias da_CLBlastZsymm = CLBlastStatusCode function(const CLBlastLayout layout, const CLBlastSide side, const CLBlastTriangle triangle,
      const size_t m, const size_t n,
      const cl_double2 alpha,
      const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
      const cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
      const cl_double2 beta,
      cl_mem c_buffer, const size_t c_offset, const size_t c_ld,
      cl_command_queue* queue, cl_event* event);
  alias da_CLBlastHsymm = CLBlastStatusCode function(const CLBlastLayout layout, const CLBlastSide side, const CLBlastTriangle triangle,
      const size_t m, const size_t n,
      const cl_half alpha,
      const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
      const cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
      const cl_half beta,
      cl_mem c_buffer, const size_t c_offset, const size_t c_ld,
      cl_command_queue* queue, cl_event* event);
}

/**
 * xHEMM: Hermitian matrix-matrix multiplication
 *
 * Same operation as xSYMM, but _A_ is an Hermitian matrix instead.
 *
 * Arguments to HEMM:
 * - `const Layout layout`: Data-layout of the matrices, either `Layout::kRowMajor` (101) for row-major layout or `Layout::kColMajor` (102) for column-major data-layout.
 * - `const Side side`: The position of the triangular matrix in the operation, either on the `Side::kLeft` (141) or `Side::kRight` (142).
 * - `const Triangle triangle`: The part of the array of the triangular matrix to be used, either `Triangle::kUpper` (121) or `Triangle::kLower` (122).
 * - `const size_t m`: Integer size argument. This value must be positive.
 * - `const size_t n`: Integer size argument. This value must be positive.
 * - `const T alpha`: Input scalar constant.
 * - `const cl_mem a_buffer`: OpenCL buffer to store the input A matrix.
 * - `const size_t a_offset`: The offset in elements from the start of the input A matrix.
 * - `const size_t a_ld`: Leading dimension of the input A matrix. This value must be greater than 0.
 * - `const cl_mem b_buffer`: OpenCL buffer to store the input B matrix.
 * - `const size_t b_offset`: The offset in elements from the start of the input B matrix.
 * - `const size_t b_ld`: Leading dimension of the input B matrix. This value must be greater than 0.
 * - `const T beta`: Input scalar constant.
 * - `cl_mem c_buffer`: OpenCL buffer to store the output C matrix.
 * - `const size_t c_offset`: The offset in elements from the start of the output C matrix.
 * - `const size_t c_ld`: Leading dimension of the output C matrix. This value must be greater than 0.
 * - `cl_command_queue* queue`: Pointer to an OpenCL command queue associated with a context and device to execute the routine on.
 * - `cl_event* event`: Pointer to an OpenCL event to be able to wait for completion of the routine's OpenCL kernel(s). This is an optional argument.
 *
 * Requirements for HEMM:
 * - When `side = Side::kLeft` then `a_ld` must be at least `m`, otherwise `a_ld` must be at least `n`.
 * - The value of `b_ld` must be at least `m`.
 * - The value of `c_ld` must be at least `m`.
 */
extern(System) @nogc nothrow {
  alias da_CLBlastChemm = CLBlastStatusCode function(const CLBlastLayout layout, const CLBlastSide side, const CLBlastTriangle triangle,
      const size_t m, const size_t n,
      const cl_float2 alpha,
      const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
      const cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
      const cl_float2 beta,
      cl_mem c_buffer, const size_t c_offset, const size_t c_ld,
      cl_command_queue* queue, cl_event* event);
  alias da_CLBlastZhemm = CLBlastStatusCode function(const CLBlastLayout layout, const CLBlastSide side, const CLBlastTriangle triangle,
      const size_t m, const size_t n,
      const cl_double2 alpha,
      const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
      const cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
      const cl_double2 beta,
      cl_mem c_buffer, const size_t c_offset, const size_t c_ld,
      cl_command_queue* queue, cl_event* event);
}

/**
 * xSYRK: Rank-K update of a symmetric matrix
 *
 * Performs the matrix product _C = alpha * A * A^T + beta * C_ or _C = alpha * A^T * A + beta * C_, in which _A_ is a general matrix and _A^T_ is its transpose, _C_ (_n_ by _n_) is the symmetric matrix to be updated, and _alpha_ and _beta_ are scalar values.
 *
 * Arguments to SYRK:
 * - `const Layout layout`: Data-layout of the matrices, either `Layout::kRowMajor` (101) for row-major layout or `Layout::kColMajor` (102) for column-major data-layout.
 * - `const Triangle triangle`: The part of the array of the triangular matrix to be used, either `Triangle::kUpper` (121) or `Triangle::kLower` (122).
 * - `const Transpose a_transpose`: Transposing the input matrix A, either `Transpose::kNo` (111), `Transpose::kYes` (112), or `Transpose::kConjugate` (113) for a complex-conjugate transpose.
 * - `const size_t n`: Integer size argument. This value must be positive.
 * - `const size_t k`: Integer size argument. This value must be positive.
 * - `const T alpha`: Input scalar constant.
 * - `const cl_mem a_buffer`: OpenCL buffer to store the input A matrix.
 * - `const size_t a_offset`: The offset in elements from the start of the input A matrix.
 * - `const size_t a_ld`: Leading dimension of the input A matrix. This value must be greater than 0.
 * - `const T beta`: Input scalar constant.
 * - `cl_mem c_buffer`: OpenCL buffer to store the output C matrix.
 * - `const size_t c_offset`: The offset in elements from the start of the output C matrix.
 * - `const size_t c_ld`: Leading dimension of the output C matrix. This value must be greater than 0.
 * - `cl_command_queue* queue`: Pointer to an OpenCL command queue associated with a context and device to execute the routine on.
 * - `cl_event* event`: Pointer to an OpenCL event to be able to wait for completion of the routine's OpenCL kernel(s). This is an optional argument.
 *
 * Requirements for SYRK:
 * - When `(transpose == Transpose::kNo && layout == Layout::kColMajor) || (transpose == Transpose::kYes && layout == Layout::kRowMajor)`, then `a_ld` must be at least `n`, otherwise `a_ld` must be at least `k`.
 * - The value of `c_ld` must be at least `m`.
 */
extern(System) @nogc nothrow {
  alias da_CLBlastSsyrk = CLBlastStatusCode function(const CLBlastLayout layout, const CLBlastTriangle triangle, const CLBlastTranspose a_transpose,
      const size_t n, const size_t k,
      const float alpha,
      const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
      const float beta,
      cl_mem c_buffer, const size_t c_offset, const size_t c_ld,
      cl_command_queue* queue, cl_event* event);
  alias da_CLBlastDsyrk = CLBlastStatusCode function(const CLBlastLayout layout, const CLBlastTriangle triangle, const CLBlastTranspose a_transpose,
      const size_t n, const size_t k,
      const double alpha,
      const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
      const double beta,
      cl_mem c_buffer, const size_t c_offset, const size_t c_ld,
      cl_command_queue* queue, cl_event* event);
  alias da_CLBlastCsyrk = CLBlastStatusCode function(const CLBlastLayout layout, const CLBlastTriangle triangle, const CLBlastTranspose a_transpose,
      const size_t n, const size_t k,
      const cl_float2 alpha,
      const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
      const cl_float2 beta,
      cl_mem c_buffer, const size_t c_offset, const size_t c_ld,
      cl_command_queue* queue, cl_event* event);
  alias da_CLBlastZsyrk = CLBlastStatusCode function(const CLBlastLayout layout, const CLBlastTriangle triangle, const CLBlastTranspose a_transpose,
      const size_t n, const size_t k,
      const cl_double2 alpha,
      const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
      const cl_double2 beta,
      cl_mem c_buffer, const size_t c_offset, const size_t c_ld,
      cl_command_queue* queue, cl_event* event);
  alias da_CLBlastHsyrk = CLBlastStatusCode function(const CLBlastLayout layout, const CLBlastTriangle triangle, const CLBlastTranspose a_transpose,
      const size_t n, const size_t k,
      const cl_half alpha,
      const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
      const cl_half beta,
      cl_mem c_buffer, const size_t c_offset, const size_t c_ld,
      cl_command_queue* queue, cl_event* event);
}

/**
 * xHERK: Rank-K update of a hermitian matrix
 *
 * Same operation as xSYRK, but _C_ is an Hermitian matrix instead.
 *
 * Arguments to HERK:
 * - `const Layout layout`: Data-layout of the matrices, either `Layout::kRowMajor` (101) for row-major layout or `Layout::kColMajor` (102) for column-major data-layout.
 * - `const Triangle triangle`: The part of the array of the triangular matrix to be used, either `Triangle::kUpper` (121) or `Triangle::kLower` (122).
 * - `const Transpose a_transpose`: Transposing the input matrix A, either `Transpose::kNo` (111), `Transpose::kYes` (112), or `Transpose::kConjugate` (113) for a complex-conjugate transpose.
 * - `const size_t n`: Integer size argument. This value must be positive.
 * - `const size_t k`: Integer size argument. This value must be positive.
 * - `const T alpha`: Input scalar constant.
 * - `const cl_mem a_buffer`: OpenCL buffer to store the input A matrix.
 * - `const size_t a_offset`: The offset in elements from the start of the input A matrix.
 * - `const size_t a_ld`: Leading dimension of the input A matrix. This value must be greater than 0.
 * - `const T beta`: Input scalar constant.
 * - `cl_mem c_buffer`: OpenCL buffer to store the output C matrix.
 * - `const size_t c_offset`: The offset in elements from the start of the output C matrix.
 * - `const size_t c_ld`: Leading dimension of the output C matrix. This value must be greater than 0.
 * - `cl_command_queue* queue`: Pointer to an OpenCL command queue associated with a context and device to execute the routine on.
 * - `cl_event* event`: Pointer to an OpenCL event to be able to wait for completion of the routine's OpenCL kernel(s). This is an optional argument.
 *
 * Requirements for HERK:
 * - When `(transpose == Transpose::kNo && layout == Layout::kColMajor) || (transpose == Transpose::kYes && layout == Layout::kRowMajor)`, then `a_ld` must be at least `n`, otherwise `a_ld` must be at least `k`.
 * - The value of `c_ld` must be at least `m`.
 */
extern(System) @nogc nothrow {
  alias da_CLBlastCherk = CLBlastStatusCode function(const CLBlastLayout layout, const CLBlastTriangle triangle, const CLBlastTranspose a_transpose,
      const size_t n, const size_t k,
      const float alpha,
      const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
      const float beta,
      cl_mem c_buffer, const size_t c_offset, const size_t c_ld,
      cl_command_queue* queue, cl_event* event);
  alias da_CLBlastZherk = CLBlastStatusCode function(const CLBlastLayout layout, const CLBlastTriangle triangle, const CLBlastTranspose a_transpose,
      const size_t n, const size_t k,
      const double alpha,
      const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
      const double beta,
      cl_mem c_buffer, const size_t c_offset, const size_t c_ld,
      cl_command_queue* queue, cl_event* event);
}

/**
 * xSYR2K: Rank-2K update of a symmetric matrix
 *
 * Performs the matrix product _C = alpha * A * B^T + alpha * B * A^T + beta * C_ or _C = alpha * A^T * B + alpha * B^T * A + beta * C_, in which _A_ and _B_ are general matrices and _A^T_ and _B^T_ are their transposed versions, _C_ (_n_ by _n_) is the symmetric matrix to be updated, and _alpha_ and _beta_ are scalar values.
 *
 * Arguments to SYR2K:
 * - `const Layout layout`: Data-layout of the matrices, either `Layout::kRowMajor` (101) for row-major layout or `Layout::kColMajor` (102) for column-major data-layout.
 * - `const Triangle triangle`: The part of the array of the triangular matrix to be used, either `Triangle::kUpper` (121) or `Triangle::kLower` (122).
 * - `const Transpose ab_transpose`: Transposing the packed input matrix AP, either `Transpose::kNo` (111), `Transpose::kYes` (112), or `Transpose::kConjugate` (113) for a complex-conjugate transpose.
 * - `const size_t n`: Integer size argument. This value must be positive.
 * - `const size_t k`: Integer size argument. This value must be positive.
 * - `const T alpha`: Input scalar constant.
 * - `const cl_mem a_buffer`: OpenCL buffer to store the input A matrix.
 * - `const size_t a_offset`: The offset in elements from the start of the input A matrix.
 * - `const size_t a_ld`: Leading dimension of the input A matrix. This value must be greater than 0.
 * - `const cl_mem b_buffer`: OpenCL buffer to store the input B matrix.
 * - `const size_t b_offset`: The offset in elements from the start of the input B matrix.
 * - `const size_t b_ld`: Leading dimension of the input B matrix. This value must be greater than 0.
 * - `const T beta`: Input scalar constant.
 * - `cl_mem c_buffer`: OpenCL buffer to store the output C matrix.
 * - `const size_t c_offset`: The offset in elements from the start of the output C matrix.
 * - `const size_t c_ld`: Leading dimension of the output C matrix. This value must be greater than 0.
 * - `cl_command_queue* queue`: Pointer to an OpenCL command queue associated with a context and device to execute the routine on.
 * - `cl_event* event`: Pointer to an OpenCL event to be able to wait for completion of the routine's OpenCL kernel(s). This is an optional argument.
 *
 * Requirements for SYR2K:
 * - When `(transpose == Transpose::kNo && layout == Layout::kColMajor) || (transpose == Transpose::kYes && layout == Layout::kRowMajor)`, then `a_ld` must be at least `n`, otherwise `a_ld` must be at least `k`.
 * - When `(transpose == Transpose::kNo && layout == Layout::kColMajor) || (transpose == Transpose::kYes && layout == Layout::kRowMajor)`, then `b_ld` must be at least `n`, otherwise `b_ld` must be at least `k`.
 * - The value of `c_ld` must be at least `n`.
 */
extern(System) @nogc nothrow {
  alias da_CLBlastSsyr2k = CLBlastStatusCode function(const CLBlastLayout layout, const CLBlastTriangle triangle, const CLBlastTranspose ab_transpose,
      const size_t n, const size_t k,
      const float alpha,
      const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
      const cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
      const float beta,
      cl_mem c_buffer, const size_t c_offset, const size_t c_ld,
      cl_command_queue* queue, cl_event* event);
  alias da_CLBlastDsyr2k = CLBlastStatusCode function(const CLBlastLayout layout, const CLBlastTriangle triangle, const CLBlastTranspose ab_transpose,
      const size_t n, const size_t k,
      const double alpha,
      const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
      const cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
      const double beta,
      cl_mem c_buffer, const size_t c_offset, const size_t c_ld,
      cl_command_queue* queue, cl_event* event);
  alias da_CLBlastCsyr2k = CLBlastStatusCode function(const CLBlastLayout layout, const CLBlastTriangle triangle, const CLBlastTranspose ab_transpose,
      const size_t n, const size_t k,
      const cl_float2 alpha,
      const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
      const cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
      const cl_float2 beta,
      cl_mem c_buffer, const size_t c_offset, const size_t c_ld,
      cl_command_queue* queue, cl_event* event);
  alias da_CLBlastZsyr2k = CLBlastStatusCode function(const CLBlastLayout layout, const CLBlastTriangle triangle, const CLBlastTranspose ab_transpose,
      const size_t n, const size_t k,
      const cl_double2 alpha,
      const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
      const cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
      const cl_double2 beta,
      cl_mem c_buffer, const size_t c_offset, const size_t c_ld,
      cl_command_queue* queue, cl_event* event);
  alias da_CLBlastHsyr2k = CLBlastStatusCode function(const CLBlastLayout layout, const CLBlastTriangle triangle, const CLBlastTranspose ab_transpose,
      const size_t n, const size_t k,
      const cl_half alpha,
      const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
      const cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
      const cl_half beta,
      cl_mem c_buffer, const size_t c_offset, const size_t c_ld,
      cl_command_queue* queue, cl_event* event);
}

/**
 * xHER2K: Rank-2K update of a hermitian matrix
 *
 * Same operation as xSYR2K, but _C_ is an Hermitian matrix instead.
 *
 * Arguments to HER2K:
 * - `const Layout layout`: Data-layout of the matrices, either `Layout::kRowMajor` (101) for row-major layout or `Layout::kColMajor` (102) for column-major data-layout.
 * - `const Triangle triangle`: The part of the array of the triangular matrix to be used, either `Triangle::kUpper` (121) or `Triangle::kLower` (122).
 * - `const Transpose ab_transpose`: Transposing the packed input matrix AP, either `Transpose::kNo` (111), `Transpose::kYes` (112), or `Transpose::kConjugate` (113) for a complex-conjugate transpose.
 * - `const size_t n`: Integer size argument. This value must be positive.
 * - `const size_t k`: Integer size argument. This value must be positive.
 * - `const T alpha`: Input scalar constant.
 * - `const cl_mem a_buffer`: OpenCL buffer to store the input A matrix.
 * - `const size_t a_offset`: The offset in elements from the start of the input A matrix.
 * - `const size_t a_ld`: Leading dimension of the input A matrix. This value must be greater than 0.
 * - `const cl_mem b_buffer`: OpenCL buffer to store the input B matrix.
 * - `const size_t b_offset`: The offset in elements from the start of the input B matrix.
 * - `const size_t b_ld`: Leading dimension of the input B matrix. This value must be greater than 0.
 * - `const U beta`: Input scalar constant.
 * - `cl_mem c_buffer`: OpenCL buffer to store the output C matrix.
 * - `const size_t c_offset`: The offset in elements from the start of the output C matrix.
 * - `const size_t c_ld`: Leading dimension of the output C matrix. This value must be greater than 0.
 * - `cl_command_queue* queue`: Pointer to an OpenCL command queue associated with a context and device to execute the routine on.
 * - `cl_event* event`: Pointer to an OpenCL event to be able to wait for completion of the routine's OpenCL kernel(s). This is an optional argument.
 *
 * Requirements for HER2K:
 * - When `(transpose == Transpose::kNo && layout == Layout::kColMajor) || (transpose == Transpose::kYes && layout == Layout::kRowMajor)`, then `a_ld` must be at least `n`, otherwise `a_ld` must be at least `k`.
 * - When `(transpose == Transpose::kNo && layout == Layout::kColMajor) || (transpose == Transpose::kYes && layout == Layout::kRowMajor)`, then `b_ld` must be at least `n`, otherwise `b_ld` must be at least `k`.
 * - The value of `c_ld` must be at least `n`.
 */
extern(System) @nogc nothrow {
  alias da_CLBlastCher2k = CLBlastStatusCode function(const CLBlastLayout layout, const CLBlastTriangle triangle, const CLBlastTranspose ab_transpose,
      const size_t n, const size_t k,
      const cl_float2 alpha,
      const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
      const cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
      const float beta,
      cl_mem c_buffer, const size_t c_offset, const size_t c_ld,
      cl_command_queue* queue, cl_event* event);
  alias da_CLBlastZher2k = CLBlastStatusCode function(const CLBlastLayout layout, const CLBlastTriangle triangle, const CLBlastTranspose ab_transpose,
      const size_t n, const size_t k,
      const cl_double2 alpha,
      const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
      const cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
      const double beta,
      cl_mem c_buffer, const size_t c_offset, const size_t c_ld,
      cl_command_queue* queue, cl_event* event);
}

/**
 * xTRMM: Triangular matrix-matrix multiplication
 *
 * Performs the matrix product _B = alpha * A * B_ or _B = alpha * B * A_, in which _A_ is a unit or non-unit triangular matrix, _B_ (_m_ by _n_) is the general matrix to be updated, and _alpha_ is a scalar value.
 *
 * Arguments to TRMM:
 * - `const Layout layout`: Data-layout of the matrices, either `Layout::kRowMajor` (101) for row-major layout or `Layout::kColMajor` (102) for column-major data-layout.
 * - `const Side side`: The position of the triangular matrix in the operation, either on the `Side::kLeft` (141) or `Side::kRight` (142).
 * - `const Triangle triangle`: The part of the array of the triangular matrix to be used, either `Triangle::kUpper` (121) or `Triangle::kLower` (122).
 * - `const Transpose a_transpose`: Transposing the input matrix A, either `Transpose::kNo` (111), `Transpose::kYes` (112), or `Transpose::kConjugate` (113) for a complex-conjugate transpose.
 * - `const Diagonal diagonal`: The property of the diagonal matrix, either `Diagonal::kNonUnit` (131) for non-unit values on the diagonal or `Diagonal::kUnit` (132) for unit values on the diagonal.
 * - `const size_t m`: Integer size argument. This value must be positive.
 * - `const size_t n`: Integer size argument. This value must be positive.
 * - `const T alpha`: Input scalar constant.
 * - `const cl_mem a_buffer`: OpenCL buffer to store the input A matrix.
 * - `const size_t a_offset`: The offset in elements from the start of the input A matrix.
 * - `const size_t a_ld`: Leading dimension of the input A matrix. This value must be greater than 0.
 * - `cl_mem b_buffer`: OpenCL buffer to store the output B matrix.
 * - `const size_t b_offset`: The offset in elements from the start of the output B matrix.
 * - `const size_t b_ld`: Leading dimension of the output B matrix. This value must be greater than 0.
 * - `cl_command_queue* queue`: Pointer to an OpenCL command queue associated with a context and device to execute the routine on.
 * - `cl_event* event`: Pointer to an OpenCL event to be able to wait for completion of the routine's OpenCL kernel(s). This is an optional argument.
 *
 * Requirements for TRMM:
 * - When `side = Side::kLeft` then `a_ld` must be at least `m`, otherwise `a_ld` must be at least `n`.
 * - The value of `b_ld` must be at least `m`.
 */
extern(System) @nogc nothrow {
  alias da_CLBlastStrmm = CLBlastStatusCode function(const CLBlastLayout layout, const CLBlastSide side, const CLBlastTriangle triangle, const CLBlastTranspose a_transpose, const CLBlastDiagonal diagonal,
      const size_t m, const size_t n,
      const float alpha,
      const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
      cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
      cl_command_queue* queue, cl_event* event);
  alias da_CLBlastDtrmm = CLBlastStatusCode function(const CLBlastLayout layout, const CLBlastSide side, const CLBlastTriangle triangle, const CLBlastTranspose a_transpose, const CLBlastDiagonal diagonal,
      const size_t m, const size_t n,
      const double alpha,
      const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
      cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
      cl_command_queue* queue, cl_event* event);
  alias da_CLBlastCtrmm = CLBlastStatusCode function(const CLBlastLayout layout, const CLBlastSide side, const CLBlastTriangle triangle, const CLBlastTranspose a_transpose, const CLBlastDiagonal diagonal,
      const size_t m, const size_t n,
      const cl_float2 alpha,
      const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
      cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
      cl_command_queue* queue, cl_event* event);
  alias da_CLBlastZtrmm = CLBlastStatusCode function(const CLBlastLayout layout, const CLBlastSide side, const CLBlastTriangle triangle, const CLBlastTranspose a_transpose, const CLBlastDiagonal diagonal,
      const size_t m, const size_t n,
      const cl_double2 alpha,
      const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
      cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
      cl_command_queue* queue, cl_event* event);
  alias da_CLBlastHtrmm = CLBlastStatusCode function(const CLBlastLayout layout, const CLBlastSide side, const CLBlastTriangle triangle, const CLBlastTranspose a_transpose, const CLBlastDiagonal diagonal,
      const size_t m, const size_t n,
      const cl_half alpha,
      const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
      cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
      cl_command_queue* queue, cl_event* event);
}

/**
 * xTRSM: Solves a triangular system of equations
 *
 * Solves the equation _A * X = alpha * B_ for the unknown _m_ by _n_ matrix X, in which _A_ is an _n_ by _n_ unit or non-unit triangular matrix and B is an _m_ by _n_ matrix. The matrix _B_ is overwritten by the solution _X_.
 *
 * Arguments to TRSM:
 * - `const Layout layout`: Data-layout of the matrices, either `Layout::kRowMajor` (101) for row-major layout or `Layout::kColMajor` (102) for column-major data-layout.
 * - `const Side side`: The position of the triangular matrix in the operation, either on the `Side::kLeft` (141) or `Side::kRight` (142).
 * - `const Triangle triangle`: The part of the array of the triangular matrix to be used, either `Triangle::kUpper` (121) or `Triangle::kLower` (122).
 * - `const Transpose a_transpose`: Transposing the input matrix A, either `Transpose::kNo` (111), `Transpose::kYes` (112), or `Transpose::kConjugate` (113) for a complex-conjugate transpose.
 * - `const Diagonal diagonal`: The property of the diagonal matrix, either `Diagonal::kNonUnit` (131) for non-unit values on the diagonal or `Diagonal::kUnit` (132) for unit values on the diagonal.
 * - `const size_t m`: Integer size argument. This value must be positive.
 * - `const size_t n`: Integer size argument. This value must be positive.
 * - `const T alpha`: Input scalar constant.
 * - `const cl_mem a_buffer`: OpenCL buffer to store the input A matrix.
 * - `const size_t a_offset`: The offset in elements from the start of the input A matrix.
 * - `const size_t a_ld`: Leading dimension of the input A matrix. This value must be greater than 0.
 * - `cl_mem b_buffer`: OpenCL buffer to store the output B matrix.
 * - `const size_t b_offset`: The offset in elements from the start of the output B matrix.
 * - `const size_t b_ld`: Leading dimension of the output B matrix. This value must be greater than 0.
 * - `cl_command_queue* queue`: Pointer to an OpenCL command queue associated with a context and device to execute the routine on.
 * - `cl_event* event`: Pointer to an OpenCL event to be able to wait for completion of the routine's OpenCL kernel(s). This is an optional argument.
 */
extern(System) @nogc nothrow {
  alias da_CLBlastStrsm = CLBlastStatusCode function(const CLBlastLayout layout, const CLBlastSide side, const CLBlastTriangle triangle, const CLBlastTranspose a_transpose, const CLBlastDiagonal diagonal,
      const size_t m, const size_t n,
      const float alpha,
      const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
      cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
      cl_command_queue* queue, cl_event* event);
  alias da_CLBlastDtrsm = CLBlastStatusCode function(const CLBlastLayout layout, const CLBlastSide side, const CLBlastTriangle triangle, const CLBlastTranspose a_transpose, const CLBlastDiagonal diagonal,
      const size_t m, const size_t n,
      const double alpha,
      const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
      cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
      cl_command_queue* queue, cl_event* event);
  alias da_CLBlastCtrsm = CLBlastStatusCode function(const CLBlastLayout layout, const CLBlastSide side, const CLBlastTriangle triangle, const CLBlastTranspose a_transpose, const CLBlastDiagonal diagonal,
      const size_t m, const size_t n,
      const cl_float2 alpha,
      const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
      cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
      cl_command_queue* queue, cl_event* event);
  alias da_CLBlastZtrsm = CLBlastStatusCode function(const CLBlastLayout layout, const CLBlastSide side, const CLBlastTriangle triangle, const CLBlastTranspose a_transpose, const CLBlastDiagonal diagonal,
      const size_t m, const size_t n,
      const cl_double2 alpha,
      const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
      cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
      cl_command_queue* queue, cl_event* event);
}

/**
 * xHAD: Element-wise vector product (Hadamard)
 *
 * Performs the Hadamard element-wise product _z = alpha * x * y + beta * z_, in which _x_, _y_, and _z_ are vectors and _alpha_ and _beta_ are scalar constants.
 *
 * Arguments to HAD:
 * - `const size_t n`: Integer size argument. This value must be positive.
 * - `const T alpha`: Input scalar constant.
 * - `const cl_mem x_buffer`: OpenCL buffer to store the input x vector.
 * - `const size_t x_offset`: The offset in elements from the start of the input x vector.
 * - `const size_t x_inc`: Stride/increment of the input x vector. This value must be greater than 0.
 * - `const cl_mem y_buffer`: OpenCL buffer to store the input y vector.
 * - `const size_t y_offset`: The offset in elements from the start of the input y vector.
 * - `const size_t y_inc`: Stride/increment of the input y vector. This value must be greater than 0.
 * - `const T beta`: Input scalar constant.
 * - `cl_mem z_buffer`: OpenCL buffer to store the output z vector.
 * - `const size_t z_offset`: The offset in elements from the start of the output z vector.
 * - `const size_t z_inc`: Stride/increment of the output z vector. This value must be greater than 0.
 * - `cl_command_queue* queue`: Pointer to an OpenCL command queue associated with a context and device to execute the routine on.
 * - `cl_event* event`: Pointer to an OpenCL event to be able to wait for completion of the routine's OpenCL kernel(s). This is an optional argument.
 */
extern(System) @nogc nothrow {
  alias da_CLBlastShad = CLBlastStatusCode function(const size_t n,
      const float alpha,
      const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
      const cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
      const float beta,
      cl_mem z_buffer, const size_t z_offset, const size_t z_inc,
      cl_command_queue* queue, cl_event* event);
  alias da_CLBlastDhad = CLBlastStatusCode function(const size_t n,
      const double alpha,
      const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
      const cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
      const double beta,
      cl_mem z_buffer, const size_t z_offset, const size_t z_inc,
      cl_command_queue* queue, cl_event* event);
  alias da_CLBlastChad = CLBlastStatusCode function(const size_t n,
      const cl_float2 alpha,
      const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
      const cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
      const cl_float2 beta,
      cl_mem z_buffer, const size_t z_offset, const size_t z_inc,
      cl_command_queue* queue, cl_event* event);
  alias da_CLBlastZhad = CLBlastStatusCode function(const size_t n,
      const cl_double2 alpha,
      const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
      const cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
      const cl_double2 beta,
      cl_mem z_buffer, const size_t z_offset, const size_t z_inc,
      cl_command_queue* queue, cl_event* event);
  alias da_CLBlastHhad = CLBlastStatusCode function(const size_t n,
      const cl_half alpha,
      const cl_mem x_buffer, const size_t x_offset, const size_t x_inc,
      const cl_mem y_buffer, const size_t y_offset, const size_t y_inc,
      const cl_half beta,
      cl_mem z_buffer, const size_t z_offset, const size_t z_inc,
      cl_command_queue* queue, cl_event* event);
}

/**
 * xOMATCOPY: Scaling and out-place transpose/copy (non-BLAS function)
 *
 * Performs scaling and out-of-place transposition/copying of matrices according to _B = alpha*op(A)_, in which _A_ is an input matrix (_m_ rows by _n_ columns), _B_ an output matrix, and _alpha_ a scalar value. The operation _op_ can be a normal matrix copy, a transposition or a conjugate transposition.
 *
 * Arguments to OMATCOPY:
 * - `const Layout layout`: Data-layout of the matrices, either `Layout::kRowMajor` (101) for row-major layout or `Layout::kColMajor` (102) for column-major data-layout.
 * - `const Transpose a_transpose`: Transposing the input matrix A, either `Transpose::kNo` (111), `Transpose::kYes` (112), or `Transpose::kConjugate` (113) for a complex-conjugate transpose.
 * - `const size_t m`: Integer size argument. This value must be positive.
 * - `const size_t n`: Integer size argument. This value must be positive.
 * - `const T alpha`: Input scalar constant.
 * - `const cl_mem a_buffer`: OpenCL buffer to store the input A matrix.
 * - `const size_t a_offset`: The offset in elements from the start of the input A matrix.
 * - `const size_t a_ld`: Leading dimension of the input A matrix. This value must be greater than 0.
 * - `cl_mem b_buffer`: OpenCL buffer to store the output B matrix.
 * - `const size_t b_offset`: The offset in elements from the start of the output B matrix.
 * - `const size_t b_ld`: Leading dimension of the output B matrix. This value must be greater than 0.
 * - `cl_command_queue* queue`: Pointer to an OpenCL command queue associated with a context and device to execute the routine on.
 * - `cl_event* event`: Pointer to an OpenCL event to be able to wait for completion of the routine's OpenCL kernel(s). This is an optional argument.
 *
 * Requirements for OMATCOPY:
 * - The value of `a_ld` must be at least `m`.
 * - The value of `b_ld` must be at least `n`.
 */
extern(System) @nogc nothrow {
  alias da_CLBlastSomatcopy = CLBlastStatusCode function(const CLBlastLayout layout, const CLBlastTranspose a_transpose,
      const size_t m, const size_t n,
      const float alpha,
      const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
      cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
      cl_command_queue* queue, cl_event* event);
  alias da_CLBlastDomatcopy = CLBlastStatusCode function(const CLBlastLayout layout, const CLBlastTranspose a_transpose,
      const size_t m, const size_t n,
      const double alpha,
      const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
      cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
      cl_command_queue* queue, cl_event* event);
  alias da_CLBlastComatcopy = CLBlastStatusCode function(const CLBlastLayout layout, const CLBlastTranspose a_transpose,
      const size_t m, const size_t n,
      const cl_float2 alpha,
      const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
      cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
      cl_command_queue* queue, cl_event* event);
  alias da_CLBlastZomatcopy = CLBlastStatusCode function(const CLBlastLayout layout, const CLBlastTranspose a_transpose,
      const size_t m, const size_t n,
      const cl_double2 alpha,
      const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
      cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
      cl_command_queue* queue, cl_event* event);
  alias da_CLBlastHomatcopy = CLBlastStatusCode function(const CLBlastLayout layout, const CLBlastTranspose a_transpose,
      const size_t m, const size_t n,
      const cl_half alpha,
      const cl_mem a_buffer, const size_t a_offset, const size_t a_ld,
      cl_mem b_buffer, const size_t b_offset, const size_t b_ld,
      cl_command_queue* queue, cl_event* event);
}

/**
 * xIM2COL: Im2col function (non-BLAS function)
 *
 * Performs the im2col algorithm, in which _im_ is the input matrix and _col_ is the output matrix. Overwrites any existing values in the _col_ buffer
 *
 * Arguments to IM2COL:
 * - `const KernelMode kernel_mode`: The kernel mode, either `KernelMode::kCrossCorrelation` for the normal mode, or `KernelMode::kConvolution` for the convolution mode that flips a kernel along `h` and `w` axes.
 * - `const size_t channels`: Integer size argument. This value must be positive.
 * - `const size_t height`: Integer size argument. This value must be positive.
 * - `const size_t width`: Integer size argument. This value must be positive.
 * - `const size_t kernel_h`: Integer size argument. This value must be positive.
 * - `const size_t kernel_w`: Integer size argument. This value must be positive.
 * - `const size_t pad_h`: Integer size argument. This value must be positive.
 * - `const size_t pad_w`: Integer size argument. This value must be positive.
 * - `const size_t stride_h`: Integer size argument. This value must be positive.
 * - `const size_t stride_w`: Integer size argument. This value must be positive.
 * - `const size_t dilation_h`: Integer size argument. This value must be positive.
 * - `const size_t dilation_w`: Integer size argument. This value must be positive.
 * - `const cl_mem im_buffer`: OpenCL buffer to store the input im tensor.
 * - `const size_t im_offset`: The offset in elements from the start of the input im tensor.
 * - `cl_mem col_buffer`: OpenCL buffer to store the output col tensor.
 * - `const size_t col_offset`: The offset in elements from the start of the output col tensor.
 * - `cl_command_queue* queue`: Pointer to an OpenCL command queue associated with a context and device to execute the routine on.
 * - `cl_event* event`: Pointer to an OpenCL event to be able to wait for completion of the routine's OpenCL kernel(s). This is an optional argument.
 */
extern(System) @nogc nothrow {
  alias da_CLBlastSim2col = CLBlastStatusCode function(const CLBlastKernelMode kernel_mode,
      const size_t channels, const size_t height, const size_t width, const size_t kernel_h, const size_t kernel_w, const size_t pad_h, const size_t pad_w, const size_t stride_h, const size_t stride_w, const size_t dilation_h, const size_t dilation_w,
      const cl_mem im_buffer, const size_t im_offset,
      cl_mem col_buffer, const size_t col_offset,
      cl_command_queue* queue, cl_event* event);
  alias da_CLBlastDim2col = CLBlastStatusCode function(const CLBlastKernelMode kernel_mode,
      const size_t channels, const size_t height, const size_t width, const size_t kernel_h, const size_t kernel_w, const size_t pad_h, const size_t pad_w, const size_t stride_h, const size_t stride_w, const size_t dilation_h, const size_t dilation_w,
      const cl_mem im_buffer, const size_t im_offset,
      cl_mem col_buffer, const size_t col_offset,
      cl_command_queue* queue, cl_event* event);
  alias da_CLBlastCim2col = CLBlastStatusCode function(const CLBlastKernelMode kernel_mode,
      const size_t channels, const size_t height, const size_t width, const size_t kernel_h, const size_t kernel_w, const size_t pad_h, const size_t pad_w, const size_t stride_h, const size_t stride_w, const size_t dilation_h, const size_t dilation_w,
      const cl_mem im_buffer, const size_t im_offset,
      cl_mem col_buffer, const size_t col_offset,
      cl_command_queue* queue, cl_event* event);
  alias da_CLBlastZim2col = CLBlastStatusCode function(const CLBlastKernelMode kernel_mode,
      const size_t channels, const size_t height, const size_t width, const size_t kernel_h, const size_t kernel_w, const size_t pad_h, const size_t pad_w, const size_t stride_h, const size_t stride_w, const size_t dilation_h, const size_t dilation_w,
      const cl_mem im_buffer, const size_t im_offset,
      cl_mem col_buffer, const size_t col_offset,
      cl_command_queue* queue, cl_event* event);
  alias da_CLBlastHim2col = CLBlastStatusCode function(const CLBlastKernelMode kernel_mode,
      const size_t channels, const size_t height, const size_t width, const size_t kernel_h, const size_t kernel_w, const size_t pad_h, const size_t pad_w, const size_t stride_h, const size_t stride_w, const size_t dilation_h, const size_t dilation_w,
      const cl_mem im_buffer, const size_t im_offset,
      cl_mem col_buffer, const size_t col_offset,
      cl_command_queue* queue, cl_event* event);
}

/**
 * xCOL2IM: Col2im function (non-BLAS function)
 *
 * Performs the col2im algorithm, in which _col_ is the input matrix and _im_ is the output matrix. Accumulates results on top of the existing values in the _im_ buffer.
 *
 * Arguments to COL2IM:
 * - `const KernelMode kernel_mode`: The kernel mode, either `KernelMode::kCrossCorrelation` for the normal mode, or `KernelMode::kConvolution` for the convolution mode that flips a kernel along `h` and `w` axes.
 * - `const size_t channels`: Integer size argument. This value must be positive.
 * - `const size_t height`: Integer size argument. This value must be positive.
 * - `const size_t width`: Integer size argument. This value must be positive.
 * - `const size_t kernel_h`: Integer size argument. This value must be positive.
 * - `const size_t kernel_w`: Integer size argument. This value must be positive.
 * - `const size_t pad_h`: Integer size argument. This value must be positive.
 * - `const size_t pad_w`: Integer size argument. This value must be positive.
 * - `const size_t stride_h`: Integer size argument. This value must be positive.
 * - `const size_t stride_w`: Integer size argument. This value must be positive.
 * - `const size_t dilation_h`: Integer size argument. This value must be positive.
 * - `const size_t dilation_w`: Integer size argument. This value must be positive.
 * - `const cl_mem col_buffer`: OpenCL buffer to store the input col tensor.
 * - `const size_t col_offset`: The offset in elements from the start of the input col tensor.
 * - `cl_mem im_buffer`: OpenCL buffer to store the output im tensor.
 * - `const size_t im_offset`: The offset in elements from the start of the output im tensor.
 * - `cl_command_queue* queue`: Pointer to an OpenCL command queue associated with a context and device to execute the routine on.
 * - `cl_event* event`: Pointer to an OpenCL event to be able to wait for completion of the routine's OpenCL kernel(s). This is an optional argument.
 */
extern(System) @nogc nothrow {
  alias da_CLBlastScol2im = CLBlastStatusCode function(const CLBlastKernelMode kernel_mode,
      const size_t channels, const size_t height, const size_t width, const size_t kernel_h, const size_t kernel_w, const size_t pad_h, const size_t pad_w, const size_t stride_h, const size_t stride_w, const size_t dilation_h, const size_t dilation_w,
      const cl_mem col_buffer, const size_t col_offset,
      cl_mem im_buffer, const size_t im_offset,
      cl_command_queue* queue, cl_event* event);
  alias da_CLBlastDcol2im = CLBlastStatusCode function(const CLBlastKernelMode kernel_mode,
      const size_t channels, const size_t height, const size_t width, const size_t kernel_h, const size_t kernel_w, const size_t pad_h, const size_t pad_w, const size_t stride_h, const size_t stride_w, const size_t dilation_h, const size_t dilation_w,
      const cl_mem col_buffer, const size_t col_offset,
      cl_mem im_buffer, const size_t im_offset,
      cl_command_queue* queue, cl_event* event);
  alias da_CLBlastCcol2im = CLBlastStatusCode function(const CLBlastKernelMode kernel_mode,
      const size_t channels, const size_t height, const size_t width, const size_t kernel_h, const size_t kernel_w, const size_t pad_h, const size_t pad_w, const size_t stride_h, const size_t stride_w, const size_t dilation_h, const size_t dilation_w,
      const cl_mem col_buffer, const size_t col_offset,
      cl_mem im_buffer, const size_t im_offset,
      cl_command_queue* queue, cl_event* event);
  alias da_CLBlastZcol2im = CLBlastStatusCode function(const CLBlastKernelMode kernel_mode,
      const size_t channels, const size_t height, const size_t width, const size_t kernel_h, const size_t kernel_w, const size_t pad_h, const size_t pad_w, const size_t stride_h, const size_t stride_w, const size_t dilation_h, const size_t dilation_w,
      const cl_mem col_buffer, const size_t col_offset,
      cl_mem im_buffer, const size_t im_offset,
      cl_command_queue* queue, cl_event* event);
  alias da_CLBlastHcol2im = CLBlastStatusCode function(const CLBlastKernelMode kernel_mode,
      const size_t channels, const size_t height, const size_t width, const size_t kernel_h, const size_t kernel_w, const size_t pad_h, const size_t pad_w, const size_t stride_h, const size_t stride_w, const size_t dilation_h, const size_t dilation_w,
      const cl_mem col_buffer, const size_t col_offset,
      cl_mem im_buffer, const size_t im_offset,
      cl_command_queue* queue, cl_event* event);
}

/**
 * xCONVGEMM: Batched convolution as GEMM (non-BLAS function)
 *
 * Integrates im2col and GEMM for batched 3D convolution, in which _im_ is the 4D input tensor (NCHW - batch-channelin-height-width), _kernel_ the 4D kernel weights tensor (KCHW - channelout-channelin-height-width), and _result_ the 4D output tensor (NCHW - batch-channelout-height-width).
 *
 * Arguments to CONVGEMM:
 * - `const KernelMode kernel_mode`: The kernel mode, either `KernelMode::kCrossCorrelation` for the normal mode, or `KernelMode::kConvolution` for the convolution mode that flips a kernel along `h` and `w` axes.
 * - `const size_t channels`: Integer size argument. This value must be positive.
 * - `const size_t height`: Integer size argument. This value must be positive.
 * - `const size_t width`: Integer size argument. This value must be positive.
 * - `const size_t kernel_h`: Integer size argument. This value must be positive.
 * - `const size_t kernel_w`: Integer size argument. This value must be positive.
 * - `const size_t pad_h`: Integer size argument. This value must be positive.
 * - `const size_t pad_w`: Integer size argument. This value must be positive.
 * - `const size_t stride_h`: Integer size argument. This value must be positive.
 * - `const size_t stride_w`: Integer size argument. This value must be positive.
 * - `const size_t dilation_h`: Integer size argument. This value must be positive.
 * - `const size_t dilation_w`: Integer size argument. This value must be positive.
 * - `const size_t num_kernels`: Integer size argument. This value must be positive.
 * - `const size_t batch_count`: Integer size argument. This value must be positive.
 * - `const cl_mem im_buffer`: OpenCL buffer to store the input im tensor.
 * - `const size_t im_offset`: The offset in elements from the start of the input im tensor.
 * - `const cl_mem kernel_buffer`: OpenCL buffer to store the input kernel tensor.
 * - `const size_t kernel_offset`: The offset in elements from the start of the input kernel tensor.
 * - `cl_mem result_buffer`: OpenCL buffer to store the output result tensor.
 * - `const size_t result_offset`: The offset in elements from the start of the output result tensor.
 * - `cl_command_queue* queue`: Pointer to an OpenCL command queue associated with a context and device to execute the routine on.
 * - `cl_event* event`: Pointer to an OpenCL event to be able to wait for completion of the routine's OpenCL kernel(s). This is an optional argument.
 */
extern(System) @nogc nothrow {
  alias da_CLBlastSconvgemm = CLBlastStatusCode function(const CLBlastKernelMode kernel_mode,
      const size_t channels, const size_t height, const size_t width, const size_t kernel_h, const size_t kernel_w, const size_t pad_h, const size_t pad_w, const size_t stride_h, const size_t stride_w, const size_t dilation_h, const size_t dilation_w, const size_t num_kernels, const size_t batch_count,
      const cl_mem im_buffer, const size_t im_offset,
      const cl_mem kernel_buffer, const size_t kernel_offset,
      cl_mem result_buffer, const size_t result_offset,
      cl_command_queue* queue, cl_event* event);
  alias da_CLBlastDconvgemm = CLBlastStatusCode function(const CLBlastKernelMode kernel_mode,
      const size_t channels, const size_t height, const size_t width, const size_t kernel_h, const size_t kernel_w, const size_t pad_h, const size_t pad_w, const size_t stride_h, const size_t stride_w, const size_t dilation_h, const size_t dilation_w, const size_t num_kernels, const size_t batch_count,
      const cl_mem im_buffer, const size_t im_offset,
      const cl_mem kernel_buffer, const size_t kernel_offset,
      cl_mem result_buffer, const size_t result_offset,
      cl_command_queue* queue, cl_event* event);
  alias da_CLBlastHconvgemm = CLBlastStatusCode function(const CLBlastKernelMode kernel_mode,
      const size_t channels, const size_t height, const size_t width, const size_t kernel_h, const size_t kernel_w, const size_t pad_h, const size_t pad_w, const size_t stride_h, const size_t stride_w, const size_t dilation_h, const size_t dilation_w, const size_t num_kernels, const size_t batch_count,
      const cl_mem im_buffer, const size_t im_offset,
      const cl_mem kernel_buffer, const size_t kernel_offset,
      cl_mem result_buffer, const size_t result_offset,
      cl_command_queue* queue, cl_event* event);
}

/**
 * xAXPYBATCHED: Batched version of AXPY
 *
 * As AXPY, but multiple operations are batched together for better performance.
 *
 * Arguments to AXPYBATCHED:
 *
 * - `const size_t n`: Integer size argument. This value must be positive.
 * - `const T *alphas`: Input scalar constants.
 * - `const cl_mem x_buffer`: OpenCL buffer to store the input x vector.
 * - `const size_t *x_offsets`: The offsets in elements from the start of the input x vector.
 * - `const size_t x_inc`: Stride/increment of the input x vector. This value must be greater than 0.
 * - `cl_mem y_buffer`: OpenCL buffer to store the output y vector.
 * - `const size_t *y_offsets`: The offsets in elements from the start of the output y vector.
 * - `const size_t y_inc`: Stride/increment of the output y vector. This value must be greater than 0.
 * - `const size_t batch_count`: Number of batches. This value must be positive.
 * - `cl_command_queue* queue`: Pointer to an OpenCL command queue associated with a context and device to execute the routine on.
 * - `cl_event* event`: Pointer to an OpenCL event to be able to wait for completion of the routine's OpenCL kernel(s). This is an optional argument.
 */
extern(System) @nogc nothrow {
  alias da_CLBlastSaxpyBatched = CLBlastStatusCode function(const size_t n,
      const float *alphas,
      const cl_mem x_buffer, const size_t *x_offsets, const size_t x_inc,
      cl_mem y_buffer, const size_t *y_offsets, const size_t y_inc,
      const size_t batch_count,
      cl_command_queue* queue, cl_event* event);
  alias da_CLBlastDaxpyBatched = CLBlastStatusCode function(const size_t n,
      const double *alphas,
      const cl_mem x_buffer, const size_t *x_offsets, const size_t x_inc,
      cl_mem y_buffer, const size_t *y_offsets, const size_t y_inc,
      const size_t batch_count,
      cl_command_queue* queue, cl_event* event);
  alias da_CLBlastCaxpyBatched = CLBlastStatusCode function(const size_t n,
      const cl_float2 *alphas,
      const cl_mem x_buffer, const size_t *x_offsets, const size_t x_inc,
      cl_mem y_buffer, const size_t *y_offsets, const size_t y_inc,
      const size_t batch_count,
      cl_command_queue* queue, cl_event* event);
  alias da_CLBlastZaxpyBatched = CLBlastStatusCode function(const size_t n,
      const cl_double2 *alphas,
      const cl_mem x_buffer, const size_t *x_offsets, const size_t x_inc,
      cl_mem y_buffer, const size_t *y_offsets, const size_t y_inc,
      const size_t batch_count,
      cl_command_queue* queue, cl_event* event);
  alias da_CLBlastHaxpyBatched = CLBlastStatusCode function(const size_t n,
      const cl_half *alphas,
      const cl_mem x_buffer, const size_t *x_offsets, const size_t x_inc,
      cl_mem y_buffer, const size_t *y_offsets, const size_t y_inc,
      const size_t batch_count,
      cl_command_queue* queue, cl_event* event);
}

/**
 * xGEMMBATCHED: Batched version of GEMM
 *
 * As GEMM, but multiple operations are batched together for better performance.
 *
 * Arguments to GEMMBATCHED:
 * - `const Layout layout`: Data-layout of the matrices, either `Layout::kRowMajor` (101) for row-major layout or `Layout::kColMajor` (102) for column-major data-layout.
 * - `const Transpose a_transpose`: Transposing the input matrix A, either `Transpose::kNo` (111), `Transpose::kYes` (112), or `Transpose::kConjugate` (113) for a complex-conjugate transpose.
 * - `const Transpose b_transpose`: Transposing the input matrix B, either `Transpose::kNo` (111), `Transpose::kYes` (112), or `Transpose::kConjugate` (113) for a complex-conjugate transpose.
 * - `const size_t m`: Integer size argument. This value must be positive.
 * - `const size_t n`: Integer size argument. This value must be positive.
 * - `const size_t k`: Integer size argument. This value must be positive.
 * - `const T *alphas`: Input scalar constants.
 * - `const cl_mem a_buffer`: OpenCL buffer to store the input A matrix.
 * - `const size_t *a_offsets`: The offsets in elements from the start of the input A matrix.
 * - `const size_t a_ld`: Leading dimension of the input A matrix. This value must be greater than 0.
 * - `const cl_mem b_buffer`: OpenCL buffer to store the input B matrix.
 * - `const size_t *b_offsets`: The offsets in elements from the start of the input B matrix.
 * - `const size_t b_ld`: Leading dimension of the input B matrix. This value must be greater than 0.
 * - `const T *betas`: Input scalar constants.
 * - `cl_mem c_buffer`: OpenCL buffer to store the output C matrix.
 * - `const size_t *c_offsets`: The offsets in elements from the start of the output C matrix.
 * - `const size_t c_ld`: Leading dimension of the output C matrix. This value must be greater than 0.
 * - `const size_t batch_count`: Number of batches. This value must be positive.
 * - `cl_command_queue* queue`: Pointer to an OpenCL command queue associated with a context and device to execute the routine on.
 * - `cl_event* event`: Pointer to an OpenCL event to be able to wait for completion of the routine's OpenCL kernel(s). This is an optional argument.
 *
 * Requirements for GEMMBATCHED:
 * - When `(transpose_a == Transpose::kNo && layout == Layout::kColMajor) || (transpose_a == Transpose::kYes && layout == Layout::kRowMajor)`, then `a_ld` must be at least `m`, otherwise `a_ld` must be at least `k`.
 * - When `(transpose_b == Transpose::kNo && layout == Layout::kColMajor) || (transpose_b == Transpose::kYes && layout == Layout::kRowMajor)`, then `b_ld` must be at least `k`, otherwise `b_ld` must be at least `n`.
 * - The value of `c_ld` must be at least `m`.
 */
extern(System) @nogc nothrow {
  alias da_CLBlastSgemmBatched = CLBlastStatusCode function(const CLBlastLayout layout, const CLBlastTranspose a_transpose, const CLBlastTranspose b_transpose,
      const size_t m, const size_t n, const size_t k,
      const float *alphas,
      const cl_mem a_buffer, const size_t *a_offsets, const size_t a_ld,
      const cl_mem b_buffer, const size_t *b_offsets, const size_t b_ld,
      const float *betas,
      cl_mem c_buffer, const size_t *c_offsets, const size_t c_ld,
      const size_t batch_count,
      cl_command_queue* queue, cl_event* event);
  alias da_CLBlastDgemmBatched = CLBlastStatusCode function(const CLBlastLayout layout, const CLBlastTranspose a_transpose, const CLBlastTranspose b_transpose,
      const size_t m, const size_t n, const size_t k,
      const double *alphas,
      const cl_mem a_buffer, const size_t *a_offsets, const size_t a_ld,
      const cl_mem b_buffer, const size_t *b_offsets, const size_t b_ld,
      const double *betas,
      cl_mem c_buffer, const size_t *c_offsets, const size_t c_ld,
      const size_t batch_count,
      cl_command_queue* queue, cl_event* event);
  alias da_CLBlastCgemmBatched = CLBlastStatusCode function(const CLBlastLayout layout, const CLBlastTranspose a_transpose, const CLBlastTranspose b_transpose,
      const size_t m, const size_t n, const size_t k,
      const cl_float2 *alphas,
      const cl_mem a_buffer, const size_t *a_offsets, const size_t a_ld,
      const cl_mem b_buffer, const size_t *b_offsets, const size_t b_ld,
      const cl_float2 *betas,
      cl_mem c_buffer, const size_t *c_offsets, const size_t c_ld,
      const size_t batch_count,
      cl_command_queue* queue, cl_event* event);
  alias da_CLBlastZgemmBatched = CLBlastStatusCode function(const CLBlastLayout layout, const CLBlastTranspose a_transpose, const CLBlastTranspose b_transpose,
      const size_t m, const size_t n, const size_t k,
      const cl_double2 *alphas,
      const cl_mem a_buffer, const size_t *a_offsets, const size_t a_ld,
      const cl_mem b_buffer, const size_t *b_offsets, const size_t b_ld,
      const cl_double2 *betas,
      cl_mem c_buffer, const size_t *c_offsets, const size_t c_ld,
      const size_t batch_count,
      cl_command_queue* queue, cl_event* event);
  alias da_CLBlastHgemmBatched = CLBlastStatusCode function(const CLBlastLayout layout, const CLBlastTranspose a_transpose, const CLBlastTranspose b_transpose,
      const size_t m, const size_t n, const size_t k,
      const cl_half *alphas,
      const cl_mem a_buffer, const size_t *a_offsets, const size_t a_ld,
      const cl_mem b_buffer, const size_t *b_offsets, const size_t b_ld,
      const cl_half *betas,
      cl_mem c_buffer, const size_t *c_offsets, const size_t c_ld,
      const size_t batch_count,
      cl_command_queue* queue, cl_event* event);
}

/**
 * xGEMMSTRIDEDBATCHED: StridedBatched version of GEMM
 *
 * As GEMM, but multiple strided operations are batched together for better performance.
 *
 * Arguments to GEMMSTRIDEDBATCHED:
 * - `const Layout layout`: Data-layout of the matrices, either `Layout::kRowMajor` (101) for row-major layout or `Layout::kColMajor` (102) for column-major data-layout.
 * - `const Transpose a_transpose`: Transposing the input matrix A, either `Transpose::kNo` (111), `Transpose::kYes` (112), or `Transpose::kConjugate` (113) for a complex-conjugate transpose.
 * - `const Transpose b_transpose`: Transposing the input matrix B, either `Transpose::kNo` (111), `Transpose::kYes` (112), or `Transpose::kConjugate` (113) for a complex-conjugate transpose.
 * - `const size_t m`: Integer size argument. This value must be positive.
 * - `const size_t n`: Integer size argument. This value must be positive.
 * - `const size_t k`: Integer size argument. This value must be positive.
 * - `const T alpha`: Input scalar constant.
 * - `const cl_mem a_buffer`: OpenCL buffer to store the input A matrix.
 * - `const size_t a_offset`: The offset in elements from the start of the input A matrix.
 * - `const size_t a_ld`: Leading dimension of the input A matrix. This value must be greater than 0.
 * - `const size_t a_stride`: The (fixed) stride between two batches of the A matrix.
 * - `const cl_mem b_buffer`: OpenCL buffer to store the input B matrix.
 * - `const size_t b_offset`: The offset in elements from the start of the input B matrix.
 * - `const size_t b_ld`: Leading dimension of the input B matrix. This value must be greater than 0.
 * - `const size_t b_stride`: The (fixed) stride between two batches of the B matrix.
 * - `const T beta`: Input scalar constant.
 * - `cl_mem c_buffer`: OpenCL buffer to store the output C matrix.
 * - `const size_t c_offset`: The offset in elements from the start of the output C matrix.
 * - `const size_t c_ld`: Leading dimension of the output C matrix. This value must be greater than 0.
 * - `const size_t c_stride`: The (fixed) stride between two batches of the C matrix.
 * - `const size_t batch_count`: Number of batches. This value must be positive.
 * - `cl_command_queue* queue`: Pointer to an OpenCL command queue associated with a context and device to execute the routine on.
 * - `cl_event* event`: Pointer to an OpenCL event to be able to wait for completion of the routine's OpenCL kernel(s). This is an optional argument.
 *
 * Requirements for GEMMSTRIDEDBATCHED:
 * - When `(transpose_a == Transpose::kNo && layout == Layout::kColMajor) || (transpose_a == Transpose::kYes && layout == Layout::kRowMajor)`, then `a_ld` must be at least `m`, otherwise `a_ld` must be at least `k`.
 * - When `(transpose_b == Transpose::kNo && layout == Layout::kColMajor) || (transpose_b == Transpose::kYes && layout == Layout::kRowMajor)`, then `b_ld` must be at least `k`, otherwise `b_ld` must be at least `n`.
 * - The value of `c_ld` must be at least `m`.
 */
extern(System) @nogc nothrow {
  alias da_CLBlastSgemmStridedBatched = CLBlastStatusCode function(const CLBlastLayout layout, const CLBlastTranspose a_transpose, const CLBlastTranspose b_transpose,
      const size_t m, const size_t n, const size_t k,
      const float alpha,
      const cl_mem a_buffer, const size_t a_offset, const size_t a_ld, const size_t a_stride,
      const cl_mem b_buffer, const size_t b_offset, const size_t b_ld, const size_t b_stride,
      const float beta,
      cl_mem c_buffer, const size_t c_offset, const size_t c_ld, const size_t c_stride,
      const size_t batch_count,
      cl_command_queue* queue, cl_event* event);
  alias da_CLBlastDgemmStridedBatched = CLBlastStatusCode function(const CLBlastLayout layout, const CLBlastTranspose a_transpose, const CLBlastTranspose b_transpose,
      const size_t m, const size_t n, const size_t k,
      const double alpha,
      const cl_mem a_buffer, const size_t a_offset, const size_t a_ld, const size_t a_stride,
      const cl_mem b_buffer, const size_t b_offset, const size_t b_ld, const size_t b_stride,
      const double beta,
      cl_mem c_buffer, const size_t c_offset, const size_t c_ld, const size_t c_stride,
      const size_t batch_count,
      cl_command_queue* queue, cl_event* event);
  alias da_CLBlastCgemmStridedBatched = CLBlastStatusCode function(const CLBlastLayout layout, const CLBlastTranspose a_transpose, const CLBlastTranspose b_transpose,
      const size_t m, const size_t n, const size_t k,
      const cl_float2 alpha,
      const cl_mem a_buffer, const size_t a_offset, const size_t a_ld, const size_t a_stride,
      const cl_mem b_buffer, const size_t b_offset, const size_t b_ld, const size_t b_stride,
      const cl_float2 beta,
      cl_mem c_buffer, const size_t c_offset, const size_t c_ld, const size_t c_stride,
      const size_t batch_count,
      cl_command_queue* queue, cl_event* event);
  alias da_CLBlastZgemmStridedBatched = CLBlastStatusCode function(const CLBlastLayout layout, const CLBlastTranspose a_transpose, const CLBlastTranspose b_transpose,
      const size_t m, const size_t n, const size_t k,
      const cl_double2 alpha,
      const cl_mem a_buffer, const size_t a_offset, const size_t a_ld, const size_t a_stride,
      const cl_mem b_buffer, const size_t b_offset, const size_t b_ld, const size_t b_stride,
      const cl_double2 beta,
      cl_mem c_buffer, const size_t c_offset, const size_t c_ld, const size_t c_stride,
      const size_t batch_count,
      cl_command_queue* queue, cl_event* event);
  alias da_CLBlastHgemmStridedBatched = CLBlastStatusCode function(const CLBlastLayout layout, const CLBlastTranspose a_transpose, const CLBlastTranspose b_transpose,
      const size_t m, const size_t n, const size_t k,
      const cl_half alpha,
      const cl_mem a_buffer, const size_t a_offset, const size_t a_ld, const size_t a_stride,
      const cl_mem b_buffer, const size_t b_offset, const size_t b_ld, const size_t b_stride,
      const cl_half beta,
      cl_mem c_buffer, const size_t c_offset, const size_t c_ld, const size_t c_stride,
      const size_t batch_count,
      cl_command_queue* queue, cl_event* event);
}

/**
 * GemmTempBufferSize: Retrieves the size of the temporary buffer for GEMM (auxiliary function)
 *
 * Retrieves the required size of the temporary buffer for the GEMM kernel for specific arguments and for a specific device/platform and tuning parameters. This could be 0 in case no temporary buffer is required. Arguments are similar to those for GEMM.
 *
 * Arguments to GemmTempBufferSize:
 * - `const Layout layout`: Data-layout of the matrices, either `Layout::kRowMajor` (101) for row-major layout or `Layout::kColMajor` (102) for column-major data-layout.
 * - `const Transpose a_transpose`: Transposing the input matrix A, either `Transpose::kNo` (111), `Transpose::kYes` (112), or `Transpose::kConjugate` (113) for a complex-conjugate transpose.
 * - `const Transpose b_transpose`: Transposing the input matrix B, either `Transpose::kNo` (111), `Transpose::kYes` (112), or `Transpose::kConjugate` (113) for a complex-conjugate transpose.
 * - `const size_t m`: Integer size argument. This value must be positive.
 * - `const size_t n`: Integer size argument. This value must be positive.
 * - `const size_t k`: Integer size argument. This value must be positive.
 * - `const size_t a_offset`: The offset in elements from the start of the input A matrix.
 * - `const size_t a_ld`: Leading dimension of the input A matrix. This value must be greater than 0.
 * - `const size_t b_offset`: The offset in elements from the start of the input B matrix.
 * - `const size_t b_ld`: Leading dimension of the input B matrix. This value must be greater than 0.
 * - `const size_t c_offset`: The offset in elements from the start of the output C matrix.
 * - `const size_t c_ld`: Leading dimension of the output C matrix. This value must be greater than 0.
 * - `cl_command_queue* queue`: Pointer to an OpenCL command queue associated with a context and device to execute the routine on.
 * - `size_t& temp_buffer_size`: The result of this function: the required buffer size.
 */
extern(System) @nogc nothrow {
  alias da_CLBlastSGemmTempBufferSize = CLBlastStatusCode function(const CLBlastLayout layout, const CLBlastTranspose a_transpose, const CLBlastTranspose b_transpose,
      const size_t m, const size_t n, const size_t k,
      const size_t a_offset, const size_t a_ld,
      const size_t b_offset, const size_t b_ld,
      const size_t c_offset, const size_t c_ld,
      cl_command_queue* queue, size_t* temp_buffer_size);

  alias da_CLBlastDGemmTempBufferSize = CLBlastStatusCode function(const CLBlastLayout layout, const CLBlastTranspose a_transpose, const CLBlastTranspose b_transpose,
      const size_t m, const size_t n, const size_t k,
      const size_t a_offset, const size_t a_ld,
      const size_t b_offset, const size_t b_ld,
      const size_t c_offset, const size_t c_ld,
      cl_command_queue* queue, size_t* temp_buffer_size);

  alias da_CLBlastCGemmTempBufferSize = CLBlastStatusCode function(const CLBlastLayout layout, const CLBlastTranspose a_transpose, const CLBlastTranspose b_transpose,
      const size_t m, const size_t n, const size_t k,
      const size_t a_offset, const size_t a_ld,
      const size_t b_offset, const size_t b_ld,
      const size_t c_offset, const size_t c_ld,
      cl_command_queue* queue, size_t* temp_buffer_size);

  alias da_CLBlastZGemmTempBufferSize = CLBlastStatusCode function(const CLBlastLayout layout, const CLBlastTranspose a_transpose, const CLBlastTranspose b_transpose,
      const size_t m, const size_t n, const size_t k,
      const size_t a_offset, const size_t a_ld,
      const size_t b_offset, const size_t b_ld,
      const size_t c_offset, const size_t c_ld,
      cl_command_queue* queue, size_t* temp_buffer_size);

  alias da_CLBlastHGemmTempBufferSize = CLBlastStatusCode function(const CLBlastLayout layout, const CLBlastTranspose a_transpose, const CLBlastTranspose b_transpose,
      const size_t m, const size_t n, const size_t k,
      const size_t a_offset, const size_t a_ld,
      const size_t b_offset, const size_t b_ld,
      const size_t c_offset, const size_t c_ld,
      cl_command_queue* queue, size_t* temp_buffer_size);
}

extern(System) @nogc nothrow {
  /**
   * ClearCache: Resets the cache of compiled binaries (auxiliary function)
   *
   * CLBlast stores binaries of compiled kernels into a cache in case the same kernel is used later on for the same device. This cache can be cleared to free up system memory or it can be useful in case of debugging.
   */
  alias da_CLBlastClearCache = CLBlastStatusCode function();

  /**
   * FillCache: Populates the cache of compiled binaries for a specific device (auxiliary function)
   *
   * CLBlast stores binaries of compiled kernels into a cache in case the same kernel is used later on for the same device. This cache is automatically populated whenever a new binary is created. Thus, the first run of a specific kernel could take extra time. For debugging or performance evaluation purposes, it might be useful to populate the cache upfront. This function populates the cache for all kernels in CLBlast for all precisions, but for a specific device only.
   */
  alias da_CLBlastFillCache = CLBlastStatusCode function(const cl_device_id device);

  /**
   * OverrideParameters: Override tuning parameters (auxiliary function)
   *
   * This function overrides tuning parameters for a specific device-precision-kernel combination. The next time the target routine is called it will be re-compiled and use the new parameters. All further times (until `OverrideParameters` is called again) it will load the kernel from the cache and thus continue to use the new parameters. Note that the first time after calling `OverrideParameters` a performance drop can be observable due to the re-compilation of the kernel. See [tuning.md](tuning.md) for more details on which kernel names and parameters are valid.
   */
  alias da_CLBlastOverrideParameters = CLBlastStatusCode function(const cl_device_id device, const char* kernel_name,
      const CLBlastPrecision precision, const size_t num_parameters,
      const char** parameters_names, const size_t* parameters_values);
}

__gshared {
  da_CLBlastSswap CLBlastSswap;
  da_CLBlastDswap CLBlastDswap;
  da_CLBlastCswap CLBlastCswap;
  da_CLBlastZswap CLBlastZswap;
  da_CLBlastHswap CLBlastHswap;
  da_CLBlastSscal CLBlastSscal;
  da_CLBlastDscal CLBlastDscal;
  da_CLBlastCscal CLBlastCscal;
  da_CLBlastZscal CLBlastZscal;
  da_CLBlastHscal CLBlastHscal;
  da_CLBlastScopy CLBlastScopy;
  da_CLBlastDcopy CLBlastDcopy;
  da_CLBlastCcopy CLBlastCcopy;
  da_CLBlastZcopy CLBlastZcopy;
  da_CLBlastHcopy CLBlastHcopy;
  da_CLBlastSaxpy CLBlastSaxpy;
  da_CLBlastDaxpy CLBlastDaxpy;
  da_CLBlastCaxpy CLBlastCaxpy;
  da_CLBlastZaxpy CLBlastZaxpy;
  da_CLBlastHaxpy CLBlastHaxpy;
  da_CLBlastSdot CLBlastSdot;
  da_CLBlastDdot CLBlastDdot;
  da_CLBlastHdot CLBlastHdot;
  da_CLBlastCdotu CLBlastCdotu;
  da_CLBlastZdotu CLBlastZdotu;
  da_CLBlastCdotc CLBlastCdotc;
  da_CLBlastZdotc CLBlastZdotc;
  da_CLBlastSnrm2 CLBlastSnrm2;
  da_CLBlastDnrm2 CLBlastDnrm2;
  da_CLBlastScnrm2 CLBlastScnrm2;
  da_CLBlastDznrm2 CLBlastDznrm2;
  da_CLBlastHnrm2 CLBlastHnrm2;
  da_CLBlastSasum CLBlastSasum;
  da_CLBlastDasum CLBlastDasum;
  da_CLBlastScasum CLBlastScasum;
  da_CLBlastDzasum CLBlastDzasum;
  da_CLBlastHasum CLBlastHasum;
  da_CLBlastSsum CLBlastSsum;
  da_CLBlastDsum CLBlastDsum;
  da_CLBlastScsum CLBlastScsum;
  da_CLBlastDzsum CLBlastDzsum;
  da_CLBlastHsum CLBlastHsum;
  da_CLBlastiSamax CLBlastiSamax;
  da_CLBlastiDamax CLBlastiDamax;
  da_CLBlastiCamax CLBlastiCamax;
  da_CLBlastiZamax CLBlastiZamax;
  da_CLBlastiHamax CLBlastiHamax;
  da_CLBlastiSamin CLBlastiSamin;
  da_CLBlastiDamin CLBlastiDamin;
  da_CLBlastiCamin CLBlastiCamin;
  da_CLBlastiZamin CLBlastiZamin;
  da_CLBlastiHamin CLBlastiHamin;
  da_CLBlastiSmax CLBlastiSmax;
  da_CLBlastiDmax CLBlastiDmax;
  da_CLBlastiCmax CLBlastiCmax;
  da_CLBlastiZmax CLBlastiZmax;
  da_CLBlastiHmax CLBlastiHmax;
  da_CLBlastiSmin CLBlastiSmin;
  da_CLBlastiDmin CLBlastiDmin;
  da_CLBlastiCmin CLBlastiCmin;
  da_CLBlastiZmin CLBlastiZmin;
  da_CLBlastiHmin CLBlastiHmin;
  da_CLBlastSgemv CLBlastSgemv;
  da_CLBlastDgemv CLBlastDgemv;
  da_CLBlastCgemv CLBlastCgemv;
  da_CLBlastZgemv CLBlastZgemv;
  da_CLBlastHgemv CLBlastHgemv;
  da_CLBlastSgbmv CLBlastSgbmv;
  da_CLBlastDgbmv CLBlastDgbmv;
  da_CLBlastCgbmv CLBlastCgbmv;
  da_CLBlastZgbmv CLBlastZgbmv;
  da_CLBlastHgbmv CLBlastHgbmv;
  da_CLBlastChemv CLBlastChemv;
  da_CLBlastZhemv CLBlastZhemv;
  da_CLBlastChbmv CLBlastChbmv;
  da_CLBlastZhbmv CLBlastZhbmv;
  da_CLBlastSsymv CLBlastSsymv;
  da_CLBlastDsymv CLBlastDsymv;
  da_CLBlastHsymv CLBlastHsymv;
  da_CLBlastSsbmv CLBlastSsbmv;
  da_CLBlastDsbmv CLBlastDsbmv;
  da_CLBlastHsbmv CLBlastHsbmv;
  da_CLBlastSspmv CLBlastSspmv;
  da_CLBlastDspmv CLBlastDspmv;
  da_CLBlastHspmv CLBlastHspmv;
  da_CLBlastStrmv CLBlastStrmv;
  da_CLBlastDtrmv CLBlastDtrmv;
  da_CLBlastCtrmv CLBlastCtrmv;
  da_CLBlastZtrmv CLBlastZtrmv;
  da_CLBlastHtrmv CLBlastHtrmv;
  da_CLBlastStbmv CLBlastStbmv;
  da_CLBlastDtbmv CLBlastDtbmv;
  da_CLBlastCtbmv CLBlastCtbmv;
  da_CLBlastZtbmv CLBlastZtbmv;
  da_CLBlastHtbmv CLBlastHtbmv;
  da_CLBlastStpmv CLBlastStpmv;
  da_CLBlastDtpmv CLBlastDtpmv;
  da_CLBlastCtpmv CLBlastCtpmv;
  da_CLBlastZtpmv CLBlastZtpmv;
  da_CLBlastHtpmv CLBlastHtpmv;
  da_CLBlastStrsv CLBlastStrsv;
  da_CLBlastDtrsv CLBlastDtrsv;
  da_CLBlastCtrsv CLBlastCtrsv;
  da_CLBlastZtrsv CLBlastZtrsv;
  da_CLBlastSger CLBlastSger;
  da_CLBlastDger CLBlastDger;
  da_CLBlastHger CLBlastHger;
  da_CLBlastCgeru CLBlastCgeru;
  da_CLBlastZgeru CLBlastZgeru;
  da_CLBlastCgerc CLBlastCgerc;
  da_CLBlastZgerc CLBlastZgerc;
  da_CLBlastCher CLBlastCher;
  da_CLBlastZher CLBlastZher;
  da_CLBlastChpr CLBlastChpr;
  da_CLBlastZhpr CLBlastZhpr;
  da_CLBlastCher2 CLBlastCher2;
  da_CLBlastZher2 CLBlastZher2;
  da_CLBlastChpr2 CLBlastChpr2;
  da_CLBlastZhpr2 CLBlastZhpr2;
  da_CLBlastSsyr CLBlastSsyr;
  da_CLBlastDsyr CLBlastDsyr;
  da_CLBlastHsyr CLBlastHsyr;
  da_CLBlastSspr CLBlastSspr;
  da_CLBlastDspr CLBlastDspr;
  da_CLBlastHspr CLBlastHspr;
  da_CLBlastSsyr2 CLBlastSsyr2;
  da_CLBlastDsyr2 CLBlastDsyr2;
  da_CLBlastHsyr2 CLBlastHsyr2;
  da_CLBlastSspr2 CLBlastSspr2;
  da_CLBlastDspr2 CLBlastDspr2;
  da_CLBlastHspr2 CLBlastHspr2;
  da_CLBlastSgemm CLBlastSgemm;
  da_CLBlastDgemm CLBlastDgemm;
  da_CLBlastCgemm CLBlastCgemm;
  da_CLBlastZgemm CLBlastZgemm;
  da_CLBlastHgemm CLBlastHgemm;
  da_CLBlastSsymm CLBlastSsymm;
  da_CLBlastDsymm CLBlastDsymm;
  da_CLBlastCsymm CLBlastCsymm;
  da_CLBlastZsymm CLBlastZsymm;
  da_CLBlastHsymm CLBlastHsymm;
  da_CLBlastChemm CLBlastChemm;
  da_CLBlastZhemm CLBlastZhemm;
  da_CLBlastSsyrk CLBlastSsyrk;
  da_CLBlastDsyrk CLBlastDsyrk;
  da_CLBlastCsyrk CLBlastCsyrk;
  da_CLBlastZsyrk CLBlastZsyrk;
  da_CLBlastHsyrk CLBlastHsyrk;
  da_CLBlastCherk CLBlastCherk;
  da_CLBlastZherk CLBlastZherk;
  da_CLBlastSsyr2k CLBlastSsyr2k;
  da_CLBlastDsyr2k CLBlastDsyr2k;
  da_CLBlastCsyr2k CLBlastCsyr2k;
  da_CLBlastZsyr2k CLBlastZsyr2k;
  da_CLBlastHsyr2k CLBlastHsyr2k;
  da_CLBlastCher2k CLBlastCher2k;
  da_CLBlastZher2k CLBlastZher2k;
  da_CLBlastStrmm CLBlastStrmm;
  da_CLBlastDtrmm CLBlastDtrmm;
  da_CLBlastCtrmm CLBlastCtrmm;
  da_CLBlastZtrmm CLBlastZtrmm;
  da_CLBlastHtrmm CLBlastHtrmm;
  da_CLBlastStrsm CLBlastStrsm;
  da_CLBlastDtrsm CLBlastDtrsm;
  da_CLBlastCtrsm CLBlastCtrsm;
  da_CLBlastZtrsm CLBlastZtrsm;
  da_CLBlastShad CLBlastShad;
  da_CLBlastDhad CLBlastDhad;
  da_CLBlastChad CLBlastChad;
  da_CLBlastZhad CLBlastZhad;
  da_CLBlastHhad CLBlastHhad;
  da_CLBlastSomatcopy CLBlastSomatcopy;
  da_CLBlastDomatcopy CLBlastDomatcopy;
  da_CLBlastComatcopy CLBlastComatcopy;
  da_CLBlastZomatcopy CLBlastZomatcopy;
  da_CLBlastHomatcopy CLBlastHomatcopy;
  da_CLBlastSim2col CLBlastSim2col;
  da_CLBlastDim2col CLBlastDim2col;
  da_CLBlastCim2col CLBlastCim2col;
  da_CLBlastZim2col CLBlastZim2col;
  da_CLBlastHim2col CLBlastHim2col;
  da_CLBlastScol2im CLBlastScol2im;
  da_CLBlastDcol2im CLBlastDcol2im;
  da_CLBlastCcol2im CLBlastCcol2im;
  da_CLBlastZcol2im CLBlastZcol2im;
  da_CLBlastHcol2im CLBlastHcol2im;
  da_CLBlastSconvgemm CLBlastSconvgemm;
  da_CLBlastDconvgemm CLBlastDconvgemm;
  da_CLBlastHconvgemm CLBlastHconvgemm;
  da_CLBlastSaxpyBatched CLBlastSaxpyBatched;
  da_CLBlastDaxpyBatched CLBlastDaxpyBatched;
  da_CLBlastCaxpyBatched CLBlastCaxpyBatched;
  da_CLBlastZaxpyBatched CLBlastZaxpyBatched;
  da_CLBlastHaxpyBatched CLBlastHaxpyBatched;
  da_CLBlastSgemmBatched CLBlastSgemmBatched;
  da_CLBlastDgemmBatched CLBlastDgemmBatched;
  da_CLBlastCgemmBatched CLBlastCgemmBatched;
  da_CLBlastZgemmBatched CLBlastZgemmBatched;
  da_CLBlastHgemmBatched CLBlastHgemmBatched;
  da_CLBlastSgemmStridedBatched CLBlastSgemmStridedBatched;
  da_CLBlastDgemmStridedBatched CLBlastDgemmStridedBatched;
  da_CLBlastCgemmStridedBatched CLBlastCgemmStridedBatched;
  da_CLBlastZgemmStridedBatched CLBlastZgemmStridedBatched;
  da_CLBlastHgemmStridedBatched CLBlastHgemmStridedBatched;
  da_CLBlastSGemmTempBufferSize CLBlastSGemmTempBufferSize;
  da_CLBlastDGemmTempBufferSize CLBlastDGemmTempBufferSize;
  da_CLBlastCGemmTempBufferSize CLBlastCGemmTempBufferSize;
  da_CLBlastZGemmTempBufferSize CLBlastZGemmTempBufferSize;
  da_CLBlastHGemmTempBufferSize CLBlastHGemmTempBufferSize;
  da_CLBlastClearCache CLBlastClearCache;
  da_CLBlastFillCache CLBlastFillCache;
  da_CLBlastOverrideParameters CLBlastOverrideParameters;
}
