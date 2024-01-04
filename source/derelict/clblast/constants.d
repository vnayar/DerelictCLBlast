module derelict.clblast.constants;

/**
 * Status codes. These codes can be returned by functions declared in this header file. The error
 * codes match either the standard OpenCL error codes or the clBLAS error codes.
 */
enum CLBlastStatusCode {
  // Status codes in common with the OpenCL standard
  Success                   =   0, // CL_SUCCESS
  OpenCLCompilerNotAvailable=  -3, // CL_COMPILER_NOT_AVAILABLE
  TempBufferAllocFailure    =  -4, // CL_MEM_OBJECT_ALLOCATION_FAILURE
  OpenCLOutOfResources      =  -5, // CL_OUT_OF_RESOURCES
  OpenCLOutOfHostMemory     =  -6, // CL_OUT_OF_HOST_MEMORY
  OpenCLBuildProgramFailure = -11, // CL_BUILD_PROGRAM_FAILURE: OpenCL compilation error
  InvalidValue              = -30, // CL_INVALID_VALUE
  InvalidCommandQueue       = -36, // CL_INVALID_COMMAND_QUEUE
  InvalidMemObject          = -38, // CL_INVALID_MEM_OBJECT
  InvalidBinary             = -42, // CL_INVALID_BINARY
  InvalidBuildOptions       = -43, // CL_INVALID_BUILD_OPTIONS
  InvalidProgram            = -44, // CL_INVALID_PROGRAM
  InvalidProgramExecutable  = -45, // CL_INVALID_PROGRAM_EXECUTABLE
  InvalidKernelName         = -46, // CL_INVALID_KERNEL_NAME
  InvalidKernelDefinition   = -47, // CL_INVALID_KERNEL_DEFINITION
  InvalidKernel             = -48, // CL_INVALID_KERNEL
  InvalidArgIndex           = -49, // CL_INVALID_ARG_INDEX
  InvalidArgValue           = -50, // CL_INVALID_ARG_VALUE
  InvalidArgSize            = -51, // CL_INVALID_ARG_SIZE
  InvalidKernelArgs         = -52, // CL_INVALID_KERNEL_ARGS
  InvalidLocalNumDimensions = -53, // CL_INVALID_WORK_DIMENSION: Too many thread dimensions
  InvalidLocalThreadsTotal  = -54, // CL_INVALID_WORK_GROUP_SIZE: Too many threads in total
  InvalidLocalThreadsDim    = -55, // CL_INVALID_WORK_ITEM_SIZE: ... or for a specific dimension
  InvalidGlobalOffset       = -56, // CL_INVALID_GLOBAL_OFFSET
  InvalidEventWaitList      = -57, // CL_INVALID_EVENT_WAIT_LIST
  InvalidEvent              = -58, // CL_INVALID_EVENT
  InvalidOperation          = -59, // CL_INVALID_OPERATION
  InvalidBufferSize         = -61, // CL_INVALID_BUFFER_SIZE
  InvalidGlobalWorkSize     = -63, // CL_INVALID_GLOBAL_WORK_SIZE

  // Status codes in common with the clBLAS library
  NotImplemented            = -1024, // Routine or functionality not implemented yet
  InvalidMatrixA            = -1022, // Matrix A is not a valid OpenCL buffer
  InvalidMatrixB            = -1021, // Matrix B is not a valid OpenCL buffer
  InvalidMatrixC            = -1020, // Matrix C is not a valid OpenCL buffer
  InvalidVectorX            = -1019, // Vector X is not a valid OpenCL buffer
  InvalidVectorY            = -1018, // Vector Y is not a valid OpenCL buffer
  InvalidDimension          = -1017, // Dimensions M, N, and K have to be larger than zero
  InvalidLeadDimA           = -1016, // LD of A is smaller than the matrix's first dimension
  InvalidLeadDimB           = -1015, // LD of B is smaller than the matrix's first dimension
  InvalidLeadDimC           = -1014, // LD of C is smaller than the matrix's first dimension
  InvalidIncrementX         = -1013, // Increment of vector X cannot be zero
  InvalidIncrementY         = -1012, // Increment of vector Y cannot be zero
  InsufficientMemoryA       = -1011, // Matrix A's OpenCL buffer is too small
  InsufficientMemoryB       = -1010, // Matrix B's OpenCL buffer is too small
  InsufficientMemoryC       = -1009, // Matrix C's OpenCL buffer is too small
  InsufficientMemoryX       = -1008, // Vector X's OpenCL buffer is too small
  InsufficientMemoryY       = -1007, // Vector Y's OpenCL buffer is too small

  // Custom additional status codes for CLBlast
  InsufficientMemoryTemp    = -2050, // Temporary buffer provided to GEMM routine is too small
  InvalidBatchCount         = -2049, // The batch count needs to be positive
  InvalidOverrideKernel     = -2048, // Trying to override parameters for an invalid kernel
  MissingOverrideParameter  = -2047, // Missing override parameter(s) for the target kernel
  InvalidLocalMemUsage      = -2046, // Not enough local memory available on this device
  NoHalfPrecision           = -2045, // Half precision (16-bits) not supported by the device
  NoDoublePrecision         = -2044, // Double precision (64-bits) not supported by the device
  InvalidVectorScalar       = -2043, // The unit-sized vector is not a valid OpenCL buffer
  InsufficientMemoryScalar  = -2042, // The unit-sized vector's OpenCL buffer is too small
  DatabaseError             = -2041, // Entry for the device was not found in the database
  UnknownError              = -2040, // A catch-all error code representing an unspecified error
  UnexpectedError           = -2039, // A catch-all error code representing an unexpected exception
}

// Matrix layout and transpose types
enum CLBlastLayout {
  RowMajor = 101,
  ColMajor = 102
}

enum CLBlastTranspose {
  No = 111,
  Yes = 112,
  Conjugate = 113
}

enum CLBlastTriangle {
  Upper = 121,
  Lower = 122
}

enum CLBlastDiagonal {
  NonUnit = 131,
  Unit = 132
}

enum CLBlastSide {
  Left = 141,
  Right = 142
}

enum CLBlastKernelMode {
  CrossCorrelation = 151,
  Convolution = 152
}

// Precision enum (values in bits)
enum CLBlastPrecision {
  Half = 16,
  Single = 32,
  Double = 64,
  ComplexSingle = 3232,
  ComplexDouble = 6464
}
