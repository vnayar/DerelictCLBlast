import std.stdio;

import std.format : format;
import std.regex : matchFirst;
import std.conv : to;

// Imports related to OpenCL itself.
import derelict.opencl.types : CLVersion, cl_int, cl_uint, cl_platform_id, cl_device_id;
import derelict.opencl.cl : DerelictCL;
import derelict.opencl.functions :
    clGetPlatformIDs, clGetPlatformInfo,
  clGetDeviceIDs, clGetDeviceInfo;
import derelict.opencl.constants : CL_SUCCESS,
    CL_PLATFORM_NAME, CL_PLATFORM_VERSION,
    CL_DEVICE_TYPE_ALL, CL_DEVICE_NAME, CL_DEVICE_MAX_COMPUTE_UNITS, CL_DEVICE_MAX_CLOCK_FREQUENCY;
import derelict.clblast.clblast : DerelictCLBlast;

void main()
{
  // Load the OpenCL library.
  DerelictCL.load();
  // Now load CLBlast (which uses OpenCL).
  DerelictCLBlast.load();

  // Load vendor-specific platforms supporting devices.
  CLPlatform[] platforms = loadCLPlatforms();
  displayCLPlatforms(platforms);

  // For now, assume the first platform and the first device.
  CLPlatform activePlatform = platforms[0];
  CLDevice activeDevice = activePlatform.devices[0];

  // Reload the OpenCL library.
  DerelictCL.reload(activePlatform.clVersion);

  // Load OpenCL official extensions.
  DerelictCL.loadEXT(activePlatform.id);

  // Now OpenCL functions can be called.
  testCLBlastGemm(activeDevice);
}

/**
 * An example of loading data from the host onto the GPU, executing operations, and retrieving the
 * result.
 */
void testCLBlastGemm(CLDevice activeDevice) {
  import derelict.opencl.constants : CL_TRUE, CL_MEM_READ_WRITE;
  import derelict.opencl.types : cl_context, cl_command_queue, cl_event, cl_mem;
  import derelict.opencl.functions : clCreateContext, clCreateCommandQueue, clEnqueueWriteBuffer,
      clEnqueueReadBuffer, clReleaseEvent, clWaitForEvents, clReleaseContext, clReleaseCommandQueue,
      clReleaseMemObject, clCreateBuffer;
  // Additional imports for CLBlast.
  import derelict.clblast.functions : CLBlastSgemm;
  import derelict.clblast.constants : CLBlastStatusCode, CLBlastLayout, CLBlastTranspose;

  // Based on the example program here:
  // https://github.com/CNugteren/CLBlast/blob/master/samples/sgemm.c

  // Example SGEMM arguments
  const size_t m = 128;
  const size_t n = 64;
  const size_t k = 512;
  const float alpha = 0.7f;
  const float beta = 1.0f;
  const size_t a_ld = k;
  const size_t b_ld = n;
  const size_t c_ld = n;

  // Creates the OpenCL context, queue, and an event
  cl_context context = clCreateContext(null, 1, &activeDevice.id, null, null, null);
  scope(exit) clReleaseContext(context);
  cl_command_queue queue = clCreateCommandQueue(context, activeDevice.id, 0, null);
  scope(exit) clReleaseCommandQueue(queue);
  cl_event event = null;

  // Populate host matrices with some example data
  float[] host_a = new float[](m*k);
  float[] host_b = new float[](n*k);
  float[] host_c = new float[](m*n);
  foreach (ref a; host_a) { a = 12.193f; }
  foreach (ref b; host_b) { b = -8.199f; }
  foreach (ref c; host_c) { c = 0.0f; }

  // Create memory buffers on the compute device.
  cl_mem device_a = clCreateBuffer(context, CL_MEM_READ_WRITE, m*k*float.sizeof, null, null);
  scope(exit) clReleaseMemObject(device_a);
  cl_mem device_b = clCreateBuffer(context, CL_MEM_READ_WRITE, n*k*float.sizeof, null, null);
  scope(exit) clReleaseMemObject(device_b);
  cl_mem device_c = clCreateBuffer(context, CL_MEM_READ_WRITE, m*n*float.sizeof, null, null);
  scope(exit) clReleaseMemObject(device_c);

  // Copy the matrices to the memory buffers on the compute device
  clEnqueueWriteBuffer(queue, device_a, CL_TRUE, 0, m*k*float.sizeof, host_a.ptr, 0, null, null);
  clEnqueueWriteBuffer(queue, device_b, CL_TRUE, 0, n*k*float.sizeof, host_b.ptr, 0, null, null);
  clEnqueueWriteBuffer(queue, device_c, CL_TRUE, 0, m*n*float.sizeof, host_c.ptr, 0, null, null);

  // Call the SGEMM routine, performing C = alpha*A*X + beta*C
  // A is an (m x k) matrix, B is an (k x n) matrix, C is an (m x n) matrix
  // See https://github.com/CNugteren/CLBlast/blob/master/doc/api.md#xgemm-general-matrix-matrix-multiplication
  CLBlastStatusCode status = CLBlastSgemm(CLBlastLayout.RowMajor,
                                          CLBlastTranspose.No, CLBlastTranspose.No,
                                          m, n, k,
                                          alpha,
                                          device_a, 0, a_ld,
                                          device_b, 0, b_ld,
                                          beta,
                                          device_c, 0, c_ld,
                                          &queue, &event);

  // Copy the result back to the host.
  float[] result = new float[](m*n);
  clEnqueueReadBuffer(queue, device_c, CL_TRUE, 0, m*n*float.sizeof, result.ptr, 0, null, null);

  // Wait for completion
  if (status == CLBlastStatusCode.Success) {
    clWaitForEvents(1, &event);
    clReleaseEvent(event);
  }

  // Print a piece of the result.
  writeln("First row of result: ", result[0..n]);

  // Example completed. See "clblast_c.h" for status codes (0 -> success).
  printf("Completed SGEMM with status %d\n", status);
}

/**
 * A vendor OpenCL driver implementation, such as PoCL (Portable Computing Language), and several
 * devices, such as CPUs or GPUs, that may be used via that platform.
 */
struct CLPlatform {
  cl_platform_id id;

  /// Typically the name of the OpenCL vendor, e.g. "Portable Computing Language".
  string name;

  /**
   * The version of OpenCL supported by the combination of OpenCL driver and device.
   */
  CLVersion clVersion;
  CLDevice[] devices;
}

/**
 * A physical device able to execute paralle computations, such as CPUs, GPUs, FPGAs, or other
 * devices. Devices typically allow kernels (mini-programs to execute in parallel on
 * multi-dimentional arrays) to be loaded, and executed on data that is enqueued into a command
 * queues. A single device may offer multiple command queues.
 */
struct CLDevice {
  cl_device_id id;

  /**
   * Typically the name of the hardware vendor/model,
   * e.g. pthread-Intel(R) Core(TM) i7-3630QM CPU @ 2.40GHz
   */
  string name;

  /**
   * Each computeUnit is a group of 1 or more processors that may efficiently communicate with
   * another through shared caches, optimized busses, shared instruction caches, etc.
   */
  cl_uint maxComputeUnits;

  /**
   * The maximum clock speed of each processor in a computeUnit in MHz.
   */
  cl_uint maxClockFrequency;
}

/**
 * Dynamically loads the OpenCL (Open Compute Library), and detects which local platforms are
 * available with OpenCL drivers. If drivers are installed, OpenCL can execute on a CPU, FPGA, and
 * numerous GPUs from NVidia, AMD, ATI, Intel, and others.
 *
 * Similar information can be obtained from the `clinfo` shell command.
 */
CLPlatform[] loadCLPlatforms() {
  // Query platforms and devices
  cl_int CL_err = CL_SUCCESS;
  cl_uint numPlatforms = 0;
  cl_uint maxPlatforms = 10;  // TODO: Hard-coded max 10.
  cl_platform_id[] platformIds = new cl_platform_id[](maxPlatforms);
  CL_err = clGetPlatformIDs(maxPlatforms, platformIds.ptr, &numPlatforms);

  if (CL_err != CL_SUCCESS) {
    throw new Exception(format("OpenCL Error: method=%s, code=%d", "clGetPlatformIds", CL_err));
  }

  CLPlatform[] platforms;

  // Display informative information about the available platforms.
  // https://registry.khronos.org/OpenCL/sdk/1.0/docs/man/xhtml/clGetPlatformInfo.html
  for (int i = 0; i < numPlatforms; i++) {
    CLPlatform platform;
    platform.id = platformIds[i];

    char[100] buf;
    size_t bufSize;
    CL_err = clGetPlatformInfo(platformIds[i], CL_PLATFORM_NAME, buf.sizeof, buf.ptr, &bufSize);
    platform.name = buf[0..bufSize-1].dup;

    // The returned version has format:
    // OpenCL<space><major_version.minor_version><space><platform-specific information>
    CL_err = clGetPlatformInfo(platformIds[i], CL_PLATFORM_VERSION, buf.sizeof, buf.ptr, &bufSize);
    auto captures = matchFirst(
        buf[0..bufSize-1],
        `^OpenCL (?P<majorVersion>\d+)\.(?P<minorVersion>\d+) (?P<platformInfo>.*)$`);
    platform.clVersion = cast(CLVersion)(
        captures["majorVersion"].to!int * 10 + captures["minorVersion"].to!int);

    // Get information about devices on the platform.
    // https://registry.khronos.org/OpenCL/sdk/1.0/docs/man/xhtml/clGetDeviceIDs.html
    cl_uint maxDevices = 10;
    cl_uint numDevices = 0;
    cl_device_id[] deviceIds = new cl_device_id[](maxDevices);
    CL_err = clGetDeviceIDs(platformIds[i], CL_DEVICE_TYPE_ALL, maxDevices, deviceIds.ptr, &numDevices);
    if (CL_err != CL_SUCCESS) {
      throw new Exception(format("OpenCL Error: method=%s, code=%d", "clGetDeviceIDs", CL_err));
    }
    for (int j = 0; j < numDevices; j++) {
      CLDevice device;
      device.id = deviceIds[j];

      // https://registry.khronos.org/OpenCL/sdk/1.0/docs/man/xhtml/clGetDeviceInfo.html
      CL_err = clGetDeviceInfo(
          deviceIds[j], CL_DEVICE_NAME, buf.sizeof, buf.ptr, &bufSize);
      device.name = buf[0..bufSize-1].dup;

      CL_err = clGetDeviceInfo(
          deviceIds[j], CL_DEVICE_MAX_COMPUTE_UNITS, device.maxComputeUnits.sizeof,
          &device.maxComputeUnits, null);

      CL_err = clGetDeviceInfo(
          deviceIds[j], CL_DEVICE_MAX_CLOCK_FREQUENCY, device.maxClockFrequency.sizeof,
          &device.maxClockFrequency, null);

      platform.devices ~= device;
    }

    platforms ~= platform;
  }
  return platforms;
}

void displayCLPlatforms(CLPlatform[] platforms) {
  writeln(platforms.length, " platform(s) found");
  foreach (i, platform; platforms) {
    writeln("  Platform[", i, "].id = ", platform.id);
    writeln("  Platform[", i, "].name = ", platform.name);
    writeln("  Platform[", i, "].clVersion = ", platform.clVersion);
    writeln("  ", platform.devices.length, " device(s) found");
    foreach (j, device; platform.devices) {
      writeln("    Device[", j, "].id = ", device.id);
      writeln("    Device[", j, "].name = ", device.name);
      writeln("    Device[", j, "].maxComputeUnits = ", device.maxComputeUnits);
      writeln("    Device[", j, "].maxClockFrequency = ", device.maxClockFrequency);
    }
  }
}

