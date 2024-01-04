module derelict.clblast.types;

import derelict.opencl.types : cl_float, cl_double;

union cl_float2 {
  cl_float[2] s;
  struct { cl_float x, y; }
  struct { cl_float  s0, s1; }
  struct { cl_float  lo, hi; }
}

union cl_double2 {
  cl_double[2] s;
  struct { cl_double x, y; }
  struct { cl_double  s0, s1; }
  struct { cl_double  lo, hi; }
}
