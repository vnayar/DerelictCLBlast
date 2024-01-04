module derelict.clblast.clblast;

import derelict.util.loader;
import derelict.clblast.functions;

private {
  import derelict.util.system;

  // See official binary releases: https://github.com/CNugteren/CLBlast/releases
  static if (Derelict_OS_Windows) {
    enum libNames = "clblast.dll";
  } else static if (Derelict_OS_Mac) {
    enum libNames = "libclblast.1.dylib,libclblast.1.dylib";
  } else static if (Derelict_OS_Posix) {
    enum libNames = "libclblast.so.1,libclblast.so";
  } else {
    static assert(0, "Need to implement OpenCL libNames for this operating system.");
  }
}


class DerelictCLBlastLoader : SharedLibLoader {
protected:
  override void loadSymbols() {
    bindFunc(cast(void**)&CLBlastSswap, "CLBlastSswap");
    bindFunc(cast(void**)&CLBlastDswap, "CLBlastDswap");
    bindFunc(cast(void**)&CLBlastCswap, "CLBlastCswap");
    bindFunc(cast(void**)&CLBlastZswap, "CLBlastZswap");
    bindFunc(cast(void**)&CLBlastHswap, "CLBlastHswap");
    bindFunc(cast(void**)&CLBlastSscal, "CLBlastSscal");
    bindFunc(cast(void**)&CLBlastDscal, "CLBlastDscal");
    bindFunc(cast(void**)&CLBlastCscal, "CLBlastCscal");
    bindFunc(cast(void**)&CLBlastZscal, "CLBlastZscal");
    bindFunc(cast(void**)&CLBlastHscal, "CLBlastHscal");
    bindFunc(cast(void**)&CLBlastScopy, "CLBlastScopy");
    bindFunc(cast(void**)&CLBlastDcopy, "CLBlastDcopy");
    bindFunc(cast(void**)&CLBlastCcopy, "CLBlastCcopy");
    bindFunc(cast(void**)&CLBlastZcopy, "CLBlastZcopy");
    bindFunc(cast(void**)&CLBlastHcopy, "CLBlastHcopy");
    bindFunc(cast(void**)&CLBlastSaxpy, "CLBlastSaxpy");
    bindFunc(cast(void**)&CLBlastDaxpy, "CLBlastDaxpy");
    bindFunc(cast(void**)&CLBlastCaxpy, "CLBlastCaxpy");
    bindFunc(cast(void**)&CLBlastZaxpy, "CLBlastZaxpy");
    bindFunc(cast(void**)&CLBlastHaxpy, "CLBlastHaxpy");
    bindFunc(cast(void**)&CLBlastSdot, "CLBlastSdot");
    bindFunc(cast(void**)&CLBlastDdot, "CLBlastDdot");
    bindFunc(cast(void**)&CLBlastHdot, "CLBlastHdot");
    bindFunc(cast(void**)&CLBlastCdotu, "CLBlastCdotu");
    bindFunc(cast(void**)&CLBlastZdotu, "CLBlastZdotu");
    bindFunc(cast(void**)&CLBlastCdotc, "CLBlastCdotc");
    bindFunc(cast(void**)&CLBlastZdotc, "CLBlastZdotc");
    bindFunc(cast(void**)&CLBlastSnrm2, "CLBlastSnrm2");
    bindFunc(cast(void**)&CLBlastDnrm2, "CLBlastDnrm2");
    bindFunc(cast(void**)&CLBlastScnrm2, "CLBlastScnrm2");
    bindFunc(cast(void**)&CLBlastDznrm2, "CLBlastDznrm2");
    bindFunc(cast(void**)&CLBlastHnrm2, "CLBlastHnrm2");
    bindFunc(cast(void**)&CLBlastSasum, "CLBlastSasum");
    bindFunc(cast(void**)&CLBlastDasum, "CLBlastDasum");
    bindFunc(cast(void**)&CLBlastScasum, "CLBlastScasum");
    bindFunc(cast(void**)&CLBlastDzasum, "CLBlastDzasum");
    bindFunc(cast(void**)&CLBlastHasum, "CLBlastHasum");
    bindFunc(cast(void**)&CLBlastSsum, "CLBlastSsum");
    bindFunc(cast(void**)&CLBlastDsum, "CLBlastDsum");
    bindFunc(cast(void**)&CLBlastScsum, "CLBlastScsum");
    bindFunc(cast(void**)&CLBlastDzsum, "CLBlastDzsum");
    bindFunc(cast(void**)&CLBlastHsum, "CLBlastHsum");
    bindFunc(cast(void**)&CLBlastiSamax, "CLBlastiSamax");
    bindFunc(cast(void**)&CLBlastiDamax, "CLBlastiDamax");
    bindFunc(cast(void**)&CLBlastiCamax, "CLBlastiCamax");
    bindFunc(cast(void**)&CLBlastiZamax, "CLBlastiZamax");
    bindFunc(cast(void**)&CLBlastiHamax, "CLBlastiHamax");
    bindFunc(cast(void**)&CLBlastiSamin, "CLBlastiSamin");
    bindFunc(cast(void**)&CLBlastiDamin, "CLBlastiDamin");
    bindFunc(cast(void**)&CLBlastiCamin, "CLBlastiCamin");
    bindFunc(cast(void**)&CLBlastiZamin, "CLBlastiZamin");
    bindFunc(cast(void**)&CLBlastiHamin, "CLBlastiHamin");
    bindFunc(cast(void**)&CLBlastiSmax, "CLBlastiSmax");
    bindFunc(cast(void**)&CLBlastiDmax, "CLBlastiDmax");
    bindFunc(cast(void**)&CLBlastiCmax, "CLBlastiCmax");
    bindFunc(cast(void**)&CLBlastiZmax, "CLBlastiZmax");
    bindFunc(cast(void**)&CLBlastiHmax, "CLBlastiHmax");
    bindFunc(cast(void**)&CLBlastiSmin, "CLBlastiSmin");
    bindFunc(cast(void**)&CLBlastiDmin, "CLBlastiDmin");
    bindFunc(cast(void**)&CLBlastiCmin, "CLBlastiCmin");
    bindFunc(cast(void**)&CLBlastiZmin, "CLBlastiZmin");
    bindFunc(cast(void**)&CLBlastiHmin, "CLBlastiHmin");
    bindFunc(cast(void**)&CLBlastSgemv, "CLBlastSgemv");
    bindFunc(cast(void**)&CLBlastDgemv, "CLBlastDgemv");
    bindFunc(cast(void**)&CLBlastCgemv, "CLBlastCgemv");
    bindFunc(cast(void**)&CLBlastZgemv, "CLBlastZgemv");
    bindFunc(cast(void**)&CLBlastHgemv, "CLBlastHgemv");
    bindFunc(cast(void**)&CLBlastSgbmv, "CLBlastSgbmv");
    bindFunc(cast(void**)&CLBlastDgbmv, "CLBlastDgbmv");
    bindFunc(cast(void**)&CLBlastCgbmv, "CLBlastCgbmv");
    bindFunc(cast(void**)&CLBlastZgbmv, "CLBlastZgbmv");
    bindFunc(cast(void**)&CLBlastHgbmv, "CLBlastHgbmv");
    bindFunc(cast(void**)&CLBlastChemv, "CLBlastChemv");
    bindFunc(cast(void**)&CLBlastZhemv, "CLBlastZhemv");
    bindFunc(cast(void**)&CLBlastChbmv, "CLBlastChbmv");
    bindFunc(cast(void**)&CLBlastZhbmv, "CLBlastZhbmv");
    bindFunc(cast(void**)&CLBlastSsymv, "CLBlastSsymv");
    bindFunc(cast(void**)&CLBlastDsymv, "CLBlastDsymv");
    bindFunc(cast(void**)&CLBlastHsymv, "CLBlastHsymv");
    bindFunc(cast(void**)&CLBlastSsbmv, "CLBlastSsbmv");
    bindFunc(cast(void**)&CLBlastDsbmv, "CLBlastDsbmv");
    bindFunc(cast(void**)&CLBlastHsbmv, "CLBlastHsbmv");
    bindFunc(cast(void**)&CLBlastSspmv, "CLBlastSspmv");
    bindFunc(cast(void**)&CLBlastDspmv, "CLBlastDspmv");
    bindFunc(cast(void**)&CLBlastHspmv, "CLBlastHspmv");
    bindFunc(cast(void**)&CLBlastStrmv, "CLBlastStrmv");
    bindFunc(cast(void**)&CLBlastDtrmv, "CLBlastDtrmv");
    bindFunc(cast(void**)&CLBlastCtrmv, "CLBlastCtrmv");
    bindFunc(cast(void**)&CLBlastZtrmv, "CLBlastZtrmv");
    bindFunc(cast(void**)&CLBlastHtrmv, "CLBlastHtrmv");
    bindFunc(cast(void**)&CLBlastStbmv, "CLBlastStbmv");
    bindFunc(cast(void**)&CLBlastDtbmv, "CLBlastDtbmv");
    bindFunc(cast(void**)&CLBlastCtbmv, "CLBlastCtbmv");
    bindFunc(cast(void**)&CLBlastZtbmv, "CLBlastZtbmv");
    bindFunc(cast(void**)&CLBlastHtbmv, "CLBlastHtbmv");
    bindFunc(cast(void**)&CLBlastStpmv, "CLBlastStpmv");
    bindFunc(cast(void**)&CLBlastDtpmv, "CLBlastDtpmv");
    bindFunc(cast(void**)&CLBlastCtpmv, "CLBlastCtpmv");
    bindFunc(cast(void**)&CLBlastZtpmv, "CLBlastZtpmv");
    bindFunc(cast(void**)&CLBlastHtpmv, "CLBlastHtpmv");
    bindFunc(cast(void**)&CLBlastStrsv, "CLBlastStrsv");
    bindFunc(cast(void**)&CLBlastDtrsv, "CLBlastDtrsv");
    bindFunc(cast(void**)&CLBlastCtrsv, "CLBlastCtrsv");
    bindFunc(cast(void**)&CLBlastZtrsv, "CLBlastZtrsv");
    bindFunc(cast(void**)&CLBlastSger, "CLBlastSger");
    bindFunc(cast(void**)&CLBlastDger, "CLBlastDger");
    bindFunc(cast(void**)&CLBlastHger, "CLBlastHger");
    bindFunc(cast(void**)&CLBlastCgeru, "CLBlastCgeru");
    bindFunc(cast(void**)&CLBlastZgeru, "CLBlastZgeru");
    bindFunc(cast(void**)&CLBlastCgerc, "CLBlastCgerc");
    bindFunc(cast(void**)&CLBlastZgerc, "CLBlastZgerc");
    bindFunc(cast(void**)&CLBlastCher, "CLBlastCher");
    bindFunc(cast(void**)&CLBlastZher, "CLBlastZher");
    bindFunc(cast(void**)&CLBlastChpr, "CLBlastChpr");
    bindFunc(cast(void**)&CLBlastZhpr, "CLBlastZhpr");
    bindFunc(cast(void**)&CLBlastCher2, "CLBlastCher2");
    bindFunc(cast(void**)&CLBlastZher2, "CLBlastZher2");
    bindFunc(cast(void**)&CLBlastChpr2, "CLBlastChpr2");
    bindFunc(cast(void**)&CLBlastZhpr2, "CLBlastZhpr2");
    bindFunc(cast(void**)&CLBlastSsyr, "CLBlastSsyr");
    bindFunc(cast(void**)&CLBlastDsyr, "CLBlastDsyr");
    bindFunc(cast(void**)&CLBlastHsyr, "CLBlastHsyr");
    bindFunc(cast(void**)&CLBlastSspr, "CLBlastSspr");
    bindFunc(cast(void**)&CLBlastDspr, "CLBlastDspr");
    bindFunc(cast(void**)&CLBlastHspr, "CLBlastHspr");
    bindFunc(cast(void**)&CLBlastSsyr2, "CLBlastSsyr2");
    bindFunc(cast(void**)&CLBlastDsyr2, "CLBlastDsyr2");
    bindFunc(cast(void**)&CLBlastHsyr2, "CLBlastHsyr2");
    bindFunc(cast(void**)&CLBlastSspr2, "CLBlastSspr2");
    bindFunc(cast(void**)&CLBlastDspr2, "CLBlastDspr2");
    bindFunc(cast(void**)&CLBlastHspr2, "CLBlastHspr2");
    bindFunc(cast(void**)&CLBlastSgemm, "CLBlastSgemm");
    bindFunc(cast(void**)&CLBlastDgemm, "CLBlastDgemm");
    bindFunc(cast(void**)&CLBlastCgemm, "CLBlastCgemm");
    bindFunc(cast(void**)&CLBlastZgemm, "CLBlastZgemm");
    bindFunc(cast(void**)&CLBlastHgemm, "CLBlastHgemm");
    bindFunc(cast(void**)&CLBlastSsymm, "CLBlastSsymm");
    bindFunc(cast(void**)&CLBlastDsymm, "CLBlastDsymm");
    bindFunc(cast(void**)&CLBlastCsymm, "CLBlastCsymm");
    bindFunc(cast(void**)&CLBlastZsymm, "CLBlastZsymm");
    bindFunc(cast(void**)&CLBlastHsymm, "CLBlastHsymm");
    bindFunc(cast(void**)&CLBlastChemm, "CLBlastChemm");
    bindFunc(cast(void**)&CLBlastZhemm, "CLBlastZhemm");
    bindFunc(cast(void**)&CLBlastSsyrk, "CLBlastSsyrk");
    bindFunc(cast(void**)&CLBlastDsyrk, "CLBlastDsyrk");
    bindFunc(cast(void**)&CLBlastCsyrk, "CLBlastCsyrk");
    bindFunc(cast(void**)&CLBlastZsyrk, "CLBlastZsyrk");
    bindFunc(cast(void**)&CLBlastHsyrk, "CLBlastHsyrk");
    bindFunc(cast(void**)&CLBlastCherk, "CLBlastCherk");
    bindFunc(cast(void**)&CLBlastZherk, "CLBlastZherk");
    bindFunc(cast(void**)&CLBlastSsyr2k, "CLBlastSsyr2k");
    bindFunc(cast(void**)&CLBlastDsyr2k, "CLBlastDsyr2k");
    bindFunc(cast(void**)&CLBlastCsyr2k, "CLBlastCsyr2k");
    bindFunc(cast(void**)&CLBlastZsyr2k, "CLBlastZsyr2k");
    bindFunc(cast(void**)&CLBlastHsyr2k, "CLBlastHsyr2k");
    bindFunc(cast(void**)&CLBlastCher2k, "CLBlastCher2k");
    bindFunc(cast(void**)&CLBlastZher2k, "CLBlastZher2k");
    bindFunc(cast(void**)&CLBlastStrmm, "CLBlastStrmm");
    bindFunc(cast(void**)&CLBlastDtrmm, "CLBlastDtrmm");
    bindFunc(cast(void**)&CLBlastCtrmm, "CLBlastCtrmm");
    bindFunc(cast(void**)&CLBlastZtrmm, "CLBlastZtrmm");
    bindFunc(cast(void**)&CLBlastHtrmm, "CLBlastHtrmm");
    bindFunc(cast(void**)&CLBlastStrsm, "CLBlastStrsm");
    bindFunc(cast(void**)&CLBlastDtrsm, "CLBlastDtrsm");
    bindFunc(cast(void**)&CLBlastCtrsm, "CLBlastCtrsm");
    bindFunc(cast(void**)&CLBlastZtrsm, "CLBlastZtrsm");
    bindFunc(cast(void**)&CLBlastShad, "CLBlastShad");
    bindFunc(cast(void**)&CLBlastDhad, "CLBlastDhad");
    bindFunc(cast(void**)&CLBlastChad, "CLBlastChad");
    bindFunc(cast(void**)&CLBlastZhad, "CLBlastZhad");
    bindFunc(cast(void**)&CLBlastHhad, "CLBlastHhad");
    bindFunc(cast(void**)&CLBlastSomatcopy, "CLBlastSomatcopy");
    bindFunc(cast(void**)&CLBlastDomatcopy, "CLBlastDomatcopy");
    bindFunc(cast(void**)&CLBlastComatcopy, "CLBlastComatcopy");
    bindFunc(cast(void**)&CLBlastZomatcopy, "CLBlastZomatcopy");
    bindFunc(cast(void**)&CLBlastHomatcopy, "CLBlastHomatcopy");
    bindFunc(cast(void**)&CLBlastSim2col, "CLBlastSim2col");
    bindFunc(cast(void**)&CLBlastDim2col, "CLBlastDim2col");
    bindFunc(cast(void**)&CLBlastCim2col, "CLBlastCim2col");
    bindFunc(cast(void**)&CLBlastZim2col, "CLBlastZim2col");
    bindFunc(cast(void**)&CLBlastHim2col, "CLBlastHim2col");
    bindFunc(cast(void**)&CLBlastScol2im, "CLBlastScol2im");
    bindFunc(cast(void**)&CLBlastDcol2im, "CLBlastDcol2im");
    bindFunc(cast(void**)&CLBlastCcol2im, "CLBlastCcol2im");
    bindFunc(cast(void**)&CLBlastZcol2im, "CLBlastZcol2im");
    bindFunc(cast(void**)&CLBlastHcol2im, "CLBlastHcol2im");
    bindFunc(cast(void**)&CLBlastSconvgemm, "CLBlastSconvgemm");
    bindFunc(cast(void**)&CLBlastDconvgemm, "CLBlastDconvgemm");
    bindFunc(cast(void**)&CLBlastHconvgemm, "CLBlastHconvgemm");
    bindFunc(cast(void**)&CLBlastSaxpyBatched, "CLBlastSaxpyBatched");
    bindFunc(cast(void**)&CLBlastDaxpyBatched, "CLBlastDaxpyBatched");
    bindFunc(cast(void**)&CLBlastCaxpyBatched, "CLBlastCaxpyBatched");
    bindFunc(cast(void**)&CLBlastZaxpyBatched, "CLBlastZaxpyBatched");
    bindFunc(cast(void**)&CLBlastHaxpyBatched, "CLBlastHaxpyBatched");
    bindFunc(cast(void**)&CLBlastSgemmBatched, "CLBlastSgemmBatched");
    bindFunc(cast(void**)&CLBlastDgemmBatched, "CLBlastDgemmBatched");
    bindFunc(cast(void**)&CLBlastCgemmBatched, "CLBlastCgemmBatched");
    bindFunc(cast(void**)&CLBlastZgemmBatched, "CLBlastZgemmBatched");
    bindFunc(cast(void**)&CLBlastHgemmBatched, "CLBlastHgemmBatched");
    bindFunc(cast(void**)&CLBlastSgemmStridedBatched, "CLBlastSgemmStridedBatched");
    bindFunc(cast(void**)&CLBlastDgemmStridedBatched, "CLBlastDgemmStridedBatched");
    bindFunc(cast(void**)&CLBlastCgemmStridedBatched, "CLBlastCgemmStridedBatched");
    bindFunc(cast(void**)&CLBlastZgemmStridedBatched, "CLBlastZgemmStridedBatched");
    bindFunc(cast(void**)&CLBlastHgemmStridedBatched, "CLBlastHgemmStridedBatched");
    bindFunc(cast(void**)&CLBlastSGemmTempBufferSize, "CLBlastSGemmTempBufferSize");
    bindFunc(cast(void**)&CLBlastDGemmTempBufferSize, "CLBlastDGemmTempBufferSize");
    bindFunc(cast(void**)&CLBlastCGemmTempBufferSize, "CLBlastCGemmTempBufferSize");
    bindFunc(cast(void**)&CLBlastZGemmTempBufferSize, "CLBlastZGemmTempBufferSize");
    bindFunc(cast(void**)&CLBlastHGemmTempBufferSize, "CLBlastHGemmTempBufferSize");
    bindFunc(cast(void**)&CLBlastClearCache, "CLBlastClearCache");
    bindFunc(cast(void**)&CLBlastFillCache, "CLBlastFillCache");
    bindFunc(cast(void**)&CLBlastOverrideParameters, "CLBlastOverrideParameters");
  }

public:
  this() {
    super(libNames);
  }
}

__gshared DerelictCLBlastLoader DerelictCLBlast;

shared static this() {
  DerelictCLBlast = new DerelictCLBlastLoader();
}
