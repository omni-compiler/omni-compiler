package exc.openacc;
import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.io.Writer;
import java.util.*;

import exc.object.*;


public class ACCgpuDecompiler {
  private static ACCgpuDecompileWriter out = null;
  private static final int BUFFER_SIZE = 4096;
  private static final String GPU_SRC_EXTENSION = ".cu";
  public static final String GPU_FUNC_CONF = "OEPNACC_GPU_FUNC_CONF_PROP";
  public static final String GPU_FUNC_CONF_ASYNC = "OEPNACC_GPU_FUNC_CONF_ASYNC_PROP";
  public static final String GPU_FUNC_CONF_SHAREDMEMORY = "OEPNACC_GPU_FUNC_CONF_SHAREDMEMORY_PROP";
  public static final String GPU_STRAGE_SHARED = "OPENACC_GPU_SHARED";
  
  
  public static final String GPU_INDEX_TABLE = "OPENACC_GPU_INDEX_TABLE_PROP";

  
  void decompile(XobjectFile env, XobjectDef deviceKernelDef, Ident deviceKernelId, XobjectDef hostFuncDef, List<XobjectDef> decls) throws ACCexception{
    // write gpu_code
    try {
      if (out == null) {
        Writer w = new BufferedWriter(new FileWriter(ACCutil.removeExtension(env.getSourceFileName()) + GPU_SRC_EXTENSION), BUFFER_SIZE);
        //Writer w = new BufferedWriter(new FileWriter("test.cu"), BUFFER_SIZE);
        out = new ACCgpuDecompileWriter(w, env);
      }

      // add header include line
      out.println("#include \"acc_gpu_func.hpp\"");
      //out.println("#include \"acc_index_macro.h\"");
      out.println();
      
      for(XobjectDef declDef : decls){
        out.printDecl(declDef);
      }

      // decompile device function
      out.printDeviceFunc(deviceKernelDef, deviceKernelId);
      out.println();

      // decompile wrapping function
      out.printHostFunc(hostFuncDef);
      out.println();

      out.flush();
    } catch (IOException e) {
      throw new ACCexception("error in gpu decompiler: " + e.getMessage());
    }
  }
  

}
  
