package exc.openacc;
import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.io.Writer;

import xcodeml.util.XmOption;

import exc.object.*;


class ACCgpuDecompiler {
  private static final int BUFFER_SIZE = 4096;
  private static final String GPU_SRC_EXTENSION = ".cu";
  public static final String GPU_FUNC_CONF = "OEPNACC_GPU_FUNC_CONF_PROP";
  public static final String GPU_FUNC_CONF_ASYNC = "OEPNACC_GPU_FUNC_CONF_ASYNC_PROP";
  public static final String GPU_FUNC_CONF_SHAREDMEMORY = "OEPNACC_GPU_FUNC_CONF_SHAREDMEMORY_PROP";
  public static final String GPU_STRAGE_SHARED = "OPENACC_GPU_SHARED";

  public void decompile(ACCglobalDecl decl){
    XobjectFile env = decl.getEnv();
    XobjectFile envDevice = decl.getEnvDevice();
    
    if(envDevice.getDefs().isEmpty()){
      return;
    }
    
    envDevice.collectAllTypes();
    
    //collect ids corresponds to used types
    XobjList globalIdList = (XobjList)envDevice.getGlobalIdentList();
    for(Xtype type : envDevice.getTypeList()){
      if(type.isStruct() || type.isEnum() || type.isUnion()){
	Ident id = findIdent((XobjList)env.getGlobalIdentList(), type);
	if(id != null){
	  globalIdList.cons(id);
	  type.setTagIdent(id);
	}
      }
    }

    try{
      Writer w = new BufferedWriter(new FileWriter(ACCutil.removeExtension(env.getSourceFileName()) + GPU_SRC_EXTENSION), BUFFER_SIZE);
      ACCgpuDecompileWriter writer = new ACCgpuDecompileWriter(w, envDevice);
      
      writer.println("#include \"acc.h\"");
      writer.println("#include \"acc_gpu.h\"");
      writer.println("#include \"acc_gpu_func.hpp\"");
      if(XmOption.isXcalableMP()){
        writer.println("#include \"xmp_index_macro.h\"");
      }
      writer.println();

      writer.printAll();
      
      writer.flush();
      writer.close();
    }catch (IOException e){
      ACC.fatal("error in gpu decompiler: " + e.getMessage());
    }
  }

  private Ident findIdent(XobjList idList, Xtype type){
    for(Xobject x : idList){
      Ident id = (Ident)x;
      if(id.Type() == type){
	return id;
      }
    }
    return null;
  }
}
  
