package exc.openacc;

import java.util.Iterator;

import exc.block.*;
import exc.object.*;

public class ACCtranslateUpdate {
  private final static String GPU_COPY_DATA_FUNC_NAME = "_ACC_gpu_copy_data";
  private final static String GPU_ASYNC_FUNC_SUFFIX = "_async";
  private final static String GPU_ASYNC_DEFAULT_FUNC_SUFFIX = "_async_default";
  private final static int GPU_COPY_HOST_TO_DEVICE = 400;
  private final static int GPU_COPY_DEVICE_TO_HOST = 401;
  
  private PragmaBlock pb;
  private ACCinfo updateInfo;
  private ACCglobalDecl globalDecl;

  
  ACCtranslateUpdate(PragmaBlock pb){
    this.pb = pb;
    this.updateInfo = ACCutil.getACCinfo(pb);
    if(this.updateInfo == null){
      ACC.fatal("can't get info");
    }
    this.globalDecl = this.updateInfo.getGlobalDecl();
  }
  
  public void translate() throws ACCexception{
    ACC.debug("translate update");
    
    if(updateInfo.isDisabled()) return;
    
    BlockList replaceBody = Bcons.emptyBody();
    
    for(Iterator<ACCvar> iter = updateInfo.getVars(); iter.hasNext(); ){
      ACCvar var = iter.next();
      String varName = var.getName();
      if(! updateInfo.isVarAllocated(varName)){
        throw new ACCexception(var + " is not allocated in device memory");
      }
      
      Ident hostDescId = updateInfo.getHostDescId(varName);//var.getHostDesc();

      Xobject dirObj = null;
      if(var.copiesDtoH()){ //update host
        dirObj = Xcons.IntConstant(GPU_COPY_DEVICE_TO_HOST);
      }else if(var.copiesHtoD()){ //update device
        dirObj = Xcons.IntConstant(GPU_COPY_HOST_TO_DEVICE);
      }else{
        throw new ACCexception(var + " does not have update direction");
      }

      String copyFuncName = GPU_COPY_DATA_FUNC_NAME;
      XobjList copyFuncArgs = Xcons.List(hostDescId.Ref(), var.getOffset(), var.getSize(), dirObj);
      
      //XobjList copyFuncArgs = Xcons.List(hostDescId.Ref(), dirObj);
      if(updateInfo.isAsync()){
        Xobject asyncExp = updateInfo.getAsyncExp(); 
        if(asyncExp != null){ //async(expr)
          copyFuncName += GPU_ASYNC_FUNC_SUFFIX;
          copyFuncArgs.add(asyncExp);
        }else{ // async all
          copyFuncName += GPU_ASYNC_DEFAULT_FUNC_SUFFIX;
        }
      }

      Block copyFunc = ACCutil.createFuncCallBlock(copyFuncName, copyFuncArgs);
      replaceBody.add(copyFunc);
    }
    
    Block replaceBlock = null;
    if(updateInfo.isEnabled()){
      replaceBlock = Bcons.COMPOUND(replaceBody);
    }else{
      replaceBlock = Bcons.IF(updateInfo.getIfCond(), Bcons.COMPOUND(replaceBody), null);
    }
    
    updateInfo.setReplaceBlock(replaceBlock);
  }
}
