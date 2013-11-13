package exc.openacc;

//import java.util.*;
import exc.block.*;
import exc.object.*;

public class ACCtranslateWait {
  private static final String ACC_GPU_WAIT_FUNC_NAME = "_ACC_gpu_wait";
  private static final String ACC_GPU_WAIT_ALL_FUNC_NAME = "_ACC_gpu_wait_all";
  private PragmaBlock pb;
  private ACCinfo waitInfo;
  //private ACCglobalDecl globalDecl;

  public ACCtranslateWait(PragmaBlock pb) {
    this.pb = pb;
    this.waitInfo = ACCutil.getACCinfo(pb);
    if(this.waitInfo == null){
      ACC.fatal("can't get info");
    }
    //this.globalDecl = this.waitInfo.getGlobalDecl();
  }

  public void translate() throws ACCexception{
    ACC.debug("translate wait");

    Block waitFuncBlock;
    Xobject waitExp = waitInfo.getWaitExp();
    if(waitExp != null){ //wait
      waitFuncBlock = ACCutil.createFuncCallBlock(ACC_GPU_WAIT_FUNC_NAME, Xcons.List(waitExp));
    }else{ //wait all
      waitFuncBlock = ACCutil.createFuncCallBlock(ACC_GPU_WAIT_ALL_FUNC_NAME, Xcons.List());
    }
    
    waitInfo.setReplaceBlock(waitFuncBlock);
  }
}
