package exc.openacc;

import java.util.Iterator;

import exc.block.*;
import exc.object.*;

public class ACCtranslateUpdate {
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
        dirObj = Xcons.IntConstant(ACC.DEVICE_TO_HOST);
      }else if(var.copiesHtoD()){ //update device
        dirObj = Xcons.IntConstant(ACC.HOST_TO_DEVICE);
      }else{
        throw new ACCexception(var + " does not have update direction");
      }

      String copyFuncName = null;
      Xobject asyncExpr = Xcons.IntConstant(ACC.ACC_ASYNC_SYNC);
      if(updateInfo.isAsync()){
        Xobject expr = updateInfo.getAsyncExp(); 
        if(expr != null){ //async(expr)
	    asyncExpr = expr;
        }else{
	    asyncExpr = Xcons.IntConstant(ACC.ACC_ASYNC_NOVAL);
        }
      }

      XobjList copyFuncArgs = Xcons.List(hostDescId.Ref(), dirObj, asyncExpr);
      Block copyFunc;
      if(var.isSubarray()){
        XobjList subarrayList = var.getSubscripts();
        XobjList lowerList = Xcons.List();
        XobjList lengthList = Xcons.List();
        for(Xobject x : subarrayList){
          XobjList rangeList = (XobjList)x;
          //copyFuncArgs.add(rangeList.left());
          //copyFuncArgs.add(rangeList.right());
          lowerList.add(rangeList.left());
          lengthList.add(rangeList.right());
        }
	copyFuncName = ACC.COPY_SUBDATA_FUNC_NAME;
	copyFunc = ACCutil.createFuncCallBlockWithArrayRange(copyFuncName, copyFuncArgs, Xcons.List(lowerList, lengthList));
      }else{
	copyFuncName = ACC.COPY_DATA_FUNC_NAME;
	copyFunc = ACCutil.createFuncCallBlock(copyFuncName, copyFuncArgs);
      }

      //Block copyFunc = ACCutil.createFuncCallBlock(copyFuncName, copyFuncArgs);
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
