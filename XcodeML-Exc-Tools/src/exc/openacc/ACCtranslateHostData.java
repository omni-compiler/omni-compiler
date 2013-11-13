package exc.openacc;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

import exc.block.*;
import exc.object.*;

public class ACCtranslateHostData {
  private PragmaBlock pb;
  private ACCinfo hostDataInfo;
  private List<Block> initBlockList;
  private List<Block> copyinBlockList;
  private List<Block> copyoutBlockList;
  private List<Block> finalizeBlockList;

  ACCtranslateHostData(PragmaBlock pb){
    this.pb = pb;
    this.hostDataInfo = ACCutil.getACCinfo(pb);
    if(this.hostDataInfo == null){
      ACC.fatal("cannot get accinfo");
    }
    initBlockList = new ArrayList<Block>();
    copyinBlockList = new ArrayList<Block>();
    copyoutBlockList = new ArrayList<Block>();
    finalizeBlockList = new ArrayList<Block>();
  }
  
  public void translate() throws ACCexception{
    if(ACC.debugFlag){
      System.out.println("translate host_data");
    }
    
    //if(hostDataInfo.isDisabled()) return;
    
  }
  
  public void rewrite(){
    Iterator<ACCvar> iter = hostDataInfo.getVars();
    while(iter.hasNext()){
      rewriteVar(iter.next());
    }
  }
  
  private void rewriteVar(ACCvar var){
    String hostName = var.getName();
    Xobject deviceAddr = hostDataInfo.getDevicePtr(hostName).Ref();
    
    BasicBlockExprIterator iter = new BasicBlockExprIterator(pb.getBody());
    for (iter.init(); !iter.end(); iter.next()) {
      Xobject expr = iter.getExpr();
      topdownXobjectIterator exprIter = new topdownXobjectIterator(expr);
      for (exprIter.init(); !exprIter.end(); exprIter.next()) {
        Xobject x = exprIter.getXobject();
        switch (x.Opcode()) {
        case VAR:
        {
          String varName = x.getName();
          if(! varName.equals(hostName)) break;
          ACC.fatal("can't rewrite '" + varName + "' with that's device pointer");
        }
        case ADDR_OF:
        {
          Xobject v = x.getArg(0);
          if(! v.getName().equals(hostName)) break;
          //deviceAddr.setType(Xtype.Pointer(v.Type()));
          exprIter.setXobject(Xcons.Cast(Xtype.Pointer(v.Type()), deviceAddr));
          rewriteVar(var);
          return;
        }
//        case VAR: 
//        {
//          String varName = x.getName();
//          if(! varName.equals(hostName)) break;
//          exprIter.setXobject(deviceAddr);
//        } break;
        case ARRAY_ADDR:
        {
          String arrayName = x.getName();
          if(! arrayName.equals(hostName)) break;
          exprIter.setXobject(Xcons.Cast(x.Type(), deviceAddr));
        }break;
//        case ARRAY_REF:
//        {
//          String arrayName = x.getArg(0).getName();
//          if(! arrayName.equals(hostName)) break;
//          Xobject new_x = deviceAddr.copy();
//          new_x.setType(x.getArg(0).Type());
//          exprIter.setXobject(new_x);
//        } break;
        default:
        }
      }
    }
  }
}
