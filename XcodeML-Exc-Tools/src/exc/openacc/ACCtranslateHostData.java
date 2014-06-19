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
      XobjectIterator exprIter = new bottomupXobjectIterator(expr);
      for (exprIter.init(); !exprIter.end(); exprIter.next()) {
        Xobject x = exprIter.getXobject();
        switch (x.Opcode()) {
        case VAR:
        {
          String varName = x.getName();
          if(! varName.equals(hostName)) break;

          Xtype varType = var.getId().Type();
          Xobject new_x;
          if(varType.isArray() || varType.isPointer()){
            new_x = Xcons.Cast(varType, deviceAddr);  
          }else{
            new_x = Xcons.PointerRef(Xcons.Cast(Xtype.Pointer(varType), deviceAddr));
          }
          exprIter.setXobject(new_x);
        }break;
        case ARRAY_ADDR:
        {
          String arrayName = x.getName();
          Xtype t = x.Type();
          if(! arrayName.equals(hostName)) break;
          exprIter.setXobject(Xcons.Cast(Xtype.Pointer(x.Type().getRef()), deviceAddr));
        }break;
        case ARRAY_REF:
        {
          Xobject arrayAddr = x.getArg(0);
          if(arrayAddr.Opcode() == Xcode.ARRAY_ADDR)break;
          exprIter.setXobject(convertArrayRef(x));
        } break;
        default:
        }
      }
    }
  }

  private Xobject convertArrayRef(Xobject x)
  {
    if(x.Opcode() != Xcode.ARRAY_REF) return x;
    Xobject arrayAddr = x.getArg(0);
    XobjList indexList = (XobjList)(x.getArg(1)); 

    Xobject result = arrayAddr;
    for(Xobject idx : indexList){
      result = Xcons.PointerRef(Xcons.binaryOp(Xcode.PLUS_EXPR, result, idx));
    }
    return result;
  }
}
