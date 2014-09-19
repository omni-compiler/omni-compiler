package exc.xcalablemp;

import exc.object.*;
import exc.openacc.ACCpragma;
import exc.block.*;

public class XACCrewriteACCparallel extends XACCrewriteACCdata{

  public XACCrewriteACCparallel(XMPglobalDecl decl, PragmaBlock pb) {
    super(decl, pb);
  } 

  @Override
  public Block makeReplaceBlock(){
    XACCtranslatePragma trans = new XACCtranslatePragma(_globalDecl);
    
    if(device == null) return null;
    
    
    XobjList createArgs = Xcons.List();
    XobjList updateDeviceArgs = Xcons.List();
    XobjList updateHostArgs = Xcons.List();
    XobjList deleteArgs = Xcons.List();

    analyzeClause(createArgs, updateDeviceArgs, updateHostArgs, deleteArgs);
    
    XobjList kernelClauses = getKernelClauses();
    Block initDeviceLoopBlock = makeBeginDeviceLoop(createArgs, updateDeviceArgs);
    Block finalizeDeviceLoopBlock = makeEndDeviceLoop(deleteArgs, updateHostArgs);

    Block mainDeviceLoopBlock = makeMainDeviceLoop(kernelClauses);
    
    add(initDeviceLoopBlock);
    add(mainDeviceLoopBlock);
    add(finalizeDeviceLoopBlock);
    
    return replaceBlock;
  }
  
  private Block makeMainDeviceLoop(XobjList kernelClauses) {
    DeviceLoop deviceLoop = new DeviceLoop(device);
    
    CforBlock forBlock = (CforBlock)pb.getBody().getHead();
    if(! forBlock.isCanonical()){
                        forBlock.Canonicalize();
    }
    String loopVarName = forBlock.getInductionVar().getSym();

    try{
      int dim = on.getCorrespondingDim(loopVarName);
      if(dim >= 0){
        XMPlayout myLayout = on.getLayout();
        String layoutSym = XMPlayout.getDistMannerString(myLayout.getDistMannerAt(dim));
        Ident loopInitId = deviceLoop.declLocalIdent("_XACC_loop_init_" + loopVarName, Xtype.intType);
        Ident loopCondId = deviceLoop.declLocalIdent("_XACC_loop_cond_" + loopVarName, Xtype.intType);
        Ident loopStepId = deviceLoop.declLocalIdent("_XACC_loop_step_" + loopVarName, Xtype.intType);
        Ident schedLoopFuncId = _globalDecl.declExternFunc("_XACC_sched_loop_layout_"+ layoutSym);
        Xobject oldInit, oldCond, oldStep;
        XobjList loopIter = XMPutil.getLoopIter(forBlock, loopVarName);
        
        //get old loop iter
        if(loopIter != null){
          oldInit = ((Ident)loopIter.getArg(0)).Ref();
          oldCond = ((Ident)loopIter.getArg(1)).Ref();
          oldStep = ((Ident)loopIter.getArg(2)).Ref();
        }else{
          oldInit = forBlock.getLowerBound();
          oldCond = forBlock.getUpperBound();
          oldStep = forBlock.getStep();
        }
        XobjList schedLoopFuncArgs = 
            Xcons.List(oldInit,oldCond, oldStep,
                loopInitId.getAddr(), loopCondId.getAddr(), loopStepId.getAddr(), 
                on.getArrayDesc().Ref(), Xcons.IntConstant(dim), deviceLoop.getLoopVarId().Ref());
        deviceLoop.getBody().insert(schedLoopFuncId.Call(schedLoopFuncArgs));
        
        //rewrite loop iter
        forBlock.setLowerBound(loopInitId.Ref());
        forBlock.setUpperBound(loopCondId.Ref());
        forBlock.setStep(loopStepId.Ref());
      }
      
    } catch (XMPexception e) {
      XMP.error(pb.getLineNo(), e.getMessage());
    }
    
    deviceLoop.add(Bcons.PRAGMA(Xcode.ACC_PRAGMA, pb.getPragma(), clauses, pb.getBody()));
    
    return deviceLoop.makeBlock();
  }

  private XobjList getKernelClauses()
  {
    XobjList list = Xcons.List();
    for(Xobject x : clauses){
      if(x == null) continue;
      String clauseName = x.left().getSym();
      Xobject clauseArgs = x.right();
      ACCpragma clause = ACCpragma.valueOf(clauseName);
      switch(clause){
      case HOST:
      case DEVICE:
      case COPY:
      case COPYIN:
      case COPYOUT:
      case CREATE:
        break;
      default:
        list.add(x);
      }
    }
    return list;
  }
  
  protected void rewriteXACCClauses(XobjList clauses, DeviceLoop deviceLoop){
    for(Xobject x : clauses){
      if(x == null) continue;
      String clauseName = x.left().getSym();
      XobjList clauseArgs = (XobjList)x.right();
      ACCpragma clause = ACCpragma.valueOf(clauseName);
      rewriteXACCClause(clause, clauseArgs, deviceLoop);
    }
  }
}
