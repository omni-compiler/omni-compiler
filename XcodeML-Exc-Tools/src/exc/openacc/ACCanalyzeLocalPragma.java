package exc.openacc;
import exc.block.*;
import exc.object.*;

import java.util.*;

public class ACCanalyzeLocalPragma {
  private ACCglobalDecl   _globalDecl;
  //private XobjectDef    currentDef;

  public ACCanalyzeLocalPragma(ACCglobalDecl globalDecl) {
    _globalDecl = globalDecl;
  }

  public void analyze(FuncDefBlock def) {
    FunctionBlock fb = def.getBlock();
    //currentDef = def.getDef();

    BlockIterator i = new topdownBlockIterator(fb);
    for (i.init(); !i.end(); i.next()) {
      Block b = i.getBlock();
      if (b.Opcode() ==  Xcode.ACC_PRAGMA) {
        PragmaBlock pb = (PragmaBlock)b;
        try {
          analyzePragma(pb);
        } catch (ACCexception e) {
          ACC.error(pb.getLineNo(), e.getMessage());
        }
      /*}else if(b.Opcode() == Xcode.LIST){
        SimpleBlock sb = (SimpleBlock)b;
        BasicBlock bb = sb.getBasicBlock();
        for(Statement st = bb.getHead(); st != null; st = st.getNext()){
          Xobject exp = st.getExpr();
          if(exp.Opcode() == Xcode.FUNCTION_CALL){
            String funcName = exp.getArg(0).getName();
            XobjList funcArgs = (XobjList)exp.getArg(1);
            System.out.println("func:"+funcName);
            System.out.println("args:"+funcArgs);
            PragmaBlock pb = getOuterPragmaBlock(b);
            ACCinfo info = null;
            if(pb != null){
              info = ACCutil.getACCinfo(pb);
            }
            Iterator<Xobject> it = funcArgs.iterator();
            for(int argNum = 0; it.hasNext(); argNum++){
              Xobject var = it.next();
              Xcode type = var.Opcode();
              if(type==Xcode.ARRAY_REF){
                //XXX
              }
              String varName = var.getString();
              boolean isAllocated = false;
              if(info != null){
                isAllocated = info.isVarAllocated(varName);
              }
              _globalDecl.setCallerInfo(funcName, argNum, type, isAllocated);
            }

          }
        }*/
      }else if(b.Opcode() == Xcode.PRAGMA_LINE){
        PragmaBlock pb = (PragmaBlock)b;
        ACC.error(pb.getLineNo(), "unknown pragma : " + pb.getClauses());
      }
    }
  }
  
  private void analyzePragma(PragmaBlock pb) throws ACCexception {
    String pragmaName = pb.getPragma();

    switch (ACCpragma.valueOf(pragmaName)) {
    case PARALLEL:
      analyzeParallel(pb); break;
    case KERNELS:
      analyzeKernels(pb); break;
    case DATA:
      analyzeData(pb); break;
    case HOST_DATA:
      analyzeHostData(pb); break;
    case LOOP:
      analyzeLoop(pb); break;
    case CACHE:
      analyzeCache(pb); break;
    case PARALLEL_LOOP:
      break;
    case KERNELS_LOOP:
      break;
    case DECLARE:
      analyzeDeclare(pb); break;
    case UPDATE:
      analyzeUpdate(pb); break;
    case WAIT:
      analyzeWait(pb); break;

    default:
      throw new ACCexception("'" + pragmaName.toLowerCase() + "' directive is not supported yet");
    }
  }
  
  private void analyzeParallel(PragmaBlock pb) throws ACCexception{
    XobjList clauseList = (XobjList)pb.getClauses();
    ACCinfo accInfo = new ACCinfo(ACCpragma.PARALLEL, pb, _globalDecl);
    ACCutil.setACCinfo(pb, accInfo);
    
    if(ACC.debugFlag){
      System.out.println("parallel directive : " + clauseList);
    }
    
    for(Xobject o : clauseList){
      XobjList clause = (XobjList)o;
      ACCpragma clauseName = ACCpragma.valueOf(clause.getArg(0));
      Xobject clauseArgs = (clause.Nargs() > 1)? clause.getArg(1) : null;
            
      switch(clauseName){
      case IF:
        accInfo.setIfCond(clauseArgs); break;
      case ASYNC:
        accInfo.setAsyncExp(clauseArgs); break;
      case NUM_GANGS:
        accInfo.setNumGangsExp(clauseArgs); break;
      case NUM_WORKERS:
        accInfo.setNumWorkersExp(clauseArgs); break;
      case VECT_LEN:
        accInfo.setVectorLengthExp(clauseArgs); break;
      case PRIVATE:
      case FIRSTPRIVATE:
        for(Xobject var : (XobjList)clauseArgs) accInfo.declACCvar(var.getName(), clauseName);
        break;
      default:
        if(clauseName.isDataClause() || clauseName.isReduction()){
          for(Xobject var : (XobjList)clauseArgs) accInfo.declACCvar(var.getName(), clauseName);
        }else{
          ACC.fatal("'" + clauseName.getName() +"' clause is not allowed in 'parallel' directive");
        }
      }
    }
    
    specifyInductionVarAsPrivate(pb);
    checkUnspecifiedVars(pb, accInfo);
  }
  
  private void analyzeKernels(PragmaBlock pb) throws ACCexception{
    XobjList clauseList = (XobjList)pb.getClauses();
    ACCinfo accInfo = new ACCinfo(ACCpragma.KERNELS, pb, _globalDecl);
    ACCutil.setACCinfo(pb, accInfo);
    
    if(ACC.debugFlag){
      System.out.println("kernels directive : " + clauseList);
    }
    
    for(Xobject o : clauseList){
      XobjList clause = (XobjList)o;
      ACCpragma clauseName = ACCpragma.valueOf(clause.getArg(0));
      Xobject clauseArgs = (clause.Nargs() > 1)? clause.getArg(1) : null;
            
      switch(clauseName){
      case IF:
        accInfo.setIfCond(clauseArgs); break;
      case ASYNC:
        accInfo.setAsyncExp(clauseArgs); break;
      default:
        if(clauseName.isDataClause()){
          for(Xobject var : (XobjList)clauseArgs) accInfo.declACCvar(var.getName(), clauseName);
        }else{
          ACC.fatal("'" + clauseName +"' clause is not allowed in 'kernels' directive");
        }
      }
    }
    specifyInductionVarAsPrivate(pb);
    checkUnspecifiedVars(pb, accInfo);
  }
  
  private void analyzeData(PragmaBlock pb) throws ACCexception{
    XobjList clauseList = (XobjList)pb.getClauses();
    ACCinfo accInfo = new ACCinfo(ACCpragma.DATA, pb, _globalDecl);
    ACCutil.setACCinfo(pb, accInfo);
    
    if(ACC.debugFlag){
      System.out.println("data directive : " + clauseList);
    }
    
    for(Xobject o : clauseList){
      XobjList clause = (XobjList)o;
      ACCpragma clauseName = ACCpragma.valueOf(clause.getArg(0));
      Xobject clauseArgs = (clause.Nargs() > 1)? clause.getArg(1) : null;
            
      switch(clauseName){
      case IF:
        accInfo.setIfCond(clauseArgs); break;
      default:
        if(clauseName.isDataClause()){
          for(Xobject var : (XobjList)clauseArgs) accInfo.declACCvar(var.getName(), clauseName);
        }else{
          ACC.fatal("'" + clauseName +"' clause is not allowed in 'data' directive");
        }
      }
    }
  }
  
  private void analyzeHostData(PragmaBlock pb) throws ACCexception{
    XobjList clauseList = (XobjList)pb.getClauses();
    ACCinfo accInfo = new ACCinfo(ACCpragma.HOST_DATA, pb, _globalDecl);
    ACCutil.setACCinfo(pb, accInfo);
    
    if(ACC.debugFlag){
      System.out.println("host_data directive : " + clauseList);
    }
    
    for(Xobject o : clauseList){
      XobjList clause = (XobjList)o;
      ACCpragma clauseName = ACCpragma.valueOf(clause.getArg(0));
      Xobject clauseArgs = (clause.Nargs() > 1)? clause.getArg(1) : null;
            
      //only use_device clause
      if(clauseName == ACCpragma.USE_DEVICE){
        for(Xobject var : (XobjList)clauseArgs) accInfo.declACCvar(var.getName(), clauseName);
      }else{
        ACC.fatal("'" + clauseName +"' clause is not allowed in 'host_data' directive");
      }
    }
  }
  
  private void analyzeLoop(PragmaBlock pb) throws ACCexception{
    XobjList clauseList = (XobjList)pb.getClauses();
    ACCinfo accInfo = new ACCinfo(ACCpragma.LOOP, pb, _globalDecl);
    ACCutil.setACCinfo(pb, accInfo);
    

    //parallel or kernels? 
    ACCpragma computePragma = accInfo.getComputePragma();
    if(computePragma == null){
      throw new ACCexception("loop directive exists outside parallel/kernels region");
    }
    
    if(ACC.debugFlag){
      System.out.println("loop directive : " + clauseList);
    }
    
    for(Xobject o : clauseList){
      XobjList clause = (XobjList)o;
      ACCpragma clauseName = ACCpragma.valueOf(clause.getArg(0));
      Xobject clauseArgs = (clause.Nargs() > 1)? clause.getArg(1) : null;
      
      if(computePragma == ACCpragma.PARALLEL){
        switch(clauseName){
        case GANG:
        case WORKER:
        case VECTOR:
          if(clauseArgs == null) accInfo.addExecModel(clauseName); //accInfo.setExecLevel(clauseName);
          else ACC.fatal(clauseName + "'s arg is not allowed in parallel region");
          continue;
        }
      }else{ //computePragma == ACCpragma.KERNELS
        switch(clauseName){
        case GANG:
          accInfo.addExecModel(clauseName); //accInfo.setExecLevel(clauseName);
          accInfo.setNumGangsExp(clauseArgs); continue;
        case WORKER:
          accInfo.addExecModel(clauseName); //accInfo.setExecLevel(clauseName);
          accInfo.setNumWorkersExp(clauseArgs); continue;
        case VECTOR:
          accInfo.addExecModel(clauseName); //accInfo.setExecLevel(clauseName);
          accInfo.setVectorLengthExp(clauseArgs); continue;
        }
      }
            
      switch(clauseName){
      case COLLAPSE:
        accInfo.setCollapseNum(clauseArgs); break;
      case SEQ:
        accInfo.addExecModel(ACCpragma.SEQ); break;
      case INDEPENDENT:
        accInfo.setIndependent(); break;
      case PRIVATE:
        for(Xobject var : (XobjList)clauseArgs) accInfo.declACCvar(var.getName(), clauseName);
        break;
      default:
        if(clauseName.isReduction()){
          for(Xobject var : (XobjList)clauseArgs) accInfo.declACCvar(var.getName(), clauseName);
        }else{
          ACC.fatal("'" + clauseName +"' clause is not allowed in 'loop' directive");
        }
      }
    }
    
    //check loop
    BlockList body = pb.getBody();
    Block block = body.getHead();
    if(block.Opcode() != Xcode.FOR_STATEMENT) throw new ACCexception("for-loop must be following 'loop' directive");
    CforBlock forBlock = (CforBlock)block;
    if(! forBlock.isCanonical()){
      forBlock.Canonicalize();
      if(! forBlock.isCanonical()){
        throw new ACCexception("can't canonicalize this loop");
      }
    }
    
    /*
    //if execModel is not specified, set it AUTO.
    Iterator<ACCpragma> execModel = accInfo.getExecModels();
    if(! execModel.hasNext()){
      accInfo.addExecModel(ACCpragma._AUTO);
    }*/
  }
  
  private void analyzeCache(PragmaBlock pb) throws ACCexception{
    XobjList clauseList = (XobjList)pb.getClauses();
    ACCinfo accInfo = new ACCinfo(ACCpragma.CACHE, pb, _globalDecl);
    ACCutil.setACCinfo(pb, accInfo);
    
    if(ACC.debugFlag){
      System.out.println("cache directive : " + clauseList);
    }
    
    Xobject args = clauseList; //.getArg(1);
    for(Xobject var : (XobjList)args){
      accInfo.declACCvar(var.getName(), ACCpragma.CACHE);
    }
  }
  
  private void analyzeDeclare(PragmaBlock pb) throws ACCexception{
    XobjList clauseList = (XobjList)pb.getClauses();
    ACCinfo accInfo = new ACCinfo(ACCpragma.DECLARE, pb, _globalDecl);
    ACCutil.setACCinfo(pb, accInfo);
    
    if(ACC.debugFlag){
      System.out.println("declare directive : " + clauseList);
    }
    
    for(Xobject o : clauseList){
      XobjList clause = (XobjList)o;
      ACCpragma clauseName = ACCpragma.valueOf(clause.getArg(0));
      Xobject clauseArgs = (clause.Nargs() > 1)? clause.getArg(1) : null;
            
      if(clauseName.isDataClause() || clauseName == ACCpragma.DEVICE_RESIDENT){
        for(Xobject var : (XobjList)clauseArgs) accInfo.declACCvar(var.getName(), clauseName);
      }else{
        ACC.fatal("'" + clauseName +"' clause is not allowed in 'declare' directive");
      }
    }
  }
  
  private void analyzeUpdate(PragmaBlock pb) throws ACCexception{
    XobjList clauseList = (XobjList)pb.getClauses();
    ACCinfo accInfo = new ACCinfo(ACCpragma.UPDATE, pb, _globalDecl);
    ACCutil.setACCinfo(pb, accInfo);
    
    if(ACC.debugFlag){
      System.out.println("update directive : " + clauseList);
    }
    
    for(Xobject o : clauseList){
      XobjList clause = (XobjList)o;
      ACCpragma clauseName = ACCpragma.valueOf(clause.getArg(0));
      Xobject clauseArgs = (clause.Nargs() > 1)? clause.getArg(1) : null;
            
      switch(clauseName){
      case IF:
        accInfo.setIfCond(clauseArgs); break;
      case ASYNC:
        accInfo.setAsyncExp(clauseArgs); break;
      case HOST:
      case DEVICE:
        for(Xobject var : (XobjList)clauseArgs) accInfo.declACCvar(var.getName(), clauseName);
        break;
      default:
        ACC.fatal("'" + clauseName +"' clause is not allowed in 'update' directive");
      }
    }
  }
  
  private void analyzeWait(PragmaBlock pb) throws ACCexception{
    Xobject arg = pb.getClauses();
    ACCinfo accInfo = new ACCinfo(ACCpragma.WAIT, pb, _globalDecl);
    ACCutil.setACCinfo(pb, accInfo);
    
    if(ACC.debugFlag){
      System.out.println("wait directive : " + arg);
    }
    
    accInfo.setWaitExp(arg);
  }
  
  //Utility methods
  
  PragmaBlock getOuterPragmaBlock(Block block){
    for(Block b = block.getParentBlock(); b != null; b = b.getParentBlock()){
      if(b.Opcode() == Xcode.ACC_PRAGMA){
        return (PragmaBlock)b;
      }
    }
    return null;
  }
  
  void specifyInductionVarAsPrivate(PragmaBlock pb) throws ACCexception{
    ACCinfo info = ACCutil.getACCinfo(pb);
    
    if(ACC.debugFlag){
      ACC.debug("check induction variables");
    }
    
    topdownBlockIterator blockIter = new topdownBlockIterator(pb.getBody());
    for(blockIter.init(); !blockIter.end(); blockIter.next()){
      Block b = blockIter.getBlock();
      if(b.Opcode() == Xcode.FOR_STATEMENT){
        CforBlock forBlock = (CforBlock)b;
        if(! forBlock.isCanonical()){
          forBlock.Canonicalize();
          if(! forBlock.isCanonical()) throw new ACCexception("loop can't canonicalize");
        }

        String indVarName = forBlock.getInductionVar().getName();
        Ident indVarId = pb.findVarIdent(indVarName);
        if(indVarId != null){
          ACCvar var = info.getACCvar(indVarName);
          if(var == null){
            info.declACCvar(indVarName, ACCpragma.PRIVATE);
            if(ACC.debugFlag){
              ACC.debug("add private(" + indVarName + ")");
            }
          }
        }
      }
    }
    
  }
  
  void checkUnspecifiedVars(PragmaBlock pb, ACCinfo info) throws ACCexception{
    Set<String> checkedVars = new HashSet<String>();
    
    if(ACC.debugFlag){
      ACC.debug("check unspecified variables");
    }
    
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
          if (checkedVars.contains(varName)) break;
          
          //break if var is declared in pragma block
          if(pb.findVarIdent(varName) == null) break; 
          
          ACCvar var = info.getACCvar(varName);
          if(var != null){
            if(var.isPrivate() || var.isFirstprivate()){
              checkedVars.add(varName);
              break;
            }
          }
          
          if(! info.isVarAllocated(varName)){
            info.declACCvar(varName, ACCpragma.PRESENT_OR_COPY);
            if(ACC.debugFlag) ACC.debug("add present_or_copy(" + varName + ")");
          }
          checkedVars.add(varName);
        } break;
        case ARRAY_REF:
        {
          String varName = x.getArg(0).getName();
          if (checkedVars.contains(varName)) break;
          
          //break if array is declared in pragma block
          if(pb.findVarIdent(varName) == null) break; 

          if(! info.isVarAllocated(varName)){
            info.declACCvar(varName, ACCpragma.PRESENT_OR_COPY);
            if(ACC.debugFlag) ACC.debug("add present_or_copy(" + varName + ")");
          }

          checkedVars.add(varName);
        } break;
        default:
        }
      }
    }
  }
  
  
}
