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
      analyzeParallelLoop(pb); break;
    case KERNELS_LOOP:
      analyzeKernelsLoop(pb); break;
    case DECLARE:
      analyzeDeclare(pb); break;
    case UPDATE:
      analyzeUpdate(pb); break;
    case WAIT:
      analyzeWait(pb); break;
    case ENTER_DATA:
      analyzeEnterData(pb); break;
    case EXIT_DATA:
      analyzeExitData(pb); break;

    default:
      throw new ACCexception("'" + pragmaName.toLowerCase() + "' directive is not supported yet");
    }
  }

  private void analyzeParallel(PragmaBlock pb) throws ACCexception{
    XobjList clauseList = (XobjList)pb.getClauses();
    ACCinfo accInfo = new ACCinfo(ACCpragma.PARALLEL, pb, _globalDecl);
    ACCutil.setACCinfo(pb, accInfo);
    
    ACC.debug("parallel directive : " + clauseList);
    
    for(Xobject o : clauseList){
      XobjList clause = (XobjList)o;
      ACCpragma clauseName = ACCpragma.valueOf(clause.getArg(0));
      Xobject clauseArgs = clause.getArgOrNull(1);//(clause.Nargs() > 1)? clause.getArg(1) : null;
            
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
        //for(Xobject var : (XobjList)clauseArgs) accInfo.declACCvar(var.getName(), clauseName);
        analyzeVarList(accInfo, clauseName, (XobjList)clauseArgs);
        break;
      default:
        if(clauseName.isDataClause() || clauseName.isReduction()){
          //for(Xobject var : (XobjList)clauseArgs) accInfo.declACCvar(var.getName(), clauseName);
          analyzeVarList(accInfo, clauseName, (XobjList)clauseArgs);
        }else{
          ACC.fatal("'" + clauseName.getName() +"' clause is not allowed in 'parallel' directive");
        }
      }
    }
    
    //specifyInductionVarAsPrivate(pb);
    //specifyAttribute(pb, accInfo);
    
    if(!ACC.debugFlag){
      checkSynchronization(pb);
    }
  }

  private void analyzeKernels(PragmaBlock pb) throws ACCexception{
    XobjList clauseList = (XobjList)pb.getClauses();
    ACCinfo accInfo = new ACCinfo(ACCpragma.KERNELS, pb, _globalDecl);
    ACCutil.setACCinfo(pb, accInfo);
    
    ACC.debug("kernels directive : " + clauseList);
    
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
          //for(Xobject var : (XobjList)clauseArgs) accInfo.declACCvar(var.getName(), clauseName);
          analyzeVarList(accInfo, clauseName, (XobjList)clauseArgs);
        }else{
          ACC.fatal("'" + clauseName +"' clause is not allowed in 'kernels' directive");
        }
      }
    }
    //specifyInductionVarAsPrivate(pb);
    //specifyAttribute(pb, accInfo);
    
    if(!ACC.debugFlag){
      checkSynchronization(pb);
    }
  }
  
  private void analyzeData(PragmaBlock pb) throws ACCexception{
    XobjList clauseList = (XobjList)pb.getClauses();
    ACCinfo accInfo = new ACCinfo(ACCpragma.DATA, pb, _globalDecl);
    ACCutil.setACCinfo(pb, accInfo);
    
    ACC.debug("data directive : " + clauseList);
    
    for(Xobject o : clauseList){
      XobjList clause = (XobjList)o;
      ACCpragma clauseName = ACCpragma.valueOf(clause.getArg(0));
      Xobject clauseArgs = (clause.Nargs() > 1)? clause.getArg(1) : null;
            
      switch(clauseName){
      case IF:
        accInfo.setIfCond(clauseArgs); break;
      default:
        if(clauseName.isDataClause()){
          //for(Xobject var : (XobjList)clauseArgs) accInfo.declACCvar(var.getName(), clauseName);
          analyzeVarList(accInfo, clauseName, (XobjList)clauseArgs);
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
    
    ACC.debug("host_data directive : " + clauseList);
    
    for(Xobject o : clauseList){
      XobjList clause = (XobjList)o;
      ACCpragma clauseName = ACCpragma.valueOf(clause.getArg(0));
      Xobject clauseArgs = (clause.Nargs() > 1)? clause.getArg(1) : null;
            
      //only use_device clause
      if(clauseName == ACCpragma.USE_DEVICE){
        //for(Xobject var : (XobjList)clauseArgs) accInfo.declACCvar(var.getName(), clauseName);
        analyzeVarList(accInfo, clauseName, (XobjList)clauseArgs);
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
    
    ACC.debug("loop directive : " + clauseList);
    
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
        default:
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
        default:
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
        //for(Xobject var : (XobjList)clauseArgs) accInfo.declACCvar(var.getName(), clauseName);
        analyzeVarList(accInfo, clauseName, (XobjList)clauseArgs);
        break;
      default:
        if(clauseName.isReduction()){
          //for(Xobject var : (XobjList)clauseArgs) accInfo.declACCvar(var.getName(), clauseName);
          analyzeVarList(accInfo, clauseName, (XobjList)clauseArgs);
        }else{
          ACC.fatal("'" + clauseName +"' clause is not allowed in 'loop' directive");
        }
      }
    }
    
    //check loop
    checkLoopBody(pb);
    checkLoop(pb, accInfo);
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
    
    ACC.debug("cache directive : " + clauseList);
    
    Xobject args = clauseList; //.getArg(1);
    analyzeVarList(accInfo, ACCpragma.CACHE, (XobjList)args);
  }
  
  private void analyzeParallelLoop(PragmaBlock pb) throws ACCexception{
    XobjList clauseList = (XobjList)pb.getClauses();
    ACCinfo accInfo = new ACCinfo(ACCpragma.PARALLEL_LOOP, pb, _globalDecl);
    ACCutil.setACCinfo(pb, accInfo);
    
    ACC.debug("parallel loop directive : " + clauseList);
    
    for(Xobject o : clauseList){
      XobjList clause = (XobjList)o;
      ACCpragma clauseName = ACCpragma.valueOf(clause.getArg(0));
      Xobject clauseArgs = (clause.Nargs() > 1)? clause.getArg(1) : null;
            
      switch(clauseName){
      //for parallel
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
        //for(Xobject var : (XobjList)clauseArgs) accInfo.declACCvar(var.getName(), clauseName);
        analyzeVarList(accInfo, clauseName, (XobjList)clauseArgs);
        break;

      //for loop
      case GANG:
      case WORKER:
      case VECTOR:
        if(clauseArgs == null) accInfo.addExecModel(clauseName);
        else ACC.fatal(clauseName + "'s arg is not allowed in parallel region");
        break;
      case COLLAPSE:
        accInfo.setCollapseNum(clauseArgs); break;
      case SEQ:
        accInfo.addExecModel(ACCpragma.SEQ); break;
      case INDEPENDENT:
        accInfo.setIndependent(); break;
        
      default:
        if(clauseName.isDataClause() || clauseName.isReduction()){
          //for(Xobject var : (XobjList)clauseArgs) accInfo.declACCvar(var.getName(), clauseName);
          analyzeVarList(accInfo, clauseName, (XobjList)clauseArgs);
        }else{
          ACC.fatal("'" + clauseName.getName() +"' clause is not allowed in 'parallel loop' directive");
        }
      }
    }
    
    //specifyInductionVarAsPrivate(pb);
    //checkUnspecifiedVars(pb, accInfo);
    //specifyAttribute(pb, accInfo);
    
    checkLoopBody(pb);
    checkLoop(pb, accInfo);
    
    if(!ACC.debugFlag){
      checkSynchronization(pb);
    }
  }
  
  private void analyzeKernelsLoop(PragmaBlock pb) throws ACCexception{
    XobjList clauseList = (XobjList)pb.getClauses();
    ACCinfo accInfo = new ACCinfo(ACCpragma.KERNELS_LOOP, pb, _globalDecl);
    ACCutil.setACCinfo(pb, accInfo);
    
    ACC.debug("kernels loop directive : " + clauseList);
    
    for(Xobject o : clauseList){
      XobjList clause = (XobjList)o;
      ACCpragma clauseName = ACCpragma.valueOf(clause.getArg(0));
      Xobject clauseArgs = (clause.Nargs() > 1)? clause.getArg(1) : null;
            
      switch(clauseName){
      //for kernels
      case IF:
        accInfo.setIfCond(clauseArgs); break;
      case ASYNC:
        accInfo.setAsyncExp(clauseArgs); break;

      //for loop
      case GANG:
        accInfo.addExecModel(clauseName); //accInfo.setExecLevel(clauseName);
        accInfo.setNumGangsExp(clauseArgs); break;
      case WORKER:
        accInfo.addExecModel(clauseName); //accInfo.setExecLevel(clauseName);
        accInfo.setNumWorkersExp(clauseArgs); break;
      case VECTOR:
        accInfo.addExecModel(clauseName); //accInfo.setExecLevel(clauseName);
        accInfo.setVectorLengthExp(clauseArgs); break;
      case COLLAPSE:
        accInfo.setCollapseNum(clauseArgs); break;
      case SEQ:
        accInfo.addExecModel(ACCpragma.SEQ); break;
      case INDEPENDENT:
        accInfo.setIndependent(); break;
        
      default:
        if(clauseName.isDataClause() || clauseName.isReduction()){
          analyzeVarList(accInfo, clauseName, (XobjList)clauseArgs);
        }else{
          ACC.fatal("'" + clauseName.getName() +"' clause is not allowed in 'kernels loop' directive");
        }
      }
    }
    
    checkLoopBody(pb);
    checkLoop(pb, accInfo);
    
    if(!ACC.debugFlag){
      checkSynchronization(pb);
    }
  }
  
  private void analyzeDeclare(PragmaBlock pb) throws ACCexception{
    XobjList clauseList = (XobjList)pb.getClauses();
    ACCinfo accInfo = new ACCinfo(ACCpragma.DECLARE, pb, _globalDecl);
    ACCutil.setACCinfo(pb, accInfo);
    
    ACC.debug("declare directive : " + clauseList);
    
    for(Xobject o : clauseList){
      XobjList clause = (XobjList)o;
      ACCpragma clauseName = ACCpragma.valueOf(clause.getArg(0));
      Xobject clauseArgs = (clause.Nargs() > 1)? clause.getArg(1) : null;
            
      if(clauseName.isDataClause() || clauseName == ACCpragma.DEVICE_RESIDENT){
        //for(Xobject var : (XobjList)clauseArgs) accInfo.declACCvar(var.getName(), clauseName);
        analyzeVarList(accInfo, clauseName, (XobjList)clauseArgs);
      }else{
        ACC.fatal("'" + clauseName +"' clause is not allowed in 'declare' directive");
      }
    }
    
    ACCutil.setACCinfo(pb.getParentBlock(), accInfo); // add property to parent block
  }
  
  private void analyzeUpdate(PragmaBlock pb) throws ACCexception{
    XobjList clauseList = (XobjList)pb.getClauses();
    ACCinfo accInfo = new ACCinfo(ACCpragma.UPDATE, pb, _globalDecl);
    ACCutil.setACCinfo(pb, accInfo);
    
    ACC.debug("update directive : " + clauseList);
    
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
        //for(Xobject var : (XobjList)clauseArgs) accInfo.declACCvar(var.getName(), clauseName);
        analyzeVarList(accInfo, clauseName, (XobjList)clauseArgs);
        break;
      default:
        ACC.fatal("'" + clauseName +"' clause is not allowed in 'update' directive");
      }
    }
    
    //checkSynchronization(pb);
  }
  
  private void analyzeWait(PragmaBlock pb) throws ACCexception{
    Xobject arg = pb.getClauses();
    ACCinfo accInfo = new ACCinfo(ACCpragma.WAIT, pb, _globalDecl);
    ACCutil.setACCinfo(pb, accInfo);
    
    ACC.debug("wait directive : " + arg);
    
    accInfo.setWaitExp(arg);
  }
  
  private void analyzeEnterData(PragmaBlock pb) throws ACCexception{
    XobjList clauseList = (XobjList)pb.getClauses();
    ACCinfo accInfo = new ACCinfo(ACCpragma.ENTER_DATA, pb, _globalDecl);
    ACCutil.setACCinfo(pb, accInfo);
    
    ACC.debug("enter data directive : " + clauseList);
    
    for(Xobject o : clauseList){
      XobjList clause = (XobjList)o;
      ACCpragma clauseName = ACCpragma.valueOf(clause.getArg(0));
      Xobject clauseArgs = (clause.Nargs() > 1)? clause.getArg(1) : null;
            
      switch(clauseName){
      case IF:
        accInfo.setIfCond(clauseArgs); break;
      case ASYNC:
        accInfo.setAsyncExp(clauseArgs); break;
      case COPYIN:
      case CREATE:
      case PRESENT_OR_COPYIN:
      case PRESENT_OR_CREATE:
        analyzeVarList(accInfo, clauseName, (XobjList)clauseArgs);
        break;
      default:
        ACC.fatal("'" + clauseName +"' clause is not allowed in 'enter data' directive");
      }
    }
  }
  
  private void analyzeExitData(PragmaBlock pb) throws ACCexception{
    XobjList clauseList = (XobjList)pb.getClauses();
    ACCinfo accInfo = new ACCinfo(ACCpragma.EXIT_DATA, pb, _globalDecl);
    ACCutil.setACCinfo(pb, accInfo);
    
    ACC.debug("exit data directive : " + clauseList);
    
    for(Xobject o : clauseList){
      XobjList clause = (XobjList)o;
      ACCpragma clauseName = ACCpragma.valueOf(clause.getArg(0));
      Xobject clauseArgs = (clause.Nargs() > 1)? clause.getArg(1) : null;
            
      switch(clauseName){
      case IF:
        accInfo.setIfCond(clauseArgs); break;
      case ASYNC:
        accInfo.setAsyncExp(clauseArgs); break;
      case COPYOUT:
      case DELETE:
        analyzeVarList(accInfo, clauseName, (XobjList)clauseArgs);
        break;
      default:
        ACC.fatal("'" + clauseName +"' clause is not allowed in 'exit data' directive");
      }
    }
  }
  
  //Utility methods
  
//  PragmaBlock getOuterPragmaBlock(Block block){
//    for(Block b = block.getParentBlock(); b != null; b = b.getParentBlock()){
//      if(b.Opcode() == Xcode.ACC_PRAGMA){
//        return (PragmaBlock)b;
//      }
//    }
//    return null;
//  }
  
  private void specifyInductionVarAsPrivate(PragmaBlock pb) throws ACCexception{
    ACCinfo info = ACCutil.getACCinfo(pb);
    
    ACC.debug("check induction variables");
    
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
  
  private void checkUnspecifiedVars(PragmaBlock pb, ACCinfo info) throws ACCexception{
    Set<String> checkedVars = new HashSet<String>();
    
    ACC.debug("check unspecified variables");
    
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
  
  private class VarAttribute{
    String name = null;
    boolean isRead = false;
    boolean isWritten = false;
    boolean isArray = false;
    public VarAttribute(String name) {
      this.name = name;
    }
  }
  
  private void specifyAttribute(PragmaBlock pb,ACCinfo info) throws ACCexception{
    Map<String,VarAttribute> unspecifiedVars = getUnspecifiedVarMap(pb, info);

    BasicBlockExprIterator iter = new BasicBlockExprIterator(pb.getBody());
    for (iter.init(); !iter.end(); iter.next()) {
      Xobject expr = iter.getExpr();
      topdownXobjectIterator exprIter = new topdownXobjectIterator(expr);
      for (exprIter.init(); !exprIter.end(); exprIter.next()) {
        Xobject x = exprIter.getXobject();
        String varName = null;
        if(x.Opcode().isAsgOp()){
          Xobject lhs = x.getArg(0);
          if(lhs.Opcode() == Xcode.VAR){
            varName = lhs.getName();
          }else if(lhs.Opcode() == Xcode.ARRAY_REF){
            varName = lhs.getArg(0).getName();
          }
        }else if(x.Opcode() == Xcode.PRE_INCR_EXPR || x.Opcode() == Xcode.PRE_DECR_EXPR){
          varName = x.getArg(0).getName();
        }else{
          continue;
        }
        
        if(unspecifiedVars.containsKey(varName)){
          VarAttribute va = unspecifiedVars.get(varName);
          va.isWritten = true;
        }
      }
    }
    
    for(Map.Entry<String, VarAttribute> e : unspecifiedVars.entrySet()){
      String v = e.getKey();
      VarAttribute va = e.getValue();
      
      ACCpragma accAtt = ACCpragma.PRESENT_OR_COPY;
      
      if(va.isArray){
        
      }else{
        if(va.isWritten == false){
          accAtt = ACCpragma.FIRSTPRIVATE;
        }
      }
      info.declACCvar(v, accAtt);
    }
  }
  
  private Map<String,VarAttribute> getUnspecifiedVarMap(PragmaBlock pb,ACCinfo info){
    Set<String> checkedVars = new HashSet<String>();
    Map<String,VarAttribute> unspecifiedVars= new HashMap<String,VarAttribute>(); 

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
          checkedVars.add(varName);
          
          //check if var is declared in pragma block
          if(pb.findVarIdent(varName) == null) break;
          
          //if(iter.getBasicBlock().)
          
          //check if var is declared outer data clause
          if(info.isVarAllocated(varName)) break;
          
          ACCvar var = info.getACCvar(varName);
          if(var != null){
            if(var.isPrivate() || var.isFirstprivate()){
              break;
            }
          }

          unspecifiedVars.put(varName,new VarAttribute(varName));
        } break;
        case ARRAY_REF:
        {
          String varName = x.getArg(0).getName();
          if (checkedVars.contains(varName)) break;
          
          checkedVars.add(varName);
          
          //break if array is declared in pragma block
          if(pb.findVarIdent(varName) == null) break; 
          
          //break if array is declared outer data clause
          if(info.isVarAllocated(varName)) break;

          VarAttribute va = new VarAttribute(varName);
          va.isArray = true;
          unspecifiedVars.put(varName,va);
        } break;
        default:
        }
      }
    }
    return unspecifiedVars;    
  }
    
    
  private void checkLoopBody(PragmaBlock pb) throws ACCexception{
    //check loop
    BlockList body = pb.getBody();
    Block block = body.getHead();
    if(block.Opcode() != Xcode.FOR_STATEMENT) throw new ACCexception("for-loop must be following 'loop' directive");
    CforBlock forBlock = (CforBlock)block;
    
    if(forBlock.isCanonical()) return;
    
    forBlock.Canonicalize();
    if(! forBlock.isCanonical()){
      throw new ACCexception("can't canonicalize this loop");
    }
  }
  
  private void checkLoop(PragmaBlock pb, ACCinfo accInfo) throws ACCexception{
    int collapseNum = accInfo.getCollapseNum();
    checkCollapsedLoop(pb.getBody().getHead(), collapseNum);
  }
  
  private boolean checkCollapsedLoop(Block block, int num_collapse) throws ACCexception{    
    if(block.Opcode() != Xcode.FOR_STATEMENT){
      throw new ACCexception("lack of nested loops");
    }
    
    CforBlock forBlock = (CforBlock)block;
    if(! forBlock.isCanonical()){
      forBlock.Canonicalize();  
      if(! forBlock.isCanonical()){
        throw new ACCexception("can't canonicalize the loop");
      }
    }

    if(num_collapse < 2){
      return true;
    }
    
    BlockList forBody = forBlock.getBody();
    if(! forBody.isSingle()){
      throw new ACCexception("not tightly nested loop");
    }
    if(forBody.getIdentList() != null && ! forBody.getIdentList().isEmpty()){
      throw new ACCexception("var declare is not allowed between nested loops");
    }
    return checkCollapsedLoop(forBody.getHead(), num_collapse - 1);
  }
  
  private void checkSynchronization(PragmaBlock pb) throws ACCexception{
    ACCinfo info = ACCutil.getACCinfo(pb);
    Block prevBlock = pb.getPrev();
    if(prevBlock!=null){
      if(prevBlock.Opcode() == Xcode.ACC_PRAGMA){
        ACCinfo prevInfo = ACCutil.getACCinfo(prevBlock);
        if(prevInfo==null){/*fatal*/}
        switch(prevInfo.getPragma()){
        case PARALLEL:
        case PARALLEL_LOOP:
        case KERNELS:
        case KERNELS_LOOP:
        //case UPDATE:
          if(prevInfo.isAsync()==false && info.isAsync() == false){
            prevInfo.setAsyncExp(null);
          }
          break;
        default:
        }
      }
    }
  }
  
  private void analyzeVarList(ACCinfo accInfo, ACCpragma clause, XobjList clauseArgs) throws ACCexception{
    for(Xobject x : clauseArgs){
      if(x.Opcode() == Xcode.LIST){
        Xobject var = x.getArg(0);
        XobjList subscripts = (XobjList)x.copy();
        subscripts.removeFirstArgs();
        accInfo.declACCvar(var.getName(), subscripts, clause);
      }else{
        accInfo.declACCvar(x.getName(), clause);
      }
    }
  }
  
}
