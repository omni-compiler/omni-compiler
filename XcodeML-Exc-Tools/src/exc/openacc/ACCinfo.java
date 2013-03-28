package exc.openacc;

import java.util.*;
import exc.block.*;
import exc.object.*;


public class ACCinfo {
  private ACCinfo parent;
  private Block block; /* back link */
  private ACCglobalDecl globalDecl;
  ACCpragma pragma; /* directives */

  private List<ACCvar> varList; /* variable list */
  
  private Xobject ifCond = null;
  
  private boolean isAsync = false;
  private Xobject asyncExp = null;
  
  private Xobject waitExp = null;
  
  private Set<ACCpragma> execModels; 
  private Xobject num_gangsExp = null;
  private Xobject num_workersExp = null;
  private Xobject vect_lenExp = null;

  private int collapseNum = 0;
  
  private boolean isIndependent = false;
  
  //for rewrite
  //private List<Ident> idList = null;
  private XobjList idList = null;
  @Deprecated
  private List<Block> beginBlockList = null;
  private Block replaceBlock = null;
  @Deprecated
  private List<Block> endBlockList = null;
  private Block beginBlock = null;
  private Block endBlock = null;
  private XobjList declList = null;
  
  
  ACCinfo(ACCpragma pragma, Block b, ACCglobalDecl globalDecl){
    this.pragma = pragma;
    this.block = b; 
    this.globalDecl = globalDecl;
    
    parent = getOuterACCinfo();
    varList = new ArrayList<ACCvar>();
    execModels = new HashSet<ACCpragma>();
  }
  
  /** @return true if expression is scalar integer */
  private boolean isExpScalarInt(Xobject exp){
    if(exp == null) return false;
    
    if(exp.Opcode() == Xcode.VAR){
      String varName = exp.getName();
      Ident varId = block.findVarIdent(varName);
      if(varId.Type() == Xtype.intType){
        return true;
      }
    }else if(exp.isIntConstant()){
      return true;
    }
    return false;
  }
  
  public void setIfCond(Xobject ifCond) throws ACCexception{
    if(this.ifCond != null) throw new ACCexception("if clause is already specified"); 

    if(ifCond != null){
      if(isExpScalarInt(ifCond)){
        this.ifCond = ifCond;
      }
      else throw new ACCexception("'" + ifCond + "' is not scalar integer expression");
    }else{
      this.ifCond = null;
    }
  }
  public Xobject getIfCond(){
    return ifCond;
  }

  public void setAsyncExp(Xobject asyncExp) throws ACCexception{
    if(isAsync) throw new ACCexception("async clause is already specified");

    if(asyncExp != null){
      if(isExpScalarInt(asyncExp)){
        this.asyncExp = asyncExp;        
      }
      else throw new ACCexception("'" + asyncExp + "' is not scalar integer expression");
    }
    isAsync = true;
  }
  public boolean isAsync(){
    return isAsync;
  }
  public Xobject getAsyncExp() throws ACCexception{
    if(! isAsync) throw new ACCexception("it is not async");
    return asyncExp;
  }
  
  public void setWaitExp(Xobject waitExp) throws ACCexception{
    if(this.waitExp != null) throw new ACCexception("wait is already specified");
    
    if(waitExp != null){
      if(isExpScalarInt(waitExp)){
        this.waitExp = waitExp;
      }else{
        throw new ACCexception("'" + waitExp + "' is not scalar integer expression");
      }
    }
  }
  public Xobject getWaitExp() throws ACCexception{
    return waitExp;
  }

  public void addExecModel(ACCpragma execModel) throws ACCexception{
    if(execModels.contains(execModel)){
      ACC.warning("'" + execModel.getName() + "' is already specified");
      return;
    }
    switch(execModel){
    case GANG:
    case WORKER:
    case VECTOR:
      if(execModels.contains(ACCpragma.SEQ)){
        throw new ACCexception("this loop is already specified as seq");
      }
      execModels.add(execModel);
      break;
    case SEQ:
      if(execModels.contains(ACCpragma.GANG) || execModels.contains(ACCpragma.WORKER)
          || execModels.contains(ACCpragma.VECTOR)){
        throw new ACCexception("this loop is already specified as (gang|worker|vector)");
      }
      execModels.add(ACCpragma.SEQ);
      break;
    default:
      throw new ACCexception(execModel.getName() + " is not execModel");
    }
  }
  public Iterator<ACCpragma> getExecModels(){
    return execModels.iterator();
  }
  
  public void setNumGangsExp(Xobject num_gangsExp) throws ACCexception{
    if(this.num_gangsExp != null) throw new ACCexception("number of gangs is already specified"); 
    if(num_gangsExp == null) return;
    
    if(isExpScalarInt(num_gangsExp)){
      this.num_gangsExp = num_gangsExp;
    }else{
      throw new ACCexception("'" + num_gangsExp + "' is not scalar integer expression");
    }
  }
  public Xobject getNumGangsExp(){
    return num_gangsExp;
  }

  public void setNumWorkersExp(Xobject num_workersExp) throws ACCexception{
    if(this.num_workersExp != null) throw new ACCexception("number of workers is already specified");
    if(num_workersExp == null) return;
    
    if(isExpScalarInt(num_workersExp)){
      this.num_workersExp = num_workersExp;
    }else{
      throw new ACCexception("'" + num_workersExp + "' is not scalar integer expression");
    }
  }
  public Xobject getNumWorkersExp(){
    return num_workersExp;
  }
  
  public void setVectorLengthExp(Xobject vect_lenExp) throws ACCexception{
    if(this.vect_lenExp != null) throw new ACCexception("vector length is already specified");
    if(vect_lenExp == null) return;
    
    if(isExpScalarInt(vect_lenExp)){
      this.vect_lenExp = vect_lenExp;
    }else{
      throw new ACCexception("'" + vect_lenExp + "' is not scalar integer expression");
    }
  }
  public Xobject getVectorLengthExp(){
    return vect_lenExp;
  }


  public void setCollapseNum(Xobject collapseNum) throws ACCexception{
    if(this.collapseNum != 0) throw new ACCexception("collapse clause is already specified");
    if(collapseNum.Opcode() != Xcode.INT_CONSTANT){
      throw new ACCexception("collapse clause arg must be integer constant");
    }
    
    int n = collapseNum.getInt();
    if(n > 0){
      //FIXME check loop nest
      this.collapseNum = n;
    }else{
      throw new ACCexception("collapse clause arg must be positive integer");
    }
  }
  public int getCollapseNum(){
    return collapseNum;
  }
  
  public void setIndependent() throws ACCexception{
    if(isIndependent) throw new ACCexception("independent clause is already specified");
    isIndependent = true;
  }
  public boolean isIndependent(){
    return isIndependent;
  }
  
  public ACCpragma getComputePragma(){
    for(ACCinfo info = this; info != null; info = info.parent){
      switch(info.pragma){
      case PARALLEL:
      case PARALLEL_LOOP:
        return ACCpragma.PARALLEL;
      case KERNELS:
      case KERNELS_LOOP:
        return ACCpragma.KERNELS;
      }
    }
    return null;
  }
  
  private ACCvar findOuterACCvar(String varName){
    for(ACCinfo info = parent; info != null; info = info.parent){
      ACCvar accVar = info.getACCvar(varName);
      if(accVar != null) return accVar;
    }
    return null;
  }
  
  /** @return true if this is always enabled */
  public boolean isEnabled(){
    if(ifCond == null) return true;
    if(ifCond.isIntConstant()){
      return ! ifCond.isZeroConstant();
    }
    return false;
  }
  
  /** @return true if this is always disabled */
  public boolean isDisabled(){
    if(ifCond != null && ifCond.isZeroConstant()) return true;
    return false;
  }
  
  public boolean isVarAllocated(String varName){
    for(ACCinfo info = this; info != null; info = info.parent){
      if(info.isDisabled()) continue;

      ACCvar var = info.getACCvar(varName);
      if(var != null){
        if(var.isPresent() || var.isAllocated()) return true;
      }
    }
    return false;
  }
  
  public boolean isVarPrivate(String varName){
    for(ACCinfo info = this; info != null; info = info.parent){
      ACCvar var = info.getACCvar(varName);
      if(var != null){
        if(info.isEnabled() && var.isPrivate()) return true;
      }
    }
    return false;
  }
  public boolean isVarFirstprivate(String varName){
    for(ACCinfo info = this; info != null; info = info.parent){
      ACCvar var = info.getACCvar(varName);
      if(var != null){
        if(info.isEnabled() && var.isFirstprivate()) return true;
      }
    }
    return false;
  }
  
  public Ident getDevicePtr(String varName){
    for(ACCinfo info = this; info != null; info = info.parent){
      if(info.isDisabled()) continue;
      
      ACCvar var = info.getACCvar(varName);
      if(var != null){
        Ident deviceptr = var.getDevicePtr();
        if(deviceptr != null){
          return deviceptr;
        }
      }
    }
    return null;
  }
  
  XobjList getFuncInfo(){
    for(Block b = this.block; b != null; b = b.getParentBlock()){
      if(b.Opcode() == Xcode.FUNCTION_DEFINITION){
        FunctionBlock fb = (FunctionBlock)b;
        String funcName = fb.getName();
        Xobject funcParams = fb.getBody().getIdentList();

        return Xcons.List(Xcons.String(funcName), funcParams);
      }
    }
    ACC.fatal("cant't get func info");
    return null;
  }
      
  public void declACCvar(String varName, ACCpragma atr) throws ACCexception{
    ACCvar parentACCvar;
    ACCvar newACCvar;
    Ident newACCvarId;
    
    if(getACCvar(varName) != null){
      //FIXME allows subarray
      ACC.fatal("VAR:" + varName + " is already specified in same directive");
      return;
    }
    
    switch(atr){
    case COPY:
    case COPYIN:
    case COPYOUT:
    case CREATE:
      if(isVarAllocated(varName)){
        throw new ACCexception("'" + varName + "' is already allocated on device memory");
      }
      break;
    case PRESENT:
      if(! isVarAllocated(varName)){
        //throw new ACCexception("'" + varName + "' is not allocated yet.");
        //check func params
        XobjList funcInfo = getFuncInfo();
        //String funcName = funcInfo.getArg(0).getString();
        XobjList funcParams = (XobjList)funcInfo.getArg(1);
        /*
        int paramNum = 0;
        for(Xobject o : funcParams){
          Ident id = (Ident)o;
          if(id != null){
            if(id.getName().equals(varName)){
              Xcode code = id.getValue().Opcode();
              globalDecl.setCalleeInfo(funcName, paramNum, code, true);
              System.out.println("setfuncparaminfo:"+id + "(" + paramNum + ")");
            }
          }
          paramNum++;
        }
        if(paramNum > funcParams.Nargs()){ //not found in params
          throw new ACCexception("'" + varName + "' is not allocated.");
        }
        */
        if(! ACCutil.hasIdent(funcParams, varName)){
          throw new ACCexception("'" + varName + "' is not allocated.");
        }
      }
      break;
    case PRESENT_OR_COPY:
    case PRESENT_OR_COPYIN:
    case PRESENT_OR_COPYOUT:
    case PRESENT_OR_CREATE:
      if(isVarAllocated(varName)){
        ACC.debug("'" + varName + "' is already allocated. '" + atr.getName() + "' is same as 'present'.");
      }else{
        ACC.debug("'" + varName + "' is not allocated yet. '" + atr.getName() + "' is same as '" + atr.getName().substring("PRESENT_OR_".length()) + "'.");
      }
      break;
    case DEVICEPTR:
      if(getDevicePtr(varName) != null){
        ACC.warning("'" + varName + "' is already specified as deviceptr.");
      }   
      break;
    case HOST:
    case DEVICE:
      if(! isVarAllocated(varName)){
        throw new ACCexception("'" + varName + "' is not allocated");
      }
      break;
      
    case PRIVATE:
    case FIRSTPRIVATE:
    case CACHE:
      break;
    default:
      if(! atr.isReduction()){
        throw new ACCexception("'" + atr.getName() + "' is unknown attribute");        
      }
    }

    parentACCvar = findOuterACCvar(varName);
    if(parentACCvar != null){
      newACCvarId = parentACCvar.getId();

    }else{

      newACCvarId = block.findVarIdent(varName);
      if(newACCvarId == null){
        ACC.fatal("VAR:" + varName + " is not declared");
        return;
      }
    }
    

    newACCvar = new ACCvar(newACCvarId, atr, parentACCvar);
    varList.add(newACCvar);
    
  }
  
  public ACCvar getACCvar(String varName){
    for(ACCvar accVar : varList){
      if(accVar.getName().equals(varName)){
        return accVar;
      }
    }
    return null;
  }
  
  ACCinfo getOuterACCinfo(){
    for(Block b = block.getParentBlock(); b != null; b = b.getParentBlock()){
      if(b.Opcode() == Xcode.ACC_PRAGMA){
        Object info = b.getProp(ACC.prop);
        if(info != null) return (ACCinfo)info;
      }
    }
    return null;
  }
  
  public ACCpragma getPragma(){
    return pragma;
  }
  @Deprecated
  public void setBeginBlockList(List<Block> beginBlockList){
    this.beginBlockList = beginBlockList;
  }
  @Deprecated
  public void setEndBlockList(List<Block> endBlockList){
    this.endBlockList = endBlockList;
  }
  public void setBeginBlock(Block beginBlock){
    this.beginBlock = beginBlock;
  }
  public void setEndBlock(Block endBlock){
    this.endBlock = endBlock;
  }
  public void setReplaceBlock(Block replaceBlock){
    this.replaceBlock = replaceBlock;
  }
  @Deprecated
  public List<Block> getBeginBlockList(){
    return beginBlockList;
  }
  @Deprecated
  public List<Block> getEndBlockList(){
    return endBlockList;
  }
  public Block getBeginBlock(){
    return beginBlock;
  }
  public Block getEndBlock(){
    return endBlock;
  }
  public Block getReplaceBlock(){
    return replaceBlock;
  }
  //public void setIdList(List<Ident> idList){
  public void setIdList(XobjList idList){
    this.idList = idList;
  }
  //public List<Ident> getIdList(){
  public XobjList getIdList(){
    return idList;
  }
  public void setDeclList(XobjList declList){
    this.declList = declList;
  }
  public XobjList getDeclList(){
    return declList;
  }
  
  public ACCglobalDecl getGlobalDecl(){
    return globalDecl;
  }
  
  public Block getBlock(){
    return block;
  }
  
  public Iterator<ACCvar> getVars(){
    return varList.iterator();
  }

  
  public Ident getHostDescId(String varName) {
    for(ACCinfo info = this; info != null; info = info.parent){
      if(info.isDisabled()) continue;
      ACCvar var = info.getACCvar(varName);
      if(var != null){
        Ident hostDescId = var.getHostDesc();
        if(hostDescId != null) return hostDescId;
      }
    }
    return null;
  }
  
  public ACCinfo getParent(){
    return parent;
  }
  
  /*
  public Iterator<ACCvar> getReductionVars(){
    List<ACCvar> reductionVars = new ArrayList<ACCvar>();
    for(ACCvar v : varList){
      if(v.isReduction()){
        reductionVars.add(v);
      }
    }
    return reductionVars.iterator();
  }*/
  
}
