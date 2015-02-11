package exc.openacc;

import java.util.*;

import exc.block.*;
import exc.object.*;


public class ACCinfo {
  private ACCinfo parent;
  private Block block; /* back link */
  private ACCglobalDecl globalDecl;
  ACCpragma pragma; /* directives */
  private Xobject xObject;

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
  
  ACCinfo(ACCpragma pragma, Xobject xObj, ACCglobalDecl globalDecl){
    this.pragma = pragma;
    this.globalDecl = globalDecl;
    this.xObject = xObj;
    
    parent = null;
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
    
    if(! num_gangsExp.isIntConstant()){
      throw new ACCexception("only constant integer is allowed for num_gangs");
    }
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
    
    if(! vect_lenExp.isIntConstant()){
      throw new ACCexception("only constant integer is allowed for vector_length");
    }
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
      this.collapseNum = n;
    }else{
      throw new ACCexception("collapse clause arg must be positive integer");
    }
  }
  public int getCollapseNum(){
    if(collapseNum != 0){
      return collapseNum;
    }else{
      return 1;
    }
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
  
  /** @return true if the variable is always allocated */
  public boolean isVarAllocated(String varName){
    for(ACCinfo info = this; info != null; info = info.parent){
      if(info.isDisabled()) continue;

      if(info.isEnabled()){
        ACCvar var = info.getACCvar(varName);
        if(var != null){
          if(var.isPresent() || var.isAllocated()) return true;
        }
      }
    }
    return false;
  }
  public boolean isVarAllocated(Ident id){
    for(ACCinfo info = this; info != null; info = info.parent){
      if(info.isDisabled()) continue;
      
      if(info.isEnabled()){
        ACCvar var = info.getACCvar(id);
        if(var != null){
          if(var.isPresent() || var.isAllocated()) return true;
        }
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
  
  public boolean isVarReduction(String varName){
    ACCvar var = getACCvar(varName);
    if(var != null){
      if(var.isReduction()) return true;
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
    declACCvar(varName, null, atr);
  }
      
  public void declACCvar(String varName, XobjList subscripts, ACCpragma atr) throws ACCexception{
    ACCvar parentACCvar;
    ACCvar newACCvar;
    Ident newACCvarId;
    
//    if((newACCvar = getACCvar(varName)) != null){
//      FIXME allows subarray
//      ACC.fatal("VAR:" + varName + " is already specified in same directive");
//      return;
//    }
    
    //check conflict
//    if(conflictsWith(varName, subscripts)){
//      //ACC.fatal("VAR:" + varName + " is already specified in same directive");
//      throw new ACCexception("VAR:" + varName + " is already specified in same directive");
//    }
    
    Ident varId;
    if(block != null){
        varId = block.findVarIdent(varName);
    }else{
      varId = xObject.findVarIdent(varName);
    }
      
    switch(atr){
    case COPY:
    case COPYIN:
    case COPYOUT:
    case CREATE:
    case DELETE:
      if(isVarAllocated(varId)){
        throw new ACCexception("'" + varName + "' is already allocated on device memory");
      }
      break;
    case PRESENT:
//      if(! isVarAllocated(varName)){
//        XobjList funcInfo = getFuncInfo();
//        XobjList funcParams = (XobjList)funcInfo.getArg(1);
//
//        if(ACCutil.hasIdent(funcParams, varName)) break;
//
//        if(globalDecl.findVarIdent(varName) != null) break;
//        
//        throw new ACCexception("'" + varName + "' is not allocated.");
//      }
      break;
    case PRESENT_OR_COPY:
    case PRESENT_OR_COPYIN:
    case PRESENT_OR_COPYOUT:
    case PRESENT_OR_CREATE:
      if(isVarAllocated(varName)){ //not enough. 
        ACC.debug("'" + varName + "' is already allocated. '" + atr.getName() + "' is same as 'present'.");
      }else{
        //ACC.debug("'" + varName + "' is not allocated yet. '" + atr.getName() + "' is same as '" + atr.getName().substring("PRESENT_OR_".length()) + "'.");
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
      
    case USE_DEVICE:
      if(! isVarAllocated(varName)){
        throw new ACCexception("'" + varName + "' is not allocated");
      }
      break;
      
    case PRIVATE:
    case FIRSTPRIVATE:
      if(varId.Type().isArray()){
        //throw new ACCexception("'" + varName + "' is an array. Array is not allowed in " + atr.getName() + ".");
      }break;
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
      
      //newACCvarId = block.findVarIdent(varName);
      newACCvarId = varId;//findVarIdent(varName);
      if(newACCvarId == null){
        ACC.fatal("VAR:" + varName + " is not declared");
        return;
      }
    }
    
    //check subscripts
    XobjList newSubscripts = null;
    if(subscripts != null){
    newSubscripts = Xcons.List();
    for(Xobject subscript : subscripts){
	XobjList newSubscript = Xcons.List();
	Xobject lower = subscript.getArg(0);
	Xobject length = subscript.getArgOrNull(1);
	String vName;
	Ident vId;
	if(!lower.isIntConstant()){
	    if(lower.Opcode() == Xcode.VAR){
	    vName = lower.getName();
	    vId = block.findVarIdent(vName);
	    if(vId == null){
		throw new ACCexception("'" + vName + "' is undefined");
	    }
	    lower = vId.Ref();
	    }
	}
	newSubscript.add(lower);
	if(length != null){
	    if(!length.isIntConstant()){
	        if(length.Opcode() == Xcode.VAR){

		vName = length.getName();
		vId = block.findVarIdent(vName);
		if(vId == null){
		    throw new ACCexception("'" + vName + "' is undefined");
		}
		length = vId.Ref();
	        }
	    }
	    newSubscript.add(length);
	}
	newSubscripts.add(newSubscript);
    }
    }
    
    newACCvar = getACCvar(varName, newSubscripts);
    if(newACCvar == null || atr == ACCpragma.HOST || atr == ACCpragma.DEVICE){ //FIXME
      newACCvar = new ACCvar(newACCvarId, newSubscripts, atr, parentACCvar);
      varList.add(newACCvar);
    }else{
      newACCvar.setAttribute(atr);
    }
    //if(! subscripts.isEmpty()){
      //newACCvar.setRange(subscripts);
    //}
    
  }
  
  public ACCvar getACCvar(String varName){
    for(ACCvar accVar : varList){
      if(accVar.getName().equals(varName)){
        return accVar;
      }
    }
    return null;
  }
  //returns ACCvar that has same region described by subscripts
  public ACCvar getACCvar(String varName, XobjList subscripts){
    for(ACCvar accVar : varList){
      if(accVar.getName().equals(varName)){
        if(accVar.contains(subscripts))
        return accVar;
      }
    }
    return null;
  }
  public ACCvar getACCvar(Ident id){
    for(ACCvar accVar : varList){
      if(accVar.getId().equals(id)){
        return accVar;
      }
    }
    return null;
  }
  
  //FIXME change function name to getParentACCinfo
  ACCinfo getOuterACCinfo(){
    if(block != null){
      for(Block b = block.getParentBlock(); b != null; b = b.getParentBlock()){
        //if(b.Opcode() != Xcode.ACC_PRAGMA);
        ACCinfo info = ACCutil.getACCinfo(b); 
        if(info != null) return info;
      }
    }
    List<XobjectDef> defs = globalDecl.getEnv().getDefs();
    for(int i = defs.size() - 1; i >= 0; i--){
      XobjectDef def = defs.get(i);
      Xobject x = def.getDef();
      if(x.Opcode() == Xcode.ACC_PRAGMA){
        Object info = x.getProp(ACC.prop);
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
  
  private Ident findVarIdent(String name)
  {
      for(Block b = block; b != null; b = b.getParentBlock()){
        BlockList body = b.getBody();
        Ident id = body.findLocalIdent(name);
        if(id != null) return id;
      }
      return globalDecl.getEnv().findVarIdent(name);
  }
  
  private boolean conflictsWith(String varName, XobjList subscripts){
    for(ACCvar var : varList){
      if(!var.getName().equals(varName)) continue;
      if(var.conllidesWith(subscripts)) return true;
    }
    return false;
  }
  
  public Iterable<ACCvar> getACCvars(ACCpragma atr){
    Iterable <ACCvar> vars = new Iterable<ACCvar>(){
      public Iterator<ACCvar> iterator(){
        return new Iterator<ACCvar>(){
          private ACCvar var;
          private Iterator<ACCvar> varIter = getVars();
          @Override
          public boolean hasNext() {
            while(varIter.hasNext()){
              var = varIter.next(); 
              if(var.isPrivate()){ //一時的に
                return true;
              }
            }
            return false;
          }
          @Override
          public ACCvar next(){
            return var;
          }
          @Override
          public void remove() {}
        };
      }
    };
    
    return vars;
  }

}
