package exc.openacc;

import java.util.*;

import exc.block.*;
import exc.object.*;

public class ACCgpuKernel {
  private ACCinfo kernelInfo; //parallel or kernels info
  private ACCgpuManager gpuManager;
  private static final String ACC_GPU_FUNC_PREFIX ="_ACC_GPU_FUNC"; 
  private static final String ACC_REDUCTION_VAR_PREFIX = "_ACC_reduction_";
  private static final String ACC_CACHE_VAR_PREFIX = "_ACC_cache_";
  
  private List<XobjectDef> _varDecls = new ArrayList<XobjectDef>();
  
  public List<Block> _kernelBlocks; 
  private List<Ident> _outerIdList;
  
  private Set<Ident> _readOnlyOuterIdSet;
  private Set<Ident> _inductionVarIdSet;
  
  private ArrayDeque<Loop> loopStack = new ArrayDeque<Loop>(); 
  
  XobjList cacheSizeList = Xcons.List();
  
  private SharedMemory sharedMemory = new SharedMemory();
  
  private ReductionManager reductionManager = new ReductionManager("test_kernels");


  public ACCgpuKernel(ACCinfo kernelInfo, Block kernelBlock) {
    this.kernelInfo = kernelInfo;
    this.gpuManager = new ACCgpuManager(kernelInfo);
  }
  
  public ACCgpuKernel(ACCinfo kernelInfo, List<Block> kernelBlocks){
    this.kernelInfo = kernelInfo;
    this._kernelBlocks = kernelBlocks;
    this.gpuManager = new ACCgpuManager(kernelInfo);
  }
  
  //ok!
  public Block makeLaunchFuncCallBlock(){
    List<Block> kernelBody = _kernelBlocks;
    String launchFuncName = kernelInfo.getGlobalDecl().genSym(ACC_GPU_FUNC_PREFIX);
    
    //make deviceKernelDef
    String deviceKernelName = launchFuncName + "_DEVICE";
    XobjectDef deviceKernelDef = makeDeviceKernelDef(deviceKernelName, _outerIdList, kernelBody);
    
    //make launchFuncDef
    XobjectDef launchFuncDef = makeLaunchFuncDef(launchFuncName, _outerIdList, deviceKernelDef);
    
    //decompile deviceKernel and launchFunction
    try{
      new ACCgpuDecompiler().decompile(kernelInfo.getGlobalDecl().getEnv(), deviceKernelDef, (Ident)deviceKernelDef.getNameObj(), launchFuncDef, _varDecls);
    }catch(ACCexception e){
      ACC.fatal("makeKernelCallBlock(" + launchFuncName + e.getMessage());
    }
    
    //make launchFuncCall
    Ident launchFuncId = (Ident)launchFuncDef.getNameObj();
    XobjList launchFuncArgs = makeLaunchFuncArgs();
    Block launchFuncCallBlock = Bcons.Statement(launchFuncId.Call(launchFuncArgs));
    
    return launchFuncCallBlock;
  }
  
  //oK?
  private XobjList makeLaunchFuncArgs(){
    XobjList launchFuncArgs = Xcons.List();
    for(Ident id : _outerIdList){
      Xobject arg = null;
      switch(id.Type().getKind()){
      case Xtype.ARRAY:
      case Xtype.POINTER:
      {
        Ident devicePtr = kernelInfo.getDevicePtr(id.getName());
        arg = devicePtr.Ref();
      } break;
      case Xtype.BASIC:
      case Xtype.STRUCT:
      case Xtype.UNION:
      case Xtype.ENUM:
      {
        Ident devicePtrId;
        String idName = id.getName(); 
        if(_readOnlyOuterIdSet.contains(id)){ //is this condition appropriate?
          devicePtrId = kernelInfo.getDevicePtr(idName);
          if(devicePtrId != null){
            arg = devicePtrId.Ref();
          }else{
            arg = id.Ref();
          }
        }else{
          try{
            devicePtrId = kernelInfo.getDevicePtr(idName); 
            arg = devicePtrId.Ref();
          }catch(Exception e){
            ACC.fatal("makeLaunchFuncArgs" + e.getMessage());
          }
        }
      } break;
      default:
        ACC.fatal("unknown type");
      }
      launchFuncArgs.add(arg); 
      
      
      /*
      if(id.Type().isArray()){
        Ident devicePtr = kernelInfo.getDevicePtr(id.getName());
        launchFuncArgs.add(devicePtr);
      }else{
        if(_readOnlyOuterIdSet.contains(id)){
          Ident devicePtr = kernelInfo.getDevicePtr(id.getName());
          if(devicePtr == null){
            launchFuncArgs.add(id.Ref());
          }else{
            launchFuncArgs.add(devicePtr.Ref());
          }
        }else{
          Ident devicePtr = kernelInfo.getDevicePtr(id.getName());
          if(devicePtr == null){
            ACC.fatal("dev ptr not found");
          }
          launchFuncArgs.add(devicePtr.Ref());
        }
      }
      */

    }
    return launchFuncArgs;
  }
 
  private Set<Ident> collectOuterIdents(Block topBlock){
    Set<Ident> outerIdSet = new HashSet<Ident>();
    
    BlockIterator blockIter = new topdownBlockIterator(topBlock);
    for(blockIter.init(); !blockIter.end(); blockIter.next()){
      Block b = blockIter.getBlock();

      switch(b.Opcode()){
      case COMPOUND_STATEMENT:
      {
        //search in decls
        BlockList body = b.getBody();
        XobjList declList = (XobjList)body.getDecls();
        for(Xobject decl : declList){
          Xobject init = decl.getArgOrNull(1);
          if(init != null){
            outerIdSet.addAll(collectOuterIdents(topBlock, body, decl));
          }
        }
      }break;
      
      case FOR_STATEMENT:
        outerIdSet.addAll(collectOuterIdents(topBlock, b.getInitBBlock()));
        outerIdSet.addAll(collectOuterIdents(topBlock, b.getIterBBlock()));
      case WHILE_STATEMENT:
      case DO_STATEMENT:
      case IF_STATEMENT:
      case SWITCH_STATEMENT:
        outerIdSet.addAll(collectOuterIdents(topBlock, b.getCondBBlock()));
        break;
        
      case LIST: //simple block
        outerIdSet.addAll(collectOuterIdents(topBlock, b.getBasicBlock()));
        break;
        
      default:
      }
    }
    
    return outerIdSet;
  }
  
  private Set<Ident> collectOuterIdents(Block topBlock, BlockList body, Xobject x){
    Set<Ident> outerIdSet = new HashSet<Ident>();
    for(String varName : collectVarNames(x)){
      Ident id = findOuterBlockIdent(topBlock, body, varName);
      if(id != null){
        outerIdSet.add(id);
      }
    }
    return outerIdSet;
  }
  
  private Set<Ident> collectOuterIdents(Block topBlock, BasicBlock bb){
    return collectOuterIdents(topBlock, bb.getParent().getParent(), bb.toXobject());
  }
  
  private Ident findOuterBlockIdent(Block topBlock, BlockList body, String name)
  {
    // if the id exists between topBlock to bb, the id is not outerId
    for(BlockList b_list = body; b_list != null; b_list = b_list.getParentList()){
      if(b_list == topBlock.getParent()) break;
      if(b_list.findLocalIdent(name) != null) return null;
    }

    return topBlock.findVarIdent(name);
  }
  
  private Set<String> collectVarNames(Xobject expr){
    Set<String> varNameSet = new HashSet<String>();
    
    topdownXobjectIterator exprIter = new topdownXobjectIterator(expr);
    for (exprIter.init(); !exprIter.end(); exprIter.next()) {
      Xobject x = exprIter.getXobject();
      if(x == null) continue;
      switch (x.Opcode()) {
      case VAR: 
      {
        String varName = x.getName();
        varNameSet.add(varName);
      } break;
      case ARRAY_REF:
      {
        String arrayName = x.getArg(0).getName();
        varNameSet.add(arrayName);
      } break;
      default:
      }
    }
    return varNameSet;
  }
  
  
  private XobjectDef makeDeviceKernelDef(String deviceKernelName, List<Ident> outerIdList, List<Block> kernelBody){
    /* make deviceKernelBody */
    XobjList deviceKernelLocalIds = Xcons.IDList();
    List<Block> initBlocks = new ArrayList<Block>();
    
    //add localIds from private vars
    Iterator<ACCvar> varIter = kernelInfo.getVars();
    while(varIter.hasNext()){
      ACCvar var = varIter.next();
      if(var.isPrivate()){
        deviceKernelLocalIds.add(Ident.Local(var.getName(), var.getId().Type()));
      }
    }
    
    //make params
    XobjList deviceKernelParamIds = Xcons.IDList();
    //add paramId from outerId
    for(Ident id : outerIdList){
      if(_readOnlyOuterIdSet.contains(id) && id.Type().isArray()){
        Xtype constParamType = makeConstRestrictType(Xtype.voidType);
        Ident constParamId = Ident.Param("_ACC_cosnt_" + id.getName(), constParamType);
       
        Xtype arrayPtrType = Xtype.Pointer(id.Type().getRef());//Xtype.Pointer(id.Type());
        Ident localId = Ident.Local(id.getName(), arrayPtrType);//makeParamId_new(id);
        Xobject initialize = Xcons.Set(localId.Ref(), Xcons.Cast(arrayPtrType, constParamId.Ref()));
        deviceKernelParamIds.add(constParamId);
        deviceKernelLocalIds.add(localId);
        initBlocks.add(Bcons.Statement(initialize));
      }else{
        deviceKernelParamIds.add(makeParamId_new(id));
      }
    }
    

    XobjList additionalParams = Xcons.IDList();
    XobjList additionalLocals = Xcons.IDList();
    
    //make mainBody
    //Block deviceKernelMainBlock = makeDeviceKernelCoreBlock(initBlocks, kernelBlock.getBody(), additionalParams, additionalLocals, null, deviceKernelId, gpuManager);
    //Block deviceKernelMainBlock = makeDeviceKernelCoreBlock(initBlocks, kernelBody.get(0).getBody(), additionalParams, additionalLocals, null, deviceKernelName, gpuManager);
    Block deviceKernelMainBlock = makeCoreBlock(initBlocks, kernelBody, additionalParams, additionalLocals, null, deviceKernelName);
    rewriteReferenceType(deviceKernelMainBlock, deviceKernelParamIds); //does it need outerIdList?


    //add localId, paramId from additional
    deviceKernelLocalIds.mergeList(additionalLocals);
    deviceKernelParamIds.mergeList(additionalParams);
    
    //if extern_sm is used, add extern_sm_id & extern_sm_offset_id
    if(sharedMemory.isUsed()){
      deviceKernelLocalIds.add(sharedMemory.externSmId);
      deviceKernelLocalIds.add(sharedMemory.smOffsetId);
    }
    
    //if block reduction is used
    if(reductionManager.hasBlockReduction){
 
      deviceKernelLocalIds.mergeList(reductionManager.getBlockReductionParamIds());//temporary//deviceKernelParamIds.mergeList(reductionManager.getBlockReductionParamIds());

      
      deviceKernelLocalIds.mergeList(reductionManager.getBlockReductionLocalIds());
    }
    

    BlockList deviceKernelBody = Bcons.emptyBody();
    
    //FIXME add extern_sm init func
    if(sharedMemory.isUsed()){
      deviceKernelBody.add(sharedMemory.makeInitFunc());
    }
    if(reductionManager.hasBlockReduction){

      /*temporary*/
      Ident reductionCnt = Ident.Var(deviceKernelName + "_red_cnt", Xtype.unsignedType, Xtype.Pointer(Xtype.unsignedType), VarScope.GLOBAL);
      Ident reductionTmpPtr = Ident.Var(deviceKernelName + "_red_tmp", Xtype.voidPtrType, Xtype.Pointer(Xtype.voidPtrType), VarScope.GLOBAL);
      _varDecls.add(new XobjectDef(Xcons.List(Xcode.VAR_DECL, reductionCnt, Xcons.IntConstant(0))));
      _varDecls.add(new XobjectDef(Xcons.List(Xcode.VAR_DECL, reductionTmpPtr, Xcons.IntConstant(0))));
      deviceKernelBody.add(Bcons.Statement(Xcons.Set(reductionManager.counterPtr.Ref(), reductionCnt.getAddr())));
      deviceKernelBody.add(Bcons.Statement(Xcons.Set(reductionManager.tempPtrPtr.Ref(), reductionTmpPtr.getAddr())));
      /*end*/
      
      
      deviceKernelBody.add(reductionManager.makeInitFunc());
      deviceKernelBody.add(reductionManager.makeLocalVarInitFuncs());
    }
    
    for(Block b : initBlocks) deviceKernelBody.add(b);
    deviceKernelBody.add(deviceKernelMainBlock);
    
    if(reductionManager.hasBlockReduction){
      deviceKernelBody.add(reductionManager.makeReduceAndFinalizeFuncs());
    }
    
    deviceKernelBody.setIdentList(deviceKernelLocalIds);
    deviceKernelBody.setDecls(ACCutil.getDecls(deviceKernelLocalIds)); //is this need?
    
    Ident deviceKernelId = ACCutil.getMacroFuncId(deviceKernelName, Xtype.voidType);
    ((FunctionType)deviceKernelId.Type()).setFuncParamIdList(deviceKernelParamIds);

    XobjectDef deviceKernelDef = XobjectDef.Func(deviceKernelId, deviceKernelParamIds, null, deviceKernelBody.toXobject()); //set decls?
    return deviceKernelDef;
  }
  
  private Xtype makeConstRestrictType(Xtype xtype){
    Xtype copyType = xtype.copy();
    copyType.setIsConst(true);
    Xtype ptrType = Xtype.Pointer(copyType);
    ptrType.setIsRestrict(true);
    return ptrType;
  }
  
  public void rewriteReferenceType(Block b, XobjList paramIds){
    BasicBlockExprIterator iter = new BasicBlockExprIterator(b);
    for (iter.init(); !iter.end(); iter.next()) {
      Xobject expr = iter.getExpr();
      topdownXobjectIterator exprIter = new topdownXobjectIterator(expr);
      for (exprIter.init(); !exprIter.end(); exprIter.next()) {
        Xobject x = exprIter.getXobject();
        switch (x.Opcode()) {
        case VAR:
        {
          Ident id = ACCutil.getIdent(paramIds, x.getName()); //FIXME this should use findVarIdent
          if(id != null){
            Xtype x_type = x.Type();
            Xtype id_type = id.Type();
            if(! x.Type().equals(id.Type())){ 
              if(id.Type().equals(Xtype.Pointer(x.Type()))){
                Xobject newXobj = Xcons.PointerRef(id.Ref());
                exprIter.setXobject(newXobj);
              }else{
                ACC.fatal("type mismatch");
              }
            }
          }
        }break;
        case VAR_ADDR:
          // need to implement
        {
          Ident id = ACCutil.getIdent(paramIds, x.getName());
          if(id != null){
            if(! x.Type().equals(Xtype.Pointer(id.Type()))){ 
              if(x.Type().equals(id.Type())){
                Xobject newXobj = id.Ref();
                exprIter.setXobject(newXobj);
              }else{
                ACC.fatal("type mismatch");
              }
            }
          }
        }break;
        default:
        }
      }
    }
    return;
  }
  
  private Block makeCoreBlock(List<Block> initBlocks, Block b, XobjList paramIds, XobjList localIds, 
      String prevExecMethodName, String deviceKernelName){
    switch(b.Opcode()){
    case FOR_STATEMENT:
      return makeCoreBlockForStatement(initBlocks, (CforBlock)b, paramIds, localIds, prevExecMethodName, deviceKernelName);
    case COMPOUND_STATEMENT:
    case ACC_PRAGMA:
      return makeCoreBlock(initBlocks, b.getBody(), paramIds, localIds, prevExecMethodName, deviceKernelName);
    default:
    {
      Block resultBlock = b.copy();
      if(prevExecMethodName==null){
        Ident block_id = Ident.Local("_ACC_block_x_id", Xtype.intType); //this is not local var but macro.
        resultBlock = Bcons.IF(Xcons.binaryOp(Xcode.LOG_EQ_EXPR, block_id.Ref(),Xcons.IntConstant(0)), resultBlock, Bcons.emptyBlock()); //if(_ACC_block_x_id == 0){b}
      }else if(prevExecMethodName.equals("block_x")){
        Ident thread_id = Ident.Local("_ACC_thread_x_id", Xtype.intType); 
        resultBlock = Bcons.IF(Xcons.binaryOp(Xcode.LOG_EQ_EXPR, thread_id.Ref(),Xcons.IntConstant(0)), resultBlock, null);  //if(_ACC_thread_x_id == 0){b}
      }
      return resultBlock;
    }
    }
  }
  private Block makeCoreBlock(List<Block> initBlocks, BlockList body, XobjList paramIds, XobjList localIds, 
      String prevExecMethodName, String deviceKernelName){
    if(body == null) return Bcons.emptyBlock();
    
    Xobject ids = body.getIdentList();
    Xobject decls = body.getDecls();
    BlockList resultBody = Bcons.emptyBody(ids, decls);
    for(Block b = body.getHead(); b != null; b = b.getNext()){
      resultBody.add(makeCoreBlock(initBlocks, b, paramIds, localIds, prevExecMethodName, deviceKernelName));
    }
    return Bcons.COMPOUND(resultBody);
  }
  private Block makeCoreBlock(List<Block> initBlocks, List<Block> blockList, XobjList paramIds, XobjList localIds, 
      String prevExecMethodName, String deviceKernelName){

    BlockList resultBody = Bcons.emptyBody();
    for(Block b : blockList){
      resultBody.add(makeCoreBlock(initBlocks, b, paramIds, localIds, prevExecMethodName, deviceKernelName));
    }
    return makeBlock(resultBody); //resultBody.isEmpty()? Bcons.emptyBlock() : Bcons.COMPOUND(resultBody);
  }
  
  private Block makeBlock(BlockList blockList){
    if(blockList == null || blockList.isEmpty()){
      return Bcons.emptyBlock();
    }
    if(blockList.isSingle()){
      Xobject decls = blockList.getDecls();
      XobjList ids = blockList.getIdentList();
      if((decls == null || decls.isEmpty()) && (ids == null || ids.isEmpty())){
        return blockList.getHead();
      }
    }
    return Bcons.COMPOUND(blockList);
  }
  
  
  private Block makeCoreBlockForStatement(List<Block> initBlocks, CforBlock forBlock, XobjList paramIds, XobjList localIds, 
      String prevExecMethodName, String deviceKernelName){
    XobjList localIds_2 = Xcons.IDList();

    ACCinfo info = ACCutil.getACCinfo(forBlock);
    if(info == null || ! info.getPragma().isLoop()){
      loopStack.push(new Loop(forBlock));
      BlockList body = Bcons.blockList(makeCoreBlock(initBlocks, forBlock.getBody(), paramIds, localIds, prevExecMethodName, deviceKernelName));
      loopStack.pop();
      return Bcons.FOR(forBlock.getInitBBlock(), forBlock.getCondBBlock(), forBlock.getIterBBlock(), body);
    }

    String execMethodName = gpuManager.getMethodName_new(forBlock);
    if(execMethodName == ""){ //if execMethod is not defined or seq
      //return Bcons.FOR(forBlock.getInitBBlock(), forBlock.getCondBBlock(), forBlock.getIterBBlock(), Bcons.blockList(makeCoreBlock(initBlocks, forBlock.getBody(), paramIds, localIds, prevExecMethodName, deviceKernelName)));
      loopStack.push(new Loop(forBlock));
      BlockList body = Bcons.blockList(makeCoreBlock(initBlocks, forBlock.getBody(), paramIds, localIds, prevExecMethodName, deviceKernelName));
      loopStack.pop();
      return Bcons.FOR(forBlock.getInitBBlock(), forBlock.getCondBBlock(), forBlock.getIterBBlock(), body);
    }

    Block resultCenterBlock = Bcons.emptyBlock();
    List<Block> beginBlocks = new ArrayList<Block>();
    List<Block> endBlocks = new ArrayList<Block>();
    XobjList reductionVarIds = Xcons.IDList();
    XobjList reductionLocalVarIds = Xcons.IDList();
    List<Block> cacheLoadBlocks = new ArrayList<Block>();
    XobjList cachedIds = Xcons.IDList();
    XobjList cacheIds = Xcons.IDList();
    XobjList cacheOffsetIds = Xcons.IDList();
    XobjList cacheSizeIds = Xcons.IDList();

    //begin reduction
    Iterator<ACCvar> vars = info.getVars();
    while(vars.hasNext()){
      ACCvar var = vars.next();
      if(! var.isReduction()) continue;
        
      Ident redId = var.getId();
      
      boolean useGang=false;
      boolean useVector=false;
      
      if(execMethodName.startsWith("block")){
        useGang = true;
      }
      if(execMethodName.contains("thread")){
        useVector = true;
      }
      
      ACCpragma execMethod = null;
      if(useGang && useVector) execMethod = ACCpragma._BLOCK_THREAD;
      else if(useGang) execMethod = ACCpragma._BLOCK;
      else execMethod = ACCpragma._THREAD;
      
      
      Reduction reduction = reductionManager.addReduction(var, execMethod);
      
      if(! _readOnlyOuterIdSet.contains(var.getId()) && !useGang){ 
        // only thread reduction
        Ident localReductionVarId = reduction.getLocalReductionVarId();
        localIds_2.add(localReductionVarId);
        
        Block initReductionVarFuncCall = reduction.makeInitReductionVarFuncCall();
        beginBlocks.add(initReductionVarFuncCall);
        
        Block reductionFuncCall = reduction.makeThreadReductionFuncCall();
        endBlocks.add(reductionFuncCall);
      }
      reductionVarIds.add(var.getId());
      reductionLocalVarIds.add(reduction.localVarId);
      
      //XobjList redTmpArgs = Xcons.List();
      //if(needsTemp(var) && execMethodName.startsWith("block")){ //this condition will be merged.
        //Ident ptr_red_tmp = Ident.Param("_ACC_gpu_red_tmp_" + redId.getName(), Xtype.Pointer(redId.Type()));
        //Ident ptr_red_cnt = Ident.Param("_ACC_gpu_red_cnt_" + redId.getName(), Xtype.Pointer(Xtype.unsignedType));
        //Ident prt_red_tmp = Ident.Var(deviceKernelName + "_red_tmp", Xtype.voidType, Xtype.voidPtrType, VarScope.GLOBAL);
        //Ident ptr_red_cnt = Ident.Var(deviceKernelName + "_red_cnt_" + redId.getName(), Xtype.unsignedType, Xtype.Pointer(Xtype.unsignedType), VarScope.GLOBAL);
        

        //paramIds.add(ptr_red_tmp);
        //paramIds.add(ptr_red_cnt);
        //redTmpArgs.add(ptr_red_tmp.Ref());
        //redTmpArgs.add(Xcons.AddrOfVar(ptr_red_cnt.Ref()));
        //_tmpUsingReductionVars.add(var);
        //_varDecls.add(new XobjectDef(Xcons.List(Xcode.VAR_DECL, ptr_red_cnt, Xcons.IntConstant(0))));
      //}

      //localIds.add(localRedId);

      //int reductionKind = getReductionKindInt(var.getReductionOperator());
      
//      if(execMethodName.equals("block_x")){
//        beginBlocks.add(ACCutil.createFuncCallBlock("_ACC_gpu_init_reduction_var", Xcons.List(localRedId.getAddr(), Xcons.IntConstant(reductionKind))));
//      }
      
//      XobjList reductionFuncCallArgs = null;// = Xcons.List();
//      if(needsTemp(var)){
//        Ident redTmpPtrPtr = reductionManager.reductionTempArrayAddr;
//        Ident redCntPtr = reductionManager.reductionCounterAddr;
//        if(execMethodName.equals("block_x")){
//          //reductionFuncCallArgs = Xcons.List()
//        }else if(execMethodName.equals("thread_x")){
//          
//        }else{ //block_thread_x
//          reductionFuncCallArgs = Xcons.List(Xcons.PointerRef(redTmpPtrPtr.Ref()), localRedId.Ref(), Xcons.IntConstant(reductionKind));
//        }
//      }else{
//        if(execMethodName.equals("block_x")){
//          //reductionFuncCallArgs = Xcons.List()
//        }else if(execMethodName.equals("thread_x")){
//          
//        }else{ //block_thread_x
//          reductionFuncCallArgs = Xcons.List(localRedId.Ref(), Xcons.IntConstant(reductionKind));
//        }
//      }
      
      
      
      //XobjList funcCallArgs = Xcons.List(var.getId().getAddr(),localRedId.Ref(), Xcons.IntConstant(reductionKind));
      //funcCallArgs.mergeList(redTmpArgs);
//      endBlocks.add(ACCutil.createFuncCallBlock("_ACC_gpu_reduce_" + execMethodName, funcCallArgs)); 
    }//end reduction

    
    int collapseNum = info.getCollapseNum();
    if(collapseNum == 0){
      collapseNum = 1;
    }
    List<CforBlock> collapsedForBlockList = new ArrayList<CforBlock>();
    collapsedForBlockList.add(forBlock);

    {
      CforBlock tmpForBlock = forBlock;
      for(int i = 1; i < collapseNum; i++){
        tmpForBlock = (CforBlock)tmpForBlock.getBody().getHead();
        collapsedForBlockList.add(tmpForBlock);
      }
    }
    
    //make calc idx funcs
    List<Block> calcIdxFuncCalls = new ArrayList<Block>();
    XobjList vIdxIdList = Xcons.IDList();
    XobjList nIterIdList = Xcons.IDList();
    XobjList indVarIdList = Xcons.IDList();
    for(CforBlock tmpForBlock : collapsedForBlockList){
      String indVarName = tmpForBlock.getInductionVar().getName();
      Xobject init = tmpForBlock.getLowerBound();
      Xobject cond = tmpForBlock.getUpperBound();
      Xobject step = tmpForBlock.getStep();
      Ident vIdxId = Ident.Local("_ACC_idx_" + indVarName, Xtype.intType);
      Ident indVarId = Ident.Local(indVarName, Xtype.intType);
      Ident nIterId = Ident.Local("_ACC_niter_" + indVarName, Xtype.intType);
      Block calcNiterFuncCall = ACCutil.createFuncCallBlock("_ACC_calc_niter", Xcons.List(nIterId.getAddr(),init,cond,step));
      Block calcIdxFuncCall = ACCutil.createFuncCallBlock("_ACC_gpu_calc_idx", Xcons.List(vIdxId.Ref(), indVarId.getAddr(), init, cond, step));
      
      localIds_2.add(nIterId);
      beginBlocks.add(calcNiterFuncCall);
      
      vIdxIdList.add(vIdxId);
      nIterIdList.add(nIterId);
      indVarIdList.add(indVarId);
      calcIdxFuncCalls.add(calcIdxFuncCall);
    }
    
    
    Ident iterInit = Ident.Local("_ACC_" + execMethodName + "_init", Xtype.intType);
    Ident iterCond = Ident.Local("_ACC_" + execMethodName + "_cond", Xtype.intType);
    Ident iterStep = Ident.Local("_ACC_" + execMethodName + "_step", Xtype.intType);
    Ident iterIdx = Ident.Local("_ACC_" + execMethodName + "_idx", Xtype.intType);
    localIds_2.mergeList(Xcons.List(iterIdx, iterInit, iterCond, iterStep));
    Ident iterCnt = Ident.Local("_ACC_" + execMethodName + "_cnt", Xtype.intType);
    localIds_2.add(iterCnt);

    XobjList initIterFuncArgs = Xcons.List(iterInit.getAddr(), iterCond.getAddr(), iterStep.getAddr());
    Xobject nIterAll = Xcons.IntConstant(1);
    for(Xobject x : nIterIdList){
      Ident nIterId = (Ident)x;
      nIterAll = Xcons.binaryOp(Xcode.MUL_EXPR, nIterAll, nIterId.Ref());
    }
    initIterFuncArgs.mergeList(Xcons.List(Xcons.IntConstant(0), nIterAll, Xcons.IntConstant(1)));

    Block initIterFunc = ACCutil.createFuncCallBlock("_ACC_gpu_init_" + execMethodName + "_iter", initIterFuncArgs); 
    beginBlocks.add(initIterFunc);
    
    Block initIterCntFunc = ACCutil.createFuncCallBlock("_ACC_gpu_init_iter_cnt", Xcons.List(iterCnt.getAddr(), iterInit.Ref(), iterCond.Ref(), iterStep.Ref()));
    beginBlocks.add(initIterCntFunc);

    //make clac each idx from virtual idx
    Block calcEachVidxBlock = makeCalcIdxFuncCall(vIdxIdList, nIterIdList, iterIdx);

    
    
//    Xobject init = forBlock.getLowerBound();
//    Xobject cond = forBlock.getUpperBound();
//    Xobject step = forBlock.getStep();
//    Ident indVarId = Ident.Local(forBlock.getInductionVar().getString(), Xtype.intType);

//    Ident iterIdx = Ident.Local("_ACC_" + execMethodName + "_idx", Xtype.intType);
//    Ident iterInit = Ident.Local("_ACC_" + execMethodName + "_init", Xtype.intType);
//    Ident iterCond = Ident.Local("_ACC_" + execMethodName + "_cond", Xtype.intType);
//    Ident iterStep = Ident.Local("_ACC_" + execMethodName + "_step", Xtype.intType);
//    localIds.mergeList(Xcons.List(iterIdx, iterInit, iterCond, iterStep));
//
//    XobjList initIterFuncArgs = Xcons.List(iterInit.getAddr(), iterCond.getAddr(), iterStep.getAddr(), init, cond, step);
//    Block initIterFunc = ACCutil.createFuncCallBlock("_ACC_gpu_init_" + execMethodName + "_iter", initIterFuncArgs); 
//    beginBlocks.add(initIterFunc);
//
//    XobjList calcIdxFuncArgs = Xcons.List(iterIdx.Ref());
//    calcIdxFuncArgs.add(indVarId.getAddr());
//    calcIdxFuncArgs.mergeList(Xcons.List(init, cond, step));
//    Block calcIdxFunc = ACCutil.createFuncCallBlock("_ACC_gpu_calc_idx", calcIdxFuncArgs); 
//    
    
    
    //push Loop to stack
    Loop thisLoop = new Loop(forBlock);
    thisLoop.setAbstractIter(iterIdx, iterInit, iterCond, iterStep);
    loopStack.push(thisLoop);
    
    //begin cache
    {
      Block headBlock = forBlock.getBody().getHead();
      if(headBlock != null && headBlock.Opcode() == Xcode.ACC_PRAGMA){
        ACCinfo headInfo = ACCutil.getACCinfo(headBlock);
        if(headInfo.getPragma() == ACCpragma.CACHE){
          Iterator<ACCvar> varIter = headInfo.getVars();
          while(varIter.hasNext()){
            ACCvar var = varIter.next();
            if(! var.isCache()) continue;
          
            
            Ident cachedId = var.getId(); 
            XobjList subscripts = var.getSubscripts();
            
            Cache cache = sharedMemory.alloc(cachedId, subscripts);
            
            Block cacheInitFunc = cache.initFunc;
            beginBlocks.add(cacheInitFunc);
            Block cacheLoadBlock = cache.loadBlock;
            cacheLoadBlocks.add(cacheLoadBlock);
            
            localIds_2.add(cache.cacheId);
            localIds_2.add(cache.cacheSizeArrayId);
            localIds_2.add(cache.cacheOffsetArrayId);
          
            	//for after rewrite
            cacheIds.add(cache.cacheId);
            cachedIds.add(cachedId);
            cacheOffsetIds.add(cache.cacheOffsetArrayId);
            cacheSizeIds.add(cache.cacheSizeArrayId);
          }//end while
        }
      }
    }
    //end cache

    BlockList forBlockList = Bcons.emptyBody();
    forBlockList.add(calcEachVidxBlock);
    
    //forBlockList.add(calcIdxFunc);
    for(Block b : calcIdxFuncCalls) forBlockList.add(b);
    
    // add cache load funcs
    for(Block b : cacheLoadBlocks){
      forBlockList.add(b);
    }
    // add inner block
    BlockList innerBody = null;
    if(collapseNum > 1){
      innerBody = collapsedForBlockList.get(collapseNum - 1).getBody();
    }else{
      innerBody = forBlock.getBody();
    }
    Block coreBlock = makeCoreBlock(initBlocks, innerBody, paramIds, localIds, execMethodName, deviceKernelName);
    
    //rewirteCacheVars
    rewriteCacheVar(coreBlock, cachedIds, cacheIds, cacheOffsetIds, cacheSizeIds);
    forBlockList.add(Bcons.IF(Xcons.binaryOp(Xcode.LOG_LT_EXPR, iterIdx.Ref(), iterCond.Ref()), coreBlock, null));    //forBlockList.add(coreBlock);
    
    //add the cache barrier func
    if(! cacheLoadBlocks.isEmpty()){
      forBlockList.add(ACCutil.createFuncCallBlock("_ACC_gpu_barrier", Xcons.List()));
    }
        
    {
      XobjList forBlockListIdents = (XobjList)indVarIdList.copy();//Xcons.List(indVarId);
      forBlockListIdents.mergeList(vIdxIdList);
      //forBlockListIdents.mergeList(nIterIdList);
      ///insert
      forBlockList.setIdentList(forBlockListIdents);
      forBlockList.setDecls(ACCutil.getDecls(forBlockListIdents));
    }
    
//    resultCenterBlock = Bcons.FOR(
//        Xcons.Set(iterIdx.Ref(), iterInit.Ref()),
//        Xcons.binaryOp(Xcode.LOG_LT_EXPR, iterIdx.Ref(), iterCond.Ref()), 
//        Xcons.asgOp(Xcode.ASG_PLUS_EXPR, iterIdx.Ref(), iterStep.Ref()), 
//        Bcons.COMPOUND(forBlockList)
//        );
    resultCenterBlock = Bcons.FOR(
      Xcons.Set(iterIdx.Ref(), iterInit.Ref()),
      Xcons.binaryOp(Xcode.LOG_NEQ_EXPR, iterCnt.Ref(), Xcons.IntConstant(0)),
      Xcons.List(Xcode.COMMA_EXPR, Xcons.asgOp(Xcode.ASG_MINUS_EXPR, iterCnt.Ref(), Xcons.IntConstant(1)), Xcons.asgOp(Xcode.ASG_PLUS_EXPR, iterIdx.Ref(), iterStep.Ref())),
      Bcons.COMPOUND(forBlockList)
      );


    //rewriteReductionvar
    rewriteReductionVar(resultCenterBlock, reductionVarIds, reductionLocalVarIds);//localIds);


    //make blocklist

    //Xobject ids = body.getIdentList();
    //Xobject decls = body.getDecls();
    BlockList resultBody = Bcons.emptyBody();
    for(Block block : beginBlocks){
      resultBody.add(block);
    }
    resultBody.add(resultCenterBlock);
    for(Block block : endBlocks){
      resultBody.add(block);                  
    }
    
    //pop stack
    loopStack.pop();

    
    BlockList resultBody2 = Bcons.blockList();
    resultBody2.setIdentList(localIds_2);
    resultBody2.setDecls(ACCutil.getDecls(localIds_2));
    resultBody2.add(Bcons.COMPOUND(resultBody));
    return Bcons.COMPOUND(resultBody2);

  }
  
  private Block makeCalcIdxFuncCall(XobjList vidxIdList, XobjList nIterIdList, Ident vIdx){
    int i;
    int numVar = vidxIdList.Nargs();
    Xobject result = vIdx.Ref();
    
    Ident funcId = ACCutil.getMacroFuncId("_ACC_calc_vidx", Xtype.intType);

    for(i = numVar - 1; i > 0; i--){
      Ident indVarId = (Ident)(vidxIdList.getArg(i));
      Ident nIterId = (Ident)(nIterIdList.getArg(i));
      Block callBlock = Bcons.Statement(funcId.Call(Xcons.List(indVarId.getAddr(), nIterId.Ref(), result))); 
      result = callBlock.toXobject();
    }
    
    Ident indVarId = (Ident)(vidxIdList.getArg(0));
    result = Xcons.Set(indVarId.Ref(), result);
    
    return Bcons.Statement(result);
  }
  
  private XobjList getSimpleSubarray(Xobject s){
    Xobject loopIdx;
    Xobject constOffset;
    Xobject length;
    
    Xobject lower;
    if(s.Opcode()!=Xcode.LIST){
      lower = s;
      length = Xcons.IntConstant(1);
    }else{
      lower = s.getArg(0);
      length = s.getArgOrNull(1);
    }
      
    switch(lower.Opcode()){
    case PLUS_EXPR:
      loopIdx = lower.getArg(0);
      constOffset = lower.getArg(1);
      break;
    case MINUS_EXPR:
      loopIdx = lower.getArg(0);
      constOffset = Xcons.unaryOp(Xcode.UNARY_MINUS_EXPR, lower.getArg(1));
      break;
    case INT_CONSTANT:
      loopIdx = Xcons.IntConstant(0);
      constOffset = lower;
    default:
      loopIdx = lower;
      constOffset = Xcons.IntConstant(0);
    }
    
    return Xcons.List(loopIdx, constOffset, length);
  }
  
  private XobjList findLoopInfo(Xobject indVar, Block parent){
    String name = indVar.getName();
    
    for(Block b = parent; b != null; b=b.getParentBlock()){
      if(b.Opcode() == Xcode.FOR_STATEMENT){
        CforBlock forBlock = (CforBlock)b;
        if(forBlock.getInductionVar().getName().equals(name)){
          return Xcons.List(forBlock.getLowerBound(), forBlock.getUpperBound(), forBlock.getStep());
        }
      }
    }
    return null;
  }
  
  private Set<Ident> collectPrivateId(ACCinfo info){
    Set<Ident> privateIdSet = new HashSet<Ident>();
    Iterator<ACCvar> accVarIter = info.getVars();
    while(accVarIter.hasNext()){
      ACCvar accVar = accVarIter.next();
      if(accVar.isPrivate()){
        privateIdSet.add(accVar.getId());
      }
    }
    return privateIdSet;
  }
  
  private XobjectDef makeLaunchFuncDef(/*Ident hostFuncId*/ /*XobjList hostFuncParams,*/String launchFuncName, List<Ident> outerIdList, XobjectDef deviceKernelDef) {
    ////make parameter 
    ////make kernel launch block
    ////make body
    XobjList launchFuncParamIds = Xcons.IDList();
    XobjList launchFuncLocalIds = Xcons.IDList();
    
    //parameter
    for(Ident id : outerIdList){
      Ident paramId = makeParamId_new(id);
      launchFuncParamIds.add(paramId);
    }
    XobjList deviceKernelCallArgs = ACCutil.getRefs(launchFuncParamIds);
    
    //# of block and thread
    Ident blockXid = Ident.Local("_ACC_GPU_DIM3_block_x", Xtype.intType);
    Ident blockYid = Ident.Local("_ACC_GPU_DIM3_block_y", Xtype.intType);
    Ident blockZid = Ident.Local("_ACC_GPU_DIM3_block_z", Xtype.intType);
    Ident threadXid = Ident.Local("_ACC_GPU_DIM3_thread_x", Xtype.intType);
    Ident threadYid = Ident.Local("_ACC_GPU_DIM3_thread_y", Xtype.intType);
    Ident threadZid = Ident.Local("_ACC_GPU_DIM3_thread_z", Xtype.intType);
    launchFuncLocalIds.mergeList(Xcons.List(Xcode.ID_LIST, blockXid, blockYid, blockZid, threadXid, threadYid, threadZid));



    BlockList launchFuncBody = Bcons.emptyBody();
    List<Block> preBlocks = new ArrayList<Block>();
    List<Block> postBlocks = new ArrayList<Block>();

    //XobjList blockThreadSize = gpuManager.getBlockThreadSize();
    XobjList blockThreadSize = gpuManager.getBlockThreadSize_new();
    XobjList blockSize = (XobjList)blockThreadSize.left();
    XobjList threadSize = (XobjList)blockThreadSize.right();

    launchFuncBody.add(Bcons.Statement(Xcons.Set(blockXid.Ref(), blockSize.getArg(0))));
    launchFuncBody.add(Bcons.Statement(Xcons.Set(blockYid.Ref(), blockSize.getArg(1))));
    launchFuncBody.add(Bcons.Statement(Xcons.Set(blockZid.Ref(), blockSize.getArg(2))));
    launchFuncBody.add(Bcons.Statement(Xcons.Set(threadXid.Ref(), threadSize.getArg(0))));
    launchFuncBody.add(Bcons.Statement(Xcons.Set(threadYid.Ref(), threadSize.getArg(1))));
    launchFuncBody.add(Bcons.Statement(Xcons.Set(threadZid.Ref(), threadSize.getArg(2))));
    
    //add blockReduction cnt & tmp
    if(reductionManager.hasBlockReduction){
      //temporary

      //deviceKernelCallArgs.add(reductionCnt.getAddr());
      //deviceKernelCallArgs.add(reductionTmpPtr.getAddr());
    }


    Xobject max_num_grid = Xcons.IntConstant(65535);
    Block adjustGridFuncCall = ACCutil.createFuncCallBlock("_ACC_GPU_ADJUST_GRID", Xcons.List(Xcons.AddrOf(blockXid.Ref()), Xcons.AddrOf(blockYid.Ref()), Xcons.AddrOf(blockZid.Ref()),max_num_grid));
    launchFuncBody.add(adjustGridFuncCall);

    launchFuncBody.setIdentList(launchFuncLocalIds);
    launchFuncBody.setDecls(ACCutil.getDecls(launchFuncLocalIds));

    for(Block b:preBlocks) launchFuncBody.add(b);

    Ident deviceKernelId = (Ident)deviceKernelDef.getNameObj();
    Xobject deviceKernelCall = deviceKernelId.Call(deviceKernelCallArgs);
    //FIXME merge GPU_FUNC_CONF and GPU_FUNC_CONF_ASYNC
    deviceKernelCall.setProp(ACCgpuDecompiler.GPU_FUNC_CONF, (Object)Xcons.List(blockXid, blockYid, blockZid,threadXid, threadYid, threadZid));
    if(kernelInfo.isAsync()){
      try{
        Xobject asyncExp = kernelInfo.getAsyncExp();
        if(asyncExp != null){
          deviceKernelCall.setProp(ACCgpuDecompiler.GPU_FUNC_CONF_ASYNC, (Object)Xcons.List(asyncExp));
        }else{
          deviceKernelCall.setProp(ACCgpuDecompiler.GPU_FUNC_CONF_ASYNC, (Object)Xcons.List());
        }
      }catch(Exception e){
        ACC.fatal("can't set async prop");   
      }
    }

    if(sharedMemory.isUsed()){
//      Xobject sm_size = Xcons.IntConstant(0);
//      for(Xobject x : cacheSizeList){
//        sm_size = Xcons.binaryOp(Xcode.PLUS_EXPR, sm_size, x);
//      }
      Xobject maxSmSize = sharedMemory.getMaxSize(); 
      deviceKernelCall.setProp(ACCgpuDecompiler.GPU_FUNC_CONF_SHAREDMEMORY, (Object)maxSmSize);
    }

    launchFuncBody.add(Bcons.Statement(deviceKernelCall));

    for(Block b:postBlocks) launchFuncBody.add(b);

    ACCglobalDecl globalDecl = kernelInfo.getGlobalDecl();
    Ident launchFuncId = globalDecl.declExternIdent(launchFuncName, Xtype.Function(Xtype.voidType));  
    XobjectDef hostFuncDef = XobjectDef.Func(launchFuncId, launchFuncParamIds, null, launchFuncBody.toXobject());

    return hostFuncDef;
  }
  
  private Ident makeParamId(Ident id){
    if(id.isArray()){
      return Ident.Local(id.getName(), id.Type());
    }
    if(_readOnlyOuterIdSet.contains(id)){
      Ident devicePtrId = kernelInfo.getDevicePtr(id.getName());
      if(devicePtrId != null){
        return Ident.Local(id.getName(), Xtype.Pointer(id.Type()));
      }else{
        return Ident.Local(id.getName(), id.Type());
      }
    }
    return Ident.Local(id.getName(), Xtype.Pointer(id.Type()));
  }
  
  private Ident makeParamId_new(Ident id){
    String varName = id.getName();
    
    switch(id.Type().getKind()){
    case Xtype.ARRAY:
    case Xtype.POINTER:
    {
      return Ident.Local(varName, id.Type());
    } 
    case Xtype.BASIC:
    case Xtype.STRUCT:
    case Xtype.UNION:
    case Xtype.ENUM:
    {
      // check whether id is firstprivate!
      if(kernelInfo.isVarAllocated(varName)){
        return Ident.Local(varName, Xtype.Pointer(id.Type()));
      }else{
        return Ident.Local(varName, id.Type());          
      }
    }
    default:
	ACC.fatal("unknown type");
      return null;
    }
    
    /*
    if(! _readOnlyOuterIdSet.contains(id)){
      String varName = id.getName();
      switch(id.Type().getKind()){
      case Xtype.ARRAY:
      {
        return Ident.Local(varName, id.Type());
      } 
      case Xtype.BASIC:
      case Xtype.STRUCT:
      case Xtype.UNION:
      case Xtype.ENUM:
      {
        //Ident newId = Ident.Param(varName, Xtype.Pointer(id.Type()));
        return Ident.Local(varName, Xtype.Pointer(id.Type()));
      }
      default:
        return null;
      }
    }else{
//      if(id.isArray()){
//        ACC.fatal("firstprivate array is not supported");
//      }else{
      if(id.Type().isArray()){
        return Ident.Local(id.getName(), id.Type());
      }else{
        if(kernelInfo.isVarFirstprivate(id.getName())){
          return Ident.Local(id.getName(),  id.Type());
        }else{
          
        }
      }
        
//      }
    }*/
  }

  boolean needsTemp(ACCvar redVar){
    ACCpragma pragma = redVar.getReductionOperator();
    Ident id = redVar.getId();
    Xtype type = id.Type();
    if(! pragma.isReduction()) ACC.fatal(pragma.getName() + " is not reduction clause");
    
    if(pragma == ACCpragma.REDUCTION_MUL){
      return true;
    }
    
    if(type.equals(Xtype.floatType) || type.equals(Xtype.doubleType)){
      if(pragma == ACCpragma.REDUCTION_PLUS){
        return true;
      }
    }
    return false;
  }
  

  
  public void analyze() throws ACCexception{
    if(_kernelBlocks.size() == 1){ // implement for kernel that has more than one block. 
      Block kernelblock = _kernelBlocks.get(0);
      analyzeKernelBlock(kernelblock);
      gpuManager.finalize();
    }
    
    gpuManager.analyze();
    
    //get outerId set
    Set<Ident> outerIdSet = new HashSet();
    for(Block b : _kernelBlocks){
      Set<Ident> blockouterIdSet = collectOuterIdents(b); // = collectOuterIdents(b,b);
      blockouterIdSet.removeAll(collectPrivatizedIdSet(b));
      blockouterIdSet.removeAll(collectInductionVarIdSet(b));
      
      //blockouterIdSet.removeAll(_inductionVarIds);
      outerIdSet.addAll(blockouterIdSet);
    }
    //remove private id
    Iterator<ACCvar> varIter = kernelInfo.getVars();
    while(varIter.hasNext()){
      ACCvar var = varIter.next();
      if(var.isPrivate()){
        outerIdSet.remove(var.getId());
      }
    }
    
    //make outerId list
    _outerIdList = new ArrayList<Ident>(outerIdSet);
    
    //collect read only id 
    _readOnlyOuterIdSet = collectReadOnlyIdSet();
    
    //collect private var ids
    _inductionVarIdSet = collectInductionVarSet();
    
  }
  
  public Set<Ident> getInductionVarIdSet(){
    return _inductionVarIdSet;
  }
  
  public Set<Ident> collectInductionVarSet(){
    Set<Ident> indVarSet = new HashSet<Ident>();
    
    for(Block kernelBlock : _kernelBlocks){
      topdownBlockIterator bi = new topdownBlockIterator(kernelBlock);
      for(bi.init(); !bi.end(); bi.next()){
        Block b = bi.getBlock();
        if(b.Opcode() == Xcode.FOR_STATEMENT){
          ACCinfo info = ACCutil.getACCinfo(b);
          if(info != null){
            CforBlock forBlock = (CforBlock)b;
            Ident indVarId = forBlock.findVarIdent(forBlock.getInductionVar().getName());
            indVarSet.add(indVarId);
          }
        }
      }
    }
    
    return indVarSet;
  }
  
  private Set<Ident> collectInductionVarIdSet(Block kernelBlock){ //rename to collectInductionVarIds
    Set<Ident> indVarIdSet = new HashSet<Ident>();

    topdownBlockIterator blockIter = new topdownBlockIterator(kernelBlock);
    for(blockIter.init(); !blockIter.end(); blockIter.next()){
      Block b = blockIter.getBlock();
      if(b.Opcode() != Xcode.FOR_STATEMENT) continue;
      ACCinfo info = ACCutil.getACCinfo(b);
      if(info == null) continue;
      CforBlock forBlock = (CforBlock)b;
      for(int i = info.getCollapseNum(); i > 0; --i){
        Ident indVarId = b.findVarIdent(forBlock.getInductionVar().getName());
        indVarIdSet.add(indVarId);
        if(i > 1){
          forBlock = (CforBlock)(b.getBody().getHead());
        }
      }
    }
    
    return indVarIdSet;
  }
  
  private Set<Ident> collectPrivatizedIdSet(Block kernelBlock){
    Set<Ident> privatizedIdSet = new HashSet<Ident>();
    
    topdownBlockIterator blockIter = new topdownBlockIterator(kernelBlock);
    for(blockIter.init(); !blockIter.end(); blockIter.next()){
      Block b = blockIter.getBlock();
      if(b.Opcode() != Xcode.FOR_STATEMENT) continue;
      ACCinfo info = ACCutil.getACCinfo(b);
      if(info == null) continue;
      for(Iterator<ACCvar> accVarIter = info.getVars(); accVarIter.hasNext(); ){
        ACCvar var = accVarIter.next();
        if(var.isPrivate()){
          privatizedIdSet.add(var.getId());
        }
      }
    }
    
    return privatizedIdSet;
  }
  
  private void analyzeKernelBlock(Block block){
    if(block == null) return;

    if(block.Opcode() == Xcode.ACC_PRAGMA){
      ACCinfo info = ACCutil.getACCinfo(block);
      if(info.getPragma().isLoop()){
        CforBlock forBlock = (CforBlock)block.getBody().getHead();
        ACCutil.setACCinfo(forBlock, info);
        Iterator<ACCpragma> execModelIter = info.getExecModels();
        gpuManager.addLoop(execModelIter, forBlock);
        BlockList forBody = forBlock.getBody();
        if(forBody != null){
          for(Block b = forBody.getHead(); b != null; b = b.getNext()){
            analyzeKernelBlock(b);
          }
        }
      }/*else if(info.getPragma() == ACCpragma.CACHE){
        Block parentBlock = block.getParentBlock();
        ACCinfo parentInfo = ACCutil.getACCinfo(parentBlock);
        if(parentInfo != null){
          parentInfo.decl
        }
      }*/
    }else if(block.Opcode() == Xcode.FOR_STATEMENT){
      for(Block b = block.getBody().getHead(); b != null; b = b.getNext()){
        analyzeKernelBlock(b);
      }
//      ACCinfo info = ACCutil.getACCinfo(block);
//      if(info == null){
//        info = ACCutil.getACCinfo(block.getParentBlock());
//        if(info != null){
//          if(info.getPragma().isCompute()){
//            //ACCutil.setACCinfo(block, info);
//          }
//        }
//      }
    }else if(block.Opcode() == Xcode.COMPOUND_STATEMENT){
      for(Block b = block.getBody().getHead(); b != null; b = b.getNext()){
        analyzeKernelBlock(b);
      }
    }else{
    }
  }
  
  private boolean isLoop(ACCpragma pragma){
    switch(pragma){
    case LOOP:
    case PARALLEL_LOOP:
    case KERNELS_LOOP:
      return true;
    default:
      return false;
    }
  }
  
  public List<Ident> getOuterIdList(){
    return _outerIdList;
  }
  
  void rewriteReductionVar(Block b, XobjList reduceIds, XobjList localIds){
    BasicBlockExprIterator iter = new BasicBlockExprIterator(b);
    for (iter.init(); !iter.end(); iter.next()) {
      Xobject expr = iter.getExpr();
      topdownXobjectIterator exprIter = new topdownXobjectIterator(expr);
      for (exprIter.init(); !exprIter.end(); exprIter.next()) {
        Xobject x = exprIter.getXobject();
        switch (x.Opcode()) {
        case VAR:
        {
          String varName = x.getName();
          if(ACCutil.hasIdent(reduceIds, varName)){
            Ident localReduceId = ACCutil.getIdent(localIds, ACC_REDUCTION_VAR_PREFIX + varName); 
            Xobject newObj = localReduceId.Ref();
            exprIter.setXobject(newObj);
          }
        }
        }
      }
    }
  }
  
  private Set<Ident> collectReadOnlyIdSet(){
    //FIXME rewrite by topdownXobjectIterator
    class VarAttribute{
      String name = null;
      boolean isRead = false;
      boolean isWritten = false;
      boolean isArray = false;
      Ident id;
      public VarAttribute(Ident id) {
        this.name = id.getName();
        this.id = id;
      }
    };
    
    Map<Ident, VarAttribute> outerIdMap = new HashMap<Ident, VarAttribute>();
    for(Ident outerId : _outerIdList){
      outerIdMap.put(outerId, new VarAttribute(outerId));
    }
    
    for(Block b : _kernelBlocks){
      BasicBlockExprIterator iter = new BasicBlockExprIterator(b);
      for (iter.init(); !iter.end(); iter.next()) {
        topdownXobjectIterator exprIter = new topdownXobjectIterator(iter.getExpr());
        for (exprIter.init(); !exprIter.end(); exprIter.next()) {
          Xobject x = exprIter.getXobject();
          String varName = null;
          ///Xobject writtenObj = null;
          if(x.Opcode().isAsgOp()){
            Xobject lhs = x.getArg(0);
            //writtenObj = lhs;
            if(lhs.Opcode() == Xcode.VAR){
              //writtenObj = lhs;
              varName = lhs.getName();
            }else if(lhs.Opcode() == Xcode.ARRAY_REF){
              //writtenObj = lhs.getArg(0);
              varName = lhs.getArg(0).getName();
            }else if(lhs.Opcode() == Xcode.POINTER_REF){
                if(lhs.getArg(0).getArg(0).Opcode()==Xcode.VAR){
                  varName = lhs.getArg(0).getArg(0).getName();
              }else{
                  ACC.warning("collectReadOnlyIdSet: process was skipped.");
                  continue;
                }
            }
          }else if(x.Opcode() == Xcode.PRE_INCR_EXPR || x.Opcode() == Xcode.PRE_DECR_EXPR){
            //writtenObj = x.getArg(0);
            varName = x.getArg(0).getName();//writtenObj.getName();
          }else{
            continue;
          }
          Ident varId = iter.getBasicBlock().getParent().findVarIdent(varName);//x.findVarIdent(varName);
          
          if(outerIdMap.containsKey(varId)){
            VarAttribute va = outerIdMap.get(varId);
            va.isWritten = true;
          }
        }
      }
    }

    Set<Ident> readOnlyOuterIdSet = new HashSet<Ident>();
    for(Map.Entry<Ident, VarAttribute> e : outerIdMap.entrySet()){
      Ident varId = e.getKey();
      VarAttribute va = e.getValue();
      if(!va.isWritten){
        readOnlyOuterIdSet.add(varId);
      }
    }
    return readOnlyOuterIdSet;
  }

  
  public void setReadOnlyOuterIdSet(Set<Ident> readOnlyOuterIdSet){
    _readOnlyOuterIdSet = readOnlyOuterIdSet;
  }
  public Set<Ident> getReadOnlyOuterIdSet(){
    return _readOnlyOuterIdSet;
  }
  public Set<Ident> getOuterIdSet(){
    return new HashSet<Ident>(_outerIdList);
  }
  
  void rewriteCacheVar(Block b, XobjList cachedIds, XobjList cacheIds, XobjList offsetIds, XobjList sizeIds){
    BasicBlockExprIterator iter = new BasicBlockExprIterator(b);
    for (iter.init(); !iter.end(); iter.next()) {
      Xobject expr = iter.getExpr();
      topdownXobjectIterator exprIter = new topdownXobjectIterator(expr);
      for (exprIter.init(); !exprIter.end(); exprIter.next()) {
        Xobject x = exprIter.getXobject();
        switch (x.Opcode()) {
        case VAR:
        {
          String varName = x.getName();
          if(ACCutil.hasIdent(cachedIds, varName)){
            Ident localId = ACCutil.getIdent(cacheIds, ACC_REDUCTION_VAR_PREFIX + varName); 
            Xobject newObj = localId.Ref();
            exprIter.setXobject(newObj);
          }
        }break;
        case ARRAY_REF:
        {
          String arrayName = x.getArg(0).getName();
          if(ACCutil.hasIdent(cachedIds, arrayName)){
            Ident localId = ACCutil.getIdent(cacheIds, ACC_CACHE_VAR_PREFIX + arrayName);
            XobjList arrayIdxList = (XobjList)x.getArg(1);
            //XobjList newArrayIdxList = Xcons.List();
            Xobject arrayIdx = null;
            
            Ident offsetId = ACCutil.getIdent(offsetIds, "_ACC_cache_offset_" + arrayName);
            Ident sizeId = ACCutil.getIdent(sizeIds, "_ACC_cache_size_" + arrayName);
            int dim = 0;
            for(Xobject idx : arrayIdxList){
              //newArrayIdxList.add(Xcons.binaryOp(Xcode.PLUS_EXPR, idx, offsetId.Ref()));
              Xobject newArrayIdx = Xcons.binaryOp(Xcode.PLUS_EXPR, idx, Xcons.arrayRef(Xtype.intType, offsetId.getAddr(), Xcons.List(Xcons.IntConstant(dim))));
              if(arrayIdx == null){
                arrayIdx = newArrayIdx;
              }else{
                arrayIdx = Xcons.binaryOp(Xcode.PLUS_EXPR, Xcons.binaryOp(Xcode.MUL_EXPR, arrayIdx, Xcons.arrayRef(Xtype.intType, sizeId.getAddr(), Xcons.List(Xcons.IntConstant(dim)))), newArrayIdx);
              }
              dim++;
            }
            Xobject newObj = Xcons.arrayRef(x.Type(), localId.Ref(), Xcons.List(arrayIdx));
            exprIter.setXobject(newObj);
          }
        }break;
        }
      }
    }
  }
  
  class Loop{
    CforBlock forBlock;
    /*
    Xobject init;
    Xobject cond;
    Xobject step;
    Xobject ind;*/
    boolean isParallelized;
    boolean isParallelizable;
    Ident abstIdx;
    Ident abstInit;
    Ident abstCond;
    Ident abstStep;
    Loop(CforBlock forBlock){
      this.forBlock = forBlock;
    }
    void setAbstractIter(Ident idx, Ident init, Ident cond, Ident step){
      abstIdx= idx;abstInit=init;abstCond=cond;abstStep=step; isParallelized=true;
    }
  }
  
  class SharedMemory{
    Ident externSmId;
    Ident smOffsetId;
    ArrayDeque<Xobject> smStack = new ArrayDeque<Xobject>();
    Xobject maxSize = Xcons.IntConstant(0);
    
    boolean isUsed = false;
    
    SharedMemory(){
      Xtype externSmType = Xtype.Array(Xtype.charType, null);      
      externSmId = Ident.Var("_ACC_sm", externSmType, Xtype.Pointer(externSmType), VarScope.GLOBAL);
      externSmId.setStorageClass(StorageClass.EXTERN);
      externSmId.setProp(ACCgpuDecompiler.GPU_STRAGE_SHARED, true);

      smOffsetId = Ident.Local("_ACC_sm_offset", Xtype.intType);
      smOffsetId.setProp(ACCgpuDecompiler.GPU_STRAGE_SHARED, true);
    }
    
    public Xobject getMaxSize() {
      return maxSize;
    }

    public boolean isUsed() {
      return isUsed;
    }

    Cache alloc(Ident id, XobjList subscript){
      Cache cache = new Cache(id, subscript);
      smStack.push(cache.cacheTotalSize);

      Xobject nowSize = Xcons.IntConstant(0);
      for(Xobject s : smStack){
        nowSize = Xcons.binaryOp(Xcode.PLUS_EXPR, nowSize, s);
      }
      maxSize = Xcons.List(Xcode.CONDITIONAL_EXPR, Xcons.binaryOp(Xcode.LOG_LT_EXPR, nowSize, maxSize), Xcons.List(maxSize, nowSize));
      
      isUsed = true;
      return cache;
    }
    void free(){
      smStack.pop();
    }
    Block makeInitFunc(){
      return ACCutil.createFuncCallBlock("_ACC_gpu_init_sm_offset", Xcons.List(smOffsetId.getAddr()));
    }
  }
  
  class Cache{
    XobjList localIds = Xcons.IDList();
    //XobjList cacheSizeIds = Xcons.IDList();
    //XobjList cacheOffsetIds = Xcons.IDList();
    Ident cacheSizeArrayId;
    Ident cacheOffsetArrayId;
    Xtype elementType;
    //int dimension = 0;
    Ident cacheId;
    Ident varId;
    XobjList subscripts;
    int cacheDim;
    Xobject cacheTotalSize;
    Block initFunc;
    Block loadBlock;
    
    Cache(Ident varId, XobjList subscripts){
      this.varId = varId;
      this.subscripts = subscripts;
      elementType = (varId.Type().isArray())? (((ArrayType)(varId.Type())).getArrayElementType()) : varId.Type();
      Xtype cacheType = (varId.Type().isArray())? Xtype.Pointer(elementType) : elementType;

      cacheId = Ident.Local(ACC_CACHE_VAR_PREFIX + varId.getName(), cacheType);
      cacheDim = subscripts.Nargs();
      
      //localIds.add(cacheId);
      
      cacheSizeArrayId = Ident.Local("_ACC_cache_size_" + varId.getName(), Xtype.Array(Xtype.intType, cacheDim));
      cacheSizeArrayId.setProp(ACCgpuDecompiler.GPU_STRAGE_SHARED, true);
      
      cacheOffsetArrayId = Ident.Local("_ACC_cache_offset_" + varId.getName(), Xtype.Array(Xtype.intType, cacheDim));
      
      localIds.add(cacheOffsetArrayId);
      localIds.add(cacheSizeArrayId);
      //cacheSizeIds.add(cacheSizeArrayId);
      
      initFunc = makeCacheInitializeFunc();
      loadBlock = makeCacheLoadBlock();
    }
    
    Block makeCacheInitializeFunc(){
      //XobjList getCacheFuncArgs = Xcons.List(_externSm.Ref(), _smOffset.Ref(), cacheId.getAddr(), cacheSizeArrayId.getAddr());//, step, cacheLength);
      XobjList getCacheFuncArgs = Xcons.List(sharedMemory.externSmId.Ref(), sharedMemory.smOffsetId.Ref(), cacheId.getAddr(), cacheSizeArrayId.getAddr());//, step, cacheLength);
      
      for(Xobject s : subscripts){
        XobjList simpleSubarray = getSimpleSubarray(s);
        Xobject cacheIdx = simpleSubarray.getArg(0);
        Xobject cacheConstOffset = simpleSubarray.getArg(1);
        Xobject cacheLength = simpleSubarray.getArg(2);
        
        //find loop
        Loop loop = null;
        Iterator<Loop> iterLoop = loopStack.iterator();
        while(iterLoop.hasNext()){
          Loop tmpLoop = iterLoop.next();
          if(tmpLoop.forBlock.getInductionVar().getName().equals(cacheIdx.getName())){
            loop = tmpLoop;
            break;
          }
        }
        if(loop == null) ACC.fatal(cacheIdx.getName() + " is not loop variable");

        Xobject step = loop.forBlock.getStep(); //もしcacheがloop変数非依存なら？
        getCacheFuncArgs.mergeList(Xcons.List(step, cacheLength));

      }
      
      Block getCacheFuncCall = ACCutil.createFuncCallBlock("_ACC_gpu_init_cache", getCacheFuncArgs);
      return getCacheFuncCall;
    }
    
    Block makeCacheLoadBlock(){
      BlockList cacheLoadBody = Bcons.emptyBody();
      XobjList cacheLoadBodyIds = Xcons.IDList();
      
      Ident cacheLoadSizeArrayId = Ident.Local("_ACC_cache_load_size_" + varId.getName(), Xtype.Array(Xtype.intType, cacheDim));
      cacheLoadSizeArrayId.setProp(ACCgpuDecompiler.GPU_STRAGE_SHARED, true);
      cacheLoadBodyIds.add(cacheLoadSizeArrayId);
      

      //cacheOffsetIds.add(cacheOffsetArrayId);//?


      int dim = 0;
      Xobject totalCacheSize = Xcons.IntConstant(1);
      
      for(Xobject s : subscripts){
        XobjList simpleSubarray = getSimpleSubarray(s);
        Xobject cacheIdx = simpleSubarray.getArg(0);
        Xobject cacheConstOffset = simpleSubarray.getArg(1);
        Xobject cacheLength = simpleSubarray.getArg(2);
        
        Xobject cacheLoadSizeRef = Xcons.arrayRef(Xtype.intType, cacheLoadSizeArrayId.getAddr(), Xcons.List(Xcons.IntConstant(dim)));
        Xobject cacheOffsetRef = Xcons.arrayRef(Xtype.intType, cacheOffsetArrayId.getAddr(), Xcons.List(Xcons.IntConstant(dim)));
        
        XobjList getLoadSizeFuncArgs = Xcons.List(Xcons.AddrOf(cacheLoadSizeRef), Xcons.arrayRef(Xtype.intType, cacheSizeArrayId.getAddr(), Xcons.List(Xcons.IntConstant(dim))));
        XobjList getOffsetFuncArgs = Xcons.List(Xcons.AddrOf(cacheOffsetRef), cacheIdx, cacheConstOffset);
        
        //find loop
        Loop loop = null;
        Iterator<Loop> iterLoop = loopStack.iterator();
        while(iterLoop.hasNext()){
          Loop tmpLoop = iterLoop.next();
          if(tmpLoop.forBlock.getInductionVar().getName().equals(cacheIdx.getName())){
            loop = tmpLoop;
            break;
          }
        }
        if(loop == null) ACC.fatal(cacheIdx.getName() + " is not loop variable");

        Xobject calculatedCacheSize = null;
        if(loop.isParallelized){
          Xobject abstIdx = loop.abstIdx.Ref();
          Xobject abstCond = loop.abstCond.Ref();
          Xobject abstStep = loop.abstStep.Ref();
          Xobject concStep = loop.forBlock.getStep();
          getLoadSizeFuncArgs.mergeList(Xcons.List(abstIdx, abstCond, abstStep, concStep));

          String methodName = gpuManager.getMethodName(loop.forBlock);
          Xobject blockSize = null;
          if(methodName.endsWith("thread_x")){
            blockSize = Xcons.Symbol(Xcode.VAR, Xtype.intType, "_ACC_GPU_DIM3_thread_x");
          }else if(methodName.endsWith("thread_y")){
            blockSize = Xcons.Symbol(Xcode.VAR, Xtype.intType, "_ACC_GPU_DIM3_thread_y");
          }else if(methodName.endsWith("thread_z")){
            blockSize = Xcons.Symbol(Xcode.VAR, Xtype.intType, "_ACC_GPU_DIM3_thread_z");
          }else{
            blockSize = Xcons.IntConstant(1);
          }
          calculatedCacheSize = Xcons.binaryOp(Xcode.PLUS_EXPR, Xcons.binaryOp(Xcode.MUL_EXPR, Xcons.binaryOp(Xcode.MINUS_EXPR,blockSize,Xcons.IntConstant(1)), concStep), cacheLength);
        }else{
          Xobject zeroObj = Xcons.IntConstant(0);
          Xobject concStep = loop.forBlock.getStep();
          getLoadSizeFuncArgs.mergeList(Xcons.List(zeroObj, zeroObj, zeroObj, concStep));
          calculatedCacheSize = cacheLength;
        }
        
        Block getLoadSizeFuncCall = ACCutil.createFuncCallBlock("_ACC_gpu_get_cache_load_size", getLoadSizeFuncArgs);
        Block getOffsetFuncCall = ACCutil.createFuncCallBlock("_ACC_gpu_get_cache_offset", getOffsetFuncArgs);
        cacheLoadBody.add(getLoadSizeFuncCall);
        cacheLoadBody.add(getOffsetFuncCall);

        totalCacheSize = Xcons.binaryOp(Xcode.MUL_EXPR, totalCacheSize, calculatedCacheSize);
        dim++;
      }
      //cacheSizeList.add(Xcons.binaryOp(Xcode.MUL_EXPR, totalCacheSize, Xcons.SizeOf(elementType)));
      cacheTotalSize = Xcons.binaryOp(Xcode.MUL_EXPR, totalCacheSize, Xcons.SizeOf(elementType));
      
      //make load for loop
      Block dummyInnerMostBlock = Bcons.emptyBlock(); //dummy block
      Block loadLoopBlock = null;//dummyInnerMostBlock;
      XobjList lhsArrayRefList = Xcons.List();
      XobjList rhsArrayRefList = Xcons.List();
      for(int d = 0; d < cacheDim; d++){ //from higher dim
        Ident tmpIter = Ident.Local("_ACC_iter_idx" + d, Xtype.intType);
        cacheLoadBodyIds.add(tmpIter);
        Xobject tmpIterInit, tmpIterCond, tmpIterStep;
        tmpIterCond = Xcons.arrayRef(Xtype.intType, cacheLoadSizeArrayId.getAddr(), Xcons.List(Xcons.IntConstant(d)));
        if(d == cacheDim -1){
          Ident thread_x_id = Ident.Local("_ACC_thread_x_id", Xtype.intType); //this is macro
          tmpIterInit = thread_x_id.Ref();
          Ident block_size_x = Ident .Local("_ACC_block_size_x", Xtype.intType); //this is macro
          tmpIterStep = block_size_x.Ref();
        }else if(d == cacheDim - 2){
          Ident thread_y_id = Ident.Local("_ACC_thread_y_id", Xtype.intType); //this is macro
          tmpIterInit = thread_y_id.Ref();
          Ident block_size_y = Ident .Local("_ACC_block_size_y", Xtype.intType); //this is macro
          tmpIterStep = block_size_y.Ref();
        }else if(d == cacheDim - 3){
          Ident thread_z_id = Ident.Local("_ACC_thread_z_id", Xtype.intType); //this is macro
          tmpIterInit = thread_z_id.Ref();
          Ident block_size_z = Ident .Local("_ACC_block_size_z", Xtype.intType); //this is macro
          tmpIterStep = block_size_z.Ref();
        }else{
          tmpIterInit = Xcons.IntConstant(0);
          tmpIterStep = Xcons.IntConstant(1);
        }
        if(false){
          lhsArrayRefList.add(tmpIter.Ref());
        }else{
          if(lhsArrayRefList.isEmpty()){
            lhsArrayRefList.add(tmpIter.Ref());
          }else{
            Xobject newRef = lhsArrayRefList.getArg(0);
            newRef = Xcons.binaryOp(Xcode.PLUS_EXPR, Xcons.binaryOp(Xcode.MUL_EXPR, newRef, Xcons.arrayRef(Xtype.intType, cacheSizeArrayId.getAddr(), Xcons.List(Xcons.IntConstant(d)))), tmpIter.Ref());
            lhsArrayRefList.setArg(0, newRef);
          }
        }
        rhsArrayRefList.add(Xcons.binaryOp(Xcode.MINUS_EXPR, tmpIter.Ref(), Xcons.arrayRef(Xtype.intType, cacheOffsetArrayId.getAddr(), Xcons.List(Xcons.IntConstant(d)))));
        if(loadLoopBlock==null){
          loadLoopBlock = Bcons.FORall(tmpIter.Ref(), tmpIterInit, tmpIterCond, tmpIterStep, Xcode.LOG_LT_EXPR, Bcons.blockList(dummyInnerMostBlock));
        }else{
          Block newDummyInnerMostBlock = Bcons.emptyBlock();
          Block replaceBlock = Bcons.FORall(tmpIter.Ref(), tmpIterInit, tmpIterCond, tmpIterStep, Xcode.LOG_LT_EXPR, Bcons.blockList(newDummyInnerMostBlock));           
          dummyInnerMostBlock.replace(replaceBlock);
          dummyInnerMostBlock = newDummyInnerMostBlock;
        }

        //loadLoopBlock = Bcons.FORall(tmpIter.Ref(), tmpIterInit, tmpIterCond, tmpIterStep, Xcode.LOG_LT_EXPR, Bcons.blockList(loadLoopBlock));
      }
      Xobject innerMostObject = Xcons.Set(Xcons.arrayRef(elementType, cacheId.getAddr(), lhsArrayRefList), Xcons.arrayRef(elementType, varId.getAddr(), rhsArrayRefList));
      dummyInnerMostBlock.replace(Bcons.Statement(innerMostObject));
      
      cacheLoadBody.add(loadLoopBlock);
      cacheLoadBody.add(ACCutil.createFuncCallBlock("_ACC_gpu_barrier", Xcons.List()));
      cacheLoadBody.setIdentList(cacheLoadBodyIds);
      cacheLoadBody.setDecls(ACCutil.getDecls(cacheLoadBodyIds));
      cacheLoadBody.setIdentList(cacheLoadBodyIds);
      return Bcons.COMPOUND(cacheLoadBody);
    }
  }
  
  class ReductionManager{
    Ident counterPtr = null;
    Ident tempPtrPtr = null;
    boolean hasBlockReduction = false;
    List<Reduction> reductionList = new ArrayList<Reduction>();
    Xobject totalElementSize = Xcons.IntConstant(0);
    Map<Reduction, Xobject> offsetMap = new HashMap<Reduction, Xobject>();
    Ident isLastVar = null;
    
    ReductionManager(String deviceKernelName){
      counterPtr = Ident.Param("_ACC_GPU_RED_CNT", Xtype.Pointer(Xtype.unsignedType));//Ident.Var("_ACC_GPU_RED_CNT", Xtype.unsignedType, Xtype.Pointer(Xtype.unsignedType), VarScope.GLOBAL);
      tempPtrPtr = Ident.Param("_ACC_GPU_RED_TMP", Xtype.Pointer(Xtype.voidPtrType));//Ident.Var("_ACC_GPU_RED_TMP", Xtype.voidPtrType, Xtype.Pointer(Xtype.voidPtrType), VarScope.GLOBAL);
      isLastVar = Ident.Local("_ACC_GPU_IS_LAST_BLOCK", Xtype.intType);
      isLastVar.setProp(ACCgpuDecompiler.GPU_STRAGE_SHARED, true);
    }
    
    public XobjList getBlockReductionParamIds() {
      return Xcons.List(Xcode.ID_LIST, counterPtr, tempPtrPtr);
    }

    public Block makeLocalVarInitFuncs() {
      BlockList body = Bcons.emptyBody();
      Iterator<Reduction> blockRedIter = reductionManager.BlockReductionIterator();
      while(blockRedIter.hasNext()){
        Reduction reduction = blockRedIter.next();
        body.add(reduction.makeInitReductionVarFuncCall());
      }
      
      if(body.isSingle()){
        return body.getHead();
      }else{
        return Bcons.COMPOUND(body);
      }
    }

    public XobjList getBlockReductionLocalIds() {
      XobjList blockLocalIds = Xcons.IDList();
      Iterator<Reduction> blockRedIter = reductionManager.BlockReductionIterator();
      while(blockRedIter.hasNext()){
        Reduction reduction = blockRedIter.next();
        blockLocalIds.add(reduction.getLocalReductionVarId());
      }
      return blockLocalIds;
    }

    public Block makeReduceAndFinalizeFuncs() {
      /*
       * {
       *   __shared__ int _ACC_GPU_IS_LAST_BLOCK;
       *   
       *   _ACC_gpu_reduction_thread(...);
       *   
       *   _ACC_gpu_is_last_block(&_ACC_GPU_IS_LAST,_ACC_GPU_RED_CNT);
       *   if((_ACC_GPU_IS_LAST)!=(0)){
       *     _ACC_gpu_reduction_block(...);
       *   }
       * }
       */
      XobjList idList = Xcons.List(isLastVar);
      BlockList body = Bcons.emptyBody(idList, ACCutil.getDecls(idList));
      
      
      BlockList thenBody = Bcons.emptyBody();
      // add funcs
      Iterator<Reduction> blockRedIter = reductionManager.BlockReductionIterator();
      while(blockRedIter.hasNext()){
        Reduction reduction = blockRedIter.next();
//        Xobject tmpPtr = Xcons.PointerRef(tempPtrPtr.Ref());
        if(reduction.useThread()){
          body.add(reduction.makeThreadReductionFuncCall());
        }
        if(reduction.useBlock()){
          body.add(reduction.makeTempWriteFuncCall(tempPtrPtr, offsetMap.get(reduction)));
        }
        
        Block blockReduction = reduction.makeBlockReductionFuncCall(tempPtrPtr, offsetMap.get(reduction));
        thenBody.add(blockReduction);
      }
      
      thenBody.add(reductionManager.makeFinalizeFunc());
      
      body.add(ACCutil.createFuncCallBlock("_ACC_gpu_is_last_block", Xcons.List(isLastVar.getAddr(), counterPtr.Ref())));
      
      //Xcons.binaryOp(Xcode.LOG_NEQ_EXPR, isLastVar.Ref(), Xcons.IntConstant(0))
      body.add(Bcons.IF(isLastVar.Ref(), Bcons.COMPOUND(thenBody), null));

      return Bcons.COMPOUND(body);
    }

    public Block makeInitFunc() {
      return ACCutil.createFuncCallBlock("_ACC_gpu_init_block_reduction", Xcons.List(counterPtr.Ref(), tempPtrPtr.Ref(), totalElementSize));
    }
    
    public Block makeFinalizeFunc(){
      return ACCutil.createFuncCallBlock("_ACC_gpu_finalize_reduction", Xcons.List(counterPtr.Ref(), tempPtrPtr.Ref()));
    }

    Reduction addReduction(ACCvar var, ACCpragma execMethod){
      Reduction reduction = new Reduction(var, execMethod, tempPtrPtr);
      reductionList.add(reduction);
      if(execMethod == ACCpragma._BLOCK || execMethod == ACCpragma._BLOCK_THREAD){
        hasBlockReduction = true;
      }
      
      offsetMap.put(reduction, totalElementSize);
      
      Xtype varType = var.getId().Type();
      Xobject elementSize;
      if(varType.isPointer()){
        elementSize = Xcons.SizeOf(varType.getRef());
      }else{
        elementSize = Xcons.SizeOf(varType);
      }
      totalElementSize = Xcons.binaryOp(Xcode.PLUS_EXPR, totalElementSize, elementSize);
      return reduction;
    }
    
    Iterator<Reduction> BlockReductionIterator(){
      return new BlockReductionIterator(reductionList);
    }
    
    class BlockReductionIterator implements Iterator<Reduction>{
      Iterator<Reduction> reductionIterator;
      Reduction re;
      public BlockReductionIterator(List<Reduction> reductionList) {
        this.reductionIterator = reductionList.iterator();
      }
      @Override
      public boolean hasNext() {
        while(true){
          if(reductionIterator.hasNext()){
            re = reductionIterator.next();
            if(re.useBlock()){
              return true;
            }
          }else{
            return false;
          }
        }
      }

      @Override
      public Reduction next() {
        return re;
      }

      @Override
      public void remove() {
        //do nothing
      }
    }
  }
  
  class Reduction{
    ACCpragma execMethod;
    Ident localVarId;
    Ident varId;
    ACCvar var;
    //Ident tmpId;
    Reduction(ACCvar var, ACCpragma execMethod, Ident tmpId){
      this.var = var;
      this.varId = var.getId();
      this.execMethod = execMethod;
      
      //generate local var id
      localVarId = Ident.Local(ACC_REDUCTION_VAR_PREFIX + varId.getName(), varId.Type());
      if(execMethod == ACCpragma._BLOCK){
        localVarId.setProp(ACCgpuDecompiler.GPU_STRAGE_SHARED, true);
      }
      
    }
    public boolean useThread() {
      if(execMethod != ACCpragma._BLOCK){
        return true;
      }
      return false;
    }
    public Ident getLocalReductionVarId() {
      return localVarId;
    }
    public Block makeInitReductionVarFuncCall() {
      int reductionKind = getReductionKindInt();
      
      if(execMethod == ACCpragma._BLOCK){
        return ACCutil.createFuncCallBlock("_ACC_gpu_init_reduction_var_single", Xcons.List(localVarId.getAddr(), Xcons.IntConstant(reductionKind)));
      }else{
        return ACCutil.createFuncCallBlock("_ACC_gpu_init_reduction_var", Xcons.List(localVarId.getAddr(), Xcons.IntConstant(reductionKind)));
      }
    }
    
    public Block makeBlockReductionFuncCall(Ident tmpPtrPtrId, Xobject tmpOffsetElementSize) {
      XobjList args = Xcons.List(varId.Ref(), Xcons.IntConstant(getReductionKindInt()), Xcons.PointerRef(tmpPtrPtrId.Ref()), tmpOffsetElementSize);
      return ACCutil.createFuncCallBlock("_ACC_gpu_reduction_block", args);
    }
    
    public Block makeThreadReductionFuncCall(){
      XobjList args = Xcons.List(localVarId.Ref(), Xcons.IntConstant(getReductionKindInt()));
      if(execMethod == ACCpragma._THREAD){
        args.cons(varId.getAddr());
      }else{
        args.cons(localVarId.getAddr());
      }
      return ACCutil.createFuncCallBlock("_ACC_gpu_reduction_thread", args);
    }
    
    public Block makeTempWriteFuncCall(Ident tmpPtrPtrId, Xobject tmpOffsetElementSize){
      Xobject tmpAddr = Xcons.binaryOp(Xcode.PLUS_EXPR, Xcons.PointerRef(tmpPtrPtrId.Ref()), tmpOffsetElementSize);
      return ACCutil.createFuncCallBlock("_ACC_gpu_reduction_tmp", Xcons.List(localVarId.Ref(), Xcons.PointerRef(tmpPtrPtrId.Ref()), tmpOffsetElementSize));
    }
    
    private int getReductionKindInt(){
      ACCpragma pragma = var.getReductionOperator();
      if(! pragma.isReduction()) ACC.fatal(pragma.getName() + " is not reduction clause");
      switch(pragma){
      case REDUCTION_PLUS: return 0;
      case REDUCTION_MUL: return 1;
      case REDUCTION_MAX: return 2;
      case REDUCTION_MIN: return 3;
      case REDUCTION_BITAND: return 4;
      case REDUCTION_BITOR: return 5;
      case REDUCTION_BITXOR: return 6;
      case REDUCTION_LOGAND: return 7;
      case REDUCTION_LOGOR: return 8;
      default: return -1;
      }
    }
    public boolean useBlock(){
      if(execMethod != ACCpragma._THREAD) return true;
      else return false;
    }
  }
}
