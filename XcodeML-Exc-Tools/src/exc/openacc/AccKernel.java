/* -*- Mode: java; c-basic-offset:2 ; indent-tabs-mode:nil ; -*- */
package exc.openacc;

import exc.block.*;
import exc.object.*;
import java.util.*;

public class AccKernel {
  static final String ACC_CALC_IDX_FUNC = "_ACC_calc_idx";
  static final String ACC_INIT_ITER_FUNC_PREFIX = "_ACC_init_iter_";
  static final String ACC_MPOOL_ALLOC_FUNCNAME = "_ACC_mpool_alloc";
  static final String ACC_MPOOL_FREE_FUNCNAME = "_ACC_mpool_free";
  final ACCglobalDecl _decl;
  final PragmaBlock _pb;
  final AccInformation _kernelInfo; //parallel or kernels info
  final AccManager gpuManager;
  static final String ACC_FUNC_PREFIX = "_ACC_kernel_";
  static final String ACC_REDUCTION_VAR_PREFIX = "_ACC_reduction_";
  static final String ACC_CACHE_VAR_PREFIX = "_ACC_cache_";
  static final String ACC_REDUCTION_TMP_VAR = "_ACC_GPU_RED_TMP";
  static final String ACC_REDUCTION_CNT_VAR = "_ACC_GPU_RED_CNT";
  static final String ACC_GPU_DEVICE_FUNC_SUFFIX = "_DEVICE";
  static final String ACC_CL_KERNEL_LAUNCHER_NAME = "_ACC_launch";
  final Xobject _accThreadIndex = Xcons.Symbol(Xcode.VAR, Xtype.intType, "_ACC_thread_x_id");
  final Xobject _accThreadIndexY = Xcons.Symbol(Xcode.VAR, Xtype.intType, "_ACC_thread_y_id");
  final Xobject _accBlockIndex = Xcons.Symbol(Xcode.VAR, Xtype.intType, "_ACC_block_x_id");
  final Xobject _accSyncThreads = ACCutil.getMacroFuncId("_ACC_sync_threads", Xtype.voidType).Call();
  final Xobject _accSyncGangs = ACCutil.getMacroFuncId("_ACC_sync_gangs", Xtype.voidType).Call();
  final Xobject _accAsyncSync = Xcons.Symbol(Xcode.VAR, Xtype.intType, "ACC_ASYNC_SYNC");
  final Xobject _accFlush = ACCutil.getMacroFuncId("_ACC_flush", Xtype.voidType).Call();

  final List<Block> _kernelBlocks;
  List<Ident> _outerIdList;
  Set<Ident> _readOnlyOuterIdSet;

  // ArrayDeque<Loop> loopStack = new ArrayDeque<Loop>();
  // SharedMemory sharedMemory = new SharedMemory();
  // ReductionManager reductionManager = new ReductionManager();
  // StackMemory _globalMemoryStack = new StackMemory("_ACC_gmem_stack", Ident.Param("_ACC_gmem_stack_addr", Xtype.voidPtrType));
  ArrayDeque<Loop> loopStack;
  SharedMemory sharedMemory;
  ReductionManager reductionManager;
  StackMemory _globalMemoryStack;

  final Set<Ident> _useMemPoolOuterIdSet = new LinkedHashSet<Ident>();
  final List<XobjList> allocList = new ArrayList<XobjList>(); //for  array or reduction tmp array
  List<ACCvar> _outerVarList;
  boolean hasGangSync = false;
   List<Ident> _globalIdList = new ArrayList<Ident>();

  public AccKernel(ACCglobalDecl decl, PragmaBlock pb, AccInformation info, List<Block> kernelBlocks) {
    this._decl = decl;
    this._pb = pb;
    this._kernelInfo = info;
    this._kernelBlocks = kernelBlocks;
    this.gpuManager = new AccManager(_kernelInfo, _pb);
    hasGangSync = false /* (ACC.platform == ACC.Platform.PZCL)*/;
    initInternalClasses();
  }

  void initInternalClasses()
  {
    loopStack = new ArrayDeque<Loop>();
    sharedMemory = new SharedMemory();
    reductionManager = new ReductionManager();
    _globalMemoryStack = new StackMemory("_ACC_gmem_stack", Ident.Param("_ACC_gmem_stack_addr", Xtype.voidPtrType));
  }

  XobjList getFuncInfo(Block block) {
    for (Block b = block; b != null; b = b.getParentBlock()) {
      if (b.Opcode() == Xcode.FUNCTION_DEFINITION) {
        FunctionBlock fb = (FunctionBlock) b;
        String funcName = fb.getName();
        Xobject funcParams = fb.getBody().getIdentList();

        return Xcons.List(Xcons.String(funcName), funcParams);
      }
    }
    ACC.fatal("cant't get func info");
    return null;
  }

  void collectOuterVar() {
    _outerVarList = new ArrayList<ACCvar>();
    for (Ident id : _outerIdList) {
      ACCvar accvar = _kernelInfo.findACCvar(id.getSym());//kernelInfo.getACCvar(id);
      //Fixme implement the below
      //if(accvar == null) accvar = kernelInfo.findOuterACCvar(id.getName());

      if (accvar == null) {
        ACC.fatal(id.getName() + " not found");
      } else {
        _outerVarList.add(accvar);
        if(ACC.debug_flag) System.out.println("add outVerList id="+id);
      }
    }
  }

  public Block makeLaunchFuncCallBlock() {
    List<Block> kernelBody = _kernelBlocks;

    String funcName = getFuncInfo(_pb).getArg(0).getString();
    int lineNo = kernelBody.get(0).getLineNo().lineNo();
    String kernelMainName = ACC_FUNC_PREFIX + funcName + "_L" + lineNo;
    String launchFuncName = "";
    launchFuncName = kernelMainName;

    collectOuterVar();

    //make deviceKernelDef
    String deviceKernelName = kernelMainName + ACC_GPU_DEVICE_FUNC_SUFFIX;
    XobjectDef deviceKernelDef = makeDeviceKernelDef(deviceKernelName, _outerIdList, kernelBody);

    //add deviceKernel and launchFunction
    XobjectFile devEnv = _decl.getEnvDevice();
    devEnv.add(deviceKernelDef);

    return makeLaunchFuncBlock(launchFuncName, deviceKernelDef);
  }

  Block makeKernelLaunchBlock(String launchFuncName, String kernelFuncName, XobjList kernelFuncArgs, Ident confId, Xobject asyncExpr)
  {
    BlockList body = Bcons.emptyBody();

      XobjList argDecls = Xcons.List();
      XobjList argSizeDecls = Xcons.List();
      for(Xobject x : kernelFuncArgs){
        if(x.Opcode() != Xcode.CAST_EXPR) {
          //FIXME use AddrOFVar
          argDecls.add(Xcons.AddrOf(x));
        }else{
          argDecls.add(Xcons.AddrOfVar(x.getArg(0)));
        }
        if(x.Type().isPointer()) {
          argSizeDecls.add(Xcons.SizeOf(Xtype.voidPtrType));
        }else {
          argSizeDecls.add(Xcons.SizeOf(x.Type()));
        }
      }
      Ident argSizesId = body.declLocalIdent("_ACC_argsizes", Xtype.Array(Xtype.unsignedlonglongType, null), StorageClass.AUTO, argSizeDecls);
      Ident argsId = body.declLocalIdent("_ACC_args", Xtype.Array(Xtype.voidPtrType, null), StorageClass.AUTO, argDecls);

      Ident launchFuncId = ACCutil.getMacroFuncId(launchFuncName, Xtype.voidType);
      int kernelNum = _decl.declKernel(kernelFuncName);
      Ident programId = _decl.getProgramId();
      int numArgs = kernelFuncArgs.Nargs();

      Xobject launchFuncArgs = Xcons.List(
              programId.Ref(),
              Xcons.IntConstant(kernelNum),
              confId.Ref(),
              asyncExpr,
              Xcons.IntConstant(numArgs),
              argSizesId.Ref(),
              argsId.Ref());
    body.add(launchFuncId.Call(launchFuncArgs));
    return Bcons.COMPOUND(body);
  }

  Block makeKernelLaunchBlockCUDA(Ident deviceKernelId, XobjList kernelArgs, XobjList conf, Xobject asyncExpr)
  {
    Xobject deviceKernelCall = deviceKernelId.Call(kernelArgs);
    //FIXME merge GPU_FUNC_CONF and GPU_FUNC_CONF_ASYNC
    deviceKernelCall.setProp(ACCgpuDecompiler.GPU_FUNC_CONF, conf);
    deviceKernelCall.setProp(ACCgpuDecompiler.GPU_FUNC_CONF_ASYNC, Xcons.List(asyncExpr));
    if (sharedMemory.isUsed()) {
      Xobject maxSmSize = sharedMemory.getMaxSize();
      deviceKernelCall.setProp(ACCgpuDecompiler.GPU_FUNC_CONF_SHAREDMEMORY, maxSmSize);
    }
    return Bcons.Statement(deviceKernelCall);
  }

  Xobject makeLaunchFuncArg(ACCvar var){
    if (var.isArray()) {
      Ident devicePtrId = var.getDevicePtr();
      Xobject devicePtr = devicePtrId.Ref();

      Xtype type = var.getId().Type();
      if(! type.equals(devicePtrId.Type())){
        return Xcons.Cast(type, devicePtr);
      }
      return devicePtr;
    } else{
      Ident id = var.getId();

      if (_useMemPoolOuterIdSet.contains(id)) {
        return id.getAddr(); //host scalar pointer
      }

      if(!var.isArray() && var.isFirstprivate()){
        return id.Ref(); //host scalar data
      }

      Ident devicePtrId = var.getDevicePtr();
      Xobject devicePtr = devicePtrId.Ref();
      Xtype elmtType = var.getElementType();
      if(! elmtType.equals(devicePtrId.Type())){
        return Xcons.Cast(Xtype.Pointer(elmtType), devicePtr);
      }
      return devicePtr;
    }
  }

  Xobject getAsyncExpr(){
    if (! _kernelInfo.hasClause(ACCpragma.ASYNC)) {
      return Xcons.IntConstant(ACC.ACC_ASYNC_SYNC);
    }

    Xobject asyncExpr = _kernelInfo.getIntExpr(ACCpragma.ASYNC);
    if (asyncExpr != null) {
      return asyncExpr;
    } else {
      return Xcons.IntConstant(ACC.ACC_ASYNC_NOVAL);
    }
  }

   Ident findInnerBlockIdent(Block topBlock, BlockList body, String name) {
    // if the id exists between topBlock to bb, the id is not outerId
    for (BlockList b_list = body; b_list != null; b_list = b_list.getParentList()) {
      Ident localId = b_list.findLocalIdent(name);
      if (localId != null) return localId;
      if (b_list == topBlock.getParent()) break;
    }
    return null;
  }

   class DeviceKernelBuildInfo {
    final private List<Block> initBlockList = new ArrayList<Block>();
    final private List<Block> finalizeBlockList = new ArrayList<Block>();
    final private XobjList paramIdList = Xcons.IDList();
    final private XobjList localIdList = Xcons.IDList();

    public List<Block> getInitBlockList() {
      return initBlockList;
    }

    public List<Block> getFinalizeBlockList(){
      return finalizeBlockList;
    }

    public void addInitBlock(Block b) {
      initBlockList.add(b);
    }

    public void addFinalizeBlock(Block b){
      finalizeBlockList.add(b);
    }

    public XobjList getParamIdList() {
      return paramIdList;
    }

    public void addParamId(Ident id) {
      paramIdList.add(id);
    }

    XobjList getLocalIdList() {
      return localIdList;
    }

    public void addLocalId(Ident id) {
      localIdList.add(id);
    }
  }

  //
  // make kernel functions executed in GPU
  //
   XobjectDef makeDeviceKernelDef(String deviceKernelName, List<Ident> outerIdList, List<Block> kernelBody) {
    /* make deviceKernelBody */
    DeviceKernelBuildInfo kernelBuildInfo = new DeviceKernelBuildInfo();

    //make params
    //add paramId from outerId
    for (Ident id : outerIdList) {
      if(ACC.debug_flag) System.out.println("makeDeviceKernelDef outerIdList id="+id);
      if (ACC.device.getUseReadOnlyDataCache() && _readOnlyOuterIdSet.contains(id)
	  && (id.Type().isArray() || id.Type().isPointer())) {
        Xtype constParamType = makeConstRestrictVoidType();
        Ident constParamId = Ident.Param("_ACC_cosnt_" + id.getName(), constParamType);

        Xtype arrayPtrType = Xtype.Pointer(id.Type().getRef());
        Ident localId = Ident.Local(id.getName(), arrayPtrType);
        Xobject initialize = Xcons.Set(localId.Ref(), Xcons.Cast(arrayPtrType, constParamId.Ref()));
        kernelBuildInfo.addParamId(constParamId);

        ACCvar accvar = _kernelInfo.findACCvar(id.getSym());
        if(accvar != null && accvar.isFirstprivate())
          localId.setProp(ACCgpuDecompiler.GPU_STORAGE_SHARED, true);
        kernelBuildInfo.addLocalId(localId);
        kernelBuildInfo.addInitBlock(Bcons.Statement(initialize));
      } else {
        Ident localId = makeParamId_new(id);
        ACCvar accvar = _kernelInfo.findACCvar(id.getSym());
        if(accvar != null && accvar.isFirstprivate())
          localId.setProp(ACCgpuDecompiler.GPU_STORAGE_SHARED, true);
        kernelBuildInfo.addParamId(localId);
      }
    }

    //make mainBody
    Block deviceKernelMainBlock = makeCoreBlock(kernelBody, kernelBuildInfo);

    //add private varId only if "parallel"
    if (_kernelInfo.getPragma() == ACCpragma.PARALLEL) {
      List<ACCvar> varList = _kernelInfo.getACCvarList();
      for (ACCvar var : varList) {
        if (var.isPrivate()) {
          Ident privateId = Ident.Local(var.getName(), var.getId().Type());
          privateId.setProp(ACCgpuDecompiler.GPU_STORAGE_SHARED, true); // ???
          kernelBuildInfo.addLocalId(privateId);
        }
      }
    }

    //if extern_sm is used, add extern_sm_id & extern_sm_offset_id
    if (sharedMemory.isUsed()) {
      kernelBuildInfo.addLocalId(sharedMemory.externSmId);
      kernelBuildInfo.addLocalId(sharedMemory.smOffsetId);
    }

    //if block reduction is used
    for(Xobject x : reductionManager.getBlockReductionLocalIds()){
      kernelBuildInfo.addLocalId((Ident) x);
    }

    if (reductionManager.hasUsingTmpReduction()) {
      for(Xobject x : reductionManager.getBlockReductionParamIds()){
        kernelBuildInfo.addParamId((Ident)x);
      }
      allocList.add(Xcons.List(reductionManager.tempPtr, Xcons.IntConstant(0), reductionManager.totalElementSize));
    }

    if (_globalMemoryStack.isUsed()){
      kernelBuildInfo.addParamId(_globalMemoryStack.getBaseId());
      kernelBuildInfo.addLocalId(_globalMemoryStack.getPosId());
      kernelBuildInfo.addInitBlock(_globalMemoryStack.makeInitFunc());
      allocList.add(Xcons.List(_globalMemoryStack.getBaseId(), Xcons.IntConstant(1024 /*temporal value*/), Xcons.IntConstant(0)));
    }

    //FIXME add extern_sm init func
    if (sharedMemory.isUsed()) {
      kernelBuildInfo.addInitBlock(sharedMemory.makeInitFunc()); //deviceKernelBody.add(sharedMemory.makeInitFunc());
    }
    kernelBuildInfo.addInitBlock(reductionManager.makeLocalVarInitFuncs()); //deviceKernelBody.add(reductionManager.makeLocalVarInitFuncs());

    kernelBuildInfo.addFinalizeBlock(reductionManager.makeReduceAndFinalizeFuncs());
    //deviceKernelBody.add(reductionManager.makeReduceAndFinalizeFuncs());

    // if(ACC.device == AccDevice.PEZYSC){
    //   kernelBuildInfo.addFinalizeBlock(Bcons.Statement(_accFlush));
    // }

    BlockList result = Bcons.emptyBody(kernelBuildInfo.getLocalIdList(), null);
    for(Block b : kernelBuildInfo.getInitBlockList()){
      result.add(b);
    }
    result.add(deviceKernelMainBlock);
    for(Block b : kernelBuildInfo.getFinalizeBlockList()){
      result.add(b);
    }

    XobjList deviceKernelParamIds = kernelBuildInfo.getParamIdList();

    Block resultBlock = Bcons.COMPOUND(result);

    rewriteReferenceType(resultBlock, deviceKernelParamIds);

    Ident deviceKernelId = _decl.getEnvDevice().declGlobalIdent(deviceKernelName, Xtype.Function(Xtype.voidType));
    ((FunctionType) deviceKernelId.Type()).setFuncParamIdList(deviceKernelParamIds);

    return XobjectDef.Func(deviceKernelId, deviceKernelParamIds, null, resultBlock.toXobject());
  }

   Xtype makeConstRestrictVoidType() {
    Xtype ptrType = Xtype.voidType.copy();
    ptrType.setIsRestrict(true);
    Xtype copyType = Xtype.Pointer(ptrType);
    copyType.setIsConst(true);
    return copyType;
  }

  void rewriteReferenceType(Block b, XobjList paramIds) {
    BasicBlockExprIterator iter = new BasicBlockExprIterator(b);
    for (iter.init(); !iter.end(); iter.next()) {
      Xobject expr = iter.getExpr();
      topdownXobjectIterator exprIter = new topdownXobjectIterator(expr);
      for (exprIter.init(); !exprIter.end(); exprIter.next()) {
        Xobject x = exprIter.getXobject();
        switch (x.Opcode()) {
        case VAR: {
          String varName = x.getName();
          if (varName.startsWith("_ACC_")) break;

          // break if the declaration exists in the DeviceKernelBlock
          if (iter.getBasicBlock().getParent().findVarIdent(varName) != null) break;

          // break if the ident doesn't exist in the parameter list
          Ident id = ACCutil.getIdent(paramIds, varName);
          if (id == null) break;

          // break if type is same
          if (x.Type().equals(id.Type())) break;

          if (id.Type().equals(Xtype.Pointer(x.Type()))) {
            Xobject newXobj = Xcons.PointerRef(id.Ref());
            exprIter.setXobject(newXobj);
          } else {
            ACC.fatal("type mismatch");
          }
        }
        break;
        case VAR_ADDR:
          // need to implement
        {
          String varName = x.getName();
          if (varName.startsWith("_ACC_")) break;

          // break if the declaration exists in the DeviceKernelBlock
          if (iter.getBasicBlock().getParent().findVarIdent(varName) != null) break;

          // break if the ident doesn't exist in the parameter list
          Ident id = ACCutil.getIdent(paramIds, varName);
          if (id == null) break;

          if (!x.Type().equals(Xtype.Pointer(id.Type()))) {
            if (x.Type().equals(id.Type())) {
              Xobject newXobj = id.Ref();
              exprIter.setXobject(newXobj);
            } else {
              ACC.fatal("type mismatch");
            }
          }
        }
        break;
        default:
        }
      }
    }
  }

  Block makeCoreBlock(Block b, DeviceKernelBuildInfo deviceKernelBuildInfo) {
    Set<ACCpragma> outerParallelisms = AccLoop.getOuterParallelism(b);
    switch (b.Opcode()) {
    case FOR_STATEMENT:
      return makeCoreBlockForStatement((CforBlock) b, deviceKernelBuildInfo);
    case COMPOUND_STATEMENT:
      return makeCoreBlock(b.getBody(), deviceKernelBuildInfo);
    case ACC_PRAGMA:
      {
        PragmaBlock pb = (PragmaBlock)b;
        ACCpragma pragma = ACCpragma.valueOf(pb.getPragma());
        if(pragma == ACCpragma.ATOMIC) {
          AccAtomic atomicDirective = (AccAtomic)b.getProp(AccDirective.prop);
          try {
            return atomicDirective.makeAtomicBlock();
          } catch (ACCexception exception) {
            exception.printStackTrace();
            ACC.fatal("failed at atomic");
          }
          
          // }else if(pragma == ACCpragma.SYNC) {
          //   AccSync syncDirective = (AccSync)b.getProp(AccDirective.prop);
          //   try {
          //     return syncDirective.makeSyncBlock();
          //   } catch (ACCexception exception) {
          //     exception.printStackTrace();
          //     ACC.fatal("failed at sync");
          //   }
          // }else if(pragma == ACCpragma.FLUSH) {
          //   AccFlush flushDirective = (AccFlush)b.getProp(AccDirective.prop);
          //   try {
          //     return flushDirective.makeFlushBlock();
          //   } catch (ACCexception exception) {
          //     exception.printStackTrace();
          //     ACC.fatal("failed at flush");
          //   }
          // }else if(pragma == ACCpragma.YIELD) {
          //   AccYield yieldDirective = (AccYield)b.getProp(AccDirective.prop);
          //   try {
          //     return yieldDirective.makeYieldBlock();
          //   } catch (ACCexception exception) {
          //     exception.printStackTrace();
          //     ACC.fatal("failed at yield");
          //   }
        } else {
          return makeCoreBlock(b.getBody(), deviceKernelBuildInfo);
        }
      }
    case OMP_PRAGMA:
      return makeCoreBlock(b.getBody(), deviceKernelBuildInfo);

    case IF_STATEMENT:
      {
        if (!outerParallelisms.contains(ACCpragma.VECTOR)) {
          BlockList resultBody = Bcons.emptyBody();

          Ident sharedIfCond = resultBody.declLocalIdent("_ACC_if_cond", Xtype.charType);
          sharedIfCond.setProp(ACCgpuDecompiler.GPU_STORAGE_SHARED, true);

          Block evalCondBlock = Bcons.IF(
                                         Xcons.binaryOp(Xcode.LOG_EQ_EXPR, _accThreadIndex, Xcons.IntConstant(0)),
                                         Bcons.Statement(Xcons.Set(sharedIfCond.Ref(), b.getCondBBlock().toXobject())),
                                         null);
          Block mainIfBlock = Bcons.IF(
                                       sharedIfCond.Ref(),
                                       makeCoreBlock(b.getThenBody(), deviceKernelBuildInfo),
                                       makeCoreBlock(b.getElseBody(), deviceKernelBuildInfo));

          resultBody.add(evalCondBlock);
          resultBody.add(_accSyncThreads);
          resultBody.add(mainIfBlock);

          return Bcons.COMPOUND(resultBody);
        } else {
          return b.copy();
        }
      }
    default:
      {
        Block resultBlock = b.copy();
        Block masterBlock = makeMasterBlock(EnumSet.copyOf(outerParallelisms), resultBlock);
        Block syncBlock = makeSyncBlock(EnumSet.copyOf(outerParallelisms));
        return Bcons.COMPOUND(Bcons.blockList(masterBlock, syncBlock));
      }
    }
  }

  Block makeCoreBlock(BlockList body, DeviceKernelBuildInfo deviceKernelBuildInfo) {
    if (body == null) return Bcons.emptyBlock();

    Xobject ids = body.getIdentList();
    Xobject decls = body.getDecls();
    BlockList varInitSection = Bcons.emptyBody();
    Map<Ident, Ident> rewriteIdents = new HashMap<>();
    Set<ACCpragma> outerParallelisms = AccLoop.getOuterParallelism(body.getParent());
    if (!outerParallelisms.contains(ACCpragma.VECTOR)) {
      if (ids != null) {
        for (XobjArgs args = ids.getArgs(); args != null; args = args.nextArgs()) {
          Ident id = (Ident) args.getArg();
          id.setProp(ACCgpuDecompiler.GPU_STORAGE_SHARED, true);
        }
      }
      //move decl initializer to body
      Block childBlock = body.getHead();
      if (decls != null && !(childBlock.Opcode() == Xcode.FOR_STATEMENT && ((CforBlock)childBlock).getInitBBlock().isEmpty())) {
        List<Block> varInitBlocks = new ArrayList<Block>();
        for (Xobject x : (XobjList) decls) {
          XobjList decl = (XobjList) x;
          if (decl.right() != null) {
            String varName = decl.left().getString();
            Ident id = ACCutil.getIdent((XobjList) ids, varName);
            Xobject initializer = decl.right();
            decl.setRight(null);
            {
              varInitBlocks.add(Bcons.Statement(Xcons.Set(id.Ref(), initializer)));
            }
          }
        }
        if (!varInitBlocks.isEmpty()) {
          BlockList thenBody = Bcons.emptyBody();
          for (Block b : varInitBlocks) {
            thenBody.add(b);
          }

          varInitSection.add(makeMasterBlock(EnumSet.copyOf(outerParallelisms), Bcons.COMPOUND(thenBody)));
          varInitSection.add(makeSyncBlock(EnumSet.copyOf(outerParallelisms)));
        }
      }
    }
    BlockList resultBody = Bcons.emptyBody(ids, decls);
    for (Block b = body.getHead(); b != null; b = b.getNext()) {
      resultBody.add(makeCoreBlock(b, deviceKernelBuildInfo));
    }

    Block resultBlock = Bcons.COMPOUND(resultBody);

    if (ids != null) {
	for (XobjArgs args = ids.getArgs(); args != null; args = args.nextArgs()) {
	    Ident id = (Ident) args.getArg();
	    Ident newId = rewriteIdents.get(id);
	    if(newId != null) args.setArg(newId);
	}
    }

    for(Map.Entry<Ident, Ident> entry : rewriteIdents.entrySet()){
      replaceVar(resultBlock, entry.getKey(), entry.getValue());
    }

    resultBody.insert(Bcons.COMPOUND(varInitSection));

    return resultBlock;
  }

  Block makeCoreBlock(List<Block> blockList, DeviceKernelBuildInfo deviceKernelBuildInfo) {
    BlockList resultBody = Bcons.emptyBody();
    for (Block b : blockList) {
      resultBody.add(makeCoreBlock(b, deviceKernelBuildInfo));
    }
    return makeBlock(resultBody);
  }

   Block makeBlock(BlockList blockList) {
    if (blockList == null || blockList.isEmpty()) {
      return Bcons.emptyBlock();
    }
    if (blockList.isSingle()) {
      Xobject decls = blockList.getDecls();
      XobjList ids = blockList.getIdentList();
      if ((decls == null || decls.isEmpty()) && (ids == null || ids.isEmpty())) {
        return blockList.getHead();
      }
    }
    return Bcons.COMPOUND(blockList);
  }

   Block makeCoreBlockForStatement(CforBlock forBlock, DeviceKernelBuildInfo deviceKernelBuildInfo) {
    BlockListBuilder resultBlockBuilder = new BlockListBuilder();

    //ACCinfo info = ACCutil.getACCinfo(forBlock);
    AccInformation info = null; //= (AccInformation)forBlock.getProp(AccInformation.prop);
    Block parentBlock = forBlock.getParentBlock();
    AccDirective directive = (AccDirective) parentBlock.getProp(AccDirective.prop);
    if (directive != null) {
      info = directive.getInfo();
    }

    if (info == null || !info.getPragma().isLoop()) {
      return makeSequentialLoop(forBlock, deviceKernelBuildInfo, null);
    }

    Xobject numGangsExpr = info.getIntExpr(ACCpragma.NUM_GANGS); //info.getNumGangsExp();
    if (numGangsExpr == null) numGangsExpr = info.getIntExpr(ACCpragma.GANG);
    Xobject numWorkersExpr = info.getIntExpr(ACCpragma.NUM_WORKERS);
    if (numWorkersExpr == null) numWorkersExpr = info.getIntExpr(ACCpragma.WORKER);
    Xobject vectorLengthExpr = info.getIntExpr(ACCpragma.VECT_LEN); //info.getVectorLengthExp();
    if (vectorLengthExpr == null) vectorLengthExpr = info.getIntExpr(ACCpragma.VECTOR);
//    System.out.println(numGangsExpr);
    if (numGangsExpr != null) gpuManager.setNumGangs(numGangsExpr);
    if (numWorkersExpr != null) gpuManager.setNumWorkers(numWorkersExpr);
    if (vectorLengthExpr != null) gpuManager.setVectorLength(vectorLengthExpr);

    String execMethodName = gpuManager.getMethodName(forBlock);
    EnumSet<ACCpragma> execMethodSet = gpuManager.getMethodType(forBlock);
    if (execMethodSet.isEmpty() || execMethodSet.contains(ACCpragma.SEQ)) { //if execMethod is not defined or seq
      return makeSequentialLoop(forBlock, deviceKernelBuildInfo, info);
//      loopStack.push(new Loop(forBlock));
//      BlockList body = Bcons.blockList(makeCoreBlock(forBlock.getBody(), deviceKernelBuildInfo, prevExecMethodName));
//      loopStack.pop();
//      return Bcons.FOR(forBlock.getInitBBlock(), forBlock.getCondBBlock(), forBlock.getIterBBlock(), body);
    }

    List<Block> cacheLoadBlocks = new ArrayList<Block>();

    LinkedList<CforBlock> collapsedForBlockList = new LinkedList<CforBlock>();

    Set<String> indVarSet = new LinkedHashSet<String>();
    {
      CforBlock tmpForBlock = forBlock;
      collapsedForBlockList.add(forBlock);
      indVarSet.add(forBlock.getInductionVar().getSym());

      Xobject collapseNumExpr = info.getIntExpr(ACCpragma.COLLAPSE);
      int collapseNum = collapseNumExpr != null ? collapseNumExpr.getInt() : 1;
      for (int i = 1; i < collapseNum; i++) {
        tmpForBlock = AccLoop.findOutermostTightlyNestedForBlock(tmpForBlock.getBody().getHead());
        collapsedForBlockList.add(tmpForBlock);
        indVarSet.add(tmpForBlock.getInductionVar().getSym());
      }
    }

    //private
    {
      for (ACCvar var : info.getACCvarList()) {
        if (!var.isPrivate()) {
          continue;
        }
        if (indVarSet.contains(var.getSymbol())) {
          continue;
        }
        Xtype varType = var.getId().Type();
        if (execMethodSet.contains(ACCpragma.VECTOR)) {
          resultBlockBuilder.declLocalIdent(var.getName(), varType);
        } else if (execMethodSet.contains(ACCpragma.GANG)) {
          if (varType.isArray()) {
            Ident arrayPtrId = Ident.Local(var.getName(), Xtype.Pointer(varType.getRef()));
            Ident privateArrayParamId = Ident.Param("_ACC_prv_" + var.getName(), Xtype.voidPtrType);
            deviceKernelBuildInfo.addLocalId(arrayPtrId);
            deviceKernelBuildInfo.addParamId(privateArrayParamId);

            try {
              Xobject sizeObj = Xcons.binaryOp(Xcode.MUL_EXPR,
                      ACCutil.getArrayElmtCountObj(varType),
                      Xcons.SizeOf(varType.getArrayElementType()));
              XobjList initPrivateFuncArgs =
                Xcons.List(Xcons.Cast(Xtype.Pointer(Xtype.voidPtrType),
                                      arrayPtrId.getAddr()), privateArrayParamId.Ref(), sizeObj);
              Block initPrivateFuncCall = ACCutil.createFuncCallBlock("_ACC_init_private", initPrivateFuncArgs);
              deviceKernelBuildInfo.addInitBlock(initPrivateFuncCall);
              allocList.add(Xcons.List(var.getId(), Xcons.IntConstant(0), sizeObj));
            } catch (Exception e) {
              ACC.fatal(e.getMessage());
            }
          } else {
            Ident privateLocalId = Ident.Local(var.getName(), varType);
            privateLocalId.setProp(ACCgpuDecompiler.GPU_STORAGE_SHARED, true);
            // System.out.println("privateLocalId shared id="+privateLocalId);
            resultBlockBuilder.addIdent(privateLocalId);
          }
        }
      }
    }

    // for pezy-sc, safe sync
    boolean hasReduction = false;
    for (ACCvar var : info.getACCvarList()) {
      if (var.isReduction()){
        hasReduction = true;
	break;
      }
    }
    if (hasReduction){
      if (hasGangSync && execMethodSet.contains(ACCpragma.GANG)){
        resultBlockBuilder.addFinalizeBlock(Bcons.Statement(_accSyncGangs));
      }
    }

    //begin reduction
    List<Reduction> reductionList = new ArrayList<Reduction>();
    //Iterator<ACCvar> vars = info.getVars();
    //while (vars.hasNext()) {
    //ACCvar var = vars.next();
    for (ACCvar var : info.getACCvarList()) {
      if (!var.isReduction()) continue;

      Reduction reduction = reductionManager.addReduction(var, execMethodSet);
      if(_readOnlyOuterIdSet.contains(var.getId())){
        ACC.fatal("reduction variable is read-only, isn't it?");
      }
      if (! reduction.onlyKernelLast()) {
        resultBlockBuilder.addIdent(reduction.getLocalReductionVarId());
        resultBlockBuilder.addInitBlock(reduction.makeInitReductionVarFuncCall());
        resultBlockBuilder.addFinalizeBlock(reduction.makeInKernelReductionFuncCall(null));
      }
      reductionList.add(reduction);
    }//end reduction

    //make calc idx funcs
    List<Block> calcIdxFuncCalls = new ArrayList<Block>();
    XobjList vIdxIdList = Xcons.IDList();
    XobjList nIterIdList = Xcons.IDList();
    XobjList indVarIdList = Xcons.IDList();
    Boolean has64bitIndVar = false;
    for (CforBlock tmpForBlock : collapsedForBlockList) {
      String indVarName = tmpForBlock.getInductionVar().getName();
      Xtype indVarType = tmpForBlock.findVarIdent(indVarName).Type();
      Xtype idxVarType = Xtype.unsignedType;
      switch (indVarType.getBasicType()) {
      case BasicType.INT:
      case BasicType.UNSIGNED_INT:
        idxVarType = Xtype.unsignedType;
        break;
      case BasicType.LONGLONG:
      case BasicType.UNSIGNED_LONGLONG:
        idxVarType = Xtype.unsignedlonglongType;
        has64bitIndVar = true;
        break;
      }
      Xobject init = tmpForBlock.getLowerBound().copy();
      Xobject cond = tmpForBlock.getUpperBound().copy();
      Xobject step = tmpForBlock.getStep().copy();
      Ident vIdxId = Ident.Local("_ACC_idx_" + indVarName, idxVarType);
      Ident indVarId = Ident.Local(indVarName, indVarType);
      Ident nIterId = resultBlockBuilder.declLocalIdent("_ACC_niter_" + indVarName, idxVarType);
      Block calcNiterFuncCall =
        ACCutil.createFuncCallBlock("_ACC_calc_niter", Xcons.List(nIterId.getAddr(), init, cond, step));
      Block calcIdxFuncCall =
        ACCutil.createFuncCallBlock(ACC_CALC_IDX_FUNC, Xcons.List(vIdxId.Ref(), indVarId.getAddr(), init, cond, step));

      resultBlockBuilder.addInitBlock(calcNiterFuncCall);

      vIdxIdList.add(vIdxId);
      nIterIdList.add(nIterId);
      indVarIdList.add(indVarId);
      calcIdxFuncCalls.add(calcIdxFuncCall);
    }

    Xtype globalIdxType = has64bitIndVar ? Xtype.unsignedlonglongType : Xtype.unsignedType;

    Ident iterIdx = resultBlockBuilder.declLocalIdent("_ACC_" + execMethodName + "_idx", globalIdxType);
    Ident iterInit = resultBlockBuilder.declLocalIdent("_ACC_" + execMethodName + "_init", globalIdxType);
    Ident iterCond = resultBlockBuilder.declLocalIdent("_ACC_" + execMethodName + "_cond", globalIdxType);
    Ident iterStep = resultBlockBuilder.declLocalIdent("_ACC_" + execMethodName + "_step", globalIdxType);

    XobjList initIterFuncArgs = Xcons.List(iterInit.getAddr(), iterCond.getAddr(), iterStep.getAddr());
    Xobject nIterAll = Xcons.IntConstant(1);
    for (Xobject x : nIterIdList) {
      Ident nIterId = (Ident) x;
      nIterAll = Xcons.binaryOp(Xcode.MUL_EXPR, nIterAll, nIterId.Ref());
    }
    initIterFuncArgs.add(nIterAll);

    Block initIterFunc = ACCutil.createFuncCallBlock(ACC_INIT_ITER_FUNC_PREFIX + execMethodName, initIterFuncArgs);
    resultBlockBuilder.addInitBlock(initIterFunc);

    //make clac each idx from virtual idx
    Block calcEachVidxBlock = makeCalcIdxFuncCall(vIdxIdList, nIterIdList, iterIdx);

    //push Loop to stack
    Loop thisLoop = new Loop(forBlock, iterIdx, iterInit, iterCond, iterStep);
    loopStack.push(thisLoop);

    List<Cache> cacheList = new ArrayList<Cache>();

    if (false) {
      transLoopCache(forBlock, resultBlockBuilder, cacheLoadBlocks, cacheList);
    }

    BlockList parallelLoopBody = Bcons.emptyBody();
    parallelLoopBody.add(calcEachVidxBlock);

    for (Block b : calcIdxFuncCalls) parallelLoopBody.add(b);

    // add cache load funcs
    for (Block b : cacheLoadBlocks) {
      parallelLoopBody.add(b);
    }
    // add inner block
    BlockList innerBody = collapsedForBlockList.getLast().getBody();
    Block coreBlock = makeCoreBlock(innerBody, deviceKernelBuildInfo);

    //rewirteCacheVars
    for (Cache cache : cacheList) {
      cache.rewrite(coreBlock);
    }
    parallelLoopBody.add(coreBlock);

    //add the cache barrier func
    if (!cacheLoadBlocks.isEmpty()) {
      parallelLoopBody.add(_accSyncThreads);
    }

    {
      XobjList forBlockListIdents = (XobjList) indVarIdList.copy();//Xcons.List(indVarId);
      forBlockListIdents.mergeList(vIdxIdList);
      ///insert
      parallelLoopBody.setIdentList(forBlockListIdents);
    }

    Block parallelLoopBlock = Bcons.FOR(
            Xcons.Set(iterIdx.Ref(), iterInit.Ref()),
            Xcons.binaryOp(Xcode.LOG_LT_EXPR, iterIdx.Ref(), iterCond.Ref()),
            Xcons.asgOp(Xcode.ASG_PLUS_EXPR, iterIdx.Ref(), iterStep.Ref()),
            Bcons.COMPOUND(parallelLoopBody)
    );

    //rewriteReductionvar
    for (Reduction red : reductionList) {
      red.rewrite(parallelLoopBlock);
    }

    //make resultBody
    resultBlockBuilder.add(parallelLoopBlock);

    if (hasGangSync && execMethodSet.contains(ACCpragma.GANG)){
      resultBlockBuilder.addFinalizeBlock(Bcons.Statement(_accSyncGangs));
    } else if (execMethodSet.contains(ACCpragma.VECTOR)) {
      resultBlockBuilder.addFinalizeBlock(Bcons.Statement(_accSyncThreads));
    }

    //pop stack
    loopStack.pop();

    BlockList resultBody = resultBlockBuilder.build();
    return Bcons.COMPOUND(resultBody);
  }

   void transLoopCache(CforBlock forBlock, BlockListBuilder resultBlockBuilder, List<Block> cacheLoadBlocks, List<Cache> cacheList) {
    Block headBlock = forBlock.getBody().getHead();
    if(headBlock == null) return;

    AccDirective directive = (AccDirective)headBlock.getProp(AccDirective.prop);
    if (directive != null) {
      AccInformation headInfo = directive.getInfo();
      if (headInfo.getPragma() == ACCpragma.CACHE) {
        for (ACCvar var : headInfo.getACCvarList()) {
          if (!var.isCache()) continue;


          Ident cachedId = var.getId();
          XobjList subscripts = var.getSubscripts();

          Cache cache = sharedMemory.alloc(cachedId, subscripts);

          resultBlockBuilder.addInitBlock(cache.initFunc);
          cacheLoadBlocks.add(cache.loadBlock);

          resultBlockBuilder.addIdent(cache.cacheId);
          resultBlockBuilder.addIdent(cache.cacheSizeArrayId);
          resultBlockBuilder.addIdent(cache.cacheOffsetArrayId);

          //for after rewrite
          cacheList.add(cache);
        }//end while
      }
    }
  }

   Block makeSequentialLoop(CforBlock forBlock, DeviceKernelBuildInfo deviceKernelBuildInfo, AccInformation info) {
    loopStack.push(new Loop(forBlock));
    BlockList body = Bcons.blockList(makeCoreBlock(forBlock.getBody(), deviceKernelBuildInfo));
    loopStack.pop();

    forBlock.Canonicalize();
    Ident originalInductionVarId = null;
    Xobject originalInductionVar = null;
    if(forBlock.isCanonical()) {
      originalInductionVar = forBlock.getInductionVar();
      originalInductionVarId = forBlock.findVarIdent(originalInductionVar.getName());
    }else{
      ACC.fatal("non canonical loop");
    }

    //FIXME this is not good for nothing parallelism kernel
    Set<ACCpragma> outerParallelisms = AccLoop.getOuterParallelism(forBlock);
    BlockList resultBody = Bcons.emptyBody();
    if(info != null){
      for(ACCvar var : info.getACCvarList()){
        if(var.isPrivate()){
          if(var.getId() != originalInductionVarId) {
            resultBody.declLocalIdent(var.getName(), var.getId().Type());
          }
        }
      }
    }

    if (outerParallelisms.contains(ACCpragma.VECTOR)) {
      resultBody.add(Bcons.FOR(forBlock.getInitBBlock(), forBlock.getCondBBlock(), forBlock.getIterBBlock(), body));
      return Bcons.COMPOUND(resultBody);
    }

    XobjList identList = resultBody.getIdentList();
    if(identList != null){
      for(Xobject xobj : identList){
        Ident id = (Ident)xobj;
        id.setProp(ACCgpuDecompiler.GPU_STORAGE_SHARED, true);
        // System.out.println("resultBody shared id="+id);
      }
    }

    Ident inductionVarId = resultBody.declLocalIdent("_ACC_loop_iter_" + originalInductionVar.getName(),
            originalInductionVar.Type());
    Block mainLoop = Bcons.FOR(Xcons.Set(inductionVarId.Ref(), forBlock.getLowerBound()),
            Xcons.binaryOp(Xcode.LOG_LT_EXPR, inductionVarId.Ref(), forBlock.getUpperBound()),
            Xcons.asgOp(Xcode.ASG_PLUS_EXPR, inductionVarId.Ref(), forBlock.getStep()),
            Bcons.COMPOUND(body));
    resultBody.add(mainLoop);

    ACCvar var = (info != null)? info.findACCvar(originalInductionVar.getName()) : null;
    if(var == null || !(var.isPrivate() || var.isFirstprivate())) {
      Block endIf = Bcons.IF(Xcons.binaryOp(Xcode.LOG_EQ_EXPR, _accThreadIndex, Xcons.IntConstant(0)),
              Bcons.Statement(Xcons.Set(originalInductionVar, inductionVarId.Ref())), null);
      resultBody.add(endIf);
    }
    resultBody.add(_accSyncThreads);

    replaceVar(mainLoop, originalInductionVarId, inductionVarId);
    return Bcons.COMPOUND(resultBody);
  }


  private Block makeCalcIdxFuncCall(XobjList vidxIdList, XobjList nIterIdList, Ident vIdx) {
    int i;
    Xobject result = vIdx.Ref();
    Ident funcId = ACCutil.getMacroFuncId("_ACC_calc_vidx", Xtype.intType);

    for (i = vidxIdList.Nargs() - 1; i > 0; i--) {
      Ident indVarId = (Ident) (vidxIdList.getArg(i));
      Ident nIterId = (Ident) (nIterIdList.getArg(i));
      Block callBlock = Bcons.Statement(funcId.Call(Xcons.List(indVarId.getAddr(), nIterId.Ref(), result)));
      result = callBlock.toXobject();
    }

    Ident indVarId = (Ident) (vidxIdList.getArg(0));
    result = Xcons.Set(indVarId.Ref(), result);

    return Bcons.Statement(result);
  }

  // make host code to launch the kernel
  Block makeLaunchFuncBlock(String launchFuncName, XobjectDef deviceKernelDef) {
    XobjList deviceKernelCallArgs = Xcons.List();
    BlockListBuilder blockListBuilder = new BlockListBuilder();
    XobjList confDecl = gpuManager.getBlockThreadSize();

    //# of block and thread
    Ident funcId = ACCutil.getMacroFuncId("_ACC_adjust_num_gangs", Xtype.voidType);
    Xobject num_gangs_decl = funcId.Call(Xcons.List(
            confDecl.getArg(0),
            Xcons.IntConstant(ACC.device.getMaxNumGangs())));
    Ident num_gangs = blockListBuilder.declLocalIdent("_ACC_num_gangs", Xtype.intType, num_gangs_decl);
    Ident num_workers = blockListBuilder.declLocalIdent("_ACC_num_workers", Xtype.intType, confDecl.getArg(1));
    Ident vec_len = blockListBuilder.declLocalIdent("_ACC_vec_len", Xtype.intType, confDecl.getArg(2));

    Ident mpool = Ident.Local("_ACC_mpool", Xtype.voidPtrType);
    Ident mpoolPos = Ident.Local("_ACC_mpool_pos", Xtype.longlongType);

    if (!allocList.isEmpty() || !_useMemPoolOuterIdSet.isEmpty()) {
      Block initMpoolPos = Bcons.Statement(Xcons.Set(mpoolPos.Ref(), Xcons.LongLongConstant(0, 0)));
      Block getMpoolFuncCall;
      Xobject asyncExpr = _kernelInfo.getIntExpr(ACCpragma.ASYNC);
      if (asyncExpr == null) {
        asyncExpr = Xcons.IntConstant(ACC.ACC_ASYNC_SYNC);
      }
      getMpoolFuncCall = ACCutil.createFuncCallBlock("_ACC_mpool_get_async", Xcons.List(mpool.getAddr(), asyncExpr));
      blockListBuilder.addIdent(mpool);
      blockListBuilder.addIdent(mpoolPos);
      blockListBuilder.addInitBlock(initMpoolPos);
      blockListBuilder.addInitBlock(getMpoolFuncCall);
    }

    XobjList reductionKernelCallArgs = Xcons.List();
    int reductionKernelVarCount = 0;

    for (ACCvar var : _outerVarList) {
      Ident varId = var.getId();
      Xobject paramRef;
      if (_useMemPoolOuterIdSet.contains(varId)) {
        Ident devPtrId = blockListBuilder.declLocalIdent("_ACC_dev_" + varId.getName(), Xtype.voidPtrType);
        Xobject size = var.getSize();

        Block mpoolAllocFuncCall = ACCutil.createFuncCallBlock(ACC_MPOOL_ALLOC_FUNCNAME, Xcons.List(devPtrId.getAddr(), size, mpool.Ref(), mpoolPos.getAddr()));
        Block mpoolFreeFuncCall = ACCutil.createFuncCallBlock(ACC_MPOOL_FREE_FUNCNAME, Xcons.List(devPtrId.Ref(), mpool.Ref()));
        Block HtoDCopyFuncCall = ACCutil.createFuncCallBlock("_ACC_copy_async", Xcons.List(varId.getAddr(), devPtrId.Ref(), size, Xcons.IntConstant(400), getAsyncExpr()));
        Block DtoHCopyFuncCall = ACCutil.createFuncCallBlock("_ACC_copy_async", Xcons.List(varId.getAddr(), devPtrId.Ref(), size, Xcons.IntConstant(401), getAsyncExpr()));
        blockListBuilder.addInitBlock(mpoolAllocFuncCall);
        blockListBuilder.addInitBlock(HtoDCopyFuncCall);
        blockListBuilder.addFinalizeBlock(DtoHCopyFuncCall);
        blockListBuilder.addFinalizeBlock(mpoolFreeFuncCall);
        paramRef = Xcons.Cast(Xtype.Pointer(varId.Type()), devPtrId.Ref());
      } else {
        paramRef = makeLaunchFuncArg(var);
      }

      deviceKernelCallArgs.add(paramRef);
      {
        Reduction red = reductionManager.findReduction(varId);
        if (red != null && red.needsExternalReduction()) {
          reductionKernelCallArgs.add(paramRef);
          reductionKernelVarCount++;
        }
      }
    }

    for (XobjList xobjList : allocList) {
      Ident varId = (Ident) (xobjList.getArg(0));
      Xobject baseSize = xobjList.getArg(1);
      Xobject numBlocksFactor = xobjList.getArg(2);

      Ident devPtrId = blockListBuilder.declLocalIdent("_ACC_gpu_device_" + varId.getName(), Xtype.voidPtrType);
      deviceKernelCallArgs.add(devPtrId.Ref());
      if (varId.getName().equals(ACC_REDUCTION_TMP_VAR)) {
        reductionKernelCallArgs.add(devPtrId.Ref());
      }

      Xobject size = Xcons.binaryOp(Xcode.PLUS_EXPR, baseSize,
              Xcons.binaryOp(Xcode.MUL_EXPR, numBlocksFactor, num_gangs.Ref()));
      Block mpoolAllocFuncCall = ACCutil.createFuncCallBlock(ACC_MPOOL_ALLOC_FUNCNAME, Xcons.List(devPtrId.getAddr(), size, mpool.Ref(), mpoolPos.getAddr()));
      Block mpoolFreeFuncCall = ACCutil.createFuncCallBlock(ACC_MPOOL_FREE_FUNCNAME, Xcons.List(devPtrId.Ref(), mpool.Ref()));
      blockListBuilder.addInitBlock(mpoolAllocFuncCall);
      blockListBuilder.addFinalizeBlock(mpoolFreeFuncCall);
    }

    //add blockReduction cnt & tmp
    if (reductionManager.hasUsingTmpReduction()) {
      Ident blockCountId = blockListBuilder.declLocalIdent("_ACC_gpu_block_count", Xtype.Pointer(Xtype.unsignedType));
      deviceKernelCallArgs.add(blockCountId.Ref());
      Block getBlockCounterFuncCall;
      Xobject asyncExpr = _kernelInfo.getIntExpr(ACCpragma.ASYNC);
      if (asyncExpr != null) {
        getBlockCounterFuncCall =
          ACCutil.createFuncCallBlock("_ACC_gpu_get_block_count_async", Xcons.List(blockCountId.getAddr(), asyncExpr));
      } else {
        getBlockCounterFuncCall =
          ACCutil.createFuncCallBlock("_ACC_gpu_get_block_count", Xcons.List(blockCountId.getAddr()));
      }
      blockListBuilder.addInitBlock(getBlockCounterFuncCall);
    }

    Block kernelLauchBlock = Bcons.emptyBlock();
    kernelLauchBlock = makeLauncherFuncCallCUDA(launchFuncName, deviceKernelDef, deviceKernelCallArgs, num_gangs.Ref(), num_workers.Ref(), vec_len.Ref(), getAsyncExpr());
    blockListBuilder.add(kernelLauchBlock);

    /* execute reduction Ops on tempoary after executing main kernel */
    /* generate Launch kernel call on host side */
    if (reductionManager.hasUsingTmpReduction()) {
      XobjectDef reductionKernelDef =
        reductionManager.makeReductionKernelDef(launchFuncName + "_red" + ACC_GPU_DEVICE_FUNC_SUFFIX);

      XobjectFile devEnv = _decl.getEnvDevice();
      devEnv.add(reductionKernelDef);
      reductionKernelCallArgs.add(num_gangs.Ref());

      Block reductionKernelCallBlock = Bcons.emptyBlock();

      reductionKernelCallBlock = makeLauncherFuncCallCUDA(launchFuncName + "_red", reductionKernelDef,
                                                          reductionKernelCallArgs,
                                                          Xcons.IntConstant(reductionKernelVarCount),
                                                          Xcons.IntConstant(1),
                                                          Xcons.IntConstant(ACC.device.getDefaultVectorLength()),
                                                          getAsyncExpr());

      Block ifBlock = Bcons.IF(Xcons.binaryOp(Xcode.LOG_GT_EXPR, num_gangs.Ref(), Xcons.IntConstant(1)),
                               reductionKernelCallBlock, null);
      blockListBuilder.add(ifBlock);
    }

    if (!_kernelInfo.hasClause(ACCpragma.ASYNC)) {
      blockListBuilder.addFinalizeBlock(ACCutil.createFuncCallBlock("_ACC_gpu_wait",
                                      Xcons.List(Xcons.IntConstant(ACC.ACC_ASYNC_SYNC) /*_accAsyncSync*/)));
    }

    BlockList launchFuncBody = blockListBuilder.build();

    return Bcons.COMPOUND(launchFuncBody);
  }

   Block makeLauncherFuncCallCUDA(String launchFuncName, XobjectDef deviceKernelDef, XobjList deviceKernelCallArgs,
                                  Xobject num_gangs, Xobject num_workers, Xobject vec_len, Xobject asyncExpr) {
    Xobject const1 = Xcons.IntConstant(1);
    BlockList body = Bcons.emptyBody();
    XobjList conf = Xcons.List(num_gangs, const1, const1, vec_len, num_workers, const1);
    Ident confId = body.declLocalIdent("_ACC_conf", Xtype.Array(Xtype.intType, 6), StorageClass.AUTO, conf);
    XobjectDef launcherFuncDef = makeLauncherFuncDefCUDA(launchFuncName, deviceKernelDef, deviceKernelCallArgs);
    XobjectFile devEnv = _decl.getEnvDevice();
    devEnv.add(launcherFuncDef);

    Ident launcherFuncId = _decl.declExternIdent(launcherFuncDef.getName(), Xtype.Function(Xtype.voidType));
    XobjList callArgs = Xcons.List();
    for(Xobject arg : deviceKernelCallArgs){
      if(arg.Opcode() == Xcode.CAST_EXPR && arg.Type().isArray()){
        arg = Xcons.Cast(Xtype.Pointer(arg.Type().getRef()), arg.getArg(0));
      }
      callArgs.add(arg);
    }
    callArgs.add(confId.Ref());
    callArgs.add(asyncExpr);

    body.add(Bcons.Statement(launcherFuncId.Call(callArgs)));

    return Bcons.COMPOUND(body);
  }

   XobjectDef makeLauncherFuncDefCUDA(String launchFuncName, XobjectDef deviceKernelDef,
                                      XobjList deviceKernelCallArgs) {
    XobjList launcherFuncParamIds = Xcons.IDList();
    BlockList launcherFuncBody = Bcons.emptyBody();
    XobjList args = Xcons.List();
    for(Xobject arg : deviceKernelCallArgs){
      Xtype type = arg.Type();
      if(arg.Opcode() == Xcode.CAST_EXPR){
        arg = arg.getArg(0);
      }
      String varName = arg.getName();
      Ident id = Ident.Param(varName, type);
      launcherFuncParamIds.add(id);

      args.add(arg);
    }

    //confs
    int numConfs = 6;
    Ident confParamId = Ident.Param("_ACC_conf", Xtype.Array(Xtype.intType, numConfs));
    launcherFuncParamIds.add(confParamId);
    XobjList confList = Xcons.List();
    for(int i = 0; i < numConfs; i++) {
      confList.add(Xcons.arrayRef(Xtype.intType, confParamId.getAddr(), Xcons.List(Xcons.IntConstant(i))));
    }

    //asyncnum
    Ident asyncId = Ident.Param("_ACC_async_num", Xtype.intType);
    launcherFuncParamIds.add(asyncId);
    Xobject asyncExpr = asyncId.Ref();

    Ident deviceKernelId = (Ident) deviceKernelDef.getNameObj();
    Block callBlock = makeKernelLaunchBlockCUDA(deviceKernelId, args, confList, asyncExpr);
    launcherFuncBody.add(callBlock);

    Ident launcherFuncId = _decl.getEnvDevice().declGlobalIdent(launchFuncName, Xtype.Function(Xtype.voidType));
    ((FunctionType) launcherFuncId.Type()).setFuncParamIdList(launcherFuncParamIds);

    return XobjectDef.Func(launcherFuncId, launcherFuncParamIds, null, launcherFuncBody.toXobject());
  }

  Xtype makeConstArray(ArrayType at){
    if(at.getRef().isArray()){
      ArrayType ret_t = (ArrayType)(at.copy());
      ret_t.setRef(makeConstArray((ArrayType)(at.getRef())));
      return ret_t;
    } else {
      Xtype ret_t = at.copy();
      Xtype new_ref = at.getRef().copy();
      new_ref.setIsConst(true);
      ret_t.setRef(new_ref);
      return ret_t;
    }
  }

  Ident makeParamId_new(Ident id) {
    String varName = id.getName();

    switch (id.Type().getKind()) {
    case Xtype.ARRAY: {
      Xtype t = id.Type().copy();
      if(_readOnlyOuterIdSet.contains(id)){
        t = makeConstArray((ArrayType)t);
      }
      return Ident.Local(varName, t);
    }
    case Xtype.POINTER: {
      Xtype t = id.Type().copy();
      if(_readOnlyOuterIdSet.contains(id)) t.setIsConst(true);
      return Ident.Local(varName, t);
    }
    case Xtype.BASIC:
    case Xtype.STRUCT:
    case Xtype.UNION:
    case Xtype.ENUM: {
      // check whether id is firstprivate!
      ACCvar var = _kernelInfo.findACCvar(ACCpragma.FIRSTPRIVATE, varName);
      if (var == null /*kernelInfo.isVarAllocated(varName)*/ || _useMemPoolOuterIdSet.contains(id) || var.getDevicePtr() != null) {
	Xtype t = id.Type().copy();
	if(_readOnlyOuterIdSet.contains(id)) t.setIsConst(true);
        return Ident.Local(varName, Xtype.Pointer(t));
      } else {
        return Ident.Local(varName, id.Type());
      }
    }
    default:
      ACC.fatal("unknown type");
      return null;
    }
  }

  public void analyze() {

    if(ACC.debug_flag) System.out.println("AccKernel.analyze ... _kernelInfo="+_kernelInfo);
    gpuManager.analyze();

    // System.out.println("AccKernel.analyze outerID ..._kernelInfo="+_kernelInfo);
    //get outerId set
    Set<Ident> outerIdSet = new LinkedHashSet<Ident>();
    OuterIdCollector outerIdCollector = new OuterIdCollector();
    for (Block b : _kernelBlocks) {
      outerIdSet.addAll(outerIdCollector.collect(b));
    }

    if(ACC.debug_flag)
      for (Ident id : outerIdSet)  System.out.println("outIdSet id="+id);

    //collect read only id
    _readOnlyOuterIdSet = new LinkedHashSet<Ident>(outerIdSet);
    AssignedIdCollector assignedIdCollector = new AssignedIdCollector();
    for (Block b : _kernelBlocks) {
      _readOnlyOuterIdSet.removeAll(assignedIdCollector.collect(b));
    }

    if(ACC.debug_flag)
      for (Ident id : _readOnlyOuterIdSet)  System.out.println("readOnlyOutIdSet id="+id);

    //make outerId list
    _outerIdList = new ArrayList<Ident>(outerIdSet);

    //FIXME
    for (Ident id : _outerIdList) {
      ACCvar var = _kernelInfo.findACCvar(id.getSym());
      if(ACC.debug_flag) System.out.println("outerId id="+id+",var="+var);
      if (var == null) continue;
      if (var.isReduction()) {
        ACCvar parentVar = findParentVar(id);
        if (parentVar == null) {
          _useMemPoolOuterIdSet.add(id);
        }
      }
    }
  }

  //copied
  ACCvar findParentVar(Ident varId) {
    if (_pb != null) {
      for (Block b = _pb.getParentBlock(); b != null; b = b.getParentBlock()) {
        // if (b.Opcode() != Xcode.ACC_PRAGMA) continue;
        AccInformation info = ((AccDirective) b.getProp(AccDirective.prop)).getInfo();
        if(info == null) continue;
        
        ACCvar var = info.findACCvar(varId.getSym());
        if (var != null && var.getId() == varId) {
          return var;
        }
      }
    }

    return null;
  }

  boolean hasBody(Block b){
    switch(b.Opcode()){
      case DO_STATEMENT:
      case SWITCH_STATEMENT:
      case WHILE_STATEMENT:
      case FOR_STATEMENT:
      case COMPOUND_STATEMENT:
        return true;
      default:
        return false;
    }
  }

  void replaceVar(Block b, Ident fromId, Ident toId) {
    BasicBlockExprIterator iter = new BasicBlockExprIterator(b);
    for (iter.init(); !iter.end(); iter.next()) {
      Xobject expr = iter.getExpr();
      Block parentBlock = iter.getBasicBlock().getParent();
      replaceVar(b, fromId, toId, expr, parentBlock);
    }

    BlockIterator bIter = new topdownBlockIterator(b);
    for (bIter.init(); !bIter.end(); bIter.next()) {
      Block block = bIter.getBlock();
      if(hasBody(block)) {
        Xobject decls = block.getBody().getDecls();
        replaceVarInDecls(b, fromId, toId, block, (XobjList) decls);
      }else if(block.Opcode() == Xcode.IF_STATEMENT){
        replaceVarInDecls(b, fromId, toId, block, (XobjList) block.getThenBody().getDecls());
        replaceVarInDecls(b, fromId, toId, block, (XobjList) block.getElseBody().getDecls());
      }
    }
  }

   void replaceVarInDecls(Block b, Ident fromId, Ident toId, Block block, XobjList decls) {
    if(decls == null) return;
    for(Xobject decl : decls){
      Xobject declRight = decl.right();
      decl.setRight(replaceVar(b, fromId, toId, declRight, block));
    }
  }

   Xobject replaceVar(Block b, Ident fromId, Ident toId, Xobject expr, Block parentBlock) {
    topdownXobjectIterator exprIter = new topdownXobjectIterator(expr);
    for (exprIter.init(); !exprIter.end(); exprIter.next()) {
      Xobject x = exprIter.getXobject();
      switch(x.Opcode()) {
      case VAR: {
        String varName = x.getName();
        if (!fromId.getName().equals(varName)) continue;
        Ident id = findInnerBlockIdent(b, parentBlock.getParent(), "_ACC_thread_x_id");

        // if (id != fromId && id != toId) continue; //if(id != fromId) continue;

        Xobject replaceXobject = null;
        if (toId.Type().equals(fromId.Type())) {
          replaceXobject = toId.Ref();
        } else if (toId.Type().isPointer() && toId.Type().getRef().equals(fromId.Type())) {
          replaceXobject = Xcons.PointerRef(toId.Ref());
        } else {
          ACC.fatal("unexpected fromId or toId type");
        }
        if (expr == x) return replaceXobject;
        exprIter.setXobject(replaceXobject);
        break;
      }
      case VAR_ADDR: {
        String varName = x.getName();
        if (!fromId.getName().equals(varName)) continue;
        Ident id = findInnerBlockIdent(b, parentBlock.getParent(), "_ACC_thread_x_id");

        if (id != fromId && id != toId) continue; //if(id != fromId) continue;

        Xobject replaceXobject = null;
        if (toId.Type().equals(fromId.Type())) {
          replaceXobject = toId.getAddr();
        } else if (toId.Type().isPointer() && toId.Type().getRef().equals(fromId.Type())) {
          replaceXobject = toId.Ref();
        } else {
          ACC.fatal("unexpected fromId or toId type");
        }
        if (expr == x) return replaceXobject;
        exprIter.setXobject(replaceXobject);
        break;
      }
      }
    }
    return expr;
  }

  public void setReadOnlyOuterIdSet(Set<Ident> readOnlyOuterIdSet) {
    _readOnlyOuterIdSet = readOnlyOuterIdSet;
  }

  public Set<Ident> getReadOnlyOuterIdSet() {
    return _readOnlyOuterIdSet;
  }

  public Set<Ident> getOuterIdSet() {
    return new LinkedHashSet<Ident>(_outerIdList);
  }

  public List<Ident> getOuterIdList() {
    return _outerIdList;
  }

  Block makeMasterBlock(EnumSet<ACCpragma> outerParallelism, Block thenBlock){
    Xobject condition = null;
    if(outerParallelism.contains(ACCpragma.VECTOR)) {
      return thenBlock;
    }else if(outerParallelism.contains(ACCpragma.WORKER)){
      condition = Xcons.binaryOp(Xcode.LOG_EQ_EXPR, _accThreadIndexY, Xcons.IntConstant(0));
    }else if(outerParallelism.contains(ACCpragma.GANG)){
      condition = Xcons.binaryOp(Xcode.LOG_EQ_EXPR, _accThreadIndex, Xcons.IntConstant(0));
    }else{
        condition = Xcons.binaryOp(Xcode.LOG_EQ_EXPR, _accThreadIndex, Xcons.IntConstant(0));
    }
    return Bcons.IF(condition, thenBlock, null);
  }

  Block makeSyncBlock(EnumSet<ACCpragma> outerParallelism){
    if(outerParallelism.contains(ACCpragma.VECTOR)) {
      return Bcons.emptyBlock();
      //}else if(outerParallelism.contains(ACCpragma.WORKER)){
    }else if(outerParallelism.contains(ACCpragma.GANG)){
      return Bcons.Statement(_accSyncThreads);
    }else{
      return Bcons.Statement(_accSyncThreads);
    }
  }

  //
  // Outer Id Collector
  //
  class OuterIdCollector {
    public Set<Ident> collect(Block topBlock) {
      Set<Ident> outerIdSet = new LinkedHashSet<Ident>();

      collectVarIdents(topBlock, outerIdSet);
      collectlVarIdentsInDecl(topBlock, outerIdSet);

      return outerIdSet;
    }

    void collectlVarIdentsInDecl(Block topBlock, Set<Ident> outerIdSet) {
      BlockIterator bIter = new topdownBlockIterator(topBlock);
      for (bIter.init(); !bIter.end(); bIter.next()) {
        Block b = bIter.getBlock();
        
        if(hasBody(b)){
          Xobject decls = b.getBody().getDecls();
          collectVarIdentsInDecls(topBlock, outerIdSet, b, decls);
        }else if(b.Opcode() == Xcode.IF_STATEMENT){
          collectVarIdentsInDecls(topBlock, outerIdSet, b, b.getThenBody().getDecls());
          collectVarIdentsInDecls(topBlock, outerIdSet, b, b.getElseBody().getDecls());
        }
      }
    }

     void collectVarIdentsInDecls(Block topBlock, Set<Ident> outerIdSet, Block b, Xobject decls) {
      if (decls == null) return;
      Set<String> varNameSet = collectVarNames(decls);
      for (String name : varNameSet) {
        Ident id = find(topBlock, b, name);
        if (id == null) continue;
        outerIdSet.add(id);
      }
    }

     void collectVarIdents(Block topBlock, Set<Ident> outerIdSet) {
      BasicBlockExprIterator bbexprIter = new BasicBlockExprIterator(topBlock);
      for (bbexprIter.init(); !bbexprIter.end(); bbexprIter.next()) {
        for (String varName : collectVarNames(bbexprIter.getExpr())) {
          Ident id = find(topBlock, bbexprIter.getBasicBlock().getParent(), varName);
          if (id == null) continue;
          outerIdSet.add(id);
        }
      }
    }

    Set<String> collectVarNames(Xobject expr) {
      Set<String> varNameSet = new LinkedHashSet<String>();

      XobjectIterator xobjIter = new topdownXobjectIterator(expr);
      for (xobjIter.init(); !xobjIter.end(); xobjIter.next()) {
        Xobject x = xobjIter.getXobject();
        if (x == null) continue;
        switch (x.Opcode()) {
        case VAR: {
          String varName = x.getName();
          varNameSet.add(varName);
        }
        break;
        case ARRAY_REF: {
          String arrayName = x.getArg(0).getName();
          varNameSet.add(arrayName);
        }
        break;
        default:
        }
      }
      return varNameSet;
    }

    boolean isPrivate(PragmaBlock pb, String varName) {
      AccDirective directive = (AccDirective) pb.getProp(AccDirective.prop);
      AccInformation info = directive.getInfo();  //(AccInformation)pb.getProp(AccInformation.prop);
      if (info == null) return false;

      ACCvar var = info.findACCvar(varName);
      return var != null && var.isPrivate();
    }

     Ident find(Block topBlock, Block block, String name) {
      // if an id exists between root to topBlock, the id is outerId
      for (Block b = block; b != null; b = b.getParentBlock()) {
        if (b == topBlock.getParentBlock()) break;
        if ((b.Opcode() == Xcode.ACC_PRAGMA || b.Opcode() == Xcode.OMP_PRAGMA) &&
            isPrivate((PragmaBlock) b, name)) return null;
        if (hasLocalIdent(b.getBody(), name)) return null;
      }
      return topBlock.findVarIdent(name);
    }

    private boolean hasLocalIdent(BlockList body, String varName) {
      return body != null && body.findLocalIdent(varName) != null;
    }
  }

  //
  // Aassigned Id Collector
  //
  class AssignedIdCollector {
    Set<Ident> collect(Block kernelBlock) {
      Set<Ident> assignedIds = new LinkedHashSet<Ident>();

      BasicBlockExprIterator bbExprIter = new BasicBlockExprIterator(kernelBlock);
      for (bbExprIter.init(); !bbExprIter.end(); bbExprIter.next()) {
        Set<Ident> assignedIdsInBody = collect(bbExprIter.getExpr(), bbExprIter.getBasicBlock().getParent());
        assignedIds.addAll(assignedIdsInBody);
      }

      BlockIterator bIter = new topdownBlockIterator(kernelBlock);
      for (bIter.init(); !bIter.end(); bIter.next()) {
        Block b = bIter.getBlock();

        if(hasBody(b)){
          Xobject decls = b.getBody().getDecls();
          Set<Ident> assignedIdsInDecl = collect(decls, b);
          assignedIds.addAll(assignedIdsInDecl);
        }else if(b.Opcode() == Xcode.IF_STATEMENT){
          Xobject thenBodyDecls = b.getThenBody().getDecls();
          Xobject elseBodyDecls = b.getElseBody().getDecls();
          assignedIds.addAll(collect(thenBodyDecls, b));
          assignedIds.addAll(collect(elseBodyDecls, b));
        }
      }
      return assignedIds;
    }

     Set<Ident> collect(Xobject expr, Block b) {
      Set<Ident> assignedIds = new LinkedHashSet<Ident>();
      if(expr == null) return assignedIds;

      XobjectIterator xobjIter = new topdownXobjectIterator(expr);
      for (xobjIter.init(); !xobjIter.end(); xobjIter.next()) {
        Xobject x = xobjIter.getXobject();
        if (x == null) continue;
        Xobject assignedExpr = findAssignedExpr(x);
        if (assignedExpr == null) continue;
        Ident assignedVarId = getAssignedId(assignedExpr, b);
        assignedIds.add(assignedVarId);
      }
      return assignedIds;
    }

     Ident getAssignedId(Xobject expr, Block b) {
      Ident id = null;
      try {
        Xobject assignedXobject = getAssignedXobject(expr);
        String varName = assignedXobject.getName();
        id = b.findVarIdent(varName);
        if (id == null) {
          throw new ACCexception("variable '" + varName + "' is not found");
        }
      } catch (ACCexception accException) {
        ACC.fatal("getAssignedId: " + accException.getMessage());
      }
      return id;
    }

     Xobject getAssignedXobject(Xobject x) throws ACCexception {
      switch (x.Opcode()) {
      case VAR:
      case ARRAY_ADDR:
      case INT_CONSTANT:
        return x;
      case ARRAY_REF:
      case MEMBER_REF:
      case ADDR_OF:
      case POINTER_REF:
      case CAST_EXPR:
      case MEMBER_ADDR:
      case MEMBER_ARRAY_REF:
        return getAssignedXobject(x.getArg(0));
      case PLUS_EXPR:
      case MINUS_EXPR: {
        //only for pointer operation
        if (!x.Type().isPointer()) throw new ACCexception("not pointer");
        Xobject lhs = x.getArg(0);
        Xobject rhs = x.getArg(1);
        if (lhs.Type().isPointer()) {
          return getAssignedXobject(lhs);
        } else if (rhs.Type().isPointer()) {
          return getAssignedXobject(rhs);
        } else {
          throw new ACCexception("no pointer operand for PLUS or MINUS");
        }
      }
      case FUNCTION_CALL: {
        Xobject funcAddr = x.getArg(0);
        // for arrayRef generated by XMP
        if (funcAddr.getName().startsWith("_XMP_M_GET_ADDR_E")) {
          Xobject args = x.getArg(1);
          return args.getArg(0);
        }
      }
      default:
        throw new ACCexception("not supported type: " + x.Opcode());
      }
    }

     Xobject findAssignedExpr(Xobject x) {
      switch (x.Opcode()) {
      case PRE_INCR_EXPR:
      case PRE_DECR_EXPR:
      case POST_INCR_EXPR:
      case POST_DECR_EXPR:
        return x.getArg(0);
      default:
        if (x.Opcode().isAsgOp()) {
          return x.getArg(0);
        } else {
          return null;
        }
      }
    }
  }

  //
  // for Loop directive
  //
  class Loop {
    final CforBlock forBlock;
    final boolean isParallelized;
    final Ident abstIdx;
    final Ident abstInit;
    final Ident abstCond;
    final Ident abstStep;

    Loop(CforBlock forBlock) {
      this(forBlock, null, null, null, null);
    }

    Loop(CforBlock forBlock, Ident idx, Ident init, Ident cond, Ident step) {
      this.forBlock = forBlock;
      abstIdx = idx;
      abstInit = init;
      abstCond = cond;
      abstStep = step;
      isParallelized = (idx != null);
    }
  } // end of Loop

  //
  // SharedMemory for CUDA
  //
  class SharedMemory {
    final Ident externSmId;
    final Ident smOffsetId;
    final ArrayDeque<Xobject> smStack = new ArrayDeque<Xobject>();
    Xobject maxSize = Xcons.IntConstant(0);

    boolean isUsed = false;

    SharedMemory() {
      Xtype externSmType = Xtype.Array(Xtype.charType, null);
      externSmId = Ident.Var("_ACC_sm", externSmType, Xtype.Pointer(externSmType), VarScope.GLOBAL);
      externSmId.setStorageClass(StorageClass.EXTERN);
      externSmId.setProp(ACCgpuDecompiler.GPU_STORAGE_SHARED, true);

      smOffsetId = Ident.Local("_ACC_sm_offset", Xtype.intType);
      smOffsetId.setProp(ACCgpuDecompiler.GPU_STORAGE_SHARED, true);
    }

    public Xobject getMaxSize() {
      return maxSize;
    }

    public boolean isUsed() {
      return isUsed;
    }

    Cache alloc(Ident id, XobjList subscript) {
      Cache cache = new Cache(id, subscript);
      smStack.push(cache.cacheTotalSize);

      Xobject nowSize = Xcons.IntConstant(0);
      for (Xobject s : smStack) {
        nowSize = Xcons.binaryOp(Xcode.PLUS_EXPR, nowSize, s);
      }
      maxSize = Xcons.List(Xcode.CONDITIONAL_EXPR, Xcons.binaryOp(Xcode.LOG_LT_EXPR, nowSize, maxSize), Xcons.List(maxSize, nowSize));

      isUsed = true;
      return cache;
    }

// --Commented out by Inspection START (2015/02/24 21:12):
//    void free() {
//      smStack.pop();
//    }
// --Commented out by Inspection STOP (2015/02/24 21:12)

    Block makeInitFunc() {
      return ACCutil.createFuncCallBlock("_ACC_gpu_init_sm_offset", Xcons.List(smOffsetId.getAddr()));
    }
  } // end of Shared Memory

  //
  // Cache
  //
  class Cache {
    final XobjList localIds = Xcons.IDList();
    final Ident cacheSizeArrayId;
    final Ident cacheOffsetArrayId;
    final Xtype elementType;
    final Ident cacheId;
    final Ident varId;
    final XobjList subscripts;
    final int cacheDim;
    Xobject cacheTotalSize;
    final Block initFunc;
    final Block loadBlock;

    Cache(Ident varId, XobjList subscripts) {
      this.varId = varId;
      this.subscripts = subscripts;
      elementType = (varId.Type().isArray()) ? (varId.Type().getArrayElementType()) : varId.Type();
      Xtype cacheType = (varId.Type().isArray()) ? Xtype.Pointer(elementType) : elementType;

      cacheId = Ident.Local(ACC_CACHE_VAR_PREFIX + varId.getName(), cacheType);
      cacheDim = subscripts.Nargs();

      cacheSizeArrayId = Ident.Local("_ACC_cache_size_" + varId.getName(), Xtype.Array(Xtype.intType, cacheDim));
      cacheSizeArrayId.setProp(ACCgpuDecompiler.GPU_STORAGE_SHARED, true);

      cacheOffsetArrayId = Ident.Local("_ACC_cache_offset_" + varId.getName(), Xtype.Array(Xtype.intType, cacheDim));

      localIds.add(cacheOffsetArrayId);
      localIds.add(cacheSizeArrayId);

      initFunc = makeCacheInitializeFunc();
      loadBlock = makeCacheLoadBlock();
    }

    Block makeCacheInitializeFunc() {
      //XobjList getCacheFuncArgs = Xcons.List(_externSm.Ref(), _smOffset.Ref(), cacheId.getAddr(), cacheSizeArrayId.getAddr());//, step, cacheLength);
      XobjList getCacheFuncArgs = Xcons.List(sharedMemory.externSmId.Ref(), sharedMemory.smOffsetId.Ref(), cacheId.getAddr(), cacheSizeArrayId.getAddr());//, step, cacheLength);

      for (Xobject s : subscripts) {
        XobjList simpleSubarray = getSimpleSubarray(s);
        Xobject cacheIdx = simpleSubarray.getArg(0);
        //Xobject cacheConstOffset = simpleSubarray.getArg(1);
        Xobject cacheLength = simpleSubarray.getArg(2);

        //find loop
        Loop loop = null;
        for (Loop tmpLoop : loopStack) {
          if (tmpLoop.forBlock.getInductionVar().getName().equals(cacheIdx.getName())) {
            loop = tmpLoop;
            break;
          }
        }
        if (loop == null) ACC.fatal(cacheIdx.getName() + " is not loop variable");

        assert loop != null;
        Xobject step = loop.forBlock.getStep(); //もしcacheがloop変数非依存なら？
        getCacheFuncArgs.mergeList(Xcons.List(step, cacheLength));

      }

      return ACCutil.createFuncCallBlock("_ACC_gpu_init_cache", getCacheFuncArgs);
    }

    Block makeCacheLoadBlock() {
      BlockList cacheLoadBody = Bcons.emptyBody();
      XobjList cacheLoadBodyIds = Xcons.IDList();

      Ident cacheLoadSizeArrayId =
        Ident.Local("_ACC_cache_load_size_" + varId.getName(), Xtype.Array(Xtype.intType, cacheDim));
      cacheLoadSizeArrayId.setProp(ACCgpuDecompiler.GPU_STORAGE_SHARED, true);
      cacheLoadBodyIds.add(cacheLoadSizeArrayId);

      int dim = 0;
      Xobject totalCacheSize = Xcons.IntConstant(1);

      for (Xobject s : subscripts) {
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
        for (Loop tmpLoop : loopStack) {
          if (tmpLoop.forBlock.getInductionVar().getName().equals(cacheIdx.getName())) {
            loop = tmpLoop;
            break;
          }
        }
        if (loop == null) ACC.fatal(cacheIdx.getName() + " is not loop variable");

        Xobject calculatedCacheSize;
        assert loop != null;
        if (loop.isParallelized) {
          Xobject abstIdx = loop.abstIdx.Ref();
          Xobject abstCond = loop.abstCond.Ref();
          Xobject abstStep = loop.abstStep.Ref();
          Xobject concStep = loop.forBlock.getStep();
          getLoadSizeFuncArgs.mergeList(Xcons.List(abstIdx, abstCond, abstStep, concStep));

          String methodName = gpuManager.getMethodName(loop.forBlock);
          Xobject blockSize;
          if (methodName.endsWith("thread_x")) {
            blockSize = Xcons.Symbol(Xcode.VAR, Xtype.intType, "_ACC_GPU_DIM3_thread_x");
          } else if (methodName.endsWith("thread_y")) {
            blockSize = Xcons.Symbol(Xcode.VAR, Xtype.intType, "_ACC_GPU_DIM3_thread_y");
          } else if (methodName.endsWith("thread_z")) {
            blockSize = Xcons.Symbol(Xcode.VAR, Xtype.intType, "_ACC_GPU_DIM3_thread_z");
          } else {
            blockSize = Xcons.IntConstant(1);
          }
          calculatedCacheSize = Xcons.binaryOp(Xcode.PLUS_EXPR, Xcons.binaryOp(Xcode.MUL_EXPR, Xcons.binaryOp(Xcode.MINUS_EXPR, blockSize, Xcons.IntConstant(1)), concStep), cacheLength);
        } else {
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
      cacheTotalSize = Xcons.binaryOp(Xcode.MUL_EXPR, totalCacheSize, Xcons.SizeOf(elementType));

      //make load for loop
      Block dummyInnerMostBlock = Bcons.emptyBlock(); //dummy block
      Block loadLoopBlock = null;//dummyInnerMostBlock;
      XobjList lhsArrayRefList = Xcons.List();
      XobjList rhsArrayRefList = Xcons.List();
      for (int d = 0; d < cacheDim; d++) { //from higher dim
        Ident tmpIter = Ident.Local("_ACC_iter_idx" + d, Xtype.intType);
        cacheLoadBodyIds.add(tmpIter);
        Xobject tmpIterInit, tmpIterCond, tmpIterStep;
        tmpIterCond = Xcons.arrayRef(Xtype.intType, cacheLoadSizeArrayId.getAddr(), Xcons.List(Xcons.IntConstant(d)));
        if (d == cacheDim - 1) {
          Ident thread_x_id = Ident.Local("_ACC_thread_x_id", Xtype.intType); //this is macro
          tmpIterInit = thread_x_id.Ref();
          Ident block_size_x = Ident.Local("_ACC_block_size_x", Xtype.intType); //this is macro
          tmpIterStep = block_size_x.Ref();
        } else if (d == cacheDim - 2) {
          Ident thread_y_id = Ident.Local("_ACC_thread_y_id", Xtype.intType); //this is macro
          tmpIterInit = thread_y_id.Ref();
          Ident block_size_y = Ident.Local("_ACC_block_size_y", Xtype.intType); //this is macro
          tmpIterStep = block_size_y.Ref();
        } else if (d == cacheDim - 3) {
          Ident thread_z_id = Ident.Local("_ACC_thread_z_id", Xtype.intType); //this is macro
          tmpIterInit = thread_z_id.Ref();
          Ident block_size_z = Ident.Local("_ACC_block_size_z", Xtype.intType); //this is macro
          tmpIterStep = block_size_z.Ref();
        } else {
          tmpIterInit = Xcons.IntConstant(0);
          tmpIterStep = Xcons.IntConstant(1);
        }
        if (lhsArrayRefList.isEmpty()) {
          lhsArrayRefList.add(tmpIter.Ref());
        } else {
          Xobject newRef = lhsArrayRefList.getArg(0);
          newRef = Xcons.binaryOp(Xcode.PLUS_EXPR, Xcons.binaryOp(Xcode.MUL_EXPR, newRef, Xcons.arrayRef(Xtype.intType, cacheSizeArrayId.getAddr(), Xcons.List(Xcons.IntConstant(d)))), tmpIter.Ref());
          lhsArrayRefList.setArg(0, newRef);
        }
        rhsArrayRefList.add(Xcons.binaryOp(Xcode.MINUS_EXPR, tmpIter.Ref(), Xcons.arrayRef(Xtype.intType, cacheOffsetArrayId.getAddr(), Xcons.List(Xcons.IntConstant(d)))));
        if (loadLoopBlock == null) {
          loadLoopBlock = Bcons.FORall(tmpIter.Ref(), tmpIterInit, tmpIterCond, tmpIterStep, Xcode.LOG_LT_EXPR, Bcons.blockList(dummyInnerMostBlock));
        } else {
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
      //cacheLoadBody.setDecls(ACCutil.getDecls(cacheLoadBodyIds));
      cacheLoadBody.setIdentList(cacheLoadBodyIds);
      return Bcons.COMPOUND(cacheLoadBody);
    }

    void rewrite(Block b) {
      BasicBlockExprIterator iter = new BasicBlockExprIterator(b);
      for (iter.init(); !iter.end(); iter.next()) {
        Xobject expr = iter.getExpr();
        topdownXobjectIterator exprIter = new topdownXobjectIterator(expr);
        for (exprIter.init(); !exprIter.end(); exprIter.next()) {
          Xobject x = exprIter.getXobject();
          switch (x.Opcode()) {
          case VAR: {
            String varName = x.getName();
            if (varName.equals(varId.getName())) {
              exprIter.setXobject(cacheId.Ref());
            }
          }
          break;
          case ARRAY_REF: {
            String arrayName = x.getArg(0).getName();
            if (!arrayName.equals(varId.getName())) break;
            XobjList arrayIdxList = (XobjList) x.getArg(1);
            Xobject arrayIdx = null;

            int dim = 0;
            for (Xobject idx : arrayIdxList) {
              //newArrayIdxList.add(Xcons.binaryOp(Xcode.PLUS_EXPR, idx, offsetId.Ref()));
              Xobject newArrayIdx = Xcons.binaryOp(Xcode.PLUS_EXPR, idx, Xcons.arrayRef(Xtype.intType, cacheOffsetArrayId.getAddr(), Xcons.List(Xcons.IntConstant(dim))));
              if (arrayIdx == null) {
                arrayIdx = newArrayIdx;
              } else {
                arrayIdx = Xcons.binaryOp(Xcode.PLUS_EXPR, Xcons.binaryOp(Xcode.MUL_EXPR, arrayIdx, Xcons.arrayRef(Xtype.intType, cacheSizeArrayId.getAddr(), Xcons.List(Xcons.IntConstant(dim)))), newArrayIdx);
              }
              dim++;
            }
            Xobject newObj = Xcons.arrayRef(x.Type(), cacheId.Ref(), Xcons.List(arrayIdx));
            exprIter.setXobject(newObj);
          }
          break;
          }
        }
      }
    }

     XobjList getSimpleSubarray(Xobject s) {
      Xobject loopIdx;
      Xobject constOffset;
      Xobject length;

      Xobject lower;
      if (s.Opcode() != Xcode.LIST) {
        lower = s;
        length = Xcons.IntConstant(1);
      } else {
        lower = s.getArg(0);
        length = s.getArgOrNull(1);
      }

      switch (lower.Opcode()) {
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
        break;
      default:
        loopIdx = lower;
        constOffset = Xcons.IntConstant(0);
      }

      return Xcons.List(loopIdx, constOffset, length);
    }
  } // end of Cache

  //
  // reduction Manager
  //
  class ReductionManager {
    Ident counterPtr = null;
    Ident tempPtr = null;
    final List<Reduction> reductionList = new ArrayList<Reduction>();
    Xobject totalElementSize = Xcons.IntConstant(0);
    final Map<Reduction, Xobject> offsetMap = new HashMap<Reduction, Xobject>();
    Ident isLastVar = null;

    ReductionManager() {
      counterPtr = Ident.Param(ACC_REDUCTION_CNT_VAR, Xtype.Pointer(Xtype.unsignedType));//Ident.Var("_ACC_GPU_RED_CNT", Xtype.unsignedType, Xtype.Pointer(Xtype.unsignedType), VarScope.GLOBAL);
      tempPtr = Ident.Param(ACC_REDUCTION_TMP_VAR, Xtype.voidPtrType);//Ident.Var("_ACC_GPU_RED_TMP", Xtype.voidPtrType, Xtype.Pointer(Xtype.voidPtrType), VarScope.GLOBAL);
      isLastVar = Ident.Local("_ACC_GPU_IS_LAST_BLOCK", Xtype.intType);
      isLastVar.setProp(ACCgpuDecompiler.GPU_STORAGE_SHARED, true);
    }

    // make functions definition for reduction exeuted after reduction
    public XobjectDef makeReductionKernelDef(String deviceKernelName) {
      BlockList reductionKernelBody = Bcons.emptyBody();

      XobjList deviceKernelParamIds = Xcons.IDList();
      Xobject blockIdx = Xcons.Symbol(Xcode.VAR, Xtype.intType, "_ACC_block_x_id");
      Ident numBlocksId = Ident.Param("_ACC_GPU_RED_NUM", Xtype.intType);
      int count = 0;
      Iterator<Reduction> blockRedIter = reductionManager.BlockReductionIterator();
      while (blockRedIter.hasNext()) {
        Reduction reduction = blockRedIter.next();
        if (! reduction.needsExternalReduction()) continue;

        Block blockReduction;
        blockReduction = reduction.makeBlockReductionFuncCall(tempPtr, offsetMap.get(reduction), numBlocksId);
        //reduction.makeBlockReductionFuncCall(tempPtr, tmpOffsetElementSize)

        Block ifBlock = Bcons.IF(Xcons.binaryOp(Xcode.LOG_EQ_EXPR, blockIdx, Xcons.IntConstant(count)), blockReduction, null);
        reductionKernelBody.add(ifBlock);
        count++;
      }

      for (Xobject x : _outerIdList) {
        Ident id = (Ident) x;
        Reduction reduction = reductionManager.findReduction(id);
        if (reduction != null && reduction.needsExternalReduction()) {
          deviceKernelParamIds.add(makeParamId_new(id)); //getVarId();
        }
      }

      deviceKernelParamIds.add(tempPtr);
      deviceKernelParamIds.add(numBlocksId);

      Ident deviceKernelId = _decl.getEnvDevice().declGlobalIdent(deviceKernelName, Xtype.Function(Xtype.voidType));
      ((FunctionType) deviceKernelId.Type()).setFuncParamIdList(deviceKernelParamIds);
      return XobjectDef.Func(deviceKernelId, deviceKernelParamIds, null, Bcons.COMPOUND(reductionKernelBody).toXobject());
    }

    public XobjList getBlockReductionParamIds() {
      return Xcons.List(Xcode.ID_LIST, tempPtr, counterPtr);
    }

    public Block makeLocalVarInitFuncs() {
      BlockList body = Bcons.emptyBody();
      Iterator<Reduction> blockRedIter = reductionManager.BlockReductionIterator();
      while (blockRedIter.hasNext()) {
        Reduction reduction = blockRedIter.next();
	if(reduction.onlyKernelLast()){
          body.add(reduction.makeInitReductionVarFuncCall());
	}
      }

      if (body.isSingle()) {
        return body.getHead();
      } else {
        return Bcons.COMPOUND(body);
      }
    }

    public XobjList getBlockReductionLocalIds() {
      XobjList blockLocalIds = Xcons.IDList();
      Iterator<Reduction> blockRedIter = reductionManager.BlockReductionIterator();
      while (blockRedIter.hasNext()) {
        Reduction reduction = blockRedIter.next();
	if(reduction.onlyKernelLast()){
          blockLocalIds.add(reduction.getLocalReductionVarId());
          blockLocalIds.add(reduction.getLocReductionVarId());
	}
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

      BlockList body = Bcons.emptyBody();
      BlockList thenBody = Bcons.emptyBody();
      BlockList tempWriteBody = Bcons.emptyBody();

      // add funcs
      Iterator<Reduction> blockRedIter = reductionManager.BlockReductionIterator();
      while (blockRedIter.hasNext()) {
        Reduction reduction = blockRedIter.next();
	if (reduction.needsExternalReduction()) {
          Ident tmpVar = Ident.Local("_ACC_gpu_reduction_tmp_" + reduction.var.getName(), reduction.varId.Type());
          if (reduction.useThread()) {
            body.addIdent(tmpVar);
            body.add(reduction.makeInitReductionVarFuncCall(tmpVar));
            body.add(reduction.makeInKernelReductionFuncCall(tmpVar));
            tempWriteBody.add(reduction.makeTempWriteFuncCall(tmpVar, tempPtr, offsetMap.get(reduction)));
            thenBody.add(reduction.makeSingleBlockReductionFuncCall(tmpVar));
          } else {
            tempWriteBody.add(reduction.makeTempWriteFuncCall(tempPtr, offsetMap.get(reduction)));
            thenBody.add(reduction.makeSingleBlockReductionFuncCall());
          }
        } else {
          if(reduction.onlyKernelLast()) {
            body.add(reduction.makeInKernelReductionFuncCall(null));
          }
	}
      }

      if (!thenBody.isEmpty()) {
        Xobject grid_dim = Xcons.Symbol(Xcode.VAR, Xtype.unsignedType, "_ACC_grid_x_dim");
        body.add(Bcons.IF(Xcons.binaryOp(Xcode.LOG_EQ_EXPR, grid_dim, Xcons.IntConstant(1)), Bcons.COMPOUND(thenBody), Bcons.COMPOUND(tempWriteBody)));
      }

      return Bcons.COMPOUND(body);
    }

    Reduction addReduction(ACCvar var, EnumSet<ACCpragma> execMethodSet) {
      Reduction reduction = new Reduction(var, execMethodSet);
      reductionList.add(reduction);

      if (!reduction.needsExternalReduction()) return reduction;

      //tmp setting
      offsetMap.put(reduction, totalElementSize);

      Xtype varType = var.getId().Type();
      Xobject elementSize;
      if (varType.isPointer()) {
        elementSize = Xcons.SizeOf(varType.getRef());
      } else {
        elementSize = Xcons.SizeOf(varType);
      }
      totalElementSize = Xcons.binaryOp(Xcode.PLUS_EXPR, totalElementSize, elementSize);
      return reduction;
    }

    Reduction findReduction(Ident id) {
      for (Reduction red : reductionList) {
        if (red.varId == id) {
          return red;
        }
      }
      return null;
    }

    Iterator<Reduction> BlockReductionIterator() {
      return new BlockReductionIterator(reductionList);
    }

    boolean hasUsingTmpReduction() {
      return !offsetMap.isEmpty();
    }

    class BlockReductionIterator implements Iterator<Reduction> {
      final Iterator<Reduction> reductionIterator;
      Reduction re;

      public BlockReductionIterator(List<Reduction> reductionList) {
        this.reductionIterator = reductionList.iterator();
      }

      @Override
      public boolean hasNext() {
        while (true) {
          if (reductionIterator.hasNext()) {
            re = reductionIterator.next();
            if (re.useBlock()) {
              return true;
            }
          } else {
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
  } // end of Reduction Manager

  //
  // Reduction
  // 
  class Reduction {
    final EnumSet<ACCpragma> execMethodSet;  //final ACCpragma execMethod;
    final Ident localVarId;
    final Ident locVarId;  // hold location (address) for reduction
    final Ident varId;
    // --Commented out by Inspection (2015/02/24 21:12):Ident launchFuncLocalId;
    final ACCvar var;

    //Ident tmpId;
    Reduction(ACCvar var, EnumSet<ACCpragma> execMethodSet) {
      this.var = var;
      this.varId = var.getId();
      this.execMethodSet = EnumSet.copyOf(execMethodSet); //execMethod;

      //generate local var id
      String reductionVarPrefix = ACC_REDUCTION_VAR_PREFIX;

      if (execMethodSet.contains(ACCpragma.GANG)) reductionVarPrefix += "b";
      if (execMethodSet.contains(ACCpragma.VECTOR)) reductionVarPrefix += "t";

      reductionVarPrefix += "_";

      localVarId = Ident.Local(reductionVarPrefix + varId.getName(), varId.Type());
      if (execMethodSet.contains(ACCpragma.GANG) && !execMethodSet.contains(ACCpragma.VECTOR)) {
        //execMethod == ACCpragma._BLOCK) 
        localVarId.setProp(ACCgpuDecompiler.GPU_STORAGE_SHARED, true);
      }
      
      locVarId = null;
    }

    public Block makeSingleBlockReductionFuncCall(Ident tmpPtrId) {
      XobjList args = Xcons.List(varId.getAddr(), tmpPtrId.Ref(), Xcons.IntConstant(getReductionKindInt()));
      return ACCutil.createFuncCallBlock("_ACC_gpu_reduction_singleblock", args);
    }

    public Block makeSingleBlockReductionFuncCall() {
      return makeSingleBlockReductionFuncCall(localVarId);
    }

    public void rewrite(Block b) {
      BasicBlockExprIterator iter = new BasicBlockExprIterator(b);
      for (iter.init(); !iter.end(); iter.next()) {
        Xobject expr = iter.getExpr();
        topdownXobjectIterator exprIter = new topdownXobjectIterator(expr);
        for (exprIter.init(); !exprIter.end(); exprIter.next()) {
          Xobject x = exprIter.getXobject();
          switch (x.Opcode()) {
          case VAR: {
            String varName = x.getName();
            if (varName.equals(varId.getName())) {
              exprIter.setXobject(localVarId.Ref());
            }
          }
          break;
          case VAR_ADDR: {
            String varName = x.getName();
            if (varName.equals(varId.getName())) {
              exprIter.setXobject(localVarId.getAddr());
            }
          }
          break;
          }
        }
      }
    }

    public boolean useThread() {
      return execMethodSet.contains(ACCpragma.VECTOR); //execMethod != ACCpragma._BLOCK;
    }

    public Ident getLocalReductionVarId() {
      return localVarId;
    }

    public Ident getLocReductionVarId() {
      return locVarId;
    }

    // initalize reduction variable 
    public Block makeInitReductionVarFuncCall(Ident id) {
      String funcName = "_ACC_gpu_init_reduction_var";

      if (!execMethodSet.contains(ACCpragma.VECTOR)) { //execMethod == ACCpragma._BLOCK) {
        funcName += "_single";
      }

      return ACCutil.createFuncCallBlock(funcName, Xcons.List(id.getAddr(), Xcons.IntConstant(getReductionKindInt())));
    }

    public Block makeInitReductionVarFuncCall() {
      return makeInitReductionVarFuncCall(localVarId);
    }

    // reduction on block
    public Block makeBlockReductionFuncCall(Ident tmpPtrId, Xobject tmpOffsetElementSize, Ident numBlocks) {
      XobjList args = Xcons.List(varId.Ref(), Xcons.IntConstant(getReductionKindInt()),
                                 tmpPtrId.Ref(), tmpOffsetElementSize);
      if (numBlocks != null) {
        args.add(numBlocks.Ref());
      }
      return ACCutil.createFuncCallBlock("_ACC_gpu_reduction_block", args);
    }

    // reduction on shared location
    public Block makeReductionLocUpdateFuncCall(){
      XobjList args = Xcons.List(locVarId.Ref(), localVarId.Ref());
      String mang_name = "_ACC_gpu_reduction_loc_update";
      mang_name += "_"+varId.Type()+"_"+getReductionKindString();
      return ACCutil.createFuncCallBlock(mang_name, args);
    }

    // reduction value on tempoary area 
    public Block makeReductionLocBlockFuncCall(Ident tmpPtrId, Xobject tmpOffsetElementSize,Ident numBlocks) {
      XobjList args = Xcons.List(varId.Ref(), tmpPtrId.Ref(), tmpOffsetElementSize);

      if (numBlocks != null) {
        args.add(numBlocks.Ref());
      } else args.add(Xcons.IntConstant(0));
      
      String mang_name = "_ACC_gpu_reduction_loc_block";
      mang_name += "_"+varId.Type()+"_"+getReductionKindString();
      return ACCutil.createFuncCallBlock(mang_name, args);
    }

    String makeExecString(EnumSet<ACCpragma> execSet){
      StringBuilder sb = new StringBuilder();
      if(execSet.contains(ACCpragma.GANG)){
        sb.append('b');  // for gang   -> blockIdx.x
        if(execSet.contains(ACCpragma.VECTOR)) {
          sb.append('t');  // for acc loop gang vector
        }
      } else if(execSet.contains(ACCpragma.WORKER)) {
        sb.append("ty"); // for worker -> threadIdx.y
      } else if(execSet.contains(ACCpragma.VECTOR)) {
        sb.append('t');  // for vector -> threadIdx.x
      } else
        ACC.fatal("failed at parallelaization clause (available: gang, worker, vector)");
      return sb.toString();
    }

    public Block makeInKernelReductionFuncCall_CUDA(Ident dstId){
      Xobject dstArg = null;

      EnumSet<ACCpragma> execSet = EnumSet.copyOf(execMethodSet);
      dstArg = dstId != null? dstId.getAddr() : varId.getAddr();
      if(needsExternalReduction()){
        execSet.remove(ACCpragma.GANG);
        if(dstId == null){
          ACC.fatal("dstId must be specified");
        }
        //dstArg = localVarId.getAddr();
      }else{
        //dstArg = varId.Type().isPointer()? varId.Ref() : varId.getAddr();
      }

      String funcName = "_ACC_gpu_reduction_" + makeExecString(execSet);
      XobjList args = Xcons.List(dstArg, Xcons.IntConstant(getReductionKindInt()), localVarId.Ref());

      return ACCutil.createFuncCallBlock(funcName, args);
    }

    public Block makeInKernelReductionFuncCall(Ident dstId){
        return makeInKernelReductionFuncCall_CUDA(dstId);
    }

    public Block makeThreadReductionFuncCall(Ident varId) {
      XobjList args = Xcons.List(varId.getAddr(), localVarId.Ref(), Xcons.IntConstant(getReductionKindInt()));
      return ACCutil.createFuncCallBlock("_ACC_gpu_reduction_thread", args);
    }

    public Block makeTempWriteFuncCall(Ident tmpPtrId, Xobject tmpOffsetElementSize) {
      return makeTempWriteFuncCall(localVarId, tmpPtrId, tmpOffsetElementSize);
    }

    public Block makeTempWriteFuncCall(Ident id, Ident tmpPtrId, Xobject tmpOffsetElementSize) {
      return ACCutil.createFuncCallBlock("_ACC_gpu_reduction_tmp", Xcons.List(id.Ref(), tmpPtrId.Ref(), tmpOffsetElementSize));
    }

     int getReductionKindInt() {
      ACCpragma pragma = var.getReductionOperator();
      if (!pragma.isReduction()) ACC.fatal(pragma.getName() + " is not reduction clause");
      switch (pragma) {
      case REDUCTION_PLUS:
        return 0;
      case REDUCTION_MUL:
        return 1;
      case REDUCTION_MAX:
        return 2;
      case REDUCTION_MIN:
        return 3;
      case REDUCTION_BITAND:
        return 4;
      case REDUCTION_BITOR:
        return 5;
      case REDUCTION_BITXOR:
        return 6;
      case REDUCTION_LOGAND:
        return 7;
      case REDUCTION_LOGOR:
        return 8;
      default:
        ACC.fatal("getReductionKindInt: unknown reduction kind");
        return -1;
      }
    }

     String getReductionKindString() {
      ACCpragma pragma = var.getReductionOperator();
      if (!pragma.isReduction()) ACC.fatal(pragma.getName() + " is not reduction clause");
      switch (pragma) {
      case REDUCTION_PLUS:
        return "PLUS";
      case REDUCTION_MUL:
        return "MUL";
      case REDUCTION_MAX:
        return "MAX";
      case REDUCTION_MIN:
        return "MIN";
      case REDUCTION_BITAND:
        return "BITAND";
      case REDUCTION_BITOR:
        return "BITOR";
      case REDUCTION_BITXOR:
        return "BITXOR";
      case REDUCTION_LOGAND:
        return "LOGAND";
      case REDUCTION_LOGOR:
        return "LOGOR";
      default:
        ACC.fatal("getReductionKindString: unknown reduction kind");
        return "???";
      }
    }

    public boolean useBlock() {
      return execMethodSet.contains(ACCpragma.GANG);
    }

    public boolean existsAtomicOperation(){
      ACCpragma op = var.getReductionOperator();
      switch (var.getId().Type().getBasicType()) {
      case BasicType.FLOAT:
      case BasicType.INT:
        return op != ACCpragma.REDUCTION_MUL;
      }
      return false;
    }

    public boolean onlyKernelLast() {
	//      if(ACC.device == AccDevice.PEZYSC) return false;

      return execMethodSet.contains(ACCpragma.GANG);
    }

    public boolean needsExternalReduction(){
	//      if(ACC.device == AccDevice.PEZYSC) return false;

      return !existsAtomicOperation() && execMethodSet.contains(ACCpragma.GANG);
    }
  } // end of Reduction

  //
  // Block List Builder
  //
   class BlockListBuilder {
    private final List<Block> initBlockList = new ArrayList<Block>();
    private final List<Block> finalizeBlockList = new ArrayList<Block>();
    private final List<Block> mainBlockList = new ArrayList<Block>();
    private final BlockList blockList = Bcons.emptyBody();

    public void addInitBlock(Block b) {
      initBlockList.add(b);
    }

    public void addFinalizeBlock(Block b) {
      finalizeBlockList.add(b);
    }

    public void add(Block b) {
      mainBlockList.add(b);
    }

    public Ident declLocalIdent(String name, Xtype type) {
      return blockList.declLocalIdent(name, type);
    }

    public Ident declLocalIdent(String name, Xtype type, Xobject init) { return blockList.declLocalIdent(name, type, StorageClass.AUTO, init); }
    public void addIdent(Ident id) {
      blockList.addIdent(id);
    }

    public BlockList build() {
      BlockList body = blockList.copy();
      for (Block b : initBlockList) body.add(b);
      for (Block b : mainBlockList) body.add(b);
      for (Block b : finalizeBlockList) body.add(b);
      return body;
    }
  }

  //
  // Stack Memory
  //
  class StackMemory{
    private final Ident baseId;
    private final Ident posId;

    private boolean isUsed = false;

    StackMemory(String name, Ident baseId) {
      /*
      XobjList idList = Xcons.IDList();
      Xtype stackStructType = new StructType(null,
              false,
              null,
              idList,
              0,
              null);
              */
      this.baseId = baseId;
      this.posId = Ident.Local(name + "_pos", Xtype.unsignedlongType);
    }

    Block makePushFunc(Xobject target, Xobject size){
      //make "_ACC_stack_push(base, &pos, size);"
      isUsed = true;
      Ident funcId = ACCutil.getMacroFuncId("_ACC_stack_push", Xtype.voidPtrType);
      XobjList args = Xcons.List(baseId.Ref(), posId.getAddr(), size);
      return Bcons.Statement(Xcons.Set(target, Xcons.Cast(target.Type(), funcId.Call(args))));
    }

    Block makeInitFunc() {
      //make "_ACC_stack_init(&pos);"
      return ACCutil.createFuncCallBlock("_ACC_stack_init", Xcons.List(posId.getAddr()));
    }

    boolean isUsed()
    {
      return isUsed;
    }

    Ident getBaseId(){
      return baseId;
    }

    Ident getPosId(){
      return posId;
    }
  }
}
