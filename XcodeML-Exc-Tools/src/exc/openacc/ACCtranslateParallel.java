package exc.openacc;

import java.util.*;
import exc.block.*;
import exc.object.*;


public class ACCtranslateParallel {
  private static final String ACC_INIT_VAR_PREFIX = "_ACC_loop_init_";
  private static final String ACC_COND_VAR_PREFIX = "_ACC_loop_cond_";
  private static final String ACC_STEP_VAR_PREFIX = "_ACC_loop_step_";
  private static final String ACC_GPU_FUNC_PREFIX ="_ACC_GPU_FUNC"; 
  
  private PragmaBlock pb;
  private ACCinfo parallelInfo;
  private ACCglobalDecl globalDecl;
  private ACCgpuManager gpuManager;
 
  public ACCtranslateParallel(PragmaBlock pb) {
    this.pb = pb;
    this.parallelInfo = ACCutil.getACCinfo(pb);
    if(this.parallelInfo == null){
      ACC.fatal("can't get info");
    }
    this.globalDecl = this.parallelInfo.getGlobalDecl();
  }
  
  public void translate() throws ACCexception{
    if(ACC.debugFlag){
      System.out.println("translate data");
    }

    //get outer ids
    XobjList outerIds = getOuterIds();
    
    //translate data
    ACCtranslateData dataTranslator = new ACCtranslateData(pb);
    dataTranslator.translate();
    
    //check private and firstprivate variable
    XobjList firstprivateIds = Xcons.IDList();
    XobjList privateIds = Xcons.IDList();
    XobjList normalIds = Xcons.IDList();
    divideIds(outerIds, normalIds, privateIds, firstprivateIds);
    
    //translate
    BlockList parallelBody = pb.getBody();
    Ident funcId = globalDecl.declExternIdent(globalDecl.genSym(ACC_GPU_FUNC_PREFIX), Xtype.Function(Xtype.voidType));  
    
    XobjList funcParamArgs = makeFuncParamArgs(normalIds, firstprivateIds);
    XobjList funcParams = (XobjList)funcParamArgs.left();
    XobjList funcArgs = (XobjList)funcParamArgs.right();
    
    createGpuFuncs(funcId, funcParams, parallelBody, parallelInfo);
    ((FunctionType)funcId.Type()).setFuncParamIdList(funcParams);  //if need
    Block funcCallBlock = ACCutil.createFuncCallBlock(funcId.getName(), funcArgs);
    parallelInfo.setReplaceBlock(funcCallBlock) ;
  }
  
  //returns XobjList( params, args )
  private XobjList makeFuncParamArgs(XobjList normalIds, XobjList firstprivateIds) {
    XobjList params = Xcons.IDList();
    XobjList args = Xcons.List();
    
    for(Xobject x : normalIds){
      Ident id = (Ident)x;
      String varName = id.getName();
      switch(id.Type().getKind()){
      case Xtype.ARRAY:
      {
        Ident newId = (Ident)id.copy();
        id.setScope(VarScope.PARAM);
        params.add(newId);
        args.add(parallelInfo.getDevicePtr(varName).Ref());
      } break;
      case Xtype.BASIC:
      case Xtype.STRUCT:
      case Xtype.UNION:
      case Xtype.ENUM:
      {
        Ident newId = Ident.Param(varName, Xtype.Pointer(id.Type()));
        params.add(newId);
        args.add(parallelInfo.getDevicePtr(varName).Ref());
      } break;
      }
    }
    
    for(Xobject x : firstprivateIds){
      Ident id = (Ident)x;
      if(id.isArray()){
        ACC.fatal("firstprivate array is not supported");
      }else{
        Ident newId = (Ident)id.copy();
        newId.setScope(VarScope.PARAM);
        params.add(newId);
        args.add(id.Ref());
      }
    }
    
    return Xcons.List(params, args);
  }

  private void divideIds(XobjList outerIds, XobjList normalIds, XobjList privateIds, XobjList firstprivateIds) {
    for(Xobject x : outerIds){
      Ident id = (Ident)x;
      String varName = id.getName();
      if(parallelInfo.isVarPrivate(varName)){
        privateIds.add(id);
      }else if(parallelInfo.isVarFirstprivate(varName)){
        firstprivateIds.add(id);
      }else{
        normalIds.add(id);
      }
    }
  }

  /** returns idents which is referenced in pragma block and defined out pragma block.*/ 
  XobjList getOuterIds(){
    XobjList outerIds = Xcons.IDList();

    Set<String> checkedVars = new HashSet<String>();

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

          Ident id = pb.findVarIdent(varName);
          if(id != null) { //id exists out pb
            outerIds.add(id);
          }
          checkedVars.add(varName);
        } break;
        case ARRAY_REF:
        {
          String varName = x.getArg(0).getName();
          if (checkedVars.contains(varName)) break;

          Ident id = pb.findVarIdent(varName);
          if(id != null) { //id exists out pb
            outerIds.add(id);
          }
          checkedVars.add(varName);
        } break;
        }
      }
    }
    
    return outerIds; 
  }

  

  public void createGpuFuncs(Ident funcId, XobjList funcParams, BlockList parallelBody, ACCinfo info) throws ACCexception {
    String hostFuncName = funcId.getName();
    Ident hostFuncId = funcId;
    gpuManager = new ACCgpuManager();

    //analyze
    analyzeParallelBody(parallelBody);
    gpuManager.finalize();
    

    //create device kernel
    XobjList deviceKernelParamArgs = makeDeviceKernelParamArgs(funcParams);
    XobjList deviceKernelParams = (XobjList)deviceKernelParamArgs.left();
    XobjList deviceKernelArgs = (XobjList)deviceKernelParamArgs.right();
    String deviceKernelName = hostFuncName + "_DEVICE";
    Ident deviceKernelId = ACCutil.getMacroFuncId(deviceKernelName, Xtype.voidType);
    XobjectDef deviceKernelDef = createDeviceKernelDef(deviceKernelId, deviceKernelParams, parallelBody);

    ((FunctionType)deviceKernelId.Type()).setFuncParamIdList(deviceKernelParams);
    Bcons.Statement(deviceKernelId.Call(deviceKernelArgs));
    
    // create host function
    XobjectDef hostFuncDef = createHostFuncDef(hostFuncId, funcParams, deviceKernelDef, deviceKernelId);

    new ACCgpuDecompiler().decompile(info.getGlobalDecl().getEnv(), deviceKernelDef, deviceKernelId, hostFuncDef);
    
  }

  private XobjList makeDeviceKernelParamArgs(XobjList funcParams) {
    //temporary
    return Xcons.List(funcParams, ACCutil.getRefs(funcParams));
  }
  
  private XobjectDef createDeviceKernelDef(Ident deviceKernelId, XobjList deviceKernelParamIds, BlockList kernelBody) throws ACCexception {
    //create device function
    XobjList deviceKernelLocalIds = Xcons.IDList();
    BlockList deviceKernelBody = Bcons.emptyBody();
    
    //set private var id as local id
    Iterator<ACCvar> varIter = parallelInfo.getVars();
    while(varIter.hasNext()){
      ACCvar var = varIter.next();
      if(var.isPrivate()){
        deviceKernelLocalIds.add(Ident.Local(var.getName(), Xtype.intType));
      }
    }
    
    List<Block> initBlocks = new ArrayList<Block>();
    
    XobjList additionalParams = Xcons.IDList();
    XobjList additionalLocals = Xcons.IDList();
    Block deviceKernelBlock = makeDeviceKernelCoreBlock(initBlocks, kernelBody, additionalParams, additionalLocals);
    deviceKernelLocalIds.mergeList(additionalLocals);

    for(Block b : initBlocks){
      deviceKernelBody.add(b);
    }
    deviceKernelBody.add(deviceKernelBlock);
  
    deviceKernelBody.setIdentList(deviceKernelLocalIds);
    deviceKernelBody.setDecls(ACCutil.getDecls(deviceKernelLocalIds));
    
    //((FunctionType)deviceKernelId.Type()).setFuncParamIdList(deviceKernelParamIds);
    XobjectDef deviceKernelDef = XobjectDef.Func(deviceKernelId, deviceKernelParamIds, null, deviceKernelBody.toXobject());

    return deviceKernelDef;
  }
  
  private Block makeDeviceKernelCoreBlock(List<Block> initBlocks, BlockList kernelBody, XobjList paramIds, XobjList localIds) throws ACCexception {
    Block b = kernelBody.getHead();
    
    Xobject ids = kernelBody.getIdentList();
    Xobject decls = kernelBody.getDecls();
    BlockList resultBody = Bcons.emptyBody(ids, decls);
    
    while(b != null){
      if(b.Opcode() == Xcode.ACC_PRAGMA){
        ACCinfo info = ACCutil.getACCinfo(b);
        if(info.getPragma() == ACCpragma.LOOP){
          CforBlock forBlock = (CforBlock)b.getBody().getHead();
          //Iterator<ACCpragma> execModels = info.getExecModels();
          String execMethodName = gpuManager.getMethodName(forBlock);
          
          Xobject init = forBlock.getLowerBound();
          Xobject cond = forBlock.getUpperBound();
          Xobject step = forBlock.getStep();
          Ident indVarId = Ident.Local(forBlock.getInductionVar().getString(), Xtype.intType);
          
          Ident iterIdx = Ident.Local("_ACC_" + execMethodName + "_idx", Xtype.intType);
          Ident iterInit = Ident.Local("_ACC_" + execMethodName + "_init", Xtype.intType);
          Ident iterCond = Ident.Local("_ACC_" + execMethodName + "_cond", Xtype.intType);
          Ident iterStep = Ident.Local("_ACC_" + execMethodName + "_step", Xtype.intType);
          localIds.mergeList(Xcons.List(iterIdx, iterInit, iterCond, iterStep));

          XobjList initIterFuncArgs = Xcons.List(iterInit.getAddr(), iterCond.getAddr(), iterStep.getAddr());
          initIterFuncArgs.mergeList(Xcons.List(init, cond, step));
          Block initIterFunc = ACCutil.createFuncCallBlock("_ACC_gpu_init_" + execMethodName + "_iter", initIterFuncArgs); 
          initBlocks.add(initIterFunc);

          XobjList calcIdxFuncArgs = Xcons.List(iterIdx.Ref());
          calcIdxFuncArgs.add(indVarId.getAddr());
          calcIdxFuncArgs.mergeList(Xcons.List(init, cond, step));
          Block calcIdxFunc = ACCutil.createFuncCallBlock("_ACC_gpu_calc_idx", calcIdxFuncArgs); 

          Block resultBlock = Bcons.FOR(
              Xcons.Set(iterIdx.Ref(), iterInit.Ref()),
              Xcons.binaryOp(Xcode.LOG_LT_EXPR, iterIdx.Ref(), iterCond.Ref()), 
              Xcons.asgOp(Xcode.ASG_PLUS_EXPR, iterIdx.Ref(), iterStep.Ref()), 
              Bcons.COMPOUND(
                  Bcons.blockList(
                      calcIdxFunc,
                      makeDeviceKernelCoreBlock(initBlocks, forBlock.getBody(), paramIds, localIds)
                      )
                  )
              );
          resultBody.add(resultBlock);
        }
      }else if(b.Opcode() == Xcode.FOR_STATEMENT){
      }else{
        resultBody.add(b);
      }
      b = b.getNext();
    }
    
    return Bcons.COMPOUND(resultBody);
  }

  private void analyzeParallelBody(BlockList body) throws ACCexception {
    Block b = body.getHead();
    
    while(b != null){
      if(b.Opcode() == Xcode.ACC_PRAGMA){
        ACCinfo info = ACCutil.getACCinfo(b);
        if(info.getPragma() == ACCpragma.LOOP){
          CforBlock forBlock = (CforBlock)b.getBody().getHead();
          ACCutil.setACCinfo(forBlock, info);
          Iterator<ACCpragma> execModelIter = info.getExecModels();
          gpuManager.setLoop(execModelIter, forBlock);
          
          analyzeParallelBody(forBlock.getBody());
        }
      }else if(b.Opcode() == Xcode.FOR_STATEMENT){
      }else{
      }
      b = b.getNext();
    }
    
    return;
  }
  
  private XobjectDef createHostFuncDef(Ident hostFuncId, XobjList hostFuncParams, XobjectDef deviceKernelDef, Ident deviceKernelId) {
    Ident blockXid = Ident.Local("_ACC_GPU_DIM3_block_x", Xtype.intType);
    Ident blockYid = Ident.Local("_ACC_GPU_DIM3_block_y", Xtype.intType);
    Ident blockZid = Ident.Local("_ACC_GPU_DIM3_block_z", Xtype.intType);
    Ident threadXid = Ident.Local("_ACC_GPU_DIM3_thread_x", Xtype.intType);
    Ident threadYid = Ident.Local("_ACC_GPU_DIM3_thread_y", Xtype.intType);
    Ident threadZid = Ident.Local("_ACC_GPU_DIM3_thread_z", Xtype.intType);
    
    XobjList hostFuncLocalIds = Xcons.List(Xcode.ID_LIST, blockXid, blockYid, blockZid, threadXid, threadYid, threadZid);
    XobjList hostFuncParamIds = hostFuncParams;
    XobjList deviceKernelCallArgs = ACCutil.getRefs(hostFuncParamIds);
    
    BlockList hostFuncBody = Bcons.emptyBody();
    
    XobjList blockThreadSize = gpuManager.getBlockThreadSize();
    XobjList blockSize = (XobjList)blockThreadSize.left();
    XobjList threadSize = (XobjList)blockThreadSize.right();
    
    hostFuncBody.add(Bcons.Statement(Xcons.Set(blockXid.Ref(), blockSize.getArg(0))));
    hostFuncBody.add(Bcons.Statement(Xcons.Set(blockYid.Ref(), blockSize.getArg(1))));
    hostFuncBody.add(Bcons.Statement(Xcons.Set(blockZid.Ref(), blockSize.getArg(2))));
    hostFuncBody.add(Bcons.Statement(Xcons.Set(threadXid.Ref(), threadSize.getArg(0))));
    hostFuncBody.add(Bcons.Statement(Xcons.Set(threadYid.Ref(), threadSize.getArg(1))));
    hostFuncBody.add(Bcons.Statement(Xcons.Set(threadZid.Ref(), threadSize.getArg(2))));
    
    hostFuncBody.setIdentList(hostFuncLocalIds);
    hostFuncBody.setDecls(ACCutil.getDecls(hostFuncLocalIds));
    
    Xobject deviceKernelCall = deviceKernelId.Call(deviceKernelCallArgs);
    deviceKernelCall.setProp(ACCgpuDecompiler.GPU_FUNC_CONF, (Object)Xcons.List(blockXid, blockYid, blockZid,threadXid, threadYid, threadZid));
    hostFuncBody.add(Bcons.Statement(deviceKernelCall));

    XobjectDef hostFuncDef = XobjectDef.Func(hostFuncId, hostFuncParamIds, null, hostFuncBody.toXobject());
    
    return hostFuncDef;
  }
}

  