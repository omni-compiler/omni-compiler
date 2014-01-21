package exc.openacc;

import java.util.*;

import exc.block.*;
import exc.object.*;


public class ACCtranslateParallel {
  private static final String ACC_INIT_VAR_PREFIX = "_ACC_loop_init_";
  private static final String ACC_COND_VAR_PREFIX = "_ACC_loop_cond_";
  private static final String ACC_STEP_VAR_PREFIX = "_ACC_loop_step_";
  private static final String ACC_GPU_FUNC_PREFIX ="_ACC_GPU_FUNC"; 
  private static final String ACC_REDUCTION_VAR_PREFIX = "_ACC_reduction_";
  
  private PragmaBlock pb;
  private ACCinfo parallelInfo;
  private ACCglobalDecl globalDecl;
  private ACCgpuManager gpuManager;
  //private List<ACCvar> reductionVars;
  private List<ACCvar> tmpUsingReductionVars = new ArrayList<ACCvar>();
  private List<XobjectDef> varDecls = new ArrayList<XobjectDef>();
 
  public ACCtranslateParallel(PragmaBlock pb) {
    this.pb = pb;
    this.parallelInfo = ACCutil.getACCinfo(pb);
    if(this.parallelInfo == null){
      ACC.fatal("can't get info");
    }
    this.globalDecl = this.parallelInfo.getGlobalDecl();
  }
  
  public void translate() throws ACCexception{
    ACC.debug("translate parallel");
    
    if(parallelInfo.isDisabled()){
      return;
    }
    
    //List<List<Block>> kernelBodyList = null;//divideBlocksBetweenKernels(pb);
    //BlockList kernelBlocks = pb.getBody();

    List<Block> kernelBody = new ArrayList<Block>(); 
    if(parallelInfo.getPragma() == ACCpragma.PARALLEL_LOOP){
      kernelBody.add(pb);
    }else{
//      for(Block b = pb.getBody().getHead(); b != null; b = b.getNext()){
//        kernelBody.add(b);
//      }
        kernelBody.add(pb);
    }
    
    //analyze and complete clause for kernel
    
    ACCgpuKernel gpuKernel = new ACCgpuKernel(parallelInfo, kernelBody);
    gpuKernel.analyze();
    
    //get readonly id set
    Set<Ident> readOnlyOuterIdSet = gpuKernel.getReadOnlyOuterIdSet();//collectReadOnlyOuterIdSet(kernelList);
    
    //kernel内のincudtionVarのid
    //Set<Ident> inductionVarIdSet = gpuKernel.getInductionVarIdSet();

    //set unspecified var's attribute from outerIdSet
    Set<Ident> outerIdSet = new HashSet<Ident>(gpuKernel.getOuterIdList());
    for(Ident id : outerIdSet){
      String varName = id.getName();
      //if(parallelInfo.getACCvar(varName) != null) continue; 
      if(parallelInfo.isVarAllocated(varName)) continue;
      //if(parallelInfo.isVarPrivate(varName)) continue;
      if(parallelInfo.isVarFirstprivate(varName)) continue; //this is need for only parallel construct
      //if(parallelInfo.getDevicePtr(varName) != null) continue;
      if(parallelInfo.isVarReduction(varName)) continue;
      
      if(readOnlyOuterIdSet.contains(id) && !id.Type().isArray()) continue; //firstprivateは除く
      //if(inductionVarIdSet.contains(id)) continue;
      
      parallelInfo.declACCvar(id.getName(), ACCpragma.PRESENT_OR_COPY);
    }
    
    //translate data
    ACCtranslateData dataTranslator = new ACCtranslateData(pb);
    dataTranslator.translate();
    
    //make kernels list of block(kernel call , sync) 
    
    Block parallelBlock = gpuKernel.makeLaunchFuncCallBlock();
    Block replaceBlock = null;
    if(parallelInfo.isEnabled()){
      replaceBlock = parallelBlock;
    }else{
      replaceBlock = Bcons.IF(parallelInfo.getIfCond(), parallelBlock, Bcons.COMPOUND(pb.getBody()));
    }

    //set replace block
    parallelInfo.setReplaceBlock(replaceBlock);
  }

  public void translate_old() throws ACCexception{
    if(ACC.debugFlag){
      System.out.println("translate parallel");
    }

    //translate data
    ACCtranslateData dataTranslator = new ACCtranslateData(pb);
    dataTranslator.translate();
    
    //get outer ids
    XobjList outerIds = getOuterIds();
    
    //check private and firstprivate variable
    XobjList firstprivateIds = Xcons.IDList();
    XobjList privateIds = Xcons.IDList();
    XobjList normalIds = Xcons.IDList();
    divideIds(outerIds, normalIds, privateIds, firstprivateIds);
    
    //translate
    Ident funcId = globalDecl.declExternIdent(globalDecl.genSym(ACC_GPU_FUNC_PREFIX), Xtype.Function(Xtype.voidType));  
    
    XobjList funcParamArgs = makeFuncParamArgs(normalIds, firstprivateIds);
    XobjList funcParams = (XobjList)funcParamArgs.left();
    XobjList funcArgs = (XobjList)funcParamArgs.right();
    
    createGpuFuncs(funcId, funcParams, pb, parallelInfo);
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

  /** @return identList which is referenced in pragma block and defined out pragma block.*/ 
  private XobjList getOuterIds(){
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
            //if(iter.getBasicBlock().getParent().findVarIdent(varName) == null){
              outerIds.add(id);
            //}
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
        default:
        }
      }
    }
    
    return outerIds; 
  }

  

  private void createGpuFuncs(Ident funcId, XobjList funcParams, Block parallelBlock, ACCinfo info) throws ACCexception {
    String hostFuncName = funcId.getName();
    Ident hostFuncId = funcId;
    gpuManager = new ACCgpuManager(info);
    //reductionVars = new ArrayList<ACCvar>();

    //analyze
    analyzeParallelBlock(parallelBlock);
    //analyzeParallelBody(parallelBlock.getBody());
    gpuManager.finalize();
    

    //create device kernel
    XobjList deviceKernelParamArgs = makeDeviceKernelParamArgs(funcParams);
    XobjList deviceKernelParams = (XobjList)deviceKernelParamArgs.left();
    XobjList deviceKernelArgs = (XobjList)deviceKernelParamArgs.right();
    String deviceKernelName = hostFuncName + "_DEVICE";
    Ident deviceKernelId = ACCutil.getMacroFuncId(deviceKernelName, Xtype.voidType);
    XobjectDef deviceKernelDef = createDeviceKernelDef(deviceKernelId, deviceKernelParams, parallelBlock/*parallelBody*/);

    ((FunctionType)deviceKernelId.Type()).setFuncParamIdList(deviceKernelParams);
    Bcons.Statement(deviceKernelId.Call(deviceKernelArgs));
    
    // create host function
    XobjectDef hostFuncDef = createHostFuncDef(hostFuncId, funcParams, deviceKernelDef, deviceKernelId);

    //new ACCgpuDecompiler().decompile(info.getGlobalDecl().getEnv(), deviceKernelDef, deviceKernelId, hostFuncDef, varDecls);
    
  }

  private XobjList makeDeviceKernelParamArgs(XobjList funcParams) {
    //temporary
    return Xcons.List(funcParams.copy(), ACCutil.getRefs(funcParams));
  }
  
  private XobjectDef createDeviceKernelDef(Ident deviceKernelId, XobjList deviceKernelParamIds, Block kernelBlock/*BlockList kernelBody*/) throws ACCexception {
    //create device function
    XobjList deviceKernelLocalIds = Xcons.IDList();
    BlockList deviceKernelBody = Bcons.emptyBody();
    
    //set private var id as local id
    Iterator<ACCvar> varIter = parallelInfo.getVars();
    while(varIter.hasNext()){
      ACCvar var = varIter.next();
      if(var.isPrivate()){
        deviceKernelLocalIds.add(Ident.Local(var.getName(), var.getId().Type()/*Xtype.intType*/));
      }
    }
    
    List<Block> initBlocks = new ArrayList<Block>();
    
    XobjList additionalParams = Xcons.IDList();
    XobjList additionalLocals = Xcons.IDList();
    //BlockList copyBody = kernelBody.copy();
    Block deviceKernelBlock = makeDeviceKernelCoreBlock(initBlocks, kernelBlock.getBody(), additionalParams, additionalLocals, null, deviceKernelId);//, Xcons.IDList());
    rewriteReferenceType(deviceKernelBlock, deviceKernelParamIds);
    deviceKernelLocalIds.mergeList(additionalLocals);
    deviceKernelParamIds.mergeList(additionalParams);
    
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
  
  private Block makeDeviceKernelCoreBlock(List<Block> initBlocks, BlockList kernelBody, XobjList paramIds, XobjList localIds, String prevExecMethodName, Ident deviceKernelId){//, XobjList reduceIds) throws ACCexception {
    Block b = kernelBody.getHead();
    
    Xobject ids = kernelBody.getIdentList();
    Xobject decls = kernelBody.getDecls();
    BlockList resultBody = Bcons.emptyBody(ids, decls);
    
    while(b != null){
      if(b.Opcode() == Xcode.FOR_STATEMENT){
        ACCinfo info = ACCutil.getACCinfo(b);
        if(info != null && (info.getPragma() == ACCpragma.LOOP || info.getPragma() == ACCpragma.PARALLEL_LOOP)){
          CforBlock forBlock = (CforBlock)b;
          String execMethodName = gpuManager.getMethodName(forBlock);
          Block resultBlock = Bcons.emptyBlock();
          
          List<Block> beginBlocks = new ArrayList<Block>();
          List<Block> endBlocks = new ArrayList<Block>();
          
          XobjList reductionVarIds = Xcons.IDList();
          XobjList reductionLocalVarIds = Xcons.IDList();
          
          if(execMethodName == ""){ //if execMethod is not defined or seq
            resultBlock=Bcons.FOR(forBlock.getInitBBlock(), forBlock.getCondBBlock(), forBlock.getIterBBlock(), 
                Bcons.blockList(makeDeviceKernelCoreBlock(initBlocks, forBlock.getBody(), paramIds, localIds, prevExecMethodName, deviceKernelId/*, reduceIds*/)));
            resultBody.add(resultBlock);
          }else{
            
            //reduction
            Iterator<ACCvar> vars = info.getVars();
            while(vars.hasNext()){
              ACCvar var = vars.next();
              if(var.isReduction()){
                //reductionVars.add(var);
                Ident localRedId = Ident.Local(ACC_REDUCTION_VAR_PREFIX + var.getName(), var.getId().Type());
                XobjList redTmpArgs = Xcons.List();
                if(needsTemp(var)){
                  Ident redId = var.getId();
                  if(needsTemp(var) && execMethodName.startsWith("block")){
                    Ident ptr_red_tmp = Ident.Param("_ACC_gpu_red_tmp_" + redId.getName(), Xtype.Pointer(redId.Type()));
                    //Ident ptr_red_cnt = Ident.Param("_ACC_gpu_red_cnt_" + redId.getName(), Xtype.Pointer(Xtype.unsignedType));
                    Ident ptr_red_cnt = Ident.Var(deviceKernelId.getName() + "_red_cnt_" + redId.getName(), Xtype.unsignedType, Xtype.Pointer(Xtype.unsignedType), VarScope.GLOBAL);
                    paramIds.add(ptr_red_tmp);
                    //paramIds.add(ptr_red_cnt);
                    redTmpArgs.add(ptr_red_tmp.Ref());
                    redTmpArgs.add(Xcons.AddrOfVar(ptr_red_cnt.Ref()));
                    tmpUsingReductionVars.add(var);
                    varDecls.add(new XobjectDef(Xcons.List(Xcode.VAR_DECL, ptr_red_cnt, Xcons.IntConstant(0))));
                  }
                }
                localIds.add(localRedId);
                reductionLocalVarIds.add(localRedId);
                reductionVarIds.add(var.getId());
                int reductionKind = getReductionKindInt(var.getReductionOperator());
                beginBlocks.add(ACCutil.createFuncCallBlock("_ACC_gpu_init_reduction_var", Xcons.List(localRedId.getAddr(), Xcons.IntConstant(reductionKind))));
                XobjList funcCallArgs = Xcons.List(var.getId().getAddr(),localRedId.Ref(), Xcons.IntConstant(reductionKind));
                funcCallArgs.mergeList(redTmpArgs);
                endBlocks.add(ACCutil.createFuncCallBlock("_ACC_gpu_reduce_" + execMethodName, funcCallArgs)); 
              }
            }

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
            //initBlocks.add(initIterFunc);
            beginBlocks.add(initIterFunc);

            XobjList calcIdxFuncArgs = Xcons.List(iterIdx.Ref());
            calcIdxFuncArgs.add(indVarId.getAddr());
            calcIdxFuncArgs.mergeList(Xcons.List(init, cond, step));
            Block calcIdxFunc = ACCutil.createFuncCallBlock("_ACC_gpu_calc_idx", calcIdxFuncArgs); 

            resultBlock = Bcons.FOR(
                Xcons.Set(iterIdx.Ref(), iterInit.Ref()),
                Xcons.binaryOp(Xcode.LOG_LT_EXPR, iterIdx.Ref(), iterCond.Ref()), 
                Xcons.asgOp(Xcode.ASG_PLUS_EXPR, iterIdx.Ref(), iterStep.Ref()), 
                Bcons.COMPOUND(
                    Bcons.blockList(
                        calcIdxFunc,
                        makeDeviceKernelCoreBlock(initBlocks, forBlock.getBody(), paramIds, localIds, execMethodName, deviceKernelId/*, reduceIds*/)
                        )
                    )
                );
          }
          
          //rewriteReductionvar
          rewriteReductionVar(resultBlock, reductionVarIds, localIds);

          //make blocklist
          for(Block block : beginBlocks) resultBody.add(block);
          resultBody.add(resultBlock);
          for(Block block : endBlocks) resultBody.add(block);          
        }else{
          CforBlock forBlock = (CforBlock)b;
          Block resultBlock=Bcons.FOR(forBlock.getInitBBlock(), forBlock.getCondBBlock(), forBlock.getIterBBlock(), 
              Bcons.blockList(makeDeviceKernelCoreBlock(initBlocks, forBlock.getBody(), paramIds, localIds, prevExecMethodName, deviceKernelId/*, reduceIds*/)));
          resultBody.add(resultBlock);
        }
      }else if(b.Opcode() == Xcode.COMPOUND_STATEMENT){
        Block resultBlock = makeDeviceKernelCoreBlock(initBlocks, b.getBody(), paramIds, localIds, prevExecMethodName, deviceKernelId/*, reduceIds*/);
        resultBody.add(resultBlock);
      }else if(b.Opcode() == Xcode.ACC_PRAGMA){
        ACCinfo info = ACCutil.getACCinfo(b);
        ACC.debug("directive in parallel : "+info.getPragma().getName());
        resultBody.add(makeDeviceKernelCoreBlock(initBlocks, b.getBody(), paramIds, localIds, prevExecMethodName, deviceKernelId));
      }else{
        Block newBlock; 
        if(prevExecMethodName==null){
          newBlock=null;
//          /blockidx.x==0
          Ident block_id = Ident.Local("_ACC_block_x_id", Xtype.intType);
          newBlock = Bcons.IF(Xcons.binaryOp(Xcode.LOG_EQ_EXPR, block_id.Ref(),Xcons.IntConstant(0)), b.copy(),null);
        }else if(prevExecMethodName.equals("block_x")){
          Ident thread_id = Ident.Local("_ACC_thread_x_id", Xtype.intType); 
          newBlock = Bcons.IF(Xcons.binaryOp(Xcode.LOG_EQ_EXPR, thread_id.Ref(),Xcons.IntConstant(0)), b.copy(), null);
          //threadidx.x==0
        }else{
          newBlock = b.copy();
        }
        
        //replace reduction_var with local_reduction_var
        //rewriteReductionVar(newBlock, reduceIds, localIds);
        resultBody.add(newBlock);
      }
      b = b.getNext();
    }
    
    return Bcons.COMPOUND(resultBody);
  }

  private void analyzeParallelBody(BlockList body) throws ACCexception {
    Block b = body.getHead();
    
   // b = b.getParentBlock();
    
    while(b != null){
      if(b.Opcode() == Xcode.ACC_PRAGMA){
        ACCinfo info = ACCutil.getACCinfo(b);
        if(info.getPragma() == ACCpragma.LOOP || info.getPragma() == ACCpragma.PARALLEL_LOOP){
          CforBlock forBlock = (CforBlock)b.getBody().getHead();
          ACCutil.setACCinfo(forBlock, info);
          Iterator<ACCpragma> execModelIter = info.getExecModels();
          gpuManager.addLoop(execModelIter, forBlock);
          
          analyzeParallelBody(forBlock.getBody());
        }
      }else if(b.Opcode() == Xcode.FOR_STATEMENT){
        
      }else if(b.Opcode() == Xcode.COMPOUND_STATEMENT){
        analyzeParallelBody(b.getBody());
      }else{
        
      }
      b = b.getNext();
    }
    
    return;
  }
  
  private void analyzeParallelBlock(Block block) throws ACCexception{
    if(block == null) return;

    Block b = block;
    if(b.Opcode() == Xcode.ACC_PRAGMA){
      ACCinfo info = ACCutil.getACCinfo(b);
      if(info.getPragma() == ACCpragma.LOOP || info.getPragma() == ACCpragma.PARALLEL_LOOP){
        CforBlock forBlock = (CforBlock)b.getBody().getHead();
        ACCutil.setACCinfo(forBlock, info);
        Iterator<ACCpragma> execModelIter = info.getExecModels();
        gpuManager.addLoop(execModelIter, forBlock);
        BlockList body = forBlock.getBody();
        if(body != null){
          b = body.getHead();
          while(b != null){
            analyzeParallelBlock(b);
            b = b.getNext();
          }
        }
      }
    }else if(b.Opcode() == Xcode.FOR_STATEMENT){

    }else if(b.Opcode() == Xcode.COMPOUND_STATEMENT){
      //analyzeParallelBody(b.getBody());
    }else{

    }
    
//    BlockList body = block.getBody();
//    if(body != null){
//      b = body.getHead();
//      while(b != null){
//        analyzeParallelBlock(b);
//        b = b.getNext();
//      }
//    }
  }
  
  
  private void analyzeParallelBlock_back(Block block) throws ACCexception{
    if(block == null) return;

    Block b = block;
    if(b.Opcode() == Xcode.ACC_PRAGMA){
      ACCinfo info = ACCutil.getACCinfo(b);
      if(info.getPragma() == ACCpragma.LOOP || info.getPragma() == ACCpragma.PARALLEL_LOOP){
        CforBlock forBlock = (CforBlock)b.getBody().getHead();
        ACCutil.setACCinfo(forBlock, info);
        Iterator<ACCpragma> execModelIter = info.getExecModels();
        gpuManager.addLoop(execModelIter, forBlock);
      }
    }else if(b.Opcode() == Xcode.FOR_STATEMENT){

    }else if(b.Opcode() == Xcode.COMPOUND_STATEMENT){
      //analyzeParallelBody(b.getBody());
    }else{

    }
    
    BlockList body = block.getBody();
    if(body != null){
      b = body.getHead();
      while(b != null){
        analyzeParallelBlock(b);
        b = b.getNext();
      }
    }
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
    List<Block> preBlocks = new ArrayList<Block>();
    List<Block> postBlocks = new ArrayList<Block>();
    
    XobjList blockThreadSize = gpuManager.getBlockThreadSize();
    XobjList blockSize = (XobjList)blockThreadSize.left();
    XobjList threadSize = (XobjList)blockThreadSize.right();
    
    hostFuncBody.add(Bcons.Statement(Xcons.Set(blockXid.Ref(), blockSize.getArg(0))));
    hostFuncBody.add(Bcons.Statement(Xcons.Set(blockYid.Ref(), blockSize.getArg(1))));
    hostFuncBody.add(Bcons.Statement(Xcons.Set(blockZid.Ref(), blockSize.getArg(2))));
    hostFuncBody.add(Bcons.Statement(Xcons.Set(threadXid.Ref(), threadSize.getArg(0))));
    hostFuncBody.add(Bcons.Statement(Xcons.Set(threadYid.Ref(), threadSize.getArg(1))));
    hostFuncBody.add(Bcons.Statement(Xcons.Set(threadZid.Ref(), threadSize.getArg(2))));
    
    //add reduction_tmp & reduction_cnt
    for(ACCvar redVar : tmpUsingReductionVars){
      if(needsTemp(redVar)){
        Ident redVarId = redVar.getId();
        Ident ptr_red_tmp = Ident.Local("_ACC_gpu_red_tmp_" + redVarId.getName(), Xtype.Pointer(redVarId.Type()));
        //Ident ptr_red_cnt = Ident.Local("_ACC_gpu_red_cnt_" + redVarId.getName(), Xtype.Pointer(Xtype.unsignedType));
        Xtype voidPtrPtr = Xtype.Pointer(Xtype.voidPtrType);
        Block mallocCall = ACCutil.createFuncCallBlock("_ACC_gpu_malloc", Xcons.List(Xcons.Cast(voidPtrPtr, ptr_red_tmp.getAddr()), Xcons.binaryOp(Xcode.MUL_EXPR, Xcons.SizeOf(Xtype.doubleType/*redVarId.Type()*/), blockXid.Ref())));
        //Block callocCall = ACCutil.createFuncCallBlock("_ACC_gpu_calloc", Xcons.List(Xcons.Cast(voidPtrPtr, ptr_red_cnt.getAddr()), Xcons.SizeOf(Xtype.unsignedType)));
        Block tmpFreeCall = ACCutil.createFuncCallBlock("_ACC_gpu_free", Xcons.List(ptr_red_tmp.Ref()));
        //Block cntFreeCall = ACCutil.createFuncCallBlock("_ACC_gpu_free", Xcons.List(ptr_red_cnt.Ref()));
        hostFuncLocalIds.add(ptr_red_tmp);
        //hostFuncLocalIds.add(ptr_red_cnt);
        deviceKernelCallArgs.add(ptr_red_tmp.Ref());
        //deviceKernelCallArgs.add(ptr_red_cnt.Ref());
        preBlocks.add(mallocCall);
        //preBlocks.add(callocCall);
        postBlocks.add(tmpFreeCall);
        //postBlocks.add(cntFreeCall);
      }
    }
    
    Xobject max_num_grid = Xcons.IntConstant(2147483647/*65535*/);
    Block adjustGridFuncCall = ACCutil.createFuncCallBlock("_ACC_GPU_ADJUST_GRID", Xcons.List(Xcons.AddrOf(blockXid.Ref()), Xcons.AddrOf(blockYid.Ref()), Xcons.AddrOf(blockZid.Ref()),max_num_grid));
    //hostFuncBody.add(adjustGridFuncCall);
    
    hostFuncBody.setIdentList(hostFuncLocalIds);
    hostFuncBody.setDecls(ACCutil.getDecls(hostFuncLocalIds));
    
    for(Block b:preBlocks) hostFuncBody.add(b);
    
    Xobject deviceKernelCall = deviceKernelId.Call(deviceKernelCallArgs);
    //FIXME merge GPU_FUNC_CONF and GPU_FUNC_CONF_ASYNC
    deviceKernelCall.setProp(ACCgpuDecompiler.GPU_FUNC_CONF, (Object)Xcons.List(blockXid, blockYid, blockZid,threadXid, threadYid, threadZid));
    if(parallelInfo.isAsync()){
      try{
        Xobject asyncExp = parallelInfo.getAsyncExp();
        if(asyncExp != null){
          deviceKernelCall.setProp(ACCgpuDecompiler.GPU_FUNC_CONF_ASYNC, (Object)Xcons.List(asyncExp));
        }else{
          deviceKernelCall.setProp(ACCgpuDecompiler.GPU_FUNC_CONF_ASYNC, (Object)Xcons.List());
        }
      }catch(Exception e){
        ACC.fatal("can't set async prop");   
      }
    }
    hostFuncBody.add(Bcons.Statement(deviceKernelCall));

    for(Block b:postBlocks) hostFuncBody.add(b);
    
    XobjectDef hostFuncDef = XobjectDef.Func(hostFuncId, hostFuncParamIds, null, hostFuncBody.toXobject());
    
    return hostFuncDef;
  }
  
  void rewriteReferenceType(Block b, XobjList paramIds){
    BasicBlockExprIterator iter = new BasicBlockExprIterator(b.getBody());
    for (iter.init(); !iter.end(); iter.next()) {
      Xobject expr = iter.getExpr();
      topdownXobjectIterator exprIter = new topdownXobjectIterator(expr);
      for (exprIter.init(); !exprIter.end(); exprIter.next()) {
        Xobject x = exprIter.getXobject();
        switch (x.Opcode()) {
        case VAR:
        {
          Ident id = ACCutil.getIdent(paramIds, x.getName());
          if(id != null){
            //Xtype x_type = x.Type();
            //Xtype p_type = Xtype.Pointer(x_type);
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
        }
        }
      }
    }
    return;
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
  
  private int getReductionKindInt(ACCpragma pragma){
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
  
}

  