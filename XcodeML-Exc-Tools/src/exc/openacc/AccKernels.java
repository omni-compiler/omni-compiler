package exc.openacc;

import exc.block.*;
import exc.object.*;
import java.util.*;

class AccKernels extends AccData {
  private final List<Block> _kernelBlocks = new ArrayList<Block>();
  private final List<AccKernel> _accKernelList = new ArrayList<AccKernel>();

  AccKernels(ACCglobalDecl decl, AccInformation info, PragmaBlock pb) {
    super(decl, info, pb);

    //divide to kernels -> list of kernel head block
    List<List<Block>> kernelBodyList = divideBlocksBetweenKernels(_pb);

    for(List<Block> kernelBody : kernelBodyList){
      AccKernel accKernel = new AccKernel(_decl, _pb, _info, kernelBody);
      _accKernelList.add(accKernel);
    }
  }

  @Override
  void analyze() throws ACCexception {
    if(isDisabled()){
      return;
    }
    completeParallelism();


    //analyze and complete clause for kernel
    for(AccKernel accKernel : _accKernelList) {
      accKernel.analyze();
    }

    /*
    //set unspecified var's attribute from outerIdSet
    //TODO do these process at analyze
    Set<Ident> readOnlyOuterIdSet = _accKernel.getReadOnlyOuterIdSet();
    for (Ident id : _accKernel.getOuterIdList()) {
      String varName = id.getName();
      if(_info.isDeclared(varName)) continue; //if declared in same directive

      if (readOnlyOuterIdSet.contains(id) && !id.Type().isArray()) {
        _info.setVar(ACCpragma.FIRSTPRIVATE, Xcons.Symbol(Xcode.VAR, varName));
      }else {
        _info.setVar(ACCpragma.PRESENT_OR_COPY, Xcons.Symbol(Xcode.VAR, varName));
      }
    }
    */
    ///////

    //get intersection of readonly id set in each kernel
    Set<Ident> readOnlyOuterIdSet = collectReadOnlyOuterIdSet(_accKernelList);
    for(AccKernel accKernel: _accKernelList){
      accKernel.setReadOnlyOuterIdSet(readOnlyOuterIdSet);
    }

    //set unspecified var's attribute from outerIdSet
    Set<Ident> outerIdSet = new HashSet<Ident>();
    for(AccKernel gpuKernel : _accKernelList){
      List<Ident> kernelOuterId = gpuKernel.getOuterIdList();
      outerIdSet.addAll(kernelOuterId);
    }
    for(Ident id : outerIdSet){
      String varName = id.getSym();
      if(_info.isDeclared(varName)) continue;
      _info.addVar(ACCpragma.PRESENT_OR_COPY, Xcons.Symbol(Xcode.VAR, varName));
    }

    /////////


    //this is the end of analyze
    super.analyze();
  }

  void completeParallelism() throws ACCexception{
    BlockIterator blockIterator = new topdownBlockIterator(_pb.getBody());
    for(blockIterator.init(); !blockIterator.end(); blockIterator.next()){
      Block b = blockIterator.getBlock();
      if(b.Opcode() != Xcode.ACC_PRAGMA) continue;
      AccDirective directive = (AccDirective)b.getProp(AccDirective.prop);
      directive.analyze();
    }
  }

  @Override
  void generate() throws ACCexception {
    if(isDisabled()){
      return;
    }



    //generate data
    super.generate();

    //make kernels list of block(kernel call , sync)
    for(AccKernel gpuKernel : _accKernelList){
      Block kernelCallBlock = gpuKernel.makeLaunchFuncCallBlock();
      _kernelBlocks.add(kernelCallBlock);
    }
  }

  @Override
  void rewrite() throws ACCexception{
    if(isDisabled()) {
      _pb.replace(Bcons.COMPOUND(_pb.getBody()));
      return;
    }

    //build
    BlockList beginBody = Bcons.emptyBody();
    for(Block b : initBlockList) beginBody.add(b);
    for(Block b : copyinBlockList) beginBody.add(b);
    BlockList endBody = Bcons.emptyBody();
    for(Block b : copyoutBlockList) endBody.add(b);
    for(Block b : finalizeBlockList) endBody.add(b);

    Block beginBlock = Bcons.COMPOUND(beginBody);
    Block endBlock = Bcons.COMPOUND(endBody);

    BlockList kernelsBody = Bcons.emptyBody();
    for(Block b : _kernelBlocks){
      kernelsBody.add(b);
    }
    Block kernelsBlock = Bcons.COMPOUND(kernelsBody);

    BlockList resultBody = Bcons.emptyBody();
    for(Xobject x: idList){
      resultBody.addIdent((Ident)x);
    }

    Xobject ifExpr = _info.getIntExpr(ACCpragma.IF);
    boolean isEnabled = (ifExpr == null || (ifExpr.isIntConstant() && !ifExpr.isZeroConstant()));
    if(isEnabled){
      resultBody.add(beginBlock);
      resultBody.add(kernelsBlock);
      resultBody.add(endBlock);
    }else {
      Ident condId = resultBody.declLocalIdent("_ACC_DATA_IF_COND", Xtype.charType, StorageClass.AUTO, ifExpr);
      resultBody.add(Bcons.IF(condId.Ref(), beginBlock, null));
      resultBody.add(Bcons.IF(condId.Ref(), kernelsBlock, Bcons.COMPOUND(_pb.getBody())));
      resultBody.add(Bcons.IF(condId.Ref(), endBlock, null));
    }

    _pb.replace(Bcons.COMPOUND(resultBody));
  }


  private List<List<Block>> divideBlocksBetweenKernels(PragmaBlock pb) {
    List<List<Block>> blockListList = new ArrayList<List<Block>>();

    /*
    if(_info.getPragma() == ACCpragma.KERNELS){
      BlockList pbBody = pb.getBody();
      for(Block b = pbBody.getHead(); b != null; b = b.getNext()){
        List<Block> blockList = new ArrayList<Block>();
        blockList.add(b);
        blockListList.add(blockList);
      }
    }else{ //ACCpragma.KERNELS_LOOP
      List<Block> blockList = new ArrayList<Block>();
      blockList.add(pb);
      blockListList.add(blockList);
    }
*/
    if(! _pb.getBody().isSingle()){
      for(Block b = pb.getBody().getHead(); b != null; b = b.getNext()){
        List<Block> blockList = new ArrayList<Block>();
        blockList.add(b);
        blockListList.add(blockList);
      }
    }else{
      List<Block> blockList = new ArrayList<Block>();
      blockList.add(pb);
      blockListList.add(blockList);
    }

    return blockListList;
  }

  private Set<Ident> collectReadOnlyOuterIdSet(List<AccKernel> kernelList){
    if(kernelList.size() == 1){
      return kernelList.get(0).getReadOnlyOuterIdSet();
    }

    Iterator<AccKernel> kernelIter = kernelList.iterator();
    AccKernel kernel = kernelIter.next();
    Set<Ident> readOnlyOuterIdSet = kernel.getReadOnlyOuterIdSet();
    while(kernelIter.hasNext()){
      kernel = kernelIter.next();
      readOnlyOuterIdSet.addAll(kernel.getReadOnlyOuterIdSet());
    }

    for(AccKernel kern : kernelList){
      Set<Ident> outerIdSet = new HashSet<Ident>(kern.getOuterIdSet());
      outerIdSet.removeAll(kern.getReadOnlyOuterIdSet());
      readOnlyOuterIdSet.removeAll(outerIdSet);
    }

    return readOnlyOuterIdSet;
  }

  boolean isAcceptableClause(ACCpragma clauseKind) {
    switch (clauseKind){
    case IF:
    case ASYNC:
      return true;
    default:
      return clauseKind.isDataClause();
    }
  }
}
