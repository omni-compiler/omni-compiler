package exc.openacc;

import exc.block.*;
import exc.object.*;
import java.util.*;

class AccParallel extends AccData{
  private Block _parallelBlock;
  private final AccKernel _accKernel;

  AccParallel(ACCglobalDecl decl, AccInformation info, PragmaBlock pb) {
    super(decl, info, pb);

    List<Block> kernelBody = new ArrayList<Block>();
    kernelBody.add(_pb);
    _accKernel = new AccKernel(_decl, _pb, _info, kernelBody);
  }

  @Override
  void analyze() throws ACCexception {
    if(isDisabled()){
      return;
    }
    completeParallelism();


    //analyze and complete clause for kernel
    _accKernel.analyze();

    //set unspecified var's attribute from outerIdSet
    //TODO do these process at analyze
    Set<Ident> readOnlyOuterIdSet = _accKernel.getReadOnlyOuterIdSet();
    for (Ident id : _accKernel.getOuterIdList()) {
      String varName = id.getName();
      if(_info.isDeclared(varName)) continue; //if declared in same directive

      if (readOnlyOuterIdSet.contains(id) && !id.Type().isPointer()) {
        _info.addVar(ACCpragma.FIRSTPRIVATE, Xcons.Symbol(Xcode.VAR, varName));
      }else {
        _info.addVar(ACCpragma.PRESENT_OR_COPY, Xcons.Symbol(Xcode.VAR, varName));
      }
    }

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

    _parallelBlock = _accKernel.makeLaunchFuncCallBlock();
  }

  @Override
  void rewrite() throws ACCexception {
    //build
    BlockList beginBody = Bcons.emptyBody();
    for(Block b : initBlockList) beginBody.add(b);
    for(Block b : copyinBlockList) beginBody.add(b);
    BlockList endBody = Bcons.emptyBody();
    for(Block b : copyoutBlockList) endBody.add(b);
    for(Block b : finalizeBlockList) endBody.add(b);

    Block beginBlock = Bcons.COMPOUND(beginBody);
    Block endBlock = Bcons.COMPOUND(endBody);

    BlockList resultBody = Bcons.emptyBody();
    for(Xobject x: idList){
      resultBody.addIdent((Ident)x);
    }

    Xobject ifExpr = _info.getIntExpr(ACCpragma.IF);
    boolean isEnabled = (ifExpr == null || (ifExpr.isIntConstant() && !ifExpr.isZeroConstant()));
    if(isEnabled){
      resultBody.add(beginBlock);
      resultBody.add(_parallelBlock);
      resultBody.add(endBlock);
    }else {
      Ident condId = resultBody.declLocalIdent("_ACC_DATA_IF_COND", Xtype.charType, StorageClass.AUTO, ifExpr);
      resultBody.add(Bcons.IF(condId.Ref(), beginBlock, null));
      resultBody.add(Bcons.IF(condId.Ref(), _parallelBlock, Bcons.COMPOUND(_pb.getBody())));
      resultBody.add(Bcons.IF(condId.Ref(), endBlock, null));
    }

    _pb.replace(Bcons.COMPOUND(resultBody));
  }

  boolean isAcceptableClause(ACCpragma clauseKind){
    switch (clauseKind) {
    case IF:
    case ASYNC:
    case NUM_GANGS:
    case NUM_WORKERS:
    case VECT_LEN:
    case PRIVATE:
    case FIRSTPRIVATE:
      return true;
    default:
      return clauseKind.isDataClause() || clauseKind.isReduction();
    }
  }
}
