/* -*- Mode: java; c-basic-offset:2 ; indent-tabs-mode:nil ; -*- */
package exc.openacc;

import exc.block.*;
import exc.object.*;
import java.util.*;

public class AccParallel extends AccData {
  private Block _parallelBlock;
  private AccKernel _accKernel;

  public AccParallel(ACCglobalDecl decl, AccInformation info, PragmaBlock pb) {
    super(decl, info, pb);

    List<Block> kernelBody = new ArrayList<Block>();
    kernelBody.add(_pb);
    _accKernel = ACC.getAccKernel(_decl, _pb, _info, kernelBody);
  }

  @Override
  void analyze() throws ACCexception {
    if(isDisabled()){
      return;
    }

    completeParallelism();

    //analyze and complete clause for kernel
    if(ACC.debug_flag) System.out.println("AccParallel _accKernel.analyze ...");
    _accKernel.analyze();

    if(ACC.debug_flag) System.out.println("AccParallel search var set ...");
    //set unspecified var's attribute from outerIdSet
    //TODO do these process at analyze
    Set<Ident> readOnlyOuterIdSet = _accKernel.getReadOnlyOuterIdSet();
    ACCpragma default_var_attr = getDefaultVarAttr();
    for (Ident id : _accKernel.getOuterIdList()) {
      String varName = id.getName();
      if(ACC.debug_flag) System.out.println("AccParallel OuterIdList id="+id);
      if(_info.isDeclared(varName)) continue; //if declared in same directive
      
      ACCvar parentVar = findParentVar(id);
      ACCvar var = _info.findACCvar(varName);

      boolean isReductionVariableInKernel = isReductionVariableInKernel(id);

      if (!id.Type().isPointer()
          && (ACC.version >= 20 || readOnlyOuterIdSet.contains(id))
          && parentVar == null /* not appeared in outer data clause*/
          && (var == null || !var.isReduction()) /* not reduction variable in the directive */
          && !isReductionVariableInKernel /* not reduction variable in the kernel*/ ) {
        if(ACC.debug_flag) System.out.println("AccParallel OuterIdList FIRSTPRIVATE id="+id);
        _info.addVar(ACCpragma.FIRSTPRIVATE, Xcons.Symbol(Xcode.VAR, varName));
      }else {
        if(ACC.debug_flag) System.out.println("AccParallel OuterIdList  default_attr="+default_var_attr+" id="+id);
        if(default_var_attr == ACCpragma.DEFAULT_NONE)
          throw new ACCexception("Variable attribute '"+varName+"' must be specified due to default(none)");
        else 
          _info.addVar(default_var_attr, Xcons.Symbol(Xcode.VAR, varName)); 
      }
    }
    
    if(ACC.debug_flag) System.out.println("AccParallel _accKernel.analyze ... end");

    //this is the end of analyze
    super.analyze();

    if(ACC.debug_flag) System.out.println("AccParallel _accKernel.analyze ... ret");
  }

  private boolean isReductionVariableInKernel(Ident id)
  {
    BlockIterator blockIterator = new topdownBlockIterator(_pb.getBody());
    for(blockIterator.init(); !blockIterator.end(); blockIterator.next()){
      Block b = blockIterator.getBlock();
      // if(b.Opcode() != Xcode.ACC_PRAGMA) continue;
      AccDirective directive = (AccDirective)b.getProp(AccDirective.prop);
      if(directive == null) continue;
      AccInformation info = directive.getInfo();
      ACCvar var = info.findReductionACCvar(id.getName());
      if(var != null && var.getId() == id){
        return true;
      }
    }
    return false;
  }

  void completeParallelism() throws ACCexception{
    BlockIterator blockIterator = new topdownBlockIterator(_pb.getBody());
    for(blockIterator.init(); !blockIterator.end(); blockIterator.next()){
      Block b = blockIterator.getBlock();
      // if(b.Opcode() != Xcode.ACC_PRAGMA) continue;
      AccDirective directive = (AccDirective)b.getProp(AccDirective.prop);
      if(directive == null) continue;
      directive.analyze();
    }
  }

  @Override
  void generate() throws ACCexception {
    if(isDisabled()){
      return;
    }

    if(ACC.debug_flag) System.out.println("AccParallel geneator _info="+_info);

    //generate data
    super.generate();

    _parallelBlock = _accKernel.makeLaunchFuncCallBlock();
  }

  @Override
  void rewrite() throws ACCexception {
    if(isDisabled()){
      _pb.replace(Bcons.COMPOUND(_pb.getBody()));
      return;
    }

    if(ACC.debug_flag) System.out.println("AccParallel rewrite _info="+_info);
    
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
    case WAIT:
    case WAIT_CLAUSE:
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
