/* 
 * $TSUKUBA_Release: Omni XcalableMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
package exc.xmpF;

import exc.object.*;
import exc.util.MachineDep;
import exc.block.*;

import java.io.File;
import java.util.*;

/**
 * AST transformation by XcalableMP directives.
 */
public class XMPtransPragma
{
  protected XobjectDef current_def;
  private XMPenv  xmp_env;
  protected XobjectFile env;
    
  public XMPtransPragma() { }

  // pass3: do transformation for XMP pragma
  public void run(FuncDefBlock def, XMPenv env) {
    Block b;
    Block fblock = def.getBlock();
    current_def = def.getDef();
    this.env = current_def.getFile();
    this.xmp_env = env; 

    XMP.debug("pass3:");
    // scan by bottom-up iterator
    BlockIterator i = new bottomupBlockIterator(fblock);
    for(i.init(); !i.end(); i.next()) {
      b = i.getBlock();
      if(b.Opcode() == Xcode.XMP_PRAGMA) {
	b = transPragma((PragmaBlock)b);
	if(b != null) i.setBlock(b);
      }
    }
    // constructor for XMPobjects
    fblock.getBody().getHead().
      getBody().getHead().insert(localConstructorBlock(fblock));
  }

  public Block localConstructorBlock(Block fblock){
    BlockList body = Bcons.emptyBody();
    XMPsymbolTable table = XMPenv.localXMPsymbolTable(fblock);
    if(table != null){
      for(XMPobject o: table.getXMPobjects()){
	Block b1 = o.buildConstructor(current_def);
	if(b1 != null) body.add(b1);
      }
      for(XMParray a: table.getXMParrays()){
	Block b2 = a.buildConstructor(current_def);
	if(b2 != null) body.add(b2);
      }
    }
    return Bcons.COMPOUND(body);
  }

  // pass3:
  // write pragma
  Block transPragma(PragmaBlock pb) {
    XMPinfo info = (XMPinfo)pb.getProp(XMP.prop);
    if(info == null) return null;

    switch (info.pragma){
    case NODES:
    case TEMPLATE:
    case DISTRIBUTE:
    case ALIGN:
    case SHADOW:
    case COARRAY:
      /* declaration directives, do nothing */
      return Bcons.emptyBlock();

    case LOOP:
      return translateLoop(pb,info);

    case REFLECT:
      return translateReflect(pb,info);
    case BARRIER:
      return translateBarrier(pb,info);
    case REDUCTION:
      return translateReduction(pb,info);
    case BCAST:
      return translateBcast(pb,info);     

    case TASK:
      return translateTask(pb,info);
    case TASKS:
      return translateTasks(pb,info);
    case GMOVE:
      return translateGmove(pb,info);
    default:
      // XMP.fatal("unknown pragma");
      // ignore it
      return null;
    }
  }

  static boolean loop_opt_enable = true;

  private Block translateLoop(PragmaBlock pb, XMPinfo info){
    BlockList ret_body = Bcons.emptyBody();
    XMPobjectsRef on_ref = info.getOnRef();

    // generate on_ref object
    // create on_desc, only use loop_on_ref
    ret_body.add(on_ref.buildConstructor(current_def));

    if(!loop_opt_enable){
      // default loop transformation (non-opt)
      // just guard body
      // DO I = lb, up, step ; if(xmp_is_loop_1(i,on_desc)) body
      ForBlock for_block = (ForBlock)pb.getBody().getHead();
      BlockList loopBody = info.getBody();
      FdoBlock for_inner = (FdoBlock)loopBody.getParent();
      Xobject test_expr = on_ref.buildLoopTestFuncCall(current_def,info);
      Block new_body = 
	Bcons.IF(BasicBlock.Cond(test_expr),loopBody,null);
      for_inner.setBody(new BlockList(new_body));
      ret_body.add((Block)for_block);
      return Bcons.COMPOUND(ret_body);
    }

    // default loop transformation (convert to local index)
    // DO I = lb, up, step ; body
    // to:----
    // set on_ref object 
    // sched_loop(lb,up,step,local_lb,local_up,local_step)
    // DO i_local = loca_lb, local_up, local_step ;
    
    Block entry_block = Bcons.emptyBlock();
    BasicBlock entry_bb = entry_block.getBasicBlock();

    for(int k = 0; k < info.getLoopDim(); k++){
      XMPdimInfo d_info = info.getLoopDimInfo(k);
      Ident local_loop_var = d_info.getLoopLocalVar();

      ForBlock for_block = d_info.getLoopBlock();
      Xtype step_type = Xtype.intType;

      if(local_loop_var == null){
	// cannot be localized
	Xobject test_expr = 
	  on_ref.buildLoopTestSkipFuncCall(current_def,info,k);
	Block skip_block = 
	  Bcons.IF(BasicBlock.Cond(test_expr),
		   Bcons.blockList(Bcons.Fcycle()),null);
	for_block.getBody().insert(skip_block);
	continue;
      }

      // transform
      Xtype btype = local_loop_var.Type();
      Ident lb_var = Ident.Local(env.genSym("lb"), btype);
      Ident ub_var = Ident.Local(env.genSym("ub"), btype);
      Ident step_var = Ident.Local(env.genSym("step"), step_type);
      
      Xobject org_loop_ind_var = for_block.getInductionVar();

      // note, in case of C, must replace induction variable with local one
      ((FdoBlock)for_block).setInductionVar(local_loop_var.Ref());
      
      entry_bb.add(Xcons.Set(lb_var.Ref(), for_block.getLowerBound()));
      entry_bb.add(Xcons.Set(ub_var.Ref(), for_block.getUpperBound()));
      entry_bb.add(Xcons.Set(step_var.Ref(), for_block.getStep()));

      Ident schd_f = 
	current_def.declExternIdent(XMP.loop_sched_f,Xtype.FsubroutineType);
      Xobject args = Xcons.List(lb_var.Ref(), ub_var.Ref(), step_var.Ref(),
				Xcons.IntConstant(k),
				on_ref.getDescId().Ref());
      entry_bb.add(schd_f.callSubroutine(args));

      for_block.setLowerBound(lb_var.Ref());
      for_block.setUpperBound(ub_var.Ref());
      for_block.setStep(step_var.Ref());
      
      if(isVarUsed(for_block.getBody(),org_loop_ind_var)){
	// if global variable is used in this block, convert local to global
	Ident l2g_f = 
	  current_def.declExternIdent(XMP.l2g_f,Xtype.FsubroutineType);
	args = Xcons.List(org_loop_ind_var,
			  local_loop_var.Ref(),
			  Xcons.IntConstant(k),
			  on_ref.getDescId().Ref());
	for_block.getBody().insert(l2g_f.callSubroutine(args));
      }
    }

    ret_body.add(entry_block);
    ret_body.add(pb.getBody().getHead()); // loop
    return Bcons.COMPOUND(ret_body);
  }


  boolean isVarUsed(BlockList body,Xobject v){
    BasicBlockExprIterator iter = new BasicBlockExprIterator(body);
    for (iter.init(); !iter.end(); iter.next()) {
      Xobject expr = iter.getExpr();
      if(expr != null){
	XobjectIterator expr_iter = new bottomupXobjectIterator(expr);
	for(expr_iter.init(); !expr_iter.end(); expr_iter.next()){
	  Xobject e = expr_iter.getXobject();
	  if(e == null) continue;
	  switch(e.Opcode()){
	  case VAR:
	  case VAR_ADDR:
	    if(e.getName().equals(v.getName())) return true;
	  }
	}
      }
    }
    return false;
  }

  private Block translateReflect(PragmaBlock pb, XMPinfo info){
    Block b = Bcons.emptyBlock();
    BasicBlock bb = b.getBasicBlock();
    Ident f;

    f = current_def.declExternIdent(XMP.reflect_f,Xtype.FsubroutineType);
    
    // no width option is supported yet.
    Vector<XMParray> reflectArrays = info.getReflectArrays();
    for(XMParray a: reflectArrays){
      bb.add(f.callSubroutine(Xcons.List(a.getDescId().Ref())));
    }

    return b;
  }

  private Block translateBarrier(PragmaBlock pb, XMPinfo i) {
    Block b = Bcons.emptyBlock();
    BasicBlock bb = b.getBasicBlock();
    Ident f = current_def.declExternIdent(XMP.barrier_f,
					  Xtype.FsubroutineType);
    // don't care about on_ref
    bb.add(f.callSubroutine(Xcons.List(Xcons.IntConstant(0))));
    return b;
  }

  private Block translateReduction(PragmaBlock pb, XMPinfo info){
    Block b = Bcons.emptyBlock();
    BasicBlock bb = b.getBasicBlock();

    // object size
    int op = info.getReductionOp();
    Ident f = current_def.declExternIdent(XMP.reduction_f,
					  Xtype.FsubroutineType);
    for(Ident id: info.getInfoVarIdents()){
      Xtype type = id.Type();
      Xobject size_expr = Xcons.IntConstant(1);
      if(type.isFarray()){
	for(Xobject s: type.getFarraySizeExpr()){
	  size_expr = Xcons.binaryOp(Xcode.MUL_EXPR,size_expr,s);
	}
	type = type.getRef();
      }
      if(!type.isBasic()){
	XMP.fatal("reduction for non-basic type ="+type);
      }
      Xobject args = Xcons.List(id.Ref(),size_expr,
				Xcons.IntConstant(type.getBasicType()),
				Xcons.IntConstant(op),
				Xcons.IntConstant(0));
      bb.add(f.callSubroutine(args));
    }
    return b;
  }

  private Block translateBcast(PragmaBlock pb, XMPinfo info){
    Block b = Bcons.emptyBlock();
    BasicBlock bb = b.getBasicBlock();

    Ident f = current_def.declExternIdent(XMP.bcast_f,
					  Xtype.FsubroutineType);
    for(Ident id: info.getInfoVarIdents()){
      Xtype type = id.Type();
      Xobject size_expr = Xcons.IntConstant(1);
      if(type.isFarray()){
	for(Xobject s: type.getFarraySizeExpr()){
	  size_expr = Xcons.binaryOp(Xcode.MUL_EXPR,size_expr,s);
	}
	type = type.getRef();
      }
      if(!type.isBasic()){
	XMP.fatal("bcast for non-basic type ="+type);
      }
      Xobject args = Xcons.List(id.Ref(),size_expr,
				Xcons.IntConstant(type.getBasicType()),
				Xcons.IntConstant(0),
				Xcons.IntConstant(0));
      bb.add(f.callSubroutine(args));
    }
    return b;
  }

  private Block translateTask(PragmaBlock pb, XMPinfo info){
    XMP.fatal("translateTask");
    return null;
  }

  private Block translateTasks(PragmaBlock pb, XMPinfo i) {
    XMP.fatal("translateTasks");
    return null;
  }

  private Block translateGmove(PragmaBlock pb, XMPinfo i) {
    XMP.fatal("translateGmove");
    return null;
  }

// declare func and return Ident with type
  public Ident XMPfuncIdent(String name) {
    if(name == null)
      throw new IllegalArgumentException("may be generating unsupported function call");
        
    Xtype t;
//     if(name == getMaxThreadsFunc || name == getThreadNumFunc ||
//        name == getNumThreadsFunc || name == sectionIdFunc)
//       t = Xtype.FintFunctionType;
//     else if(name == isLastFunc || name == isMasterFunc ||
// 	    name == doSingleFunc ||
// 	    name == staticShedNextFunc || name == dynamicShedNextFunc ||
// 	    name == guidedShedNextFunc || name == runtimeShedNextFunc ||
// 	    name == affinityShedNextFunc)
//       t = Xtype.FlogicalFunctionType;
//     else
    t = Xtype.FsubroutineType;
    return current_def.declExternIdent(name, t);
  }
}


