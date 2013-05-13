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
  private XMPenv  env;
    
  public XMPtransPragma() { }

  // pass3: do transformation for XMP pragma
  public void run(FuncDefBlock def, XMPenv env) {
    this.env = env; 
    env.setCurrentDef(def);

    Block b;
    FunctionBlock fblock = def.getBlock();

    XMP.debug("pass3:");
    
    if(XMP.debugFlag) System.out.println("pass3 +fblock="+fblock);
    // scan by bottom-up iterator
    BlockIterator i = new bottomupBlockIterator(fblock.getBody().getHead());
    for(i.init(); !i.end(); i.next()) {
      b = i.getBlock();
      if(b.Opcode() == Xcode.XMP_PRAGMA) {
	b = transPragma((PragmaBlock)b);
	if(b != null) i.setBlock(b);
      }
    }
    
    Xobject f_name = fblock.getNameObj();
    BlockList prolog = Bcons.emptyBody();
    BlockList epilog = Bcons.emptyBody();
    buildXMPobjectBlock(prolog, epilog);
    
    if(env.currentDefIsModule()){
      Xtype save_logical = Xtype.FlogicalType.copy();
      save_logical.setIsFsave(true);
      Ident init_flag_var = env.declIdent("xmpf_init_flag",save_logical);
      prolog.add(Xcons.Set(init_flag_var.Ref(),
			   Xcons.FlogicalConstant(true)));

      Ident prolog_f = 
	env.declIdent(env.currentDefName()+"_xmpf_module_init_",
		      Xtype.FsubroutineType);
      XobjectDef prolog_def = 
	XobjectDef.Func(prolog_f, null, null, prolog.toXobject());
      env.getCurrentDef().getDef().getChildren().addFirst(prolog_def);
      prolog_def.setParent(env.getCurrentDef().getDef());
    } else {
      // fblock = (FunckBlock ident <param_env> [BlockList 
      //   (CompoundBlock <local_env> [BlockList statment ...]))
      BlockList f_body = fblock.getBody().getHead().getBody();
      
      // not need to call init_module__
//       XobjectDef parent = env.getCurrentDef().getDef().getParent();
//       if(parent != null && parent.isFmoduleDef()){
// 	Ident init_flag_var = env.findVarIdent("xmpf_init_flag",null);
// 	Ident init_func = env.findVarIdent("xmpf_module_init__",null);
// 	if(init_flag_var == null || init_func == null){
// 	  System.out.println("var="+init_flag_var+",func="+init_func);
// 	  XMP.fatal("cannot find init_var or init_func for moudle");
// 	}
// 	prolog.insert(Bcons.IF(Xcons.unaryOp(Xcode.LOG_NOT_EXPR,
// 					     init_flag_var.Ref()),
// 			       init_func.callSubroutine(null),null));
//       }
      f_body.insert(Bcons.COMPOUND(prolog));
      f_body.add(Bcons.COMPOUND(epilog));
    }
  }

  void buildXMPobjectBlock(BlockList prolog, BlockList epilog){
    XMPsymbolTable table = env.getXMPsymbolTable();
    if(table != null){
      for(XMPobject o: table.getXMPobjects()){
	o.buildConstructor(prolog,env);
	o.buildDestructor(epilog,env);
      }
      for(XMParray a: table.getXMParrays()){
	a.buildConstructor(prolog,env);
	a.buildDestructor(epilog,env);
      }
    }
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
    case WAIT_ASYNC:
      return translateWaitAsync(pb,info);     
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
    ret_body.add(on_ref.buildLoopConstructor(env));

    if(!loop_opt_enable){
      // default loop transformation (non-opt)
      // just guard body
      // DO I = lb, up, step ; if(xmp_is_loop_1(i,on_desc)) body
      ForBlock for_block = (ForBlock)pb.getBody().getHead();
      BlockList loopBody = info.getBody();
      FdoBlock for_inner = (FdoBlock)loopBody.getParent();
      Xobject test_expr = on_ref.buildLoopTestFuncCall(env,info);
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
	  on_ref.buildLoopTestSkipFuncCall(env,info,k);
	Block skip_block = 
	  Bcons.IF(BasicBlock.Cond(test_expr),
		   Bcons.blockList(Bcons.Fcycle()),null);
	for_block.getBody().insert(skip_block);
	continue;
      }

      // transform
      Xtype btype = local_loop_var.Type();
      Ident lb_var = env.declIdent(XMP.genSym("XMP_loop_lb"), btype,pb);
      Ident ub_var = env.declIdent(XMP.genSym("XMP_loop_ub"), btype,pb);
      Ident step_var = env.declIdent(XMP.genSym("XMP_loop_step"), step_type,pb);
      
      Xobject org_loop_ind_var = for_block.getInductionVar();

      // note, in case of C, must replace induction variable with local one
      ((FdoBlock)for_block).setInductionVar(local_loop_var.Ref());
      
      entry_bb.add(Xcons.Set(lb_var.Ref(), for_block.getLowerBound()));
      entry_bb.add(Xcons.Set(ub_var.Ref(), for_block.getUpperBound()));
      entry_bb.add(Xcons.Set(step_var.Ref(), for_block.getStep()));

      Ident schd_f = 
	env.declExternIdent(XMP.loop_sched_f,Xtype.FsubroutineType);
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
	  env.declExternIdent(XMP.l2g_f,Xtype.FsubroutineType);
	args = Xcons.List(org_loop_ind_var,
			  local_loop_var.Ref(),
			  Xcons.IntConstant(k),
			  on_ref.getDescId().Ref());
	for_block.getBody().insert(l2g_f.callSubroutine(args));
      }
    }

    ret_body.add(entry_block);

    if(pb.getPrev() != null && 
       pb.getPrev().Opcode() == Xcode.F_STATEMENT_LIST &&
       pb.getPrev().getBasicBlock() != null){
      Statement tail = pb.getPrev().getBasicBlock().getTail();
      if(tail != null && tail.getExpr().Opcode() == Xcode.PRAGMA_LINE){
	ret_body.add(tail.getExpr());
	tail.remove();
      }
    }

    ret_body.add(pb.getBody().getHead()); // loop

    if(info.getReductionOp() != XMP.REDUCE_NONE){
      ret_body.add(translateReduction(pb,info));
    }

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
    Ident f, g, h;

    g = env.declExternIdent(XMP.set_reflect_f,Xtype.FsubroutineType);
    f = env.declExternIdent(XMP.reflect_f,Xtype.FsubroutineType);
    h = env.declExternIdent(XMP.reflect_async_f,Xtype.FsubroutineType);

    Vector<XMParray> reflectArrays = info.getReflectArrays();
    for(XMParray a: reflectArrays){

      for (int i = 0; i < info.widthList.size(); i++){
	  XMPdimInfo w = info.widthList.get(i);
	  Xobject args = Xcons.List(a.getDescId().Ref(), Xcons.IntConstant(i),
				    w.getLower(), w.getUpper(), w.getStride());
	  bb.add(g.callSubroutine(args));
      }

      if (info.getAsyncId() != null){
	  bb.add(h.callSubroutine(Xcons.List(a.getDescId().Ref(), info.getAsyncId())));
      }
      else {
	  bb.add(f.callSubroutine(Xcons.List(a.getDescId().Ref())));
      }
    }

    return b;
  }

  private Block translateBarrier(PragmaBlock pb, XMPinfo i) {
    Block b = Bcons.emptyBlock();
    BasicBlock bb = b.getBasicBlock();
    Ident f = env.declExternIdent(XMP.barrier_f,
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
    Ident f = env.declExternIdent(XMP.reduction_f,
				  Xtype.FsubroutineType);
    for(Ident id: info.getReductionVars()){
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
				Xcons.IntConstant(XMP.reduceBasicType(type)),
				Xcons.IntConstant(op),
				Xcons.IntConstant(0));
      bb.add(f.callSubroutine(args));
    }
    return b;
  }

  private Block translateBcast(PragmaBlock pb, XMPinfo info){
    Block b = Bcons.emptyBlock();
    BasicBlock bb = b.getBasicBlock();

    Ident f = env.declExternIdent(XMP.bcast_f,
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

  private Block translateWaitAsync(PragmaBlock pb, XMPinfo info){
    Block b = Bcons.emptyBlock();
    BasicBlock bb = b.getBasicBlock();

    Ident f = env.declExternIdent(XMP.wait_async_f, Xtype.FsubroutineType);
    bb.add(f.callSubroutine(Xcons.List(info.async_id)));
    return b;
  }

  private Block translateTask(PragmaBlock pb, XMPinfo info){
    BlockList ret_body = Bcons.emptyBody();
    XMPobjectsRef on_ref = info.getOnRef();

    ret_body.add(on_ref.buildConstructor(env));
    Ident f = env.declExternIdent(XMP.test_task_on_f,
				  Xtype.FlogicalFunctionType);
    Xobject cond = f.Call(Xcons.List(on_ref.getDescId().Ref()));
    ret_body.add(Bcons.IF(cond,Bcons.COMPOUND(pb.getBody()),null));

    f = env.declExternIdent(XMP.end_task_f,Xtype.FsubroutineType);
    pb.getBody().add(f.Call(Xcons.List()));
    
    return Bcons.COMPOUND(ret_body);
  }

  private Block translateTasks(PragmaBlock pb, XMPinfo i) {
    XMP.fatal("translateTasks");
    return null;
  }

  /* gmove sequence:
   * For global array,
   *  	CALL xmp_gmv_g_alloc_(desc,XMP_DESC_a)
   * 	CALL xmp_gmv_g_info(desc,#i_dim,kind,lb,ub,stride)
   * For local array
   *  	CALL xmp_gmv_l_alloc_(desc,array,a_dim)
   *    CALL xmp_gmv_l_info(desc,#i_dim,a_lb,a_ub,kind,lb,ub,stride)
   *
   * kind = 2 -> ub, up, stride
   *        1 -> index
   *        0 -> all (:)
   * And, followed by:
   *    CALL xmp_gmv_do(left,right,collective(0)/in(1)/out(2))
   * Note: data type must be describe one of global side
   */

  private final static int GMOVE_ALL   = 0;
  private final static int GMOVE_INDEX = 1;
  private final static int GMOVE_RANGE = 2;
  
  private final static int GMOVE_COLL   = 0;
  private final static int GMOVE_IN = 1;
  private final static int GMOVE_OUT = 2;

  private Block translateGmove(PragmaBlock pb, XMPinfo i) {
    Block b = Bcons.emptyBlock();
    BasicBlock bb = b.getBasicBlock();

    Xobject left = i.getGmoveLeft();
    Xobject right = i.getGmoveRight();
    
    Ident left_desc = buildGmoveDesc(left, bb, pb);
    Ident right_desc = buildGmoveDesc(right, bb, pb);

    Ident f = env.declExternIdent(XMP.gmove_do_f, Xtype.FsubroutineType);
    Xobject args = Xcons.List(left_desc.Ref(), right_desc.Ref(),
			      Xcons.IntConstant(GMOVE_COLL));
    bb.add(f.callSubroutine(args));
	    
    return b;
  }

  private Ident buildGmoveDesc(Xobject x, BasicBlock bb, PragmaBlock pb) {
    Ident descId = env.declObjectId(XMP.genSym("gmv"),pb);
    Ident f; 
    Xobject args;
    
    switch(x.Opcode()){
    case F_ARRAY_REF:
      Xobject a = x.getArg(0).getArg(0);
      XMParray array = (XMParray)a.getProp(XMP.RWprotected);
      if(array != null){
	f = env.declExternIdent(XMP.gmove_g_alloc_f, Xtype.FsubroutineType);
	args = Xcons.List(descId.Ref(), array.getDescId().Ref());
	bb.add(f.callSubroutine(args));
	
	// System.out.println("idx args="+x.getArg(1));
	f = env.declExternIdent(XMP.gmove_g_dim_info_f, Xtype.FsubroutineType);
 	int idx = 0;
 	for(Xobject e: (XobjList) x.getArg(1)){
 	  switch(e.Opcode()){
	  case F_ARRAY_INDEX:
	    args = Xcons.List(descId.Ref(),Xcons.IntConstant(idx),
			      Xcons.IntConstant(GMOVE_INDEX),
			      e.getArg(0),
			      Xcons.IntConstant(0),Xcons.IntConstant(0));
	    break;
	  case F_INDEX_RANGE:
	    if(e.getArg(0) == null && e.getArg(1) == null){
	      args = Xcons.List(descId.Ref(),Xcons.IntConstant(idx),
				Xcons.IntConstant(GMOVE_ALL),
				Xcons.IntConstant(0),
				Xcons.IntConstant(0),Xcons.IntConstant(0));
	    } else {
	      Xobject stride = e.getArg(2);
	      if(stride == null) stride = Xcons.IntConstant(1);
	      args = Xcons.List(descId.Ref(),Xcons.IntConstant(idx),
				Xcons.IntConstant(GMOVE_RANGE),
				e.getArg(0),e.getArg(1),stride);
	    }
	    break;
	  default:
	    XMP.fatal("buildGmoveDec: unknown F_ARRAY_REF element");
	  }
	  bb.add(f.callSubroutine(args));
 	  idx++;
	}
      } else {
	f = env.declExternIdent(XMP.gmove_l_alloc_f, Xtype.FsubroutineType);
	Xtype type = a.Type();
	if(!type.isFarray()) 
	  XMP.fatal("buildGmoveDesc:F_ARRAY_REF for not Farray");
	args = Xcons.List(descId.Ref(),a,
			  Xcons.IntConstant(type.getNumDimensions()));
	bb.add(f.callSubroutine(args));

	Xobject lower_b = 
	  env.declIntrinsicIdent("lbound",Xtype.FintFunctionType).
	  Call(Xcons.List(a));
	Xobject upper_b = 
	  env.declIntrinsicIdent("ubound",Xtype.FintFunctionType).
	  Call(Xcons.List(a));
	
	f = env.declExternIdent(XMP.gmove_l_dim_info_f, Xtype.FsubroutineType);
 	int idx = 0;
 	for(Xobject e: (XobjList) x.getArg(1)){
 	  switch(e.Opcode()){
	  case F_ARRAY_INDEX:
	    args = Xcons.List(descId.Ref(),Xcons.IntConstant(idx),
			      lower_b, upper_b,
			      Xcons.IntConstant(GMOVE_INDEX),
			      e.getArg(0),
			      Xcons.IntConstant(0),Xcons.IntConstant(0));
	    break;
	  case F_INDEX_RANGE:
	    if(e.getArg(0) == null && e.getArg(1) == null){
	      args = Xcons.List(descId.Ref(),Xcons.IntConstant(idx),
				lower_b,upper_b, 
				Xcons.IntConstant(GMOVE_ALL),
				Xcons.IntConstant(0),
				Xcons.IntConstant(0),Xcons.IntConstant(0));
	    } else {
	      Xobject stride = e.getArg(2);
	      if(stride == null) stride = Xcons.IntConstant(1);
	      args = Xcons.List(descId.Ref(),Xcons.IntConstant(idx),
				lower_b,upper_b,
				Xcons.IntConstant(GMOVE_RANGE),
				e.getArg(0),e.getArg(1),stride);
	    }
	    break;
	  default:
	    XMP.fatal("buildGmoveDec: unknown F_ARRAY_REF element");
	  }
	  bb.add(f.callSubroutine(args));
 	  idx++;
	}
      }
      break;
    case VAR: /* it ok */
      f = env.declExternIdent(XMP.gmove_l_alloc_f, Xtype.FsubroutineType);
      args = Xcons.List(descId.Ref(),x,Xcons.IntConstant(0));
      bb.add(f.callSubroutine(args));
      break;
    default:
      XMP.errorAt(pb,"gmove must be followed by simple assignment");
    }
    
    return descId;
  }
}


