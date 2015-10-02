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
    
    // move OMP_THREADPRIVATE to the head of prolog
    // NOTE: Hereafter any addition of statements to prolog must be done
    //       at its tail.
    BlockIterator j = new bottomupBlockIterator(fblock.getBody().getHead());
    for(j.init(); !j.end(); j.next()){
      b = j.getBlock();
      if (b.Opcode() == Xcode.OMP_PRAGMA &&
	  ((PragmaBlock)b).getPragma().equals("THREADPRIVATE")){
	b.remove();
	prolog.insert(b);
      }
    }

    if(env.currentDefIsModule()){

      Xtype save_logical = Xtype.FlogicalType.copy();
      save_logical.setIsFsave(true);
      //save_logical.setIsFprivate(true);
      //Ident init_flag_var = env.declIdent("xmpf_init_flag", save_logical);
      BlockList body= env.getCurrentDef().getBlock().getBody();
      Ident init_flag_var = body.declLocalIdent("xmpf_init_flag_"+env.currentDefName(), save_logical,
      						StorageClass.FSAVE,
      						Xcons.List(Xcode.F_VALUE, Xcons.FlogicalConstant(false)));

      // NOTE: it is guaranteed that no threadprivate exists.

      Vector<XMPmodule> modules = env.getModules();
      for (int m = modules.size() - 1; m >= 0; m--){
	Ident f = env.declIdent("xmpf_traverse_module_"+modules.get(m).getModuleName(),
				Xtype.FsubroutineType);
	prolog.insert(f.callSubroutine(Xcons.List()));
      }

      prolog.insert(Bcons.IF(init_flag_var.Ref(), Xcons.List(Xcode.RETURN_STATEMENT), null));
      prolog.add(Xcons.Set(init_flag_var.Ref(),
			   Xcons.FlogicalConstant(true)));

      // Ident prolog_f = 
      // 	env.declIdent(env.currentDefName()+"_xmpf_module_init_",
      // 		      Xtype.FsubroutineType);
      // XobjectDef prolog_def = 
      // 	XobjectDef.Func(prolog_f, null, null, prolog.toXobject());
      // env.getCurrentDef().getDef().getChildren().addFirst(prolog_def);
      // prolog_def.setParent(env.getCurrentDef().getDef());

      Ident prolog_f = 
      	env.declIdent("xmpf_traverse_module_"+env.currentDefName(),
		      Xtype.FsubroutineType);
      XobjString modName = Xcons.Symbol(Xcode.IDENT, env.getCurrentDef().getDef().getName());
      Xobject decls = Xcons.List(Xcons.List(Xcode.F_USE_DECL, modName));
      XobjectDef prolog_def = 
      	XobjectDef.Func(prolog_f, null, decls, prolog.toXobject());
      env.getEnv().add(prolog_def);

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
    epilog.add(Xcons.StatementLabel(XMP.epilog_label_f));
    epilog.add(Xcons.List(Xcode.F_CONTINUE_STATEMENT));
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
    case LOCAL_ALIAS:
    case COARRAY:
    case SAVE_DESC:
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
    case TEMPLATE_FIX:
      return translateTemplateFix(pb, info);
    case ARRAY:
      // should not reaach here.

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
	env.declInternIdent(XMP.loop_sched_f,Xtype.FsubroutineType);
      Xobject args = Xcons.List(lb_var.Ref(), ub_var.Ref(), step_var.Ref(),
				Xcons.IntConstant(k),
				on_ref.getDescId().Ref());
      entry_bb.add(schd_f.callSubroutine(args));

      for_block.setLowerBound(lb_var.Ref());
      for_block.setUpperBound(ub_var.Ref());

      XMPtemplate t = on_ref.getTemplate();
      int t_idx = on_ref.getLoopOnIndex(k);
      if (for_block.getStep().isOneConstant() && (t.getDistMannerAt(t_idx) != XMPtemplate.CYCLIC ||
						  t.getDistArgAt(t_idx) == null ||
						  t.getDistArgAt(t_idx).isOneConstant())){
	for_block.setStep(Xcons.IntConstant(1));
      }
      else {
	for_block.setStep(step_var.Ref());
      }
      
      if(isVarUsed(for_block.getBody(),org_loop_ind_var)){
	// if global variable is used in this block, convert local to global
	Ident l2g_f = 
	  env.declInternIdent(XMP.l2g_f,Xtype.FsubroutineType);
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

    Ident f = env.declInternIdent(XMP.ref_dealloc_f, Xtype.FsubroutineType);
    ret_body.add(f.callSubroutine(Xcons.List(on_ref.getDescId())));

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

    f = env.declInternIdent(XMP.reflect_f,Xtype.FsubroutineType);

    Vector<XMParray> reflectArrays = info.getReflectArrays();
    for(XMParray a: reflectArrays){

      for (int i = 0; i < info.widthList.size(); i++){
	  g = env.declInternIdent(XMP.set_reflect_f,Xtype.FsubroutineType);
	  XMPdimInfo w = info.widthList.get(i);

	  // Here the stride means the periodic flag.
	  // check wheter the shadow is full.
	  if (w.getStride().getInt() == 1 && a.isFullShadow(i)){
	    XMP.error("Periodic reflect cannot be specified for a dimension with full shadow.");
	  }

	  Xobject args = Xcons.List(a.getDescId().Ref(), Xcons.IntConstant(i),
				    w.getLower(), w.getUpper(), w.getStride());
	  bb.add(g.callSubroutine(args));
      }

      if (info.getAsyncId() != null){
	  h = env.declInternIdent(XMP.reflect_async_f,Xtype.FsubroutineType);
	  bb.add(h.callSubroutine(Xcons.List(a.getDescId().Ref(), info.getAsyncId())));
      }
      else {
	  bb.add(f.callSubroutine(Xcons.List(a.getDescId().Ref())));
      }
    }

    return b;
  }

  private Block translateBarrier(PragmaBlock pb, XMPinfo info) {

    //Block b = Bcons.emptyBlock();
    //BasicBlock bb = b.getBasicBlock();

    BlockList ret_body = Bcons.emptyBody();

    XMPobjectsRef on_ref = info.getOnRef();
    Xobject on_ref_arg;

    Ident xmp_null = env.findVarIdent("XMP_NULL", pb);
    if (xmp_null == null){
	xmp_null = env.declObjectId("XMP_NULL", null,
				    Xcons.Cast(Xtype.voidPtrType, Xcons.IntConstant(0)));
    }

    if (on_ref != null){
      ret_body.add(on_ref.buildConstructor(env));
      on_ref_arg = on_ref.getDescId().Ref();
    }
    else on_ref_arg = xmp_null;

    Ident f = env.declInternIdent(XMP.barrier_f,
				  Xtype.FsubroutineType);
    ret_body.add(f.callSubroutine(Xcons.List(on_ref_arg)));

    if (on_ref != null){
      Ident g = env.declInternIdent(XMP.ref_dealloc_f, Xtype.FsubroutineType);
      ret_body.add(g.callSubroutine(Xcons.List(on_ref.getDescId())));
    }

    return Bcons.COMPOUND(ret_body);
  }

  private Block translateReduction(PragmaBlock pb, XMPinfo info){

    //Block b = Bcons.emptyBlock();
    //BasicBlock bb = b.getBasicBlock();

    BlockList ret_body = Bcons.emptyBody();

    XMPobjectsRef on_ref = info.getOnRef();
    Xobject on_ref_arg;

    Ident xmp_null = env.findVarIdent("XMP_NULL", pb);
    if (xmp_null == null){
	xmp_null = env.declObjectId("XMP_NULL", null,
				    Xcons.Cast(Xtype.voidPtrType, Xcons.IntConstant(0)));
    }

    if (info.getAsyncId() != null){
      Xobject arg = Xcons.List(info.getAsyncId());
      Ident g = env.declInternIdent(XMP.init_async_f, Xtype.FsubroutineType);
      ret_body.add(g.callSubroutine(arg));
    }

    if (on_ref != null){
      if (info.pragma != XMPpragma.LOOP){
	ret_body.add(on_ref.buildConstructor(env));
	on_ref_arg = on_ref.getDescId().Ref();
      }
      else {
	XMPobjectsRef on_ref_copy = on_ref.convertLoopToReduction();
	ret_body.add(on_ref_copy.buildConstructor(env));
	on_ref_arg = on_ref_copy.getDescId().Ref();
      }
    }
    else on_ref_arg = xmp_null;

    // object size
    int op = info.getReductionOp();
    // boolean reduce_minus = false;
    // if (op == XMP.REDUCE_MINUS){
    //   op = XMP.REDUCE_SUM;
    //   reduce_minus = true;
    // }

    Ident f = env.declInternIdent(XMP.reduction_f, Xtype.FsubroutineType);
    Ident f2 = env.declInternIdent(XMP.reduction_loc_f, Xtype.FsubroutineType);

    //for(Ident id: info.getReductionVars()){
    for (int i = 0; i < info.getReductionVars().size(); i++){

      Ident id = info.getReductionVars().elementAt(i);
      Vector<Ident> pos_vars = info.getReductionPosVars().elementAt(i);

      Xtype type = id.Type();
      Xobject size_expr = Xcons.IntConstant(1);
      if(type.isFarray()){
	for(Xobject s: type.getFarraySizeExpr()){
          Xobject length = Xcons.binaryOp(Xcode.PLUS_EXPR, Xcons.IntConstant(1),
                                          Xcons.binaryOp(Xcode.MINUS_EXPR, s.getArg(1), s.getArg(0)));
	  size_expr = Xcons.binaryOp(Xcode.MUL_EXPR,size_expr, length);
	}
	type = type.getRef();
      }
      if(!type.isBasic()){
	XMP.fatal("reduction for non-basic type ="+type);
      }

      Xobject args = Xcons.List(id.Ref(),size_expr,
				XMP.typeIntConstant(type),
				Xcons.IntConstant(op),
				on_ref_arg);

      args.add(Xcons.IntConstant(pos_vars.size()));

      int j = 0;
      for (Ident pos: pos_vars){
	Xobject args2 = Xcons.List(Xcons.IntConstant(j), pos.Ref(), XMP.typeIntConstant(pos.Type()));
	ret_body.add(f2.callSubroutine(args2));
	j++;
      }

      // if (reduce_minus){
      // 	ret_body.add(Xcons.Set(id.Ref(), Xcons.unaryOp(Xcode.UNARY_MINUS_EXPR, id.Ref())));
      // }

      ret_body.add(f.callSubroutine(args));
    }

    if (info.getAsyncId() != null){
      Xobject arg = Xcons.List(info.getAsyncId());
      Ident g = env.declInternIdent(XMP.start_async_f, Xtype.FsubroutineType);
      ret_body.add(g.callSubroutine(arg));
    }

    if (on_ref != null){
      Ident g = env.declInternIdent(XMP.ref_dealloc_f, Xtype.FsubroutineType);
      ret_body.add(g.callSubroutine(Xcons.List(on_ref.getDescId())));
    }

    return Bcons.COMPOUND(ret_body);
  }

  private Block translateBcast(PragmaBlock pb, XMPinfo info){

    //Block b = Bcons.emptyBlock();
    //BasicBlock bb = b.getBasicBlock();

    BlockList ret_body = Bcons.emptyBody();

    Ident xmp_null = env.findVarIdent("XMP_NULL", pb);
    if (xmp_null == null){
	xmp_null = env.declObjectId("XMP_NULL", null,
				    Xcons.Cast(Xtype.voidPtrType, Xcons.IntConstant(0)));
    }

    if (info.getAsyncId() != null){
      Xobject arg = Xcons.List(info.getAsyncId());
      Ident g = env.declInternIdent(XMP.init_async_f, Xtype.FsubroutineType);
      ret_body.add(g.callSubroutine(arg));
    }

    XMPobjectsRef from_ref = info.getBcastFrom();
    Xobject from_ref_arg;
    if (from_ref != null){
      ret_body.add(from_ref.buildConstructor(env));
      from_ref_arg = from_ref.getDescId().Ref();
    }
    else from_ref_arg = xmp_null;

    XMPobjectsRef on_ref = info.getOnRef();
    Xobject on_ref_arg;
    if (on_ref != null){
      ret_body.add(on_ref.buildConstructor(env));
      on_ref_arg = on_ref.getDescId().Ref();
    }
    else on_ref_arg = xmp_null;

    Ident f = env.declInternIdent(XMP.bcast_f, Xtype.FsubroutineType);

    for(Ident id: info.getInfoVarIdents()){
      Xtype type = id.Type();
      Xobject size_expr = Xcons.IntConstant(1);

      if (type.isFarray()){

	if (type.isFassumedSize()){
	  XMP.fatal("assumed-size array cannot be the target of bcast.");
	}

	if (!type.isFassumedShape() && !type.isFallocatable()){
	  for (Xobject s: type.getFarraySizeExpr()){
	    //size_expr = Xcons.binaryOp(Xcode.MUL_EXPR,size_expr,s);

	    Xobject size;
	    if (s.Opcode() == Xcode.F_INDEX_RANGE){
	      Xobject lb = s.getArg(0);
	      Xobject ub = s.getArg(1);
	      size = Xcons.binaryOp(Xcode.MINUS_EXPR, ub, lb);
	      size = Xcons.binaryOp(Xcode.PLUS_EXPR, size, Xcons.IntConstant(1));
	    }
	    else {
	      size = s;
	    }

	    size_expr = Xcons.binaryOp(Xcode.MUL_EXPR, size_expr, size);
	  }

	}
	else {
	  Ident size_func = env.declIntrinsicIdent("size", Xtype.FintFunctionType);
	  size_expr = size_func.Call(Xcons.List(id.Ref()));
	}

	type = type.getRef();
      }

      if(!type.isBasic()){
	XMP.fatal("bcast for non-basic type ="+type);
      }

      Xobject args = Xcons.List(id.Ref(), size_expr,
				XMP.typeIntConstant(type),
				from_ref_arg,
				on_ref_arg);

      ret_body.add(f.callSubroutine(args));

    }

    if (info.getAsyncId() != null){
      Xobject arg = Xcons.List(info.getAsyncId());
      Ident g = env.declInternIdent(XMP.start_async_f, Xtype.FsubroutineType);
      ret_body.add(g.callSubroutine(arg));
    }

    if (on_ref != null){
      Ident g = env.declInternIdent(XMP.ref_dealloc_f, Xtype.FsubroutineType);
      ret_body.add(g.callSubroutine(Xcons.List(on_ref.getDescId())));
    }

    if (from_ref != null){
      Ident g = env.declInternIdent(XMP.ref_dealloc_f, Xtype.FsubroutineType);
      ret_body.add(g.callSubroutine(Xcons.List(from_ref.getDescId())));
    }

    return Bcons.COMPOUND(ret_body);
  }

  private Block translateWaitAsync(PragmaBlock pb, XMPinfo info){
    //Block b = Bcons.emptyBlock();
    //BasicBlock bb = b.getBasicBlock();

    BlockList ret_body = Bcons.emptyBody();

    XMPobjectsRef on_ref = info.getOnRef();
    Xobject on_ref_arg;
    if (on_ref != null){
      ret_body.add(on_ref.buildConstructor(env));
      on_ref_arg = on_ref.getDescId().Ref();
    }
    else on_ref_arg = env.getNullIdent(pb).Ref();

    Ident f = env.declInternIdent(XMP.wait_async_f, Xtype.FsubroutineType);

    for (Xobject i: info.waitAsyncIds){
      ret_body.add(f.callSubroutine(Xcons.List(i, on_ref_arg)));
    }

    return Bcons.COMPOUND(ret_body);
  }

  private Block translateTask(PragmaBlock pb, XMPinfo info){
    BlockList ret_body = Bcons.emptyBody();
    XMPobjectsRef on_ref = info.getOnRef();

    // when '*' = the executing node set is specified
    if (on_ref == null) return Bcons.COMPOUND(pb.getBody());

    Block parentBlock = pb.getParentBlock();
    boolean tasksFlag = false;
    if (parentBlock != null && parentBlock instanceof PragmaBlock){
      XMPinfo parentInfo = (XMPinfo)parentBlock.getProp(XMP.prop);
      if (parentInfo != null && parentInfo.pragma == XMPpragma.TASKS) tasksFlag = true;
    }

    Block b = on_ref.buildConstructor(env);
    BasicBlock bb = b.getBasicBlock();

    Ident taskNodesDescId = env.declObjectId(XMP.genSym("XMP_TASK_NODES"), pb);

    Ident f;
    if (!info.isNocomm()){
      f = env.declInternIdent(XMP.create_task_nodes_f, Xtype.FsubroutineType);
      bb.add(f.callSubroutine(Xcons.List(taskNodesDescId, on_ref.getDescId().Ref())));
    }

    Ident g1 = env.declInternIdent(XMP.nodes_dealloc_f, Xtype.FsubroutineType);
    Ident g2 = env.declInternIdent(XMP.ref_dealloc_f, Xtype.FsubroutineType);

    if (tasksFlag){
      //parentBlock.insert(on_ref.buildConstructor(env));
      parentBlock.insert(b);
      parentBlock.add(g1.callSubroutine(Xcons.List(taskNodesDescId)));
      parentBlock.add(g2.callSubroutine(Xcons.List(on_ref.getDescId())));
    }
    else {
      //ret_body.add(on_ref.buildConstructor(env));
      ret_body.add(b);
    }

    Xobject cond = null;
    if (!info.isNocomm()){
      f = env.declInternIdent(XMP.test_task_on_f,
			      Xtype.FlogicalFunctionType);
      //Xobject cond = f.Call(Xcons.List(on_ref.getDescId().Ref()));
      cond = f.Call(Xcons.List(taskNodesDescId.Ref()));
    }
    else {
      f = env.declInternIdent(XMP.test_task_nocomm_f, Xtype.FlogicalFunctionType);
      cond = f.Call(Xcons.List(on_ref.getDescId()));
    }

    ret_body.add(Bcons.IF(cond,Bcons.COMPOUND(pb.getBody()),null));
      
    if (!info.isNocomm()){
      f = env.declInternIdent(XMP.end_task_f,Xtype.FsubroutineType);
      pb.getBody().add(f.Call(Xcons.List()));
    }

    if (!tasksFlag){
      if (!info.isNocomm()) ret_body.add(g1.callSubroutine(Xcons.List(taskNodesDescId)));
      ret_body.add(g2.callSubroutine(Xcons.List(on_ref.getDescId())));
    }

    return Bcons.COMPOUND(ret_body);
  }

  private Block translateTasks(PragmaBlock pb, XMPinfo i) {
    //XMP.fatal("translateTasks");
    //return null;
    return Bcons.COMPOUND(pb.getBody());
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

    if (i.getAsyncId() != null){
      Xobject arg = Xcons.List(i.getAsyncId());
      Ident g = env.declInternIdent(XMP.init_async_f, Xtype.FsubroutineType);
      bb.add(g.callSubroutine(arg));
    }

    Ident f = env.declInternIdent(XMP.gmove_do_f, Xtype.FsubroutineType);
    Xobject args = Xcons.List(left_desc.Ref(), right_desc.Ref(),
			      Xcons.IntConstant(GMOVE_COLL));
    bb.add(f.callSubroutine(args));

    Ident d = env.declInternIdent(XMP.gmove_dealloc_f, Xtype.FsubroutineType);
    Xobject args_l = Xcons.List(left_desc.Ref());
    bb.add(d.callSubroutine(args_l));
    Xobject args_r = Xcons.List(right_desc.Ref());
    bb.add(d.callSubroutine(args_r));
	    
    if (i.getAsyncId() != null){
      Xobject arg = Xcons.List(i.getAsyncId());
      Ident g = env.declInternIdent(XMP.start_async_f, Xtype.FsubroutineType);
      bb.add(g.callSubroutine(arg));
    }

    return b;
  }

  private Ident buildGmoveDesc(Xobject x, BasicBlock bb, PragmaBlock pb) {
    Ident descId = env.declObjectId(XMP.genSym("gmv"),pb);
    Ident f; 
    Xobject args;
    XMParray array = null;

    switch(x.Opcode()){
    case F_ARRAY_REF:
      Xobject a = x.getArg(0).getArg(0);
      array = (XMParray)a.getProp(XMP.RWprotected);
      if(array != null){
	f = env.declInternIdent(XMP.gmove_g_alloc_f, Xtype.FsubroutineType);
	args = Xcons.List(descId.Ref(), array.getDescId().Ref());
	bb.add(f.callSubroutine(args));
	
	// System.out.println("idx args="+x.getArg(1));
	f = env.declInternIdent(XMP.gmove_g_dim_info_f, Xtype.FsubroutineType);
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
	f = env.declInternIdent(XMP.gmove_l_alloc_f, Xtype.FsubroutineType);
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
	
	f = env.declInternIdent(XMP.gmove_l_dim_info_f, Xtype.FsubroutineType);
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

    case VAR:

      array = (XMParray)x.getProp(XMP.RWprotected);

      if (array != null){
	f = env.declInternIdent(XMP.gmove_g_alloc_f, Xtype.FsubroutineType);
	args = Xcons.List(descId.Ref(), array.getDescId().Ref());
	bb.add(f.callSubroutine(args));
	
	f = env.declInternIdent(XMP.gmove_g_dim_info_f, Xtype.FsubroutineType);
 	for (int i = 0; i < array.getDim(); i++){
	  args = Xcons.List(descId.Ref(),Xcons.IntConstant(i),
			    Xcons.IntConstant(GMOVE_ALL),
			    Xcons.IntConstant(0),
			    Xcons.IntConstant(0),Xcons.IntConstant(0));
	  bb.add(f.callSubroutine(args));
	}
      }
      else {
	f = env.declInternIdent(XMP.gmove_l_alloc_f, Xtype.FsubroutineType);
	args = Xcons.List(descId.Ref(), x, Xcons.IntConstant(0));
	bb.add(f.callSubroutine(args));
      }

      break;

    default:
      XMP.errorAt(pb,"gmove must be followed by simple assignment");
    }
    
    return descId;
  }

  private Block translateTemplateFix(PragmaBlock pb, XMPinfo info){

    Ident f; 
    Xobject args;
    Block b = Bcons.emptyBlock();
    BasicBlock bb = b.getBasicBlock();

    XMPtemplate tObject = info.getTemplate();

    XobjList sizeList = info.getSizeList();
    XobjList distList = info.getDistList();

    if (sizeList == null || sizeList.isEmptyList()){
      sizeList = Xcons.List();
      for (int i = 0; i < tObject.getDim(); i++){
	sizeList.add(Xcons.List(tObject.getLowerAt(i), tObject.getUpperAt(i)));
      }
    }
    else {
      // check rank matching
      if (sizeList.Nargs() != tObject.getDim())
	XMP.fatal(pb, "the number of <template-spec> is different from that in the declaration");
    }

    if (distList == null || distList.isEmptyList()){
      distList = Xcons.List();
      for (int i = 0; i < tObject.getDim(); i++){
	distList.add(Xcons.List(Xcons.IntConstant(tObject.getDistMannerAt(i)),
				tObject.getDistArgAt(i)));
      }
    }
    else {
      // check dist-format matching
      for (int i = 0; i < tObject.getDim(); i++){
	Xobject dist = distList.getArg(i);
	int dist_decl = tObject.getDistMannerAt(i);
	if (dist == null){
	  if (dist_decl != XMPtemplate.DUPLICATION) XMP.fatal(pb, "<dist-format> not match");
	}
	else {
	  String dist_fmt = dist.getArg(0).getString();
	  if (dist_fmt.equalsIgnoreCase("BLOCK")){
	    if (dist_decl != XMPtemplate.BLOCK) XMP.fatal(pb, "<dist-format> not match");
	  }
	  else if(dist_fmt.equalsIgnoreCase("CYCLIC")){
	    if (dist_decl != XMPtemplate.CYCLIC)
	      XMP.fatal(pb, "<dist-format> not match");
	  }
	  else if(dist_fmt.equalsIgnoreCase("GBLOCK")){
	    if (dist_decl != XMPtemplate.GBLOCK) XMP.fatal(pb, "<dist-format> not match");
	    if (tObject.getDistArgAt(i) != null) XMP.fatal(pb, "<dist-format> not match");
	  }
	}
      }
    }

    /* template size */
    f = env.declInternIdent(XMP.template_dim_info_f, Xtype.FsubroutineType);
    for (int i = 0; i < tObject.getDim(); i++){

      Xobject dist = distList.getArg(i);

      int distManner = XMPtemplate.DUPLICATION;
      if (dist == null)
	distManner = XMPtemplate.DUPLICATION;
      else if(dist.getArg(0).isIntConstant())
	distManner = dist.getArg(0).getInt();
      else {
	String dist_fmt = dist.getArg(0).getString();
	if (dist_fmt.equalsIgnoreCase("BLOCK"))
	  distManner = XMPtemplate.BLOCK;
	else if(dist_fmt.equalsIgnoreCase("CYCLIC"))
	  distManner = XMPtemplate.CYCLIC;
	else if(dist_fmt.equalsIgnoreCase("GBLOCK"))
	  distManner = XMPtemplate.GBLOCK;
	else {
	  XMP.fatal(pb, "unknown distribution format," + dist_fmt);
	}
      }

      Xobject distArg;
      if (dist != null && dist.getArg(1) != null)
	distArg = dist.getArg(1);
      else
	distArg = Xcons.IntConstant(0);

      args = Xcons.List(tObject.getDescId().Ref(), Xcons.IntConstant(i),
			sizeList.getArg(i).getArg(0), sizeList.getArg(i).getArg(1),
			Xcons.IntConstant(distManner),
			distArg);
      bb.add(f.callSubroutine(args));
    }

    /* init */
    f = env.declInternIdent(XMP.template_init_f, Xtype.FsubroutineType);
    bb.add(f.callSubroutine(Xcons.List(tObject.getDescId().Ref(),
				       tObject.getOntoNodes().getDescId().Ref())));

    return b;
  }

}
