/* 
 * $TSUKUBA_Release: Omni XMP Compiler $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
package exc.xmpF;

import exc.object.*;
import exc.block.*;
import java.util.*;
import static xcodeml.util.XmLog.fatal;

/**
 * pass2: check and write variables
 */
public class XMPrewriteExpr
{
  private XMPenv  env;

  public XMPrewriteExpr(){ }

  public void run(FuncDefBlock def, XMPenv env) {
    this.env = env;
    env.setCurrentDef(def);

    XMP.debug("pass2:");
    FunctionBlock fb = def.getBlock();
    if (fb == null) return;

    // rewrite return statements
    BlockIterator iter5 = new topdownBlockIterator(fb);
    for (iter5.init(); !iter5.end(); iter5.next()){
      if (iter5.getBlock().Opcode() == Xcode.RETURN_STATEMENT){
	Block b = Bcons.GOTO(Xcons.StringConstant(XMP.epilog_label_f));
	iter5.setBlock(b);
      }
    }

    // rewrite allocate, deallocate, and stop statements
    BasicBlockIterator iter3 = new BasicBlockIterator(fb);
    for (iter3.init(); !iter3.end(); iter3.next()){
      StatementIterator iter4 = iter3.getBasicBlock().statements();
      while (iter4.hasNext()){
	Statement st = iter4.next();
	Xobject x = st.getExpr();
	if (x == null)  continue;
	switch (x.Opcode()) {
	case F_ALLOCATE_STATEMENT:
	  {
            // must be performed before rewriteExpr
	    Iterator<Xobject> y = ((XobjList)x.getArg(1)).iterator();
	    while (y.hasNext()){
	      XobjList alloc = (XobjList)y.next();
	      Xobject obj = alloc.getArg(0);

	      while (obj.Opcode() == Xcode.MEMBER_REF ||
		     obj.Opcode() == Xcode.F_ARRAY_REF ||
		     obj.Opcode() == Xcode.F_VAR_REF)
		  obj = obj.getArg(0).getArg(0);

	      Ident id = env.findVarIdent(obj.getName(), fb);
	      if (id == null) break;
	      XMParray array = XMParray.getArray(id);
	      if (array == null) break;

	      array.rewriteAllocate(alloc, st, fb, env);
	    }
	    break;
	  }
	case F_DEALLOCATE_STATEMENT:
	  {
            // must be performed before rewriteExpr
	    Iterator<Xobject> y = ((XobjList)x.getArg(1)).iterator();
	    while (y.hasNext()){
	      XobjList dealloc = (XobjList)y.next();
	      Xobject obj = dealloc.getArg(0);
	      while (obj.Opcode() == Xcode.MEMBER_REF ||
		     obj.Opcode() == Xcode.F_ARRAY_REF ||
		     obj.Opcode() == Xcode.F_VAR_REF)
		obj = obj.getArg(0).getArg(0);
	      Ident id = env.findVarIdent(obj.getName(), fb);
	      if (id == null) break;
	      XMParray array = XMParray.getArray(id);
	      if (array == null) break;

	      array.rewriteDeallocate(dealloc, st, fb, env);
	    }
	    break;
	  }

	case F_STOP_STATEMENT:
	  {
	    Ident f = env.declInternIdent(XMP.finalize_all_f, Xtype.FsubroutineType);
	    Xobject call = f.callSubroutine();
	    st.insert(call);
	    break;
	  }

	case EXPR_STATEMENT: // subroutine call
	  insertSizeArray(st, fb);
	  break;

	}
      }
    }

    // rewrite expr
    BasicBlockExprIterator iter = new BasicBlockExprIterator(fb);
    for (iter.init(); !iter.end(); iter.next()) {
      Xobject expr = iter.getExpr();
      if(expr != null)  rewriteExpr(expr,iter.getBasicBlock(),fb);
    }
    
    // rewrite OMP pragma
    topdownBlockIterator iter2 = new topdownBlockIterator(fb);
    for (iter2.init(); !iter2.end(); iter2.next()){
      Block block = iter2.getBlock();
      if (block.Opcode() == Xcode.OMP_PRAGMA){
	Xobject clauses = ((PragmaBlock)block).getClauses();
	rewriteOmpClauses(clauses, (PragmaBlock)block, fb);
      }
    }

    // rewrite id_list, decles, parameters
    rewriteDecls(fb);
  }

  private void rewriteDecls(FunctionBlock funcBlock){
    Xobject decl_list = funcBlock.getBody().getDecls();
    Xobject id_list = funcBlock.getBody().getIdentList();
    Xtype f_type = funcBlock.getNameObj().Type();
    Xobject f_params = null;
    if(f_type != null && f_type.isFunction())
      f_params = f_type.getFuncParam();
    
    for(Xobject i: (XobjList) id_list){
      Ident id = (Ident)i;
      XMParray array = XMParray.getArray(id);
      if(array == null) continue;
      
      // write id
      String a_name = id.getName();
      String xmp_name = array.getLocalName();
      Xtype xmp_type = array.getLocalType();

      for(Xobject decl: (XobjList) decl_list){
	if(decl.Opcode() != Xcode.VAR_DECL) continue;
	Xobject decl_id = decl.getArg(0);
	if(decl_id.Opcode() == Xcode.IDENT && 
	   decl_id.getName().equals(a_name)){
	  decl.setArg(0,Xcons.Symbol(Xcode.IDENT,xmp_type,xmp_name));
	  break;
	}
      }
      if(f_params != null && id.getStorageClass() == StorageClass.FPARAM){
	// rewrite parameter
	for(Xobject param: (XobjList)f_params){
	  if(param.Opcode() == Xcode.IDENT &&
	     param.getName().equals(a_name)){
	    param.setName(xmp_name);
	    param.setType(xmp_type);
	  }
	}
      }
    }
  }

  /*
   * rewrite expression
   */
  private void rewriteExpr(Xobject expr, BasicBlock bb, Block block){

    bottomupXobjectIterator iter = new bottomupXobjectIterator(expr);
    for(iter.init(); !iter.end();iter.next()){
      Xobject x = iter.getXobject();
      if (x == null)  continue;
      if (x.Opcode() == null) continue;      // #060  see [Xmp-dev:4675]
      switch (x.Opcode()) {
      case VAR:
	{
	  if(x.getProp(XMP.RWprotected) != null) break;

	  Ident id = env.findVarIdent(x.getName(),block);
	  if(id == null) break;
	  XMParray array = XMParray.getArray(id);
	  if(array == null) break;
	  
	  // replace with local decl
	  Xobject var = Xcons.Symbol(Xcode.VAR,array.getLocalType(),
				                 array.getLocalName());
	  var.setProp(XMP.arrayProp,array);
	  iter.setXobject(var);
	  break;
	}
      case F_ARRAY_REF:
	{
	  Xobject a = x.getArg(0);
	  if(a.Opcode() != Xcode.F_VAR_REF)
	    XMP.fatal("not F_VAR_REF for F_ARRAY_REF");
	  a = a.getArg(0);
	  XMParray array = (XMParray) a.getProp(XMP.arrayProp);
	  if(array == null) break;

	  int dim_i = 0;
	  boolean no_leading_scalar_in_subscripts = true;
	  for(XobjArgs args = x.getArg(1).getArgs(); args != null;
	      args = args.nextArgs()){

	    // check subscripts
	    // if (array.isDistributed(dim_i) &&
	    // 	args.getArg().Opcode() == Xcode.F_ARRAY_INDEX){
	    //   no_leading_scalar_in_subscripts = false;
	    // }
	    // else if (array.isDistributed(dim_i) &&
	    // 	     args.getArg().Opcode() == Xcode.F_INDEX_RANGE &&
	    // 	     !no_leading_scalar_in_subscripts){
	    //   XMP.errorAt(block, "':' must not be lead by any int-expr in a subscript list of a global array.");
	    // }

	    Xobject index_calc = 
	      arrayIndexCalc(array,dim_i++,args.getArg(),bb,block);
	    if(index_calc != null) args.setArg(index_calc);
	  }
	  
	  if(array.isLinearized())
	    x.setArg(1,array.convertLinearIndex(x.getArg(1)));

	  break;
	}

        // XXX delete this
      case CO_ARRAY_REF:
	{
	  break;
	}

      case FUNCTION_CALL:
	{
	  String fname = x.getArg(0).getString();
	  if (fname.equalsIgnoreCase("xmp_desc_of")){

	    env.removeIdent(fname, block);

	    XMParray array = (XMParray) x.getArg(1).getArg(0).getProp(XMP.arrayProp);
	    if (array != null){
	      Xobject desc = array.getDescId();
	      iter.setXobject(desc);
	      break;
	    }

	    String objName = x.getArg(1).getArg(0).getString();
	    XMPobject obj = env.findXMPobject(objName, block);
	    if (obj != null) {
	      Xobject desc = obj.getDescId();
	      iter.setXobject(desc);
	      env.removeIdent(objName, block);
	      break;
	    }

	    XMP.errorAt(block, "xmp_desc_of applied to non-XMP object");
	    XMP.exitByError();

	    break;

	      //XMParray array = (XMParray) x.getArg(1).getArg(0).getProp(XMP.arrayProp);
	      // if (array == null){
	      // 	XMP.errorAt(block,"xmp_desc_of applied to non-global data");
	      // 	XMP.exitByError();
	      // }
	      // Xobject desc = array.getDescId();
	      // iter.setXobject(desc);

	  }
	  else if (fname.equalsIgnoreCase("xmp_transpose")){
	    XMParray array0 = (XMParray) x.getArg(1).getArg(0).getProp(XMP.arrayProp);
	    XMParray array1 = (XMParray) x.getArg(1).getArg(1).getProp(XMP.arrayProp);
	    if (array0 == null || array1 == null){
	      XMP.errorAt(block,"wrong argument of xmp_transpose");
	      XMP.exitByError();
	    }
	    x.getArg(1).setArg(0, array0.getDescId());
	    x.getArg(1).setArg(1, array1.getDescId());
	  }
	  break;
	}
      }
    }
  }

  /*
   * rewrite Pragma
   */
  private void rewriteOmpClauses(Xobject expr, PragmaBlock pragmaBlock, Block block){

    boolean private_done = false;
	  
    bottomupXobjectIterator iter = new bottomupXobjectIterator(expr);
    
    for (iter.init(); !iter.end();iter.next()){
    	
      Xobject x = iter.getXobject();
      if (x == null)  continue;
      
      if (x.Opcode() == Xcode.VAR){

	  if (x.getProp(XMP.RWprotected) != null) break;

	  Ident id = env.findVarIdent(x.getName(),block);
	  if (id == null) break;
	  
	  XMParray array = XMParray.getArray(id);

	  if (array != null){
	      // replace with local decl
	      Xobject var = Xcons.Symbol(Xcode.VAR,array.getLocalType(),
					 array.getLocalName());
	      var.setProp(XMP.arrayProp,array);
	      iter.setXobject(var);
	  }
	  /*	  else {
	      Ident local_loop_var = null;
	      // find the loop variable x
	      for (Block b = pragmaBlock.getParentBlock(); b != null; b = b.getParentBlock()){
		  XMPinfo info = (XMPinfo)b.getProp(XMP.prop);
		  if (info == null) continue;
		  if (info.pragma != XMPpragma.LOOP) continue;
		  for (int k = 0; k < info.getLoopDim(); k++){
		      if (x.getName().equals(info.getLoopVar(k).getName())){
			  local_loop_var = info.getLoopDimInfo(k).getLoopLocalVar();
			  break;
		      }
		  }
		  if (local_loop_var != null) break;
	      }
	      if (local_loop_var != null) iter.setXobject(local_loop_var.Ref());
	      }*/

      }
      else if (x.Opcode() == Xcode.LIST){
	  if (x.left() != null && x.left().Opcode() == Xcode.STRING &&
	      x.left().getString().equals("DATA_PRIVATE")){

	      private_done = true;

	      if (!pragmaBlock.getPragma().equals("FOR")) continue;

	      XobjList itemList = (XobjList)x.right();

	      // find loop variable
	      Xobject loop_var = null;
	      BasicBlockIterator i = new BasicBlockIterator(pragmaBlock.getBody());
	      for (Block b = pragmaBlock.getBody().getHead();
		   b != null;
		   b = b.getNext()){
		  if (b.Opcode() == Xcode.F_DO_STATEMENT){
		      loop_var = ((FdoBlock)b).getInductionVar();
		  }
	      }
	      if (loop_var == null) continue;

	      // check if the clause has contained the loop variable
	      boolean flag = false;
	      Iterator<Xobject> j = itemList.iterator();
	      while (j.hasNext()){
		  Xobject item = j.next();
		  if (item.getName().equals(loop_var.getName())){
		      flag = true;
		  }
	      }

	      // add the loop variable to the clause
	      if (!flag){
		  itemList.add(loop_var);
	      }

	  }
      }

    }

    if (!private_done){

      if (!pragmaBlock.getPragma().equals("FOR")) return;

      // find loop variable
      Xobject loop_var = null;
      BasicBlockIterator i = new BasicBlockIterator(pragmaBlock.getBody());
      for (Block b = pragmaBlock.getBody().getHead();
	   b != null;
	   b = b.getNext()){
	if (b.Opcode() == Xcode.F_DO_STATEMENT){
	  loop_var = ((FdoBlock)b).getInductionVar();
	}
      }
      if (loop_var == null) return;
      
      // add the loop variable to the clause
      Xobject thread_private = Xcons.List(Xcons.StringConstant("DATA_PRIVATE"),
					  Xcons.List(loop_var));
      pragmaBlock.addClauses(thread_private);

    }

  }

  Xobject localIndexOffset;

  Xobject convertLocalIndex(Xobject v, int dim_i, XMParray a, 
			    BasicBlock bb, Block block){
    localIndexOffset = null;
    int loop_idx = -1;
    XMPobjectsRef on_ref = null;
    Ident local_loop_var = null;

    // find the loop variable v
    for(Block b = bb.getParent(); b != null; b = b.getParentBlock()){
      XMPinfo info = (XMPinfo)b.getProp(XMP.prop);
      if(info == null) continue;
      if(info.pragma != XMPpragma.LOOP) continue;

      for(int k = 0; k < info.getLoopDim(); k++){
	if(v.equals(info.getLoopVar(k))){
	  loop_idx = k;
	  on_ref = info.getOnRef();
	  local_loop_var = info.getLoopDimInfo(k).getLoopLocalVar();
	  break;
	}
      }
      if(loop_idx >= 0) break;
    }

    if(XMP.debugFlag) 
      System.out.println("convertLocalIndex v="+v+" loop_idx="+loop_idx);

    if(loop_idx < 0 || local_loop_var == null) return null; // not found

    if(on_ref.getRefObject().getKind() != XMPobject.TEMPLATE) 
      return null;
    
    XMPtemplate on_tmpl = on_ref.getTemplate();
    XMPtemplate a_tmpl = a.getAlignTemplate();
    if(on_tmpl != a_tmpl && !on_tmpl.getName().equals(a_tmpl.getName())){
	return null; // different template
    }
    
    if(XMP.debugFlag) System.out.println("same template");

    int a_tmpl_idx = a.getAlignSubscriptIndexAt(dim_i);
    int on_tmpl_idx = on_ref.getLoopOnIndex(loop_idx);

    if(XMP.debugFlag) 
      System.out.println("template index a_tmpl_idx="+a_tmpl_idx+
			 ", on_tmp_idx="+on_tmpl_idx);

    if(a_tmpl_idx != on_tmpl_idx) return null;

    Xobject off1;
    int lshadow = a.getShadowLeft(dim_i);
    if (lshadow != 0){
	off1 = Xcons.IntConstant(lshadow);
    }
    else {
	off1 = null;
    }

    Xobject off2 = a.getAlignSubscriptOffsetAt(dim_i);
    //Xobject off3 = on_ref.getLoopOffset(loop_idx);
    Xobject off3 = on_ref.getLoopOffset(on_ref.getLoopOnIndex(loop_idx));
    Xobject off4 = null;
    if (off3 != null){
      if (off2 != null)
	off4 = Xcons.binaryOp(Xcode.MINUS_EXPR, off2, off3);
      else 
	off4 = Xcons.unaryOp(Xcode.UNARY_MINUS_EXPR, off3);
    }
    else if (off2 != null)
      off4 = off2;
    else
      off4 = Xcons.IntConstant(0);
	
    if (off1 == null)
      localIndexOffset = off4;
    else
      localIndexOffset = Xcons.binaryOp(Xcode.PLUS_EXPR, off1, off4);

    //Xobject off1 = a.getAlignSubscriptOffsetAt(dim_i);

    // Xobject off2 = on_ref.getLoopOffset(loop_idx);
    // if(off1 == null) localIndexOffset = off2;
    // else if(off2 == null) localIndexOffset = off1;
    // else localIndexOffset = 
    //         Xcons.binaryOp(Xcode.PLUS_EXPR,off1,off2);
    // localIndexOffset = off1;

    if(XMP.debugFlag) 
      System.out.println("check template v="+local_loop_var
			 +" off="+localIndexOffset);

    return local_loop_var.Ref();
  }

  Xobject arrayIndexCalc(XMParray a, int dim_i, Xobject i, 
			 BasicBlock bb, Block block){
    switch(i.Opcode()){
    case F_ARRAY_INDEX:
      // if not distributed, do nothing
      if(!a.isDistributed(dim_i)){
	// return null;
	  Xobject x = a.convertOffset(dim_i);
	  if (x != null)
	      i.setArg(0,Xcons.binaryOp(Xcode.MINUS_EXPR, i.getArg(0), x));
//  	i.setArg(0,Xcons.binaryOp(Xcode.MINUS_EXPR,
//  				  i.getArg(0),
//  				  a.convertOffset(dim_i)));
	return i;
      }

      if (!a.isFullShadow(dim_i)){

	Xobject e = i.getArg(0);
	Xobject x = null;

	if (e.isVariable()){
	  x = convertLocalIndex(e, dim_i, a, bb, block);
	  if (localIndexOffset != null){
	    x = Xcons.binaryOp(Xcode.PLUS_EXPR, x, localIndexOffset);
	  }
	}
	else {
	  x = e.copy();
	  int cnt = convertLocalIndexInExpression(x, dim_i, a, bb, block, 0);
	  if (cnt != 1) x = null;
	}

	if (x != null){
	  Xobject y = a.convertOffset(dim_i);
	  if (y != null)
	    x = Xcons.binaryOp(Xcode.MINUS_EXPR, x, y);
	  i.setArg(0, x);
	  return i;
	}

	// // check this expression is ver+offset
	// Xobject e = i.getArg(0);
	// // we need normalize?
	// Xobject v = null;
	// Xobject offset = null;
	// if(e.isVariable()){
	//   v = e;
	// } else {
	//   switch(e.Opcode()){
	//   case PLUS_EXPR:
	//     if(e.left().isVariable()){
	//       v = e.left();
	//       offset = e.right();
	//     } else if(e.right().isVariable()){
	//       v = e.right();
	//       offset = e.left();
	//     }
	//     break;
	//   case MINUS_EXPR:
	//     if(e.left().isVariable()){
	//       v = e.left();
	//       offset = Xcons.unaryOp(Xcode.UNARY_MINUS_EXPR,e.right());
	//     }
	//     break;
	//   }
	// }

	// if (v != null){
	//   v = convertLocalIndex(v, dim_i, a, bb, block);
	//   if (v != null){
	//       if (localIndexOffset != null){
	// 	  if (offset != null)
	// 	      offset = Xcons.binaryOp(Xcode.PLUS_EXPR,
	// 				      localIndexOffset, offset);
	// 	  else 
	// 	      offset = localIndexOffset;
	//       }
	//       if (offset != null)
	// 	  v = Xcons.binaryOp(Xcode.PLUS_EXPR, v, offset);

	//       Xobject x = a.convertOffset(dim_i);
	//       if (x != null)
	// 	  v = Xcons.binaryOp(Xcode.MINUS_EXPR, v, x);
	//       //v = Xcons.binaryOp(Xcode.MINUS_EXPR, v, a.convertOffset(dim_i));
	//       i.setArg(0, v);
	//       return i;
	//   }
	// }

      }

      Xobject x = null;
      switch (a.getDistMannerAt(dim_i)){
      case XMPtemplate.BLOCK:
      case XMPtemplate.GBLOCK:
	{
	  x = Xcons.binaryOp(Xcode.MINUS_EXPR, i.getArg(0), a.getBlkOffsetVarAt(dim_i).Ref());
	  break;
	}
      default:
	{
	  Ident f = env.declInternIdent("xmpf_local_idx_",
					Xtype.Function(Xtype.intType));
	  x = f.Call(Xcons.List(a.getDescId().Ref(),
				Xcons.IntConstant(dim_i),
				i.getArg(0)));
	  break;
	}
      }

      i.setArg(0,x);
      return i;

    case F_INDEX_RANGE:
      if (!a.isDistributed(dim_i)) return i;

      if (is_colon(i, a, dim_i)){ // NOTE: this check is not strict.
      	// if (a.hasShadow(dim_i) && dim_i != a.getDim() - 1){
      	//   XMP.errorAt(block, "a subscript of the dimension having shadow must be an int-expr unless it is the last dimension.");
      	// }
	i.setArg(0, null); i.setArg(1, null); i.setArg(2, null);
      	return i;
      }

      // what to do for other cases?

    default:
      XMP.errorAt(block,"bad expression in XMP array index");
      return null;
    }
  }

  private int convertLocalIndexInExpression(Xobject x, int dim_i, XMParray a,
					    BasicBlock bb, Block block, int cnt){

    Xobject v;

    switch (x.Opcode()){

    case PLUS_EXPR:

      if (x.right().isVariable()){
	v = convertLocalIndex(x.right(), dim_i, a, bb, block);
	if (v != null){
	  if (localIndexOffset != null)
	    v = Xcons.binaryOp(Xcode.PLUS_EXPR, v, localIndexOffset);
	  x.setRight(v);
	  cnt++;
	}
      }
      else {
	cnt = convertLocalIndexInExpression(x.right(), dim_i, a, bb, block, cnt);
      }

      // fall through

    case MINUS_EXPR:

      // how to deal with the case for a(1 - (2 - i)) ???

      if (x.left().isVariable()){
	v = convertLocalIndex(x.left(), dim_i, a, bb, block);
	if (v != null){
	  if (localIndexOffset != null)
	    v = Xcons.binaryOp(Xcode.PLUS_EXPR, v, localIndexOffset);
	  x.setLeft(v);
	  cnt++;
	}
      }
      else {
	cnt = convertLocalIndexInExpression(x.left(), dim_i, a, bb, block, cnt);
      }

      break;

    }
    
    return cnt;

  }

  private void insertSizeArray(Statement st, FunctionBlock fb){

    Xobject x = st.getExpr().getArg(0);
    if (x.Opcode() != Xcode.FUNCTION_CALL) return;

    Ident sizeArray = env.declOrGetSizeArray(fb);

    String fname = x.getArg(0).getString();
    Xtype ftype = x.getArg(0).Type();
    XobjList arg_list = (XobjList)x.getArg(1);

    //
    // get interface of each argument
    //

    XobjList param_list = null;
    if (ftype != null){
      // internal or module procedures
      param_list = (XobjList)ftype.getFuncParam();
    }

    if (param_list == null){
	    
      XobjList decl_list = (XobjList)fb.getBody().getDecls();

      // retrieve interface block
    DECLLOOP: for (Xobject decl: decl_list){
	if (decl.Opcode() == Xcode.F_INTERFACE_DECL){
	  XobjList func_list = (XobjList)decl.getArg(3);
	  for (Xobject func: func_list){
	    if (func.getArg(0).getString().equals(fname)){
	      ftype = func.getArg(0).Type();
	      param_list = (XobjList)ftype.getFuncParam();
	      break DECLLOOP;
	    }
	  }
	}
      }
    }

    if (param_list == null) return;

    int k = 0;
    for (int i = 0; i < param_list.Nargs(); i++){

      if (!param_list.getArg(i).Type().isFassumedShape()) continue;
	
      Xobject arg = arg_list.getArg(i);

      if (arg.Opcode() == Xcode.VAR ||
	  arg.Opcode() == Xcode.MEMBER_REF){ // just array name
	  
	Xtype atype;
	int arrayDim;
	Ident sizeFunc;
	Xobject arg0;
	XMParray array = null;

	if (arg.Opcode() != Xcode.MEMBER_REF){
	  Ident id = env.findVarIdent(arg.getName(), fb);
	  array = XMParray.getArray(id);
	}

	if (array != null){
	  atype = array.getType();
	  arrayDim = array.getDim();
	  //sizeFunc = env.declIntrinsicIdent("xmp_array_gsize", Xtype.FintFunctionType);
	  sizeFunc = env.declExternIdent("xmp_array_gsize", Xtype.FintFunctionType);
	  arg0 = array.getDescId().Ref();
	}
	else {
	  atype = arg.Type();
	  arrayDim = atype.getNumDimensions();
	  sizeFunc = env.declIntrinsicIdent("size", Xtype.FintFunctionType);
	  arg0 = arg;
	}

	if (atype.isFallocatable() || atype.isFassumedShape()){
	  for (int j = 0; j < arrayDim; j++){
	    Xobject lhs = Xcons.FarrayRef(sizeArray.Ref(), Xcons.IntConstant(k), Xcons.IntConstant(j));
	    Xobject rhs = sizeFunc.Call(Xcons.List(arg0, Xcons.IntConstant(j+1)));
	    st.insert(Xcons.Set(lhs, rhs));
	  }
	}
	else {
	  Xobject declSize[] = atype.getFarraySizeExpr();
	  for (int j = 0; j < arrayDim; j++){
	    Xobject lhs = Xcons.FarrayRef(sizeArray.Ref(), Xcons.IntConstant(k), Xcons.IntConstant(j));
	    Xobject rhs = Xcons.binaryOp(Xcode.PLUS_EXPR,
					 Xcons.binaryOp(Xcode.MINUS_EXPR,
							declSize[j].getArg(1),
							declSize[j].getArg(0)),
					 Xcons.IntConstant(1));
	    st.insert(Xcons.Set(lhs, rhs));
	  }

	}

      }
      else if (arg.Opcode() == Xcode.F_ARRAY_REF){ // array section

	Xtype atype;
	int arrayDim;
	Ident lbFunc, ubFunc;
	Xobject arg0;
	XMParray array = null;

	if (arg.getArg(0).getArg(0).Opcode() != Xcode.MEMBER_REF){
	  Ident id = env.findVarIdent(arg.getArg(0).getArg(0).getName(), fb);
	  array = XMParray.getArray(id);
	}

	if (array != null){
	  atype = array.getType();
	  arrayDim = array.getDim();
	  lbFunc = env.declExternIdent("xmp_array_lbound", Xtype.FintFunctionType);
	  ubFunc = env.declExternIdent("xmp_array_ubound", Xtype.FintFunctionType);
	  arg0 = array.getDescId().Ref();
	}
	else {
	  atype = arg.getArg(0).getArg(0).Type();
	  arrayDim = atype.getNumDimensions();
	  lbFunc = env.declIntrinsicIdent("lbound", Xtype.FintFunctionType);
	  ubFunc = env.declIntrinsicIdent("ubound", Xtype.FintFunctionType);
	  arg0 = arg.getArg(0).getArg(0);
	}

	XobjList subList = (XobjList)arg.getArg(1);
	for (int j = 0; j < arrayDim; j++){

	  Xobject sub = subList.getArg(j);
	  Xobject rhs = null;

	  if (sub.Opcode() == Xcode.F_ARRAY_INDEX){
	    //rhs = Xcons.IntConstant(1);
	    continue;
	  }
	  else if (sub.Opcode() == Xcode.F_INDEX_RANGE){
	    Xobject lb = sub.getArg(0);
	    if (lb == null){
	      lb = lbFunc.Call(Xcons.List(arg0, Xcons.IntConstant(j+1)));
	    }
	    Xobject ub = sub.getArg(1);
	    if (ub == null){
	      ub = ubFunc.Call(Xcons.List(arg0, Xcons.IntConstant(j+1)));
	    }

	    Xobject stride = sub.getArg(2);
	    if (stride == null){
	      rhs = Xcons.binaryOp(Xcode.PLUS_EXPR,
				   Xcons.binaryOp(Xcode.MINUS_EXPR, ub, lb),
				   Xcons.IntConstant(1));
	    }
	    else {
	      rhs = Xcons.binaryOp(Xcode.MINUS_EXPR, ub, lb);
	      rhs = Xcons.binaryOp(Xcode.PLUS_EXPR, rhs, stride);
	      rhs = Xcons.binaryOp(Xcode.DIV_EXPR, rhs, stride);
	    }
	  }

	  Xobject lhs = Xcons.FarrayRef(sizeArray.Ref(), Xcons.IntConstant(k), Xcons.IntConstant(j));
	  st.insert(Xcons.Set(lhs, rhs));
	}

      }
      else {
	continue;
      }

      k++;
      if (k > XMP.MAX_ASSUMED_SHAPE){
	XMP.fatal("too many assumed-shape arguments (MAX = " + XMP.MAX_ASSUMED_SHAPE + ").");
      }

    }

  }

  private boolean is_colon(Xobject i, XMParray a, int dim_i){

    // check lower
    if (i.getArg(0) != null){
      Xobject lb = a.getType().getFarraySizeExpr()[dim_i].getArg(0);
      if (!lb.equals(i.getArg(0))) return false;
    }

    // check upper
    if (i.getArg(1) != null){
      Xobject ub = a.getType().getFarraySizeExpr()[dim_i].getArg(1);
      if (!ub.equals(i.getArg(1))) return false;
    }

    // check stride
    if (i.getArg(2) != null){
      if (i.getArg(2).Opcode() != Xcode.INT_CONSTANT ||
  	  i.getArg(2).getInt() != 1) return false;
    }

    return true;

  }

}
