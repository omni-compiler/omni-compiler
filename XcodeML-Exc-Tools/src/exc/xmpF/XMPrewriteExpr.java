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

    // rewrite allocate and deallocate statements

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

	      Ident id = env.findVarIdent(alloc.getArg(0).getName(), fb);
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

	      Ident id = env.findVarIdent(dealloc.getArg(0).getName(), fb);
	      if (id == null) break;
	      XMParray array = XMParray.getArray(id);
	      if (array == null) break;

	      array.rewriteDeallocate(dealloc, st, fb, env);
	    }
	    break;
	  }

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
    		if (clauses != null) rewriteOmpClauses(clauses, (PragmaBlock)block, fb);
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
	  for(XobjArgs args = x.getArg(1).getArgs(); args != null;
	      args = args.nextArgs()){
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
	  System.out.println("coarray not yet: "+ x);
	  break;
	}
      case FUNCTION_CALL:
	{
	  String fname = x.getArg(0).getString();
	  if (fname.equalsIgnoreCase("xmp_desc_of")){
	      XMParray array = (XMParray) x.getArg(1).getArg(0).getProp(XMP.arrayProp);
	      if (array == null){
		XMP.errorAt(block,"xmp_desc_of applied to non-global data");
		XMP.exitByError();
	      }
	      Xobject desc = array.getDescId();
	      iter.setXobject(desc);
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
    if(on_tmpl != a_tmpl) return null; // different template
    
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

    //Xobject off1 = a.getAlignSubscriptOffsetAt(dim_i);

    // Xobject off2 = on_ref.getLoopOffset(loop_idx);
    // if(off1 == null) localIndexOffset = off2;
    // else if(off2 == null) localIndexOffset = off1;
    // else localIndexOffset = 
    //         Xcons.binaryOp(Xcode.PLUS_EXPR,off1,off2);
    localIndexOffset = off1;

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

      // check this expression is ver+offset
      Xobject e = i.getArg(0);
      // we need normalize?
      Xobject v = null;
      Xobject offset = null;
      if(e.isVariable()){
	v = e;
      } else {
	switch(e.Opcode()){
	case PLUS_EXPR:
	  if(e.left().isVariable()){
	    v = e.left();
	    offset = e.right();
	  } else if(e.right().isVariable()){
	    v = e.right();
	    offset = e.left();
	  }
	  break;
	case MINUS_EXPR:
	  if(e.left().isVariable()){
	    v = e.left();
	    offset = Xcons.unaryOp(Xcode.UNARY_MINUS_EXPR,e.right());
	  }
	  break;
	}
      }

      if (v != null){
	  v = convertLocalIndex(v, dim_i, a, bb, block);
	  if (v != null){
	      if (localIndexOffset != null){
		  if (offset != null)
		      offset = Xcons.binaryOp(Xcode.PLUS_EXPR,
					      localIndexOffset, offset);
		  else 
		      offset = localIndexOffset;
	      }
	      if (offset != null)
		  v = Xcons.binaryOp(Xcode.PLUS_EXPR, v, offset);

	      Xobject x = a.convertOffset(dim_i);
	      if (x != null)
		  v = Xcons.binaryOp(Xcode.MINUS_EXPR, v, x);
	      //v = Xcons.binaryOp(Xcode.MINUS_EXPR, v, a.convertOffset(dim_i));
	      i.setArg(0, v);
	      return i;
	  }
      }

      Ident f = env.declExternIdent("xmpf_local_idx_",
				    Xtype.Function(Xtype.intType));
      Xobject x = f.Call(Xcons.List(a.getDescId().Ref(),
				    Xcons.IntConstant(dim_i),
				    i.getArg(0)));
      i.setArg(0,x);
      return i;

      /* what to do for array_range expression */
    default:
      XMP.errorAt(block,"bad expression in XMP array index");
      return null;
    }
  }
}
