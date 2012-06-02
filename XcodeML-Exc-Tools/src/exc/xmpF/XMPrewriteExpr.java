/* 
 * $TSUKUBA_Release: Omni XMP Compiler $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
package exc.xmpF;

import exc.object.*;
import exc.block.*;

/**
 * pass2: check and write variables
 */
public class XMPrewriteExpr
{
  private XMPenv  env;
  protected XobjectDef current_def;

  public XMPrewriteExpr(){ }

  public void run(FuncDefBlock def, XMPenv env) {
    this.env = env;
    current_def = def.getDef();

    XMP.debug("pass2:");
    FunctionBlock fb = def.getBlock();
    if (fb == null) return;

    // rewrite parameters
    rewriteParams(fb);

    // rewrite expr
    BasicBlockExprIterator iter = new BasicBlockExprIterator(fb);
    for (iter.init(); !iter.end(); iter.next()) {
      Xobject expr = iter.getExpr();
      if(expr != null)  rewriteExpr(expr,iter.getBasicBlock(),fb);
    }

    // create local object descriptors, constructors and desctructors
    //env.setupObjectId(fb);
    //env.setupConstructor(fb);
    //def.Finalize();
  }

  private void rewriteParams(FunctionBlock funcBlock){
    XobjList identList = funcBlock.getBody().getIdentList();

    if(XMP.debugFlag){
      System.out.println("identList="+identList);
      System.out.println("decl="+funcBlock.getBody().getDecls());
    }

    if (identList == null) {
      return;
    } else {
      for(XobjArgs args = identList.getArgs(); args != null; 
	  args = args.nextArgs()){
	Xobject x = args.getArg();
        XMParray array = 
	  env.getXMParray(x.getName(),funcBlock);
	// if parameter is array, then rewrite
        if (array != null) {
	  // replace decls
	  Xobject decl = funcBlock.getBody().findLocalDecl(x.getName());
	  if(decl != null){
	    decl.setArg(0,Xcons.Symbol(Xcode.IDENT,array.getLocalId().getName()));
	  }
	  args.setArg(array.getLocalId());
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
	  XMParray array = env.getXMParray(x.getName(),block);
	  if (array != null){
	    // replace with local decl
	    Xobject var = array.getLocalId().Ref();
	    var.setProp(XMP.arrayProp,array);
	    iter.setXobject(var);
	  }
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
	  break;
	}
        // XXX delete this
      case CO_ARRAY_REF:
	{
	  System.out.println("coarray not yet: "+ x);
	  break;
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


    Xobject off1 = a.getAlignSubscriptOffsetAt(dim_i);
    Xobject off2 = on_ref.getLoopOffset(loop_idx);
    if(off1 == null) localIndexOffset = off2;
    else if(off2 == null) localIndexOffset = off1;
    else localIndexOffset = 
	   Xcons.binaryOp(Xcode.PLUS_EXPR,off1,off2);

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
      if(!a.isDistributed(dim_i)) return null;

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

      if(v != null){
	v = convertLocalIndex(v,dim_i,a,bb,block);
	if(v != null){
	  if(localIndexOffset != null){
	    if(offset != null)
	      offset = Xcons.binaryOp(Xcode.PLUS_EXPR,
				      localIndexOffset,offset);
	    else 
	      offset = localIndexOffset;
	  }
	  if(offset != null)
	    v = Xcons.binaryOp(Xcode.PLUS_EXPR,v,offset);
	  i.setArg(0,v);
	  return i;
	}
      }

      Ident f = current_def.declExternIdent("xmpf_local_idx_",
					    Xtype.Function(Xtype.intType));
      Xobject x = f.Call(Xcons.List(a.getDescId().Ref(),
				    Xcons.IntConstant(dim_i),
				    i.getArg(0)));
      i.setArg(0,x);
      return i;

      /* what to do for array_range expression */
    default:
	XMP.error("bad expression in XMP array index");
	return null;
    }
  }
}
