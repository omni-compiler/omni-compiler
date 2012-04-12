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
      if(expr != null)  rewriteExpr(expr, fb);
    }

    // create local object descriptors, constructors and desctructors
    //env.setupObjectId(fb);
    //env.setupConstructor(fb);
    //def.Finalize();
  }

  private void rewriteParams(FunctionBlock funcBlock){
    XobjList identList = funcBlock.getBody().getIdentList();
    System.out.println("identList="+identList);
    System.out.println("decl="+funcBlock.getBody().getDecls());
    if (identList == null) {
      return;
    } else {
      for(XobjArgs args = identList.getArgs(); args != null; args = args.nextArgs()){
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
  private void rewriteExpr(Xobject expr, Block block){
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
	  System.out.println("array_ref="+x);
	  if(a.Opcode() != Xcode.F_VAR_REF)
	    XMP.fatal("not F_VAR_REF for F_ARRAY_REF");
	  a = a.getArg(0);
	  XMParray array = (XMParray) a.getProp(XMP.arrayProp);
	  if(array == null) break;

	  int dim_i = 0;
	  for(XobjArgs args = x.getArg(1).getArgs(); args != null;
	      args = args.nextArgs()){
	    Xobject index_calc = arrayIndexCalc(array,dim_i++,args.getArg());
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

  Xobject arrayIndexCalc(XMParray a, int dim_i, Xobject i){
    System.out.println("i="+i);
    switch(i.Opcode()){
    case F_ARRAY_INDEX:
      Ident f = current_def.declExternIdent("xmpf_local_idx_",
					    Xtype.Function(Xtype.intType));
      Xobject x = f.Call(Xcons.List(a.getDescId(),
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
