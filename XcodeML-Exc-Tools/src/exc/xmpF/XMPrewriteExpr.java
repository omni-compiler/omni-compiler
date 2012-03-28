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
	  if(a.Opcode() != Xcode.F_VAR_REF)
	    XMP.fatal("not F_VAR_REF for F_ARRAY_REF");
	  a = a.getArg(0);
	  XMParray array = (XMParray) a.getProp(XMP.arrayProp);
	  if(array != null){
	    Xobject index_calc = arrayIndexCalc(array,(XobjList)x.getArg(1));
	    if(index_calc != null){
	      x.setArg(1,Xcons.List(Xcons.List(Xcode.F_ARRAY_INDEX,index_calc)));
	    }
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

  Xobject arrayIndexCalc(XMParray a, XobjList index_list){
    XobjList args = Xcons.List();
    args.add(a.getDescId());
    int i = 0;
    for(Xobject x: index_list){
      i++;
      switch(x.Opcode()){
      case F_ARRAY_INDEX:
	args.add(x.getArg(0));
	break;
	/* what to do for array_range expression */
      default:
	XMP.error("bad expression in XMP array index");
	return null;
      }
    }
    Ident f = current_def.declExternIdent("xmpf_local_idx_"+i, 
					  Xtype.Function(Xtype.intType));
    return f.Call(args);
  }
}
