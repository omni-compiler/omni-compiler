/* 
 * $TSUKUBA_Release: Omni XMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
package exc.xmpF;

import exc.object.*;
import exc.block.*;

/**
 * XcalableMP AST translator
 */
public class XMPtranslate implements XobjectDefVisitor
{
  BlockPrintWriter debug_out;
  XMPenv env;

  // alorithms
  XMPanalyzePragma anaPragma = new XMPanalyzePragma();
  XMPrewriteExpr rewriteExpr = new XMPrewriteExpr();
  XMPtransPragma transPragma = new XMPtransPragma();
    
  final String XMPmainFunc = "xmpf_main";

  public XMPtranslate() {
  }
    
  public XMPtranslate(XobjectFile env) {
    init(env);
  }

  public void init(XobjectFile env){
    this.env = new XMPenv(env);
    if(XMP.debugFlag)
      debug_out = new BlockPrintWriter(System.out);
  }
    
  public void finish() {
//     FmoduleBlock mod = 
//       (FmoduleBlock)env.getProp(XMPtransPragma.PROP_KEY_FINTERNAL_MODULE);
//     if(mod != null) {
//       XobjectDef lastMod = null;
//       for(XobjectDef d : env) {
//  	if(d.isFmoduleDef())
//  	  lastMod = d;
//       }
//       if(lastMod == null)
//  	env.insert(mod.toXobjectDef());
//       else
//  	lastMod.addAfterThis(mod.toXobjectDef());
//     }
    env.getEnv().collectAllTypes();
    env.getEnv().fixupTypeRef();
  }
    
  private void replace_main(XobjectDef d) {
    String name = d.getName();
    Ident id = env.findVarIdent(name);
    if(id == null)
      XMP.fatal("'" + name + "' not in id_list");
    id.setName(XMPmainFunc);
    d.setName(XMPmainFunc);
  }

  // do transform takes three passes
  public void doDef(XobjectDef d) {
    XMP.resetError();

    if(!d.isFuncDef()){ // declarations
      XMP.fatal("Fotran only: decl out side function");
    }
        
    Xtype ft = d.getFuncType();
    if(ft != null && ft.isFprogram()) {
      ft.setIsFprogram(false);
      replace_main(d);
    }
        
    FuncDefBlock fd = new FuncDefBlock(d);

    if(XMP.hasError())
      return;
        
    anaPragma.run(fd,env);
    if(XMP.hasError())
      return;

    rewriteExpr.run(fd, env);
    if(XMP.hasError())
      return;

    transPragma.run(fd,env);
    if(XMP.hasError())
      return;
        
    // finally, replace body
    fd.Finalize();
  }

  // not used?
  boolean haveXMPpragma(Xobject x)
  {
    XobjectIterator i = new topdownXobjectIterator(x);
    for(i.init(); !i.end(); i.next()) {
      Xobject xx = i.getXobject();
      if(xx != null && xx.Opcode() == Xcode.XMP_PRAGMA)
	return true;
    }
    return false;
  }
}
