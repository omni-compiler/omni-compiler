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

  public XMPtranslate() {  }
    
  public XMPtranslate(XobjectFile env) {
    init(env);
  }

  public void init(XobjectFile env){
    this.env = new XMPenv(env);
    if(XMP.debugFlag)
      debug_out = new BlockPrintWriter(System.out);
  }
    
  public void finish() {
    env.finalize();
  }
    
  private void replace_main(XobjectDef d) {
    String name = d.getName();
    Ident id = env.getEnv().findVarIdent(name);
    if(id == null)
      XMP.fatal("'" + name + "' not in id_list");
    id.setName(XMPmainFunc);
    d.setName(XMPmainFunc);
  }

  // do transform takes three passes
  public void doDef(XobjectDef d) {
    FuncDefBlock fd = null;
    Boolean is_module = d.isFmoduleDef();

    XMP.resetError();

    // System.out.println("def="+d.getDef());
    if(is_module){
      if(!haveXMPpragma(d.getDef())) return;
      fd = XMPmoduleBlock(d);
    } else if(d.isFuncDef()){ // declarations
      Xtype ft = d.getFuncType();
      if(ft != null && ft.isFprogram()) {
	ft.setIsFprogram(false);
	replace_main(d);
      }
      fd = new FuncDefBlock(d);
    } else 
      XMP.fatal("Fotran: unknown decls");

    if(XMP.hasError())
      return;
        
    anaPragma.run(fd,env);
    if(XMP.hasError()) return;

    rewriteExpr.run(fd, env);
    if(XMP.hasError()) return;

    transPragma.run(fd,env);
    if(XMP.hasError()) return;
        
    // finally, replace body
    fd.Finalize();

    if(XMP.debugFlag) {
      System.out.println("**** final **** "+fd.getDef());
      XobjectPrintWriter out = new XobjectPrintWriter(System.out);
      out.print(fd.getDef().getDef());
      System.out.println("---- final ---- ");
    }
  }

  FuncDefBlock XMPmoduleBlock(XobjectDef def){
    XobjList decls = Xcons.List(); // emptyList
    Xobject xmp_pragma_list = Xcons.FstatementList();

    Xobject d = def.getDef();
    // module = (F_MODULE_DEFINITION name_id id_list decls body?)
    for(Xobject decl: (XobjList)d.getArg(2)){
      if(decl.Opcode() == Xcode.XMP_PRAGMA)
	xmp_pragma_list.add(decl);
      else
	decls.add(decl);
    }
    d.setArg(2,decls);
    d.setArg(3, xmp_pragma_list); // put xmp_pragmas as a body
    // System.out.println("module fblock="+d);
    return new FuncDefBlock(def);
  }

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
