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
 * XcalableMP AST translator (for Coarray)
 */
public class XMPtransCoarray implements XobjectDefVisitor
{
  XMPenv env;
  private int pass;


  //-----------------------------------------
  //  constructor
  //-----------------------------------------

  public XMPtransCoarray() {
    this(null, 1);
  }
  public XMPtransCoarray(XobjectFile env) {
    this(env, 1);
  }
  public XMPtransCoarray(XobjectFile env, int pass) {
    this.env = new XMPenv(env);
    this.pass = pass;
  }

  public void setPass(int pass) {
    this.pass = pass;
  }

  //  public void init(XobjectFile env) {
  //    this.env = new XMPenv(env);
  //}
    
  public void finish() {
    env.finalize();
  }
    

  //-----------------------------------------
  //  do transform (called topdown, twice)
  //-----------------------------------------

  public void doDef(XobjectDef d) {
    FuncDefBlock fd;
    Boolean is_module = d.isFmoduleDef();
    XMPtransCoarrayRun transCoarrayRun;

    if (pass == 1 && !is_module) {
      fd = new FuncDefBlock(d);
      XMP.resetError();
      transCoarrayRun = new XMPtransCoarrayRun(fd, env);
      transCoarrayRun.run1();
    } else if (pass == 2 && is_module) {
      fd = new FuncDefBlock(d);
      XMP.resetError();
      transCoarrayRun = new XMPtransCoarrayRun(fd, env);
      transCoarrayRun.run2();
    } else {
      return;
    }

    if(XMP.hasError())
      return;
  }

}

