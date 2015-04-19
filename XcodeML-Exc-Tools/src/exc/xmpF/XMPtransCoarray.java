/* 
 * $TSUKUBA_Release: Omni XMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */

package exc.xmpF;

import exc.object.*;
import exc.block.*;
import java.util.*;

/**
 * XcalableMP AST translator (for Coarray)
 */
public class XMPtransCoarray implements XobjectDefVisitor
{
  XMPenv env;
  private int pass;

  private ArrayList<XMPtransCoarrayRun> pastRuns;

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
    pastRuns = new ArrayList<XMPtransCoarrayRun>();
  }

  public void setPass(int pass) {
    this.pass = pass;
  }

  public void finish() {
    env.finalize();
  }
    

  //-----------------------------------------
  //  do transform (called topdown, twice)
  //-----------------------------------------

  public void doDef(XobjectDef d) {
    Boolean is_module = d.isFmoduleDef();
    XMPtransCoarrayRun transCoarrayRun;

    if (pass == 1 && is_module || pass == 2 && !is_module)
      return;

    XMP.resetError();
    transCoarrayRun = new XMPtransCoarrayRun(d, env, pastRuns, pass);
    if (pass == 1)
      transCoarrayRun.run1();
    else
      transCoarrayRun.run2();

    // assuming top-down translation along host-association
    pastRuns.add(transCoarrayRun);

    if(XMP.hasError())
      return;
  }

}

