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
  boolean _containsCoarray;

  //-----------------------------------------
  //  constructor
  //-----------------------------------------

  public XMPtransCoarray(XobjectFile env, int pass) {
    this.env = new XMPenv(env);
    this.pass = pass;
    pastRuns = new ArrayList<XMPtransCoarrayRun>();
    _containsCoarray = false;
  }

  public void finish() {
    env.finalize();
  }
    

  //-----------------------------------------
  //  do transform for each procedure and module
  //-----------------------------------------

  public void doDef(XobjectDef d) {
    boolean is_module = d.isFmoduleDef();
    XMPtransCoarrayRun transCoarrayRun;

    XMP.resetError();

    switch (pass) {
    case 0:
      boolean contains;
      if (is_module)
        contains = errorCheck_module(d);
      else
        contains = errorCheck_procedure(d);
      if (contains)
        _containsCoarray = true;
      break;

    case 1:               // for both procedures and modules
      transCoarrayRun = new XMPtransCoarrayRun(d, env, pastRuns, 1);
      transCoarrayRun.run1();
      // assuming top-down translation along host-association
      pastRuns.add(transCoarrayRun);
      break;

    case 2:               // second pass for modules
      if (!is_module)
        return;
      transCoarrayRun = new XMPtransCoarrayRun(d, env, pastRuns, 2);
      transCoarrayRun.run2();
      break;

    default:
      return;
    }

    if(XMP.hasError())
      return;
  }


  /*
   * true if there are:
   *  - any coarray declarations
   * in the module.
   */
  private boolean errorCheck_module(XobjectDef def) {
    boolean contains = false;

    // check coarray declarations
    Xobject idList = def.getFuncIdList();
    for (Xobject obj: (XobjList)idList) {
      Ident ident = (Ident)obj;
      if (ident.isCoarray()) {
        // found it is a coarray
        contains = true;
        errorCheck_ident(ident);
      }
    }
    
    return contains;
  }

  /*
   * true if there are:
   *  - any coarray declarations, or
   *  - any reference of coindexed objects, or
   *  - any reference of coarray intrinsic procedures
   * in the procedure.
   */
  private boolean errorCheck_procedure(XobjectDef def) {
    boolean contains = false;

    // check coarray declarations
    Xobject idList = def.getFuncIdList();
    for (Xobject obj: (XobjList)idList) {
      Ident ident = (Ident)obj;
      if (ident.isCoarray()) {
        // found it is a coarray
        contains = true;
        errorCheck_ident(ident);
      }
    }
    
    // check reference of coindexed objects/variables
    // check reference of intrinsic procedures
    XobjectIterator xi = new topdownXobjectIterator(def.getFuncBody());
    for (xi.init(); !xi.end(); xi.next()) {
      Xobject xobj = xi.getXobject();
      if (xobj == null)
        continue;

      switch (xobj.Opcode()) {
      case CO_ARRAY_REF:
        contains = true;
        errorCheck_coidxObj(xobj);
        break;

      case IDENT:
        if (isCoarrayLibraryName(xobj.getName()))
          contains = true;
        break;
      }
    }

    return contains;
  }


  private void errorCheck_ident(Ident ident) {
  }

  private void errorCheck_coidxObj(Xobject xobj) {
  }


  private boolean isCoarrayLibraryName(String name) {
    // True if it is a name of coarray intrinsic procedures or
    // a library and it should be converted in run1 or run2.
    // Else, false even if it is a name of itrinsic procedure.
    return false;
  }



  public boolean containsCoarray() {
    return _containsCoarray;
  }
}

