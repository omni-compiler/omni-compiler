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
  private int version;
  private Boolean useGASNet;
  private Boolean onlyCafMode;

  private ArrayList<XMPtransCoarrayRun> pastRuns;
  int _nCoarrays = 0;
  int _nCoidxObjs = 0;
  int _nCoarrayLibs = 0;

  //-----------------------------------------
  //  constructor
  //-----------------------------------------

  public XMPtransCoarray(XobjectFile env, int pass, String suboption,
                         Boolean onlyCafMode)
  {
    this.env = new XMPenv(env);
    this.pass = pass;
    _set_version(suboption);
    this.onlyCafMode = onlyCafMode;
    pastRuns = new ArrayList<XMPtransCoarrayRun>();
  }

  public void finish()
  {
    env.finalize();
  }
    

  //-----------------------------------------
  //  set
  //-----------------------------------------

  private void _set_version(String suboption)
  {
    // default
    version = 3;
    useGASNet = false;

    if ("".equals(suboption)) {
      // default
    } else if ("4".equals(suboption)) {
      version = 4;
    } else if ("6".equals(suboption)) {
      version = 6;
    } else if ("7".equals(suboption)) {
      version = 7;
    } else if ("7g".equals(suboption)) {
      version = 7;
      useGASNet = true;
    } else {
      XMP.fatal("suboption usage: -fcoarray[={4|6|7|7g}]");
    }
  }


  //-----------------------------------------
  //  do transform for a procedure or a module
  //-----------------------------------------

  public void doDef(XobjectDef d)
  {
    boolean is_module = d.isFmoduleDef();
    XMPtransCoarrayRun transCoarrayRun;

    XMP.resetError();

    switch (pass) {
    case 0:
      if (is_module)
        errorCheck_module(d);
      else
        errorCheck_procedure(d);
      break;

    case 1:               // for both procedures and modules
      transCoarrayRun = new XMPtransCoarrayRun(d, env, pastRuns, 1, version,
                                               useGASNet, onlyCafMode);
      transCoarrayRun.run1();
      // assuming top-down translation along host-association
      pastRuns.add(transCoarrayRun);
      //transCoarrayRun.finalize();
      break;

    case 2:               // second pass for modules
      if (!is_module)
        return;
      transCoarrayRun = new XMPtransCoarrayRun(d, env, pastRuns, 2, version,
                                               useGASNet, onlyCafMode);
      transCoarrayRun.run2();
      //transCoarrayRun.finalize();
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
  private void errorCheck_module(XobjectDef def)
  {
    // check coarray declarations
    Xobject idList = def.getFuncIdList();
    for (Xobject obj: (XobjList)idList) {
      Ident ident = (Ident)obj;
      if (ident.isCoarray()) {
        // found it is a coarray
        _nCoarrays += 1;
        errorCheck_ident(ident, def);
      }
    }
  }

  /*
   * true if there are:
   *  - any coarray declarations, or
   *  - any reference of coindexed objects, or
   *  - any reference of coarray intrinsic procedures
   * in the procedure.
   */
  private void errorCheck_procedure(XobjectDef def) {
    // check coarray declarations
    Xobject idList = def.getFuncIdList();
    for (Xobject obj: (XobjList)idList) {
      Ident ident = (Ident)obj;
      if (ident.isCoarray()) {
        // found it is a coarray
        _nCoarrays += 1;
        errorCheck_ident(ident, def);
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
        _nCoidxObjs += 1;
        errorCheck_coidxObj(xobj, def);
        break;

      case IDENT:
        if (isCoarrayLibraryName(xobj.getName()))
          _nCoarrayLibs += 1;
        break;
      }
    }
  }


  private void errorCheck_ident(Ident ident, XobjectDef def)
  {
    // restriction: initialization

    // non-save non-allocatable coarray in recursive procedure
    if (_isRecursiveProcedure(def) &&
        ident.getStorageClass() != StorageClass.FSAVE &&
        !ident.Type().isFallocatable()) {
      XMP.error("Coarray \'" + ident.getName() + 
                "\' must have SAVE or ALLOCATABLE attribute in recursive procedure.");
    }
  }


  private boolean _isRecursiveProcedure(XobjectDef def)
  {
    Xobject d = def.getDef();
    if (d.Opcode() == Xcode.F_MODULE_DEFINITION)
      return false;
    Xobject name = d.getArgs().getArg();
    return name.Type().isFrecursive();
  }


  private void errorCheck_coidxObj(Xobject xobj, XobjectDef def)
  {
  }


  private boolean isCoarrayLibraryName(String name)
  {
    // True if it is a name of coarray intrinsic procedures or
    // a library and it should be converted in run1 or run2.
    // Else, false even if it is a name of itrinsic procedure.

    return XMPtransCoarrayRun.intrinsicProcedureNames.contains(name);
  }


  public boolean containsCoarray()
  {
    // check if there are any coarrays or any coarray libraries.
    if (_nCoarrays > 0 || _nCoidxObjs > 0 || _nCoarrayLibs > 0)
      return true;
    return false;
  }
}

