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
  private int version = 3;             // default (useMalloc=true, optLevel=1)
  private Boolean useMalloc = true;    // default (use RA or RS method for mamory allocation)
  private int optLevel = 0;            // default
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
    env.finalizeEnv();
  }
    

  //-----------------------------------------
  //  set
  //-----------------------------------------

  private void _set_version(String suboption)
  {
    if (suboption == null || "".equals(suboption))
      return;

    for (String token: suboption.split(",", 0)) {
      if ("-O0".equalsIgnoreCase(token)) {      // no optimization for GET communication
        optLevel = 0;
      } else if ("-O1".equalsIgnoreCase(token)) { // convert to subroutine call for GET if available
        optLevel = 1;
      } else if ("RA".equalsIgnoreCase(token)) { // RuntimeAllocation (RA) method or RS method
        version = 3;
        useMalloc = true;
      } else if ("CA".equalsIgnoreCase(token)) {  // CompilerAllocation (CA) method 
        version = 4;
        useMalloc = false;
      }
      //--- old fashioned ----
      else if ("3".equals(token)) {     // RA method or RS method
        version = 3;
        useMalloc = true;
      } else if ("4".equals(token)) {   // CA method
        version = 4;
        useMalloc = false;        
      } else if ("6".equals(token)) {   // >=6 versions are under developping
        version = 6;
        useMalloc = false;
      } else if ("7".equals(token)) {
        version = 7;
        useMalloc = false;
      } else if ("7g".equals(token)) {
        version = 7;
        useMalloc = true;
      }

      else {
        XMP.fatal("found illegal suboption of coarray. \n" +
                  "  Usage: -fcoarray={{RA|CA}|{-O0|-O1}|{3|4|6|7|7g}},...\n" +
                  "  Default: RA,-O0 and 3");
      }
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
      transCoarrayRun = new XMPtransCoarrayRun
        (d, env, pastRuns, 1, version, useMalloc, onlyCafMode, optLevel);
      transCoarrayRun.run1();
      // assuming top-down translation along host-association
      pastRuns.add(transCoarrayRun);
      //transCoarrayRun.finalize();
      break;

    case 2:               // second pass for modules
      if (!is_module)
        return;
      transCoarrayRun = new XMPtransCoarrayRun
        (d, env, pastRuns, 2, version, useMalloc, onlyCafMode, optLevel);
      transCoarrayRun.run2();
      //transCoarrayRun.finalize();
      break;

    case 3:               // for both procedures and modules
      transCoarrayRun = new XMPtransCoarrayRun
        (d, env,     null, 3, version, useMalloc, onlyCafMode, optLevel);
      transCoarrayRun.run3();
      break;

    case 4:               // for both procedures and modules
      transCoarrayRun = new XMPtransCoarrayRun
        (d, env,     null, 4, version, useMalloc, onlyCafMode, optLevel);
      transCoarrayRun.run4();
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
        if (isCoarrayIntrinsicName(xobj.getName()))
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


  private boolean isCoarrayIntrinsicName(String name)
  {
    // True if it is a name of coarray intrinsic procedures
    return XMPtransCoarrayRun.isCoarrayIntrinsicName(name);
  }


  public boolean containsCoarray()
  {
    // check if there are any coarrays or any coarray libraries.
    if (_nCoarrays > 0 || _nCoidxObjs > 0 || _nCoarrayLibs > 0)
      return true;
    return false;
  }
}

