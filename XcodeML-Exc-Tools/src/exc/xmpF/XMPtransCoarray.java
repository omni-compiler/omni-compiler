/*
 * $TSUKUBA_Release: $
 * $TSUKUBA_Copyright:
 *  $
 */

package exc.xmpF;

import exc.object.*;
import exc.block.*;
import java.util.*;

/*
 * Translate Fortran Coarray
 */
public class XMPtransCoarray 
{
  final String MALLOC_LIB_NAME = "xmp_coarray_malloc";
  final String COMMONBLOCK_PREFIX = "xmpf_cptr_";
  final String INITPROC_PREFIX = "xmpf_init";


  private FuncDefBlock def;
  // - specification part of the procedure.
  // - contains (Xobject)id_list and (Xobject)decls.
  // - useful methods are:
  //     add/removeIdent, findLocalDecl(String)

  private XMPenv env;
  // contains 
  // - XobjectFile env = getEnv();

  private String name;
  private BlockList decls;

  private Vector<XMPcoarray> staticCoarrays;
  private Vector<XMPcoarray> allocCoarrays;
  private XMPinitProcedure initProcedure;

  //------------------------------
  //  CONSTRUCTOR
  //------------------------------
  public XMPtransCoarray(FuncDefBlock def, XMPenv env) {
    this.def = def;
    this.env = env;
    FunctionBlock fblock = def.getBlock();
    name = fblock.getName();
    decls = fblock.getBody();

    // set all static/allocatable coarrays in this procedure
    staticCoarrays = new Vector<XMPcoarray>();
    allocCoarrays = new Vector<XMPcoarray>();
    Xobject idList = def.getDef().getFuncIdList();
    for (Xobject obj: (XobjList)idList) {
      Ident ident = (Ident)obj;
      if (ident.getCorank() > 0) {   // found a coarray
        XMPcoarray coarray = new XMPcoarray(ident, decls);
        if (coarray.isAllocatable())
          allocCoarrays.add(coarray);
        else
          staticCoarrays.add(coarray);
        coarray.errorCheck();
      }
    }

    XMP.exitByError();   // exit if error has found.
  }

  //------------------------------
  //  TRANSLATION
  //------------------------------
  public void run() {
    if (staticCoarrays.size() > 0) {
      // This procedure has static coarrays.
      transStaticCoarrays();
    }

  }


  //-----------------------------------------------------
  //  TRANSLATION (1) local/static coarrays
  //-----------------------------------------------------

  // convert from:
  // -------------------------------------------------------
  //     subroutine EX1
  //       real :: V1(10,20)[4,*]
  //       complex(8) :: V2[*]
  //     end subroutine
  // -------------------------------------------------------
  // to:
  // -------------------------------------------------------
  //     subroutine EX1
  //       real :: V1(10,20)                                   ! a
  //       complex(8) :: V2                                    ! a
  //       pointer (xxxx_V1, V1)                               ! b
  //       pointer (xxxx_V2, V2)                               ! b
  //       common /xmpf_yyyy_zzzz_EX1/xxxx_V1,xxxx_V2          ! c
  //     end subroutine
  //     subroutine xmpf_init_zzzz_EX1                         ! d
  //       common /xmpf_yyyy_zzzz_EX1/xxxx_V1,xxxx_V2          ! e
  //       call xmp_coarray_malloc(xxxx_V1,200,4)              ! f
  //       call xmp_coarray_malloc(xxxx_V2,1,16)               ! f
  //     end subroutine
  // -------------------------------------------------------

  private void transStaticCoarrays() {

    // prepare new init procedure (d)
    String newProcName = genNewProcName();

    // remove codimensions form coarray (a)
    for (XMPcoarray coarray: staticCoarrays)
      coarray.clearCodimensions();

    // declare ident of a cray pointer (b)
    for (XMPcoarray coarray: staticCoarrays)
      coarray.declareCrayPointer();

    // generate common stmt in this procedure (c)
    genCommonStmt(COMMONBLOCK_PREFIX + name, staticCoarrays, def);

    // generate common stmt in init procedure (e)
    //////////////////////////
    // ?????
    //genCommonStmt(COMMONBLOCK_PREFIX + name, staticCoarrays, newProc.getDef());
    //////////////////////////

    // generate call stmts. (f)
    // TEMPOERARY ... stmts not used
    //////////////////////////
    Vector<Xobject> stmts = new Vector();
    for (XMPcoarray coarray: staticCoarrays) {
      Xobject stmt = coarray.genMallocCallStmt();
      stmts.add(stmt);
    }

    // output init procedure (d)
    // TEMPOERARY 
    //////////////////////////
    genSubroutine(newProcName, COMMONBLOCK_PREFIX + name);
    //////////////////////////
  }


  //------------------------------
  //  TRANSLATION (2) local/allocatable coarray
  //------------------------------
  public void transCoarrayDeclAllocatable(XMPcoarray coarray) {
    XMP.error("Allocatable coarry is not supported yet: "+coarray.getName());
  }


  //------------------------------
  //  TRANSLATION (3) module/static coarray
  //------------------------------


  //------------------------------
  //  TRANSLATION (4) module/allocatable coarray
  //------------------------------

  //------------------------------
  //  parts
  //------------------------------
  private void genCommonStmt(String commonName, Vector<XMPcoarray> coarrays,
                             FuncDefBlock def) {

    /* cf. XMPenv.declOrGetSizeArray
     */

    // common block name
    Xobject cnameObj = Xcons.Symbol(Xcode.IDENT, commonName);

    // list of common vars
    Xobject varList = Xcons.List();
    for (XMPcoarray coarray: coarrays) {
      Ident cptrId = coarray.getCrayPointerId();
      varList.add(Xcons.FvarRef(cptrId));
    }

    // declaration 
    Xobject decls = def.getBlock().getBody().getDecls();
    decls.add(Xcons.List(Xcode.F_COMMON_DECL,
                         Xcons.List(Xcode.F_VAR_LIST, cnameObj, varList)));
  }

  private void genSubroutine(String newProcName, String commonName) {
    /////////////////
    System.out.println("I want to generate Xobject describing the following 'init-subroutine':");
    System.out.println("  -----------------------------------------------");
    /////////////////

    System.out.println("  subroutine " + newProcName);

    // common stmt
    String s =         "    COMMON / " + commonName + " /";
    String delim = " ";
    for (XMPcoarray coarray: staticCoarrays) {
      s += delim + coarray.getCrayPointerName();
      delim = " , ";
    }
    System.out.println(s);

    // malloc call stmts
    for (XMPcoarray coarray: staticCoarrays) {
      int elem = coarray.getElementLength();
      int count = coarray.getTotalArraySize();
      System.out.println("    CALL " + MALLOC_LIB_NAME + "( " +
                         coarray.getCrayPointerName() + ", " +
                         count + ", " + elem + " )");
    }

    System.out.println("  end subroutine " + newProcName);

    /////////////////
    System.out.println("  -----------------------------------------------");
    /////////////////
  }


  private String genNewProcName() {
    return genNewProcName(getHostNames());
  }

  private String genNewProcName(String ... names) { // host-to-guest order
    int n = names.length;
    String initProcName = INITPROC_PREFIX;
    for (int i = 0; i < n; i++) {
      initProcName += "_";
      StringTokenizer st = new StringTokenizer(names[i], "_");
      int n_underscore = st.countTokens() - 1;
      if (n_underscore > 0)   // '_' was found in names[i]
        initProcName += String.valueOf(n_underscore);
      initProcName += names[i];
    }
    return initProcName;
  }

  private String[] getHostNames() {
    Vector<String> list = new Vector();
    list.add(def.getDef().getName());
    XobjectDef parentDef = def.getDef().getParent();
    while (parentDef != null) {
      list.add(parentDef.getName());
      parentDef = parentDef.getParent();
    }

    int n = list.size();
    String[] names = new String[n];
    for (int i = 0; i < n; i++)
      names[i] = list.get(n-i-1);

    return names;
  }


  //------------------------------
  //  UTILITIES
  //------------------------------
  public String toString() {
    String s = "{";
    String delim = "";
    for (XMPcoarray coarray: staticCoarrays) {
      s += delim + coarray.toString();
      delim = ",";
    }
    return s + "}";
  }

  public String display() {
    String s = "[static coarrays in " + name;
    for (XMPcoarray coarray: staticCoarrays)
      s += coarray.display() + "\n";
    return s;
  }
}

