package exc.xmpF;

import exc.object.*;
import exc.block.*;
import java.util.*;

/*
 * Translate Fortran Coarray
 */
public class XMPtransCoarray 
{
  final static String COMMON_PREFIX = "xmpf_cptr_";
  final static String INITPROC_PREFIX = "xmpf_init";
  final static String CRAYPOINTER_PREFIX = "xmpf_cptr_";
  final static String MALLOC_LIB_NAME = "xmpf_coarray_malloc";


  //private FuncDefBlock def;          // (XobjectDef)def.getDef() is more useful????
  // contains
  //  - XobjectDef def
  //  - FunbtionBlock fblock
  // useful methods are:
  //     add/removeIdent, findLocalDecl(String)

  private XMPenv env;

  private String name;
  private BlockList decls;

  private XobjectDef def;
  private FunctionBlock fblock;

  private Vector<XMPcoarray> staticCoarrays;
  private Vector<XMPcoarray> allocCoarrays;

  //private XMPinitProcedure initProcedure;
  private String initProcTextForFile;

  //CollectInit mediator
  XMPcollectInit collectInit;

  private String commonName;

  //------------------------------
  //  CONSTRUCTOR
  //------------------------------
  public XMPtransCoarray(FuncDefBlock funcDef, XMPenv env) {
    def = funcDef.getDef();
    fblock = funcDef.getBlock();
    this.env = env;
    name = fblock.getName();
    commonName = COMMON_PREFIX + name;

    // set all static/allocatable coarrays in this procedure
    staticCoarrays = new Vector<XMPcoarray>();
    allocCoarrays = new Vector<XMPcoarray>();
    Xobject idList = def.getFuncIdList();
    for (Xobject obj: (XobjList)idList) {
      Ident ident = (Ident)obj;
      if (ident.getCorank() > 0) {   // found a coarray
        XMPcoarray coarray = new XMPcoarray(ident, def, fblock);
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
    // initialize for the file
    collectInit = new XMPcollectInit(MALLOC_LIB_NAME);

    if (staticCoarrays.size() > 0) {  // static coarrays included
      transStaticCoarrays();
    }

    // write Tail Text into env
    collectInit.finalize(env);
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
  //
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
    genCommonStmt(commonName, staticCoarrays, def);

    // output init procedure (d)
    //genInitRoutine(newProcName, commonName);
    genInitRoutine_string(newProcName);

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
                             XobjectDef def) {

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
    Xobject decls = fblock.getBody().getDecls();
    decls.add(Xcons.List(Xcode.F_COMMON_DECL,
                         Xcons.List(Xcode.F_VAR_LIST, cnameObj, varList)));
  }

  /*
   * suspended
   *  This method could be better...
   */
  private void genInitRoutine(String newProcName, String commonName) {

    Xtype type = Xtype.FexternalSubroutineType;
    Ident ident = Ident.Fident(newProcName, Xtype.FexternalSubroutineType);
    /* or using Xcons.Symbol or Xcons.Ident */
    Xobject id_list = Xcons.List();
    Xobject decls = Xcons.List();
    Block body = new Block(Xcode.FUNCTION_DEFINITION, null);
    //Xobject body = Xcons.List();
    //BlockList blockList = Bcons.emptyBody();

    FunctionBlock newFblock = new FunctionBlock((Xobject)ident, id_list, decls, body,
                                                (Xobject)null/*gcc_attrs*/,
                                                fblock.getEnv());

    //XobjectDef newFunc = XobjectDef.Func(ident, id_list, decls, body);
    //FunctionBlock newFblock = Bcons.buildFunctionBlock(newFunc);

    //newBody.setIdentList(idList);
    //newBody.setDecls(declList);


    Xobject newFblockObj = newFblock.toXobject();
    env.getEnv().add(newFblockObj);
  }      

  /*
   * temporary but reliable method
   */
  private void genInitRoutine_string(String newProcName) {
    // init for each init routine
    collectInit.openProcText(newProcName, commonName);

    // malloc call stmts
    for (XMPcoarray coarray: staticCoarrays) {
      int elem = coarray.getElementLength();
      int count = coarray.getTotalArraySize();
      String cptrName = coarray.getCrayPointerName();
      collectInit.addForVarText(cptrName, count, elem);
    }

    // finalize for each init routine
    collectInit.closeProcText();
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
    list.add(def.getName());
    XobjectDef parentDef = def.getParent();
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

