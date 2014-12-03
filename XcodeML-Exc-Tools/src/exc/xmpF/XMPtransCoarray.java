package exc.xmpF;

import exc.object.*;
import exc.block.*;
import java.util.*;

/*
 * Translate Fortran Coarray
 */
public class XMPtransCoarray 
{
  private Boolean DEBUG = false;       // change me in debugger

  // constants
  final static String DESCRIPTOR_PREFIX  = "xmpf_codescr";
  final static String CRAYPOINTER_PREFIX = "xmpf_crayptr";
  final static String INITPROC_PREFIX = "xmpf_traverse_initcoarray";

  //private FuncDefBlock def;          // (XobjectDef)def.getDef() is more useful????
  // contains
  //  - XobjectDef def
  //  - FunbtionBlock fblock
  // useful methods are:
  //     add/removeIdent, findLocalDecl(String)

  private XMPenv env;

  private String name;

  private XobjectDef def;
  private FunctionBlock fblock;

  private Vector<XMPcoarray> staticCoarrays;
  private Vector<XMPcoarray> allocCoarrays;

  //private XMPinitProcedure initProcedure;
  private String initProcTextForFile;

  private String newProcName;
  private String commonName1, commonName2;

  //------------------------------
  //  CONSTRUCTOR
  //------------------------------
  public XMPtransCoarray(FuncDefBlock funcDef, XMPenv env) {
    def = funcDef.getDef();
    fblock = funcDef.getBlock();
    this.env = env;
    name = fblock.getName();
    newProcName = genNewProcName();
    commonName1 = DESCRIPTOR_PREFIX + "_" + name;
    commonName2 = CRAYPOINTER_PREFIX + "_" + name;

    // set all static/allocatable coarrays in this procedure
    staticCoarrays = new Vector<XMPcoarray>();
    allocCoarrays = new Vector<XMPcoarray>();
    Xobject idList = def.getFuncIdList();
    for (Xobject obj: (XobjList)idList) {
      Ident ident = (Ident)obj;
      if (ident.getCorank() > 0) {   // found a coarray
        XMPcoarray coarray = new XMPcoarray(ident, funcDef, env);
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
    Vector<XMPcoarray> allCoarrays = new Vector<XMPcoarray>();
    allCoarrays.addAll(staticCoarrays);
    allCoarrays.addAll(allocCoarrays);

    // (0-1) declare idents of cray pointers & descriptors
    for (XMPcoarray coarray: allCoarrays)
      coarray.declareIdents(CRAYPOINTER_PREFIX, DESCRIPTOR_PREFIX);

    // (0-2) generate common stmt in this procedure
    if (! allCoarrays.isEmpty())
      genCommonStmt(commonName1, commonName2, allCoarrays, def);

    // (1) replace coindexed variable assignment stmts with call stmts
    replaceCoidxVarStmts(allCoarrays);

    // (2) replace coindexed objects with function references
    replaceCoidxObjs(allCoarrays);

    // (3) replace static coarrays & generate allocation
    if (! staticCoarrays.isEmpty())
      transStaticCoarrays(staticCoarrays);

    // (4) replace allocatable coarrays & generate allocation
    if (! allocCoarrays.isEmpty())
      transAllocCoarrays(allocCoarrays);
  }


  //-----------------------------------------------------
  //  TRANSLATION (1)
  //  convert statements whose LHS are coindexed variables
  //  to subroutine calls
  //-----------------------------------------------------
  private void replaceCoidxVarStmts(Vector<XMPcoarray> coarrays) {
    /*****************************
    XobjectIterator xi = new bottomupXobjectIterator(def.getFuncBody());
    for (xi.init(); !xi.end(); xi.next()) {
      Xobject xobj = xi.getXobject();
      if (xobj == null)
        continue;

      if (xobj.Opcode() == Xcode.F_ASSIGN_STATEMENT &&
          xobj.getArg(0).Opcode() == Xcode.CO_ARRAY_REF) {

        // found statement to be converted
        Xobject callExpr = genCallStmt_putArray(xobj, coarrays);
        xi.setXobject(callExpr);
      }

    }
    *******************************/

    BlockIterator bi = new topdownBlockIterator(fblock);

    for (bi.init(); !bi.end(); bi.next()) {
      BasicBlock bb = bi.getBlock().getBasicBlock();
      if (bb == null) continue;
      for (Statement s = bb.getHead(); s != null; s = s.getNext()) {
        Xobject assignExpr = s.getExpr();
        if (assignExpr == null)
          continue;

        if (_isCoidxVarStmt(assignExpr)) {
          // found -- convert the statement
          Xobject callExpr = genCallStmt_putArray(assignExpr, coarrays);
          //s.insert(callExpr);
          //s.remove();
          s.setExpr(callExpr);
        }
      }
    }
  }


  private Boolean _isCoidxVarStmt(Xobject xobj) {
    if (xobj.Opcode() == Xcode.F_ASSIGN_STATEMENT) {
      Xobject lhs = xobj.getArg(0);
      if (lhs.Opcode() == Xcode.CO_ARRAY_REF)
        return true;
    }
    return false;
  }


  /*
   * convert a statement:
   *    v(s1,s2,...)[cs1,cs2,...] = rhs
   * to:
   *    external :: PutCommLibName
   *    call PutCommLibName(..., rhs)
   */
  private Xobject genCallStmt_putArray(Xobject assignExpr,
                                     Vector<XMPcoarray> coarrays) {
    Xobject lhs = assignExpr.getArg(0);
    Xobject rhs = assignExpr.getArg(1);

    XMPcoindexObj coidxObj = new XMPcoindexObj(lhs, coarrays);
    return coidxObj.genCallStmt_putArray(rhs);
  }

  //-----------------------------------------------------
  //  TRANSLATION (2)
  //  convert coindexed objects to function references
  //-----------------------------------------------------
  private void replaceCoidxObjs(Vector<XMPcoarray> coarrays) {
    XobjectIterator xi = new topdownXobjectIterator(def.getFuncBody());

    for (xi.init(); !xi.end(); xi.next()) {
      Xobject xobj = xi.getXobject();
      if (xobj == null)
        continue;

      if (xobj.Opcode() == Xcode.CO_ARRAY_REF) {
        // found coindexed object to be converted to the function reference

        ///////
        // TEMPORARY 
        ///////
        if (xi.getParent().Opcode() == Xcode.F_ASSIGN_STATEMENT &&
            xi.getParent().getArg(0) == xobj) {
          // found zombi coindexed object
          if (DEBUG) System.out.println("found zombi: "+xobj);
          // do nothing 
        }

        else {
          Xobject funcCall = genFuncRef_getArray(xobj, coarrays);
          xi.setXobject(funcCall);
        }
      }
    }
  }

  /*
   * convert expression:
   *    v(s1,s2,...)[cs1,cs2,...]
   * to:
   *    type,external,dimension(:,:,..) :: commGetLibName_M
   *    commGetLibName_M(...)
   */
  private Xobject genFuncRef_getArray(Xobject funcRef,
                                     Vector<XMPcoarray> coarrays) {
    XMPcoindexObj coidxObj = new XMPcoindexObj(funcRef, coarrays);
    return coidxObj.genFuncRef_getArray();
  }


  //-----------------------------------------------------
  //  TRANSLATION (3)
  //  malloc static coarrays
  //-----------------------------------------------------
  //
  // convert from:
  // --------------------------------------------
  //     subroutine EX1
  //       real :: V1(10,20)[4,*]
  //       complex(8) :: V2[*]
  //     end subroutine
  // --------------------------------------------
  // to:
  // --------------------------------------------
  //     subroutine EX1
  //       real :: V1(10,20)                                  ! a
  //       complex(8) :: V2                                   ! a
  //       integer :: desc_V1                                 ! b
  //       integer :: desc_V2                                 ! b
  //       pointer (ptr_V1, V1)                               ! b
  //       pointer (ptr_V2, V2)                               ! b
  //       common /xmpf_desc_EX1/desc_V1,desc_V2              ! c
  //       common /xmpf_ptr_EX1/ptr_V1,ptr_V2                 ! c
  //
  //     end subroutine
  // --------------------------------------------
  // and generate and add an initialization routine into the
  // same file (see XMPcoarrayInitProcedure)
  //
  private void transStaticCoarrays(Vector<XMPcoarray> coarrays) {

    // remove codimensions form coarray declaration (a)
    for (XMPcoarray coarray: coarrays)
      coarray.removeCodimensions();

    // output init procedure
    XMPcoarrayInitProcedure coarrayInit = new XMPcoarrayInitProcedure();
    coarrayInit.genInitRoutine(coarrays, newProcName,
                               commonName1, commonName2);

    // finalize the init procedure
    coarrayInit.finalize(env);
  }


  //-----------------------------------------------------
  //  TRANSLATION (4)
  //  malloc allocatable coarrays
  //-----------------------------------------------------
  private void transAllocCoarrays(Vector<XMPcoarray> coarrays) {
    for (XMPcoarray coarray: coarrays)
      XMP.error("Not supported: allocatable coarry: "+coarray.getName());
  }


  //-----------------------------------------------------
  //  parts
  //-----------------------------------------------------
  private void genCommonStmt(String commonName1, String commonName2,
                             Vector<XMPcoarray> coarrays, XobjectDef def) {

    /* cf. XMPenv.declOrGetSizeArray */

    // common block name
    Xobject cnameObj1 = Xcons.Symbol(Xcode.IDENT, commonName1);
    Xobject cnameObj2 = Xcons.Symbol(Xcode.IDENT, commonName2);

    // list of common vars
    Xobject varList1 = Xcons.List();
    Xobject varList2 = Xcons.List();
    for (XMPcoarray coarray: coarrays) {
      Ident descrId = coarray.getDescriptorId();
      Ident cptrId = coarray.getCrayPointerId();
      varList1.add(Xcons.FvarRef(descrId));
      varList2.add(Xcons.FvarRef(cptrId));
    }

    // declaration 
    Xobject decls = fblock.getBody().getDecls();
    decls.add(Xcons.List(Xcode.F_COMMON_DECL,
                         Xcons.List(Xcode.F_VAR_LIST, cnameObj1, varList1)));
    decls.add(Xcons.List(Xcode.F_COMMON_DECL,
                         Xcons.List(Xcode.F_VAR_LIST, cnameObj2, varList2)));
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

}

