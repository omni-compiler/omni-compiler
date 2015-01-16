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

  // the current class instances corresponding to the ancestor host procedures
  static ArrayList<XMPtransCoarray> current_hosts = new ArrayList<XMPtransCoarray>();

  //private FuncDefBlock def;          // (XobjectDef)def.getDef() is more useful????
  // contains
  //  - XobjectDef def
  //  - FunbtionBlock fblock
  // useful methods are:
  //     add/removeIdent, findLocalDecl(String)

  private XMPenv env;

  private String name;

  private FuncDefBlock funcDef;
  private XobjectDef def;
  private FunctionBlock fblock;

  private Vector<XMPcoarray> staticLocalCoarrays, allocLocalCoarrays;
  private Vector<XMPcoarray> visibleCoarrays;

  //private XMPinitProcedure initProcedure;
  private String initProcTextForFile;

  private String newProcName;
  private String commonName1, commonName2;

  //------------------------------
  //  CONSTRUCTOR
  //------------------------------
  public XMPtransCoarray(FuncDefBlock funcDef, XMPenv env) {
    this.funcDef = funcDef;
    def = funcDef.getDef();
    fblock = funcDef.getBlock();
    this.env = env;
    env.setCurrentDef(funcDef);      // needed if this is called before XMPrewriteExpr ???
    name = fblock.getName();
    newProcName = genNewProcName();
    commonName1 = DESCRIPTOR_PREFIX + "_" + name;
    commonName2 = CRAYPOINTER_PREFIX + "_" + name;

    XMP.exitByError();   // exit if error has found.

    _setCoarrays();
  }

  private void _setCoarrays() {
    // set coarrays declared in the current procedure
    staticLocalCoarrays = new Vector<XMPcoarray>();
    allocLocalCoarrays = new Vector<XMPcoarray>();

    Xobject idList = def.getFuncIdList();
    for (Xobject obj: (XobjList)idList) {
      Ident ident = (Ident)obj;
      if (ident.wasCoarray()) {
        // found it is a coarray or a variable converted from a coarray
        XMPcoarray coarray = new XMPcoarray(ident, funcDef, env);
        if (coarray.isAllocatable())
          allocLocalCoarrays.add(coarray);
        else
          staticLocalCoarrays.add(coarray);
        coarray.errorCheck();
      }
    }

    // renew the list of the current hosts
    // (assuming top-down analysis)
    XobjectDef pdef = def.getParent();
    if (pdef == null) {
      // no host procedure of mine.  I.e., This is an external procedure.
      current_hosts.clear();
    } else {
      for (int i = current_hosts.size() - 1; i >= 0; i--) {
        if (pdef == current_hosts.get(i).def) {
          // found the host (my parent) procedure
          break;
        }
        current_hosts.remove(i);
      }
    }
    current_hosts.add(this);

    // set all coarrays declared in the current and the host procedures
    visibleCoarrays = new Vector<XMPcoarray>();
    visibleCoarrays.addAll(staticLocalCoarrays);
    visibleCoarrays.addAll(allocLocalCoarrays);
    if (current_hosts.size() > 1) {
      // host association
      XMPtransCoarray host = current_hosts.get(current_hosts.size() - 2);
      visibleCoarrays.addAll(host.visibleCoarrays);
    }

    /****************************
    for (XobjectDef pdef = def.getParent();
         pdef != null; pdef = pdef.getParent()) {
      idList = pdef.getFuncIdList();
      for (Xobject obj: (XobjList)idList) {
        Ident ident = (Ident)obj;
        if (ident.wasCoarray()) {
          // found it is a variable converted from a coarray in the host procedure
          XMPcoarray coarray = new XMPcoarray(ident, funcDef, env);

          // check if it has an overriden name
          String name = ident.getName();
          boolean is_new = true;
          for (XMPcoarray coarray1 : visibleCoarrays) {
            if (name == coarray1.getName()) {
              is_new = false;
              break;
            }
          }
          if (is_new)
            // the name is not found yet.
            visibleCoarrays.add(coarray);
        }
      }
    }
    ********************************/
  }

  //------------------------------
  //  TRANSLATION
  //------------------------------
  public void run() {
    // a. declare cray-pointers and descriptors and
    //     generate common stmt inside this procedure
    genCommonStmt(commonName1, commonName2, staticLocalCoarrays, def);
    genCommonStmt(commonName1, commonName2, allocLocalCoarrays, def);

    // e. replace coindexed objects with function references
    replaceCoidxObjs(visibleCoarrays);

    // d. replace coindexed variable assignment stmts with call stmts
    replaceCoidxVarStmts(visibleCoarrays);

    // b. replace static coarrays & generate allocation
    transStaticCoarrays(staticLocalCoarrays);

    // c. replace allocatable coarrays & generate allocation
    transAllocCoarrays(allocLocalCoarrays);
  }


  //-----------------------------------------------------
  //  TRANSLATION a.
  //  declare cray-pointers and descriptors and
  //  generate common stmt in this procedure
  //-----------------------------------------------------
  private void genCommonStmt(String commonName1, String commonName2,
                             Vector<XMPcoarray> coarrays, XobjectDef def) {
    // do nothing if no coarrays are declared.
    if (coarrays.isEmpty())
      return;

    // declare idents of cray pointers & descriptors
    for (XMPcoarray coarray: coarrays)
      coarray.declareIdents(CRAYPOINTER_PREFIX, DESCRIPTOR_PREFIX);

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


  //-----------------------------------------------------
  //  TRANSLATION d. (PUT)
  //  convert statements whose LHS are coindexed variables
  //  to subroutine calls
  //-----------------------------------------------------
  private void replaceCoidxVarStmts(Vector<XMPcoarray> coarrays) {
    /************************
    //XobjectIterator xi = new bottomupXobjectIterator(def.getFuncBody());
    XobjectIterator xi = new topdownXobjectIterator(def.getFuncBody());

    for (xi.init(); !xi.end(); xi.next()) {
      Xobject xobj = xi.getXobject();

      if (xobj == null) {
        continue;
      }

      if (xobj.Opcode() == Xcode.F_ASSIGN_STATEMENT) {
        // found an assignment statement

        if (xobj.getArg(0).Opcode() == Xcode.CO_ARRAY_REF) {
          // found its left-hand side is a coindexed variable
          Xobject callExpr = coidxVarStmtToCallStmt(xobj, coarrays);
          xi.setXobject(callExpr);
        }
      }

    }
    *********************/

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
          Xobject callExpr = coidxVarStmtToCallStmt(assignExpr, coarrays);
          //s.insert(callExpr);
          //s.remove();
          s.setExpr(callExpr);
        }
      }

      /***************
      Xobject stmt = bi.getBlock().toXobject();
      if (_isCoidxVarStmt(stmt)) {
        // found an assignment stmt whose LHS is a coindexed variable
        Xobject callExpr = coidxVarStmtToCallStmt(stmt, coarrays);
        bi.setBlock(Bcons.Statement(callExpr));
      }
      ***************/
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
  private Xobject coidxVarStmtToCallStmt(Xobject assignExpr,
                                         Vector<XMPcoarray> coarrays) {
    Xobject lhs = assignExpr.getArg(0);
    Xobject rhs = assignExpr.getArg(1);

    XMPcoindexObj coidxObj = new XMPcoindexObj(lhs, coarrays);
    return coidxObj.toCallStmt(rhs);
  }


  //-----------------------------------------------------
  //  TRANSLATION e. (GET)
  //  convert coindexed objects to function references
  //-----------------------------------------------------
  private void replaceCoidxObjs(Vector<XMPcoarray> coarrays) {
    // itaration to solve nested reference of coindexed object.
    while (_replaceCoidxObjs1(coarrays));
  }

  private Boolean _replaceCoidxObjs1(Vector<XMPcoarray> coarrays) {
    XobjectIterator xi = new topdownXobjectIterator(def.getFuncBody());

    Boolean done = false;
    for (xi.init(); !xi.end(); xi.next()) {
      Xobject xobj = xi.getXobject();
      if (xobj == null)
        continue;

      if (xobj.Opcode() == Xcode.CO_ARRAY_REF) {
        Xobject parent = (Xobject)xobj.getParent();

        if (parent.Opcode() == Xcode.F_ASSIGN_STATEMENT &&
            parent.getArg(0) == xobj)
          // found a coindexed variable, which is LHS of an assignment stmt.
          continue;  // do nothing 

        // found target to convert
        Xobject funcCall = coidxObjToFuncRef(xobj, coarrays);
        xi.setXobject(funcCall);
        done = true;
      }
    }

    return done;
  }

  /*
   * convert expression:
   *    v(s1,s2,...)[cs1,cs2,...]
   * to:
   *    type,external,dimension(:,:,..) :: commGetLibName_M
   *    commGetLibName_M(...)
   */
  private Xobject coidxObjToFuncRef(Xobject funcRef,
                                    Vector<XMPcoarray> coarrays) {
    XMPcoindexObj coidxObj = new XMPcoindexObj(funcRef, coarrays);
    return coidxObj.toFuncRef();
  }


  //-----------------------------------------------------
  //  TRANSLATION b.
  //  malloc static coarrays
  //-----------------------------------------------------
  //
  // convert from:
  // --------------------------------------------
  //     subroutine EX1
  //       real :: V1(10,20)[4,*]
  //       complex(8) :: V2[*]
  //       ...
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
  //       ...
  //     end subroutine
  // --------------------------------------------
  // and generate and add an initialization routine into the
  // same file (see XMPcoarrayInitProcedure)
  //
  private void transStaticCoarrays(Vector<XMPcoarray> coarrays) {
    // do nothing if no coarrays are declared.
    if (coarrays.isEmpty())
      return;

    // remove codimensions form coarray declaration
    for (XMPcoarray coarray: coarrays)
      coarray.hideCodimensions();

    // output init procedure
    XMPcoarrayInitProcedure coarrayInit = new XMPcoarrayInitProcedure();
    coarrayInit.genInitRoutine(coarrays, newProcName,
                               commonName1, commonName2);

    // finalize the init procedure
    coarrayInit.finalize(env);
  }


  //-----------------------------------------------------
  //  TRANSLATION c.
  //  malloc allocatable coarrays
  //-----------------------------------------------------
  private void transAllocCoarrays(Vector<XMPcoarray> coarrays) {
    // do nothing if no coarrays are declared.
    if (coarrays.isEmpty())
      return;

    for (XMPcoarray coarray: coarrays)
      XMP.error("Not supported: allocatable coarry: "+coarray.getName());
  }


  //-----------------------------------------------------
  //  parts
  //-----------------------------------------------------
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

