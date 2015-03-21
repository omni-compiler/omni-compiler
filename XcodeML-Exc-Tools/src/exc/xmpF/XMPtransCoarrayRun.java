package exc.xmpF;

import exc.object.*;
import exc.block.*;
import java.util.*;

/*
 * Translate Coarray Fortran (for each procedure)
 */
public class XMPtransCoarrayRun
{
  private Boolean DEBUG = false;       // change me in debugger

  // constants
  final static String DESCPOINTER_PREFIX = "xmpf_descptr";
  final static String CRAYPOINTER_PREFIX = "xmpf_crayptr";
  final static String COARRAYSIZE_PREFIX = "xmpf_traverse_countcoarray";
  final static String COARRAYINIT_PREFIX = "xmpf_traverse_initcoarray";
  final static String DEF_DESCPOINTER_NAME = "xmpf_coarray_descptr";
  final static String PROC_INIT_NAME = "xmpf_coarray_proc_init";
  final static String PROC_FINALIZE_NAME = "xmpf_coarray_proc_finalize";
  final static String TAG_NAME = "xmpf_coarray_resource_tag";
  public final static String SET_COSHAPE_NAME = "xmpf_coarray_set_coshape";
  final static String COARRAYALLOC_PREFIX = "xmpf_coarray_alloc";

  // to handle host- and use-associations
  static ArrayList<XMPtransCoarrayRun> ancestors
    = new ArrayList<XMPtransCoarrayRun>();

  private XMPenv env;

  private String name;

  private FuncDefBlock funcDef;
  private XobjectDef def;
  private FunctionBlock fblock;

  private Vector<XMPcoarray> localCoarrays;
  private Vector<XMPcoarray> useAssociatedCoarrays;
  private Vector<XMPcoarray> visibleCoarrays;

  //private XMPinitProcedure initProcedure;
  private String initProcTextForFile;

  private String sizeProcName, initProcName;
  private String commonName1, commonName2;
  private Ident resourceTagId;

  private ArrayList<Xobject> _prologStmts = new ArrayList();
  private ArrayList<Xobject> _epilogStmts = new ArrayList();


  //------------------------------------------------------------
  //  CONSTRUCTOR
  //------------------------------------------------------------
  public XMPtransCoarrayRun(FuncDefBlock funcDef, XMPenv env) {
    this.funcDef = funcDef;
    def = funcDef.getDef();
    fblock = funcDef.getBlock();
    this.env = env;
    //// I don't know. See [Xmp-dev:5185]
    env.setCurrentDef(funcDef);      // needed if this is called before XMPrewriteExpr ???
    name = fblock.getName();
    String postfix = genNewProcPostfix();
    sizeProcName = COARRAYSIZE_PREFIX + postfix;
    initProcName = COARRAYINIT_PREFIX + postfix;
    commonName1 = DESCPOINTER_PREFIX + "_" + name;
    commonName2 = CRAYPOINTER_PREFIX + "_" + name;

    _setCoarrays();
    _check_ifIncludeXmpLib();
    _setResourceIagId();

    XMP.exitByError();   // exit if error has found.
  }

  private void _setResourceIagId() {
    BlockList blist = fblock.getBody();
    resourceTagId = blist.declLocalIdent(TAG_NAME,
                                 BasicType.Fint8Type,
                                 StorageClass.FLOCAL,
                                 null);
  }

  private void _setCoarrays() {
    // set localCoarrays as coarrays declared in the current procedure
    // set useAssociatedCoarrays as coarrays declared in used modules
    _setLocalCoarrays();

    // renew the list of the current hosts
    // (assuming top-down analysis)
    XobjectDef pdef = def.getParent();
    if (pdef == null) {
      // I have no host procedure.  I.e., I am an external procedure.
      ancestors.clear();
    } else {
      for (int i = ancestors.size() - 1; i >= 0; i--) {
        if (pdef == ancestors.get(i).def) {
          // found the host (my parent) procedure
          break;
        }
        ancestors.remove(i);
      }
    }
    ancestors.add(this);

    // set all coarrays declared in the current and the host procedures
    visibleCoarrays = new Vector<XMPcoarray>();
    visibleCoarrays.addAll(localCoarrays);
    visibleCoarrays.addAll(useAssociatedCoarrays);
    if (ancestors.size() > 1) {
      // host association
      XMPtransCoarrayRun host = ancestors.get(ancestors.size() - 2);
      visibleCoarrays.addAll(host.visibleCoarrays);
    }
  }


  private void _setLocalCoarrays() {
    localCoarrays = new Vector<XMPcoarray>();
    useAssociatedCoarrays = new Vector<XMPcoarray>();

    Xobject idList = def.getFuncIdList();
    for (Xobject obj: (XobjList)idList) {
      Ident ident = (Ident)obj;
      if (ident.wasCoarray()) {
        // found it is a coarray or a variable converted from a coarray
        XMPcoarray coarray = new XMPcoarray(ident, funcDef, env);
        if (coarray.isUseAssociated())
          useAssociatedCoarrays.add(coarray);
        else
          localCoarrays.add(coarray);
      }
    }
  }



  //------------------------------------------------------------
  //  TRANSLATION
  //------------------------------------------------------------

  /*
   *  PASS 1: for each procedure that is either 
   *            - an external function/subroutine or
   *            - an internal function/subroutine or
   *            - a module function/subroutine
   *          except module
   */
  public void run1() {
    // error check for each coarray declaration
    for (XMPcoarray coarray: localCoarrays)
      coarray.errorCheck();

    // resolve use association of static coarrays
    for (XMPcoarray coarray: useAssociatedCoarrays) {
      if (coarray.isAllocatable())
        continue;
      // found a use-associated static coarray 
      Ident ident = coarray.getIdent();
      ident.setFdeclaredModule(null);
      localCoarrays.add(coarray);
    }

    // divide local coarrays into procedure-local and dummy arg
    Vector<XMPcoarray> procLocalCoarrays = new Vector<XMPcoarray>();
    Vector<XMPcoarray> dummyArgCoarrays = new Vector<XMPcoarray>();
    for (XMPcoarray coarray: localCoarrays) {
      if (coarray.isDummyArg())
        dummyArgCoarrays.add(coarray);
      else
        procLocalCoarrays.add(coarray);
    }

    // convert specification and declaration part
    Boolean sw1, sw2;
    sw1 = transDeclPart_procedureLocal(procLocalCoarrays);
    sw2 = transDeclPart_dummyArg(dummyArgCoarrays);
    transExecPart(visibleCoarrays, (sw1||sw2));

    // finalize (see XMPtranslate.java)
    ///// I don't know why this call is needed or not needed.
    ///// see [XMP-dev:5185] 2015.03.21
    funcDef.Finalize();
  }


  /*
   *  PASS 2: for each module 
   *          excluding its module functions and subroutines
   */
  public void run2() {
    // error check for each coarray declaration
    for (XMPcoarray coarray: localCoarrays)
      coarray.errorCheck();

    // convert specification and declaration part
    transDeclPart_moduleLocal(localCoarrays);

    // finalize (see XMPtranslate.java)
    ///// I don't know why this call is needed or not needed.
    ///// see [XMP-dev:5185] 2015.03.21
    funcDef.Finalize();
  }


  private void transExecPart(Vector<XMPcoarray> visibleCoarrays, Boolean initSwitch) {

    // e. convert coindexed objects to function references
    convCoidxObjsToFuncCalls(visibleCoarrays);

    // d. convert coindexed variable assignment stmts to call stmts
    convCoidxStmtsToSubrCalls(visibleCoarrays);

    // j. for allocated coarrays, convert allocate/deallocate stmts and
    //    fake intrinsic function calls
    convReferenceOfAllocCoarrays(visibleCoarrays);

    // i. initialization/finalization of local resources
    if (initSwitch)
      genCallOfInitAndFin();

    // resolve prologue/epilogue code generations
    genPrologStmts();
    genEpilogStmts();
  }


  /**
    example of procedure-local coarrays
    --------------------------------------------
      subroutine EX1
        use EX2  ! including definition of a coarray
                 !   real :: V1(10,20)[4,*]            ! static local
        real, save :: V1(10,20)[4,*]                   ! static local
        complex(8), save :: V2[0:*]                    ! static local
        integer, allocatable :: V3(:)[:,:]             ! allocatable local
        ...
        V1(1:3,j)[k1,k2] = (/1.0,2.0,3.0/)             ! put 1D
        z = V2[k]**2                                   ! get 0D
        allocate (V3(1:10))                            ! allocate
        return                             ! auto-dealloc and free resources
      end subroutine
    --------------------------------------------
    output:
    --------------------------------------------
      subroutine EX1
        use EX2
        real :: V1(1:10,1:20)                                ! f
        complex(8) :: V2                                     ! f
        integer, POINTER :: V3(:)                            ! f,h

        integer(8) :: DP_V1, DP_V2, DP_V3                    ! a
        pointer (CP_V1, V1)                                  ! c
        pointer (CP_V2, V2)                                  ! c
        common /xmpf_DP_EX1/ DP_V1, DP_V2                    ! g
        common /xmpf_CP_EX1/ CP_V1, CP_V2                    ! g
        integer(8) :: tag                                    ! i
        call xmpf_coarray_proc_init(tag)                     ! i
        ...
        call xmpf_coarray_put(DP_V1, V1(1,j), 4, &           ! d
          k1+4*(k2-1), (/1.0,2.0,3.0/), ...)      
        z = xmpf_coarray_get0d(DP_V2, V2, 16, k, 0) ** 2     ! e
        call xmpf_coarray_alloc1d(DP_V3, V3, tag, 1, 1, 10)  ! j
        call xmpf_coarray_set_coshape(DP_V3, 2, k1, k2, 0)   ! jc
        call xmpf_coarray_proc_finalize(tag)                 ! i
        return
      end subroutine

    !! Additionally, two subroutines xmpf_traverse_* will    ! b
    !! be generated into the same output file to initialize
    !! DP_Vn and CP_Vn.  See XMPcoarrayInitProcedure.
    --------------------------------------------
      DP_Vn: pointer to descriptor of each coarray Vn
      CP_Vn: cray poiter to the coarray object Vn
  */
  private Boolean transDeclPart_procedureLocal(Vector<XMPcoarray> localCoarrays) {
    Boolean initSwitch = false;

    // divide procedure-local coarrays into static and allocatable
    Vector<XMPcoarray> staticLocalCoarrays = new Vector<XMPcoarray>();
    Vector<XMPcoarray> allocatableLocalCoarrays = new Vector<XMPcoarray>();
    for (XMPcoarray coarray: localCoarrays) {
      if (coarray.isAllocatable()) {
        initSwitch = true;
        allocatableLocalCoarrays.add(coarray);
      } else {
        staticLocalCoarrays.add(coarray);
      }
    }

    // a. declare descriptor pointers
    genDeclOfDescPointer(localCoarrays);

    // c. declare cray-pointers (static coarrays only)
    genDeclOfCrayPointer(staticLocalCoarrays);

    // g. generate common stmt (static coarrays only)
    genCommonStmt(staticLocalCoarrays);

    // b. generate allocation into init procedure (static coarrays only)
    genAllocOfStaticCoarrays(staticLocalCoarrays);

    // f. remove codimensions from declarations of coarrays
    removeCodimensionsFromCoarrays(localCoarrays);

    // h. replace allocatable attributes with pointer attributes and
    //    fake intrinsic 'allocatable' (allocatable coarrays only)
    replaceAllocatableWithPointer(allocatableLocalCoarrays);

    // if there are any allocatale local coarrays, init/final is needed.
    return initSwitch;
  }


  /**
    example of dummy argument coarrays
    --------------------------------------------
      subroutine EX1(V1,V2,V3)
        real :: V1(10,20)[4,*]                    ! static dummy
        complex(8) :: V2[0:*]                     ! static dummy
        integer, allocatable :: V3(:)[:,:]        ! allocatable dummy
        ...
        V1(1:3,j)[k1,k2] = (/1.0,2.0,3.0/)        ! put 1D
        z = V2[k]**2                              ! get 0D
        allocate (V3(1:10)[k1:k12,0:*])           ! allocate
        return                                ! free resources
      end subroutine
    --------------------------------------------
    output:
    --------------------------------------------
      subroutine EX1(V1,V2,V3)
        real :: V1(1:10,1:20)                                ! f
        complex(8) :: V2                                     ! f
        integer, POINTER :: V3(:)                            ! f,h

        integer(8) :: DP_V1, DP_V2, DP_V3                    ! a
        integer(8) :: tag                                    ! i
        ...
        call xmpf_coarray_proc_init(tag)                     ! i
        call xmpf_coarray_descptr(DP_V1, V1, tag)            ! k
        call xmpf_coarray_descptr(DP_V2, V2, tag)            ! k
        call xmpf_coarray_descptr(DP_V3, V3, tag)            ! k
        call xmpf_coarray_set_coshape(DP_V1, 2, 1, 4, 1)     ! kc
        call xmpf_coarray_set_coshape(DP_V2, 1, 0)           ! kc
        call xmpf_coarray_put(DP_V1, V1(1,j), 4, &           ! d
          k1+4*(k2-1), (/1.0,2.0,3.0/), ...)      
        z = xmpf_coarray_get0d(DP_V2, V2, 16, k, 0) ** 2     ! e
        call xmpf_coarray_alloc1d(DP_V3, V3, tag, 1, 1, 10)  ! j
        call xmpf_coarray_set_coshape(DP_V3, 2, k1, k2, 0)   ! jc
        call xmpf_coarray_proc_finalize(tag)                 ! i
        return
      end subroutine
    --------------------------------------------
      DP_Vn: pointer to descriptor of each coarray Vn
      CP_Vn: cray poiter to the coarray object Vn
  */
  private Boolean transDeclPart_dummyArg(Vector<XMPcoarray> localCoarrays) {
    // if there are not dummy argument coarrays, return
    if (localCoarrays.isEmpty())
      return false;

    // select static local coarrays
    Vector<XMPcoarray> staticLocalCoarrays = new Vector<XMPcoarray>();
    Vector<XMPcoarray> allocatableLocalCoarrays = new Vector<XMPcoarray>();
    Vector<XMPcoarray> dummyLocalCoarrays = new Vector<XMPcoarray>();
    for (XMPcoarray coarray: localCoarrays) {
      if (coarray.isAllocatable())
        allocatableLocalCoarrays.add(coarray);
      if (coarray.isDummyArg())
        dummyLocalCoarrays.add(coarray);
      if (!coarray.isAllocatable() && !coarray.isDummyArg())
        staticLocalCoarrays.add(coarray);
    }

    // a. declare descriptor pointers
    genDeclOfDescPointer(localCoarrays);

    // k.& kc. generate definition of descriptor pointers (dummy coarrays only)
    genDefinitionOfDescPointer(dummyLocalCoarrays);

    // f. remove codimensions from declarations of coarrays
    removeCodimensionsFromCoarrays(localCoarrays);

    // h. replace allocatable attributes with pointer attributes and
    //    fake intrinsic 'allocatable' (allocatable coarrays only)
    replaceAllocatableWithPointer(allocatableLocalCoarrays);

    // if there are any dummy argument coarrays, init/final is needed.
    return true;
  }


  /**
    example of module-local coarrays
    --------------------------------------------
      module EX1
        real :: V1(10,20)[4,*]                ! static 
        complex(8) :: V2[0:*]                 ! static 
        integer, allocatable :: V3(:)[:,:]    ! allocatable
        ...
      end module
    --------------------------------------------
    output:
    --------------------------------------------
      subroutine EX1
       !! real :: V1(10,20)[4,*]     delete                  ! m
       !! complex(8) :: V2[0:*]      delete                  ! m
        integer, POINTER :: V3(:)                            ! f,h

        integer(8) :: DP_V3                                  ! a
        ...
      end subroutine
    --------------------------------------------
      DP_Vn: pointer to descriptor of each coarray Vn
      CP_Vn: cray poiter to the coarray object Vn
  */
  private void transDeclPart_moduleLocal(Vector<XMPcoarray> localCoarrays) {

    // select static local coarrays
    Vector<XMPcoarray> staticLocalCoarrays = new Vector<XMPcoarray>();
    Vector<XMPcoarray> allocatableLocalCoarrays = new Vector<XMPcoarray>();
    for (XMPcoarray coarray: localCoarrays) {
      if (coarray.isAllocatable())
        allocatableLocalCoarrays.add(coarray);
      else
        staticLocalCoarrays.add(coarray);
    }

    // a. declare descriptor pointers (allocatable coarrays only)
    genDeclOfDescPointer(allocatableLocalCoarrays);

    // f. remove codimensions from declarations of coarrays
    // (allocatable coarrays only)
    removeCodimensionsFromCoarrays(allocatableLocalCoarrays);

    // h. replace allocatable attributes with pointer attributes and
    //    fake intrinsic 'allocatable' (allocatable coarrays only)
    replaceAllocatableWithPointer(allocatableLocalCoarrays);

    // m. remove declarations of variables (static coarrays only)
    removeDeclOfCoarrays(staticLocalCoarrays);
  }



  //-----------------------------------------------------
  //  TRANSLATION a.
  //  declare variables of descriptor pointers
  //-----------------------------------------------------
  //
  private void genDeclOfDescPointer(Vector<XMPcoarray> coarrays) {
    for (XMPcoarray coarray: coarrays) {
      // set coarray.descPtrName and 
      // generate declaration of the variable pointing the descriptor
      coarray.genDecl_descPointer(DESCPOINTER_PREFIX);
    }
  }


  //-----------------------------------------------------
  //  TRANSLATION c.
  //  declare cray-pointers
  //-----------------------------------------------------
  //
  private void genDeclOfCrayPointer(Vector<XMPcoarray> coarrays) {
    for (XMPcoarray coarray: coarrays) {
      // set coarray.crayPtrName and
      // generate declaration of the cray pointer
      coarray.genDecl_crayPointer(CRAYPOINTER_PREFIX);
    }
  }


  //-----------------------------------------------------
  //  TRANSLATION i.
  //  generate initialization and finalization calls
  //-----------------------------------------------------
  //
  private void genCallOfInitAndFin() {
    // generate "call proc_init(tag)" and insert to the top
    Xobject args1 = Xcons.List(Xcons.FvarRef(resourceTagId));
    //// Rescriction of OMNI: blist.findIdent() cannot find the name defined
    //// in any interface block. Gave up using interface bloc
    Ident fname1 = env.declExternIdent(PROC_INIT_NAME,
                                       BasicType.FexternalSubroutineType);
    Xobject call1 = fname1.callSubroutine(args1);
    insertPrologStmt(call1);

    // generate "call proc_finalize(tag)" and add to the tail
    Xobject args2 = Xcons.List(Xcons.FvarRef(resourceTagId));
    Ident fname2 = env.declExternIdent(PROC_FINALIZE_NAME,
                                       BasicType.FexternalSubroutineType);
    Xobject call2 = fname2.callSubroutine(args2);
    addEpilogStmt(call2);
  }


  //-----------------------------------------------------
  //  TRANSLATION k. & kc.
  //  generate definition of descriptor pointers
  //-----------------------------------------------------
  //
  private void genDefinitionOfDescPointer(Vector<XMPcoarray> coarrays) {
    Xobject args, subrCall;
    Ident subr, descPtrId;

    for (XMPcoarray coarray: coarrays) {
      // k.
      descPtrId = coarray.getDescPointerId();
      args = Xcons.List(descPtrId, coarray.getIdent(),
                        Xcons.FvarRef(resourceTagId));
      subr = env.declExternIdent(DEF_DESCPOINTER_NAME,
                                 BasicType.FexternalSubroutineType);
      subrCall = subr.callSubroutine(args);
      addPrologStmt(subrCall);

      if (coarray.isAllocatable())
        continue;

      // kc.
      int corank = coarray.getCorank();
      args = Xcons.List(descPtrId, Xcons.IntConstant(corank));
      for (int i = 0; i < corank - 1; i++) {
        args.add(coarray.getLcobound(i));
        args.add(coarray.getUcobound(i));
      }
      args.add(coarray.getLcobound(corank - 1));
      subr = env.declExternIdent(SET_COSHAPE_NAME,
                                 BasicType.FexternalSubroutineType);
      subrCall = subr.callSubroutine(args);
      addPrologStmt(subrCall);
    }
  }


  //-----------------------------------------------------
  //  TRANSLATION g.
  //  generate common stmt in this procedure
  //-----------------------------------------------------
  //
  private void genCommonStmt(Vector<XMPcoarray> coarrays) {
    // do nothing if no coarrays are declared.
    if (coarrays.isEmpty())
      return;

    // common block name
    Xobject cnameObj1 = Xcons.Symbol(Xcode.IDENT, commonName1);
    Xobject cnameObj2 = Xcons.Symbol(Xcode.IDENT, commonName2);

    // list of common vars
    Xobject varList1 = Xcons.List();
    Xobject varList2 = Xcons.List();
    for (XMPcoarray coarray: coarrays) {
      Ident descPtrId = coarray.getDescPointerId();
      Ident crayPtrId = coarray.getCrayPointerId();
      varList1.add(Xcons.FvarRef(descPtrId));
      varList2.add(Xcons.FvarRef(crayPtrId));
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
  private void convCoidxStmtsToSubrCalls(Vector<XMPcoarray> coarrays) {
    BlockIterator bi = new topdownBlockIterator(fblock);

    for (bi.init(); !bi.end(); bi.next()) {

      BasicBlock bb = bi.getBlock().getBasicBlock();
      if (bb == null) continue;
      for (Statement s = bb.getHead(); s != null; s = s.getNext()) {
        Xobject assignExpr = s.getExpr();
        if (assignExpr == null)
          continue;

        if (_isCoindexVarStmt(assignExpr)) {
          // found -- convert the statement
          Xobject callExpr = coindexVarStmtToCallStmt(assignExpr, coarrays);
          //s.insert(callExpr);
          //s.remove();
          s.setExpr(callExpr);
        }
      }
    }
  }


  private Boolean _isCoindexVarStmt(Xobject xobj) {
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
  private Xobject coindexVarStmtToCallStmt(Xobject assignExpr,
                                         Vector<XMPcoarray> coarrays) {
    Xobject lhs = assignExpr.getArg(0);
    Xobject rhs = assignExpr.getArg(1);

    int condition = _getConditionOfCoarrayPut(rhs);

    XMPcoindexObj coindexObj = new XMPcoindexObj(lhs, coarrays);
    return coindexObj.toCallStmt(rhs, Xcons.IntConstant(condition));
  }

  /*
   * condition 1: It may be necessary to use buffer copy.
   *              The address of RHS may not be accessed by FJ-RDMA.
   * condition 0: Otherwise.
   */
  private int _getConditionOfCoarrayPut(Xobject rhs) {
    if (rhs.isConstant())
      return 1;

    if (rhs.Opcode() == Xcode.F_ARRAY_CONSTRUCTOR)
      return 1;

    return 0;
  }

  //-----------------------------------------------------
  //  TRANSLATION e. (GET)
  //  convert coindexed objects to function references
  //-----------------------------------------------------
  private void convCoidxObjsToFuncCalls(Vector<XMPcoarray> coarrays) {
    // itaration to solve nested reference of coindexed object.
    while (_convCoidxObjsToFuncCalls1(coarrays));
  }

  private Boolean _convCoidxObjsToFuncCalls1(Vector<XMPcoarray> coarrays) {
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
        Xobject funcCall = coindexObjToFuncRef(xobj, coarrays);
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
  private Xobject coindexObjToFuncRef(Xobject funcRef,
                                      Vector<XMPcoarray> coarrays) {
    XMPcoindexObj coindexObj = new XMPcoindexObj(funcRef, coarrays);
    return coindexObj.toFuncRef();
  }


  //-----------------------------------------------------
  //  TRANSLATION b.
  //  generate allocation of static coarrays
  //-----------------------------------------------------
  // and generate and add an initialization routine into the
  // same file (see XMPcoarrayInitProcedure)
  //
  private void genAllocOfStaticCoarrays(Vector<XMPcoarray> coarrays) {
    // do nothing if no coarrays are declared.
    if (coarrays.isEmpty())
      return;

    // output init procedure
    XMPcoarrayInitProcedure coarrayInit = 
      new XMPcoarrayInitProcedure(coarrays, sizeProcName, initProcName,
                                  commonName1, commonName2, env);
    coarrayInit.run();
  }


  //-----------------------------------------------------
  //  TRANSLATION j. & jc.
  //  convert allocate/deallocate stmts for allocated coarrays
  //-----------------------------------------------------
  //
  private void convReferenceOfAllocCoarrays(Vector<XMPcoarray> coarrays) {
    BasicBlockIterator bbi =
      new BasicBlockIterator(fblock);    // see XMPrewriteExpr iter3 loop
    
    for (bbi.init(); !bbi.end(); bbi.next()) {
      StatementIterator si = bbi.getBasicBlock().statements();
      while (si.hasNext()){
	Statement st = si.next();
	Xobject xobj = st.getExpr();
	if (xobj == null || xobj.Opcode() == null)
          continue;

	switch (xobj.Opcode()) {
        case F_ALLOCATE_STATEMENT:
          // xobj.getArg(0): stat= identifier 
          //     (Reference of a variable name is only supported.)
          // xobj.getArg(1): list of variables to be allocated
          // errmsg= identifier is not supported either.
          if (_doesListHaveCoarray(xobj.getArg(1), coarrays)) {
            ArrayList<Xobject> fstmts = conv_allocateStmt(xobj, coarrays);
            LineNo lineno = xobj.getLineNo();
            for (Xobject fstmt: fstmts) {
              fstmt.setLineNo(lineno);
              st.insert(fstmt);
            }
            st.remove();
          }
          break;

        case F_DEALLOCATE_STATEMENT:
          if (_doesListHaveCoarray(xobj.getArg(1), coarrays)) {
            ArrayList<Xobject> fstmts = conv_deallocateStmt(xobj, coarrays);
            //// SET!!!
          }
          break;
        }
      }
    }
  }


  private Boolean _doesListHaveCoarray(Xobject args,
                                       Vector<XMPcoarray> coarrays) {
    Boolean allCoarray = true;
    Boolean allNoncoarray = true;
    for (Xobject arg: (XobjList)args) {
      String varname = arg.getArg(0).getString();
      Boolean found = false;
      for (XMPcoarray coarray: coarrays) {
        if (varname.equals(coarray.getName())) {
          // found coarray
          found = true;
          break;
        }
      }

      // error check for each arg
      if (found && allCoarray)
        allNoncoarray = false;
      else if (!found && allNoncoarray)
        allCoarray = false;
      else {
        // found both coarray and non-coarray
        XMP.error("current restriction: both coarray and non-coarray cannot be in the same ALLOCATE/DEALLOCATE statement.");
      }
    }

    return allCoarray;
  }

  private Boolean _isCoarrayArg(Xobject fcall, Vector<XMPcoarray> coarrays) {
    Xobject args = fcall.getArg(1);
    Xobject arg = args.getArg(0);
    String name = arg.getName();

    for (XMPcoarray coarray: coarrays) {
      if (name.equals(coarray.getName()))
        return true;
    }
    return false;
  }

  // assumed that name is a name of intrinsic function.
  //
  private Boolean _isIntrinsic(Xobject obj) {

    // TEMPORARY JUDGEMENT: If name is registered as an ident, it 
    // is not a name of intrinsic function. Else, it is regarded
    // as a name of intrinsic function.
    Ident id = env.findVarIdent(name, fblock);
    if (id == null)
      return true;          // regarded as intrinsic 

    return id.Type().isFintrinsic();
  }

  private ArrayList<Xobject> conv_allocateStmt(Xobject x, Vector<XMPcoarray> coarrays) {
    ArrayList<Xobject> newStmts = new ArrayList<Xobject>();

    for (Xobject arg: (XobjList)x.getArg(1)) {
      Xobject varname = arg.getArg(0);
      XMPcoarray coarray = _findCoarrayInCoarrays(varname, coarrays);
      XobjList shape = Xcons.List();
      int rank = arg.getArg(1).Nargs() - 1;    //// don't know why F_Front generates such
      for (int i = 0; i < rank; i++)
        shape.add(arg.getArg(1).getArg(i));
      XobjList coshape = (XobjList)arg.getArg(1).getArg(rank);    //// weird

      // TRANSLATION j.
      newStmts.add(makeStmt_coarrayAlloc(coarray, shape));
      // TRANSLATION jc.
      newStmts.add(makeStmt_setCoshape(coarray, coshape));
    }

    return newStmts;
  }


  private Xobject makeStmt_coarrayAlloc(XMPcoarray coarray, XobjList shape) {
    int rank = coarray.getRank();
    if (rank != shape.Nargs()) {
      XMP.error("number of dimensions not matched with the declaration: "
                + rank + " and " + shape.Nargs());
      return null;
    }

    Xobject args = Xcons.List(coarray.getDescPointerId(),
                              Xcons.FvarRef(coarray.getIdent()),
                              Xcons.FvarRef(resourceTagId),
                              Xcons.IntConstant(rank));
    for (int i = 0; i < rank; i++) {
      args.add(_getLboundInIndexRange(shape.getArg(i)));
      args.add(_getUboundInIndexRange(shape.getArg(i)));
    }

    String subrName = COARRAYALLOC_PREFIX + rank + "d";
    Ident subr = env.declExternIdent(subrName,
                                     BasicType.FexternalSubroutineType);
    Xobject subrCall = subr.callSubroutine(args);
    return subrCall;
  }


  private Xobject makeStmt_setCoshape(XMPcoarray coarray, XobjList coshape) {
    int corank = coarray.getCorank();
    if (corank != coshape.Nargs()) {
      XMP.error("number of codimensions not matched with the declaration:"
                + corank + " and " + coshape.Nargs());
      return null;
    }

    Xobject args = Xcons.List(coarray.getDescPointerId(),
                              Xcons.IntConstant(corank));
    for (int i = 0; i < corank - 1; i++) {
      args.add(_getLboundInIndexRange(coshape.getArg(i)));
      args.add(_getUboundInIndexRange(coshape.getArg(i)));
    }
    args.add(_getLboundInIndexRange(coshape.getArg(corank - 1)));

    Ident subr = env.declExternIdent(SET_COSHAPE_NAME,
                                     BasicType.FexternalSubroutineType);
    Xobject subrCall = subr.callSubroutine(args);
    return subrCall;
  }


  private Xobject _getLboundInIndexRange(Xobject dimension) {
    Xobject lbound;

    if (dimension == null)
      lbound = null;
    else if (dimension.Opcode() == Xcode.F_INDEX_RANGE)
      lbound = dimension.getArg(0);
    else
      lbound = null;

    if (lbound == null)
      return Xcons.IntConstant(1);

    return lbound.cfold(fblock);
  }


  private Xobject _getUboundInIndexRange(Xobject dimension) {
    Xobject ubound;

    if (dimension == null)
      ubound = null;
    else if (dimension.Opcode() == Xcode.F_INDEX_RANGE)
      ubound = dimension.getArg(1);
    else
      ubound = dimension;

    if (ubound == null)
      XMP.error("illegal upper bound specified in ALLOCATE statement");

    return ubound.cfold(fblock);
  }


  private XMPcoarray _findCoarrayInCoarrays(Xobject varname,
                                            Vector<XMPcoarray> coarrays) {
    String name = varname.getName();
    for (XMPcoarray coarray: coarrays) {
      if (name.equals(coarray.getName())) {
        return coarray;
      }
    }
    return null;
  }



  private ArrayList<Xobject> conv_deallocateStmt(Xobject x,
                                                 Vector<XMPcoarray> coarrays) {
    System.out.println(" @@@ here conv_deallocateStmt()");
    System.out.println("   x="+x);

    return null;
  }


  //-----------------------------------------------------
  //  TRANSLATION f.
  //  remove codimensions from declaration of coarray
  //-----------------------------------------------------
  //
  private void removeCodimensionsFromCoarrays(Vector<XMPcoarray> coarrays) {
    // remove codimensions form coarray declaration
    for (XMPcoarray coarray: coarrays)
      coarray.hideCodimensions();
  }

  //-----------------------------------------------------
  //  TRANSLATION h.
  //  replace allocatable attributes with pointer attributes and
  //  fake intrinsic function 'allocated'
  //-----------------------------------------------------
  //
  private void replaceAllocatableWithPointer(Vector<XMPcoarray> coarrays) {
    // remove codimensions form coarray declaration
    for (XMPcoarray coarray: coarrays) {
      coarray.resetAllocatable();
      coarray.setPointer();
    }

    XobjectIterator xi = new topdownXobjectIterator(def.getFuncBody());
    for (xi.init(); !xi.end(); xi.next()) {
      Xobject x = xi.getXobject();
      if (x == null)
        continue;
      if (x.Opcode() == null)
        continue;

      switch (x.Opcode()) {
      case FUNCTION_CALL:
        // replace "allocated" with "associated"
        Xobject fname = x.getArg(0);
        if (fname.getString().equalsIgnoreCase("allocated") &&
            _isIntrinsic(fname) &&
            _isCoarrayArg(x, coarrays)) {
          Ident associatedId = declIntIntrinsicIdent("associated");
          x.setArg(0, associatedId);
        }
        break;
      }
    }
  }


  //-----------------------------------------------------
  //  TRANSLATION m.
  //  remove declarations of coarray variables
  //-----------------------------------------------------
  //
  private void removeDeclOfCoarrays(Vector<XMPcoarray> coarrays) {
    for (XMPcoarray coarray: coarrays) {
      coarray.unlinkIdent();
    }
  }




  //-----------------------------------------------------
  //  parts
  //-----------------------------------------------------
  private String genNewProcPostfix() {
    return genNewProcPostfix(getHostNames());
  }

  private String genNewProcPostfix(String ... names) { // host-to-guest order
    int n = names.length;
    String procPostfix = "";
    for (int i = 0; i < n; i++) {
      procPostfix += "_";
      StringTokenizer st = new StringTokenizer(names[i], "_");
      int n_underscore = st.countTokens() - 1;
      if (n_underscore > 0)   // '_' was found in names[i]
        procPostfix += String.valueOf(n_underscore);
      procPostfix += names[i];
    }
    return procPostfix;
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


  //------------------------------------------------------------
  //  ERROR CHECKING
  //------------------------------------------------------------

  /*
   * Detect error if a coarray exists and xmp_lib.h is not included.
   */
  private void _check_ifIncludeXmpLib() {
    
    if (!_isCoarrayReferred() && !_isCoarrayIntrinsicUsed()) {
      /* any coarray features are not used */
      return;
    }

    /* check a typical name defined in xmp_lib.h */
    Ident id = def.findIdent("xmpf_coarray_get0d");
    if (id == null) {
      /* xmpf_lib.h seems not included. */
      XMP.error("current restriction: " + 
                "\'xmp_lib.h\' must be included to use coarray features.");
    }
  }

  private boolean _isCoarrayReferred() {
    if (localCoarrays.isEmpty())
      return false;
    return true;
  }

  private boolean _isCoarrayIntrinsicUsed() {
    final String[] _coarrayIntrinsics = {
      "xmpf_sync_all",
      "xmpf_sync_images", 
      "xmpf_lock",
      "xmpf_unlock",
      "xmpf_critical",
      "xmpf_end_critical",
      "xmpf_sync_memory",
      "xmpf_error_stop",
      };
    final List coarrayIntrinsics = 
      Arrays.asList(_coarrayIntrinsics);

    XobjList identList = def.getDef().getIdentList();
    for (Xobject x: identList) {
      Ident id = (Ident)x;
      if (coarrayIntrinsics.contains(id.getName()))
        return true;
    }
    return false;
  }


  //------------------------------
  //  tool
  //------------------------------
  private Ident declIntIntrinsicIdent(String name) { 
    FunctionType ftype = new FunctionType(Xtype.FintType, Xtype.TQ_FINTRINSIC);
    Ident ident = env.declIntrinsicIdent(name, ftype);
    return ident;
  }


  // add at the tail of _prologStmts
  private void addPrologStmt(Xobject stmt) {
    _prologStmts.add(stmt);
  }

  // add at the top of _prologStmts
  private void insertPrologStmt(Xobject stmt) {
    _prologStmts.add(0, stmt);
  }

  // add at the tail of _epilogStmts
  private void addEpilogStmt(Xobject stmt) {
    _epilogStmts.add(stmt);
  }

  // add at the top of _epilogStmts
  private void insertEpilogStmt(Xobject stmt) {
    _epilogStmts.add(0, stmt);
  }

  private void genPrologStmts() {
    BlockList blist;
    try {
      blist = fblock.getBody().getHead().getBody();
    }
    catch(NullPointerException e) {
      /////////////////
      System.out.println("--- found NullPointerException in genPrologStmts");
      /////////////////
      return;
    }
    for (int i = _prologStmts.size() - 1; i >= 0; i--)
      blist.insert(_prologStmts.get(i));
  }

  private void genEpilogStmts() {
    BlockList blist;
    try{
      blist = fblock.getBody().getHead().getBody();
    }
    catch (NullPointerException e) {
      /////////////////
      System.out.println("--- found NullPointerException in genEpilogStmts");
      /////////////////
      return;
    }
    for (Xobject stmt: _epilogStmts)
      blist.add(stmt);
  }

}

