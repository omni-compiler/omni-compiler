package exc.xmpF;

import exc.object.*;
import exc.block.*;
import java.util.*;

/*
 * Translate Coarray Fortran (for each procedure)
 */
public class XMPtransCoarrayRun
{
  private Boolean DEBUG = true;       // change me in debugger

  // constants
  final static String VAR_TAG_NAME = "xmpf_resource_tag";
  final static String TRAV_COUNTCOARRAY_PREFIX = "xmpf_traverse_countcoarray";
  final static String TRAV_INITCOARRAY_PREFIX = "xmpf_traverse_initcoarray";
  final static String GET_DESCPOINTER_NAME   = "xmpf_coarray_get_descptr";
  final static String COARRAY_ALLOC_PREFIX   = "xmpf_coarray_alloc";
  final static String COARRAY_DEALLOC_PREFIX = "xmpf_coarray_dealloc";
  final static String THIS_IMAGE_NAME        = "xmpf_this_image";  // generic
  final static String COBOUND_NAME           = "xmpf_cobound";  // generic
  final static String IMAGE_INDEX_NAME       = "xmpf_image_index";
  final static String COARRAY_PROLOG_NAME    = "xmpf_coarray_prolog";
  final static String COARRAY_EPILOG_NAME    = "xmpf_coarray_epilog";
  final static String SYNCALL_NAME           = "xmpf_sync_all";
  final static String AUTO_SYNCALL_NAME      = "xmpf_sync_all_auto";  // another entry of syncall
  final static String FINALIZE_PROGRAM_NAME  = XMP.finalize_all_f;

  // to handle host- and use-associations
  static ArrayList<XMPtransCoarrayRun> ancestors
    = new ArrayList<XMPtransCoarrayRun>();

  private XMPenv env;

  private String name;

  private FuncDefBlock funcDef;
  private XobjectDef def;
  private FunctionBlock fblock;

  private ArrayList<XMPcoarray> useAssociatedCoarrays;
  private ArrayList<XMPcoarray> localCoarrays;

  // localCoarrays is divided into the following four
  private ArrayList<XMPcoarray> staticLocalCoarrays;
  //private ArrayList<XMPcoarray> staticLocalCoarraysEx;  // add use-acc & host-module-acc
  private ArrayList<XMPcoarray> allocatableLocalCoarrays;
  private ArrayList<XMPcoarray> staticDummyCoarrays;
  private ArrayList<XMPcoarray> allocatableDummyCoarrays;

  // the host module and the host procedure
  private String hostModuleName,hostProcedureName;
  private XMPtransCoarrayRun hostModuleRun, hostProcedureRun;

  // localCoarrays + useAssociatedCoarrays + the host's visibleCoarrays
  private ArrayList<XMPcoarray> visibleCoarrays;    // available after run1()

  //private XMPinitProcedure initProcedure;
  private String initProcTextForFile;

  private String traverseCountName, traverseInitName;
  private Ident _resourceTagId = null;

  private ArrayList<Xobject> _prologStmts = new ArrayList<Xobject>();
  private ArrayList<Xobject> _epilogStmts = new ArrayList<Xobject>();

  private Boolean _autoDealloc;


  //------------------------------------------------------------
  //  CONSTRUCTOR
  //------------------------------------------------------------
  public XMPtransCoarrayRun(XobjectDef def, XMPenv env,
                            ArrayList<XMPtransCoarrayRun> pastRuns, int pass) {
    this.def = def;
    this.env = env;
    name = def.getName();

    _setHostName();
    _setHostRun(pastRuns);

    funcDef = new FuncDefBlock(def);
    fblock = funcDef.getBlock();
    env.setCurrentDef(funcDef);

    String postfix = _genNewProcPostfix();
    traverseCountName = TRAV_COUNTCOARRAY_PREFIX + postfix;
    traverseInitName = TRAV_INITCOARRAY_PREFIX + postfix;

    _setLocalCoarrays();
    /* visibleCoarrays will be set after run1 */

    if (pass == 1) {
      _check_ifIncludeXmpLib();
    }

    XMP.exitByError();   // exit if error was found.
  }


  private void _setHostName()
  {
    hostModuleName = null;
    hostProcedureName = null;

    XobjectDef parentDef = def.getParent();
    if (parentDef == null)
      return;

    switch(parentDef.getDef().Opcode()) {
    case F_MODULE_DEFINITION:
      hostModuleName = parentDef.getName();
      break;

    case FUNCTION_DEFINITION:
      hostProcedureName = parentDef.getName();
      XobjectDef granParentDef = parentDef.getParent();
      if (granParentDef != null) {
        switch(granParentDef.getDef().Opcode()) {
        case F_MODULE_DEFINITION:
          hostModuleName = granParentDef.getName();
          break;
        default:
          XMP.fatal("INTERNAL: unexpected nest of procedures " + 
                    name + ", " + hostProcedureName + ", " + 
                    granParentDef.getName());
          break;
        }
      }
      break;

    default:
      break;
    }
  }


  private void _setHostRun(ArrayList<XMPtransCoarrayRun> pastRuns)
  {
    hostModuleRun = null;
    hostProcedureRun = null;

    XobjectDef pdef = def.getParent();
    if (pdef == null)
      return;

    switch (pdef.getDef().Opcode()) {
    case FUNCTION_DEFINITION:
      _setHostRun_proc(pdef, pastRuns);
      break;
    case F_MODULE_DEFINITION:
      _setHostRun_module(pdef, pastRuns);
      break;
    default:
      XMP.fatal("INTERNAL: illegal opcode of parent XobjectDef: " +
                pdef.getDef().Opcode());
      break;
    }
  }

  private void _setHostRun_proc(XobjectDef pdef,
                                ArrayList<XMPtransCoarrayRun> pastRuns)
  {
    for (XMPtransCoarrayRun run: pastRuns) {
      if (pdef == run.def) {
        hostProcedureRun = run;
        break;
      }
    }

    if (hostProcedureRun == null)
      XMP.fatal("INTERNAL: Host procedure \'" + hostProcedureName +
                "\' is expected in pastRuns before \'" + name + "\'");

    XobjectDef ppdef = pdef.getParent();
    if (ppdef == null)
      return;

    switch (ppdef.getDef().Opcode()) {
    case F_MODULE_DEFINITION:
      _setHostRun_module(ppdef, pastRuns);
      break;
    default:
      XMP.fatal("INTERNAL: illegal opcode of ground-parent XobjectDef: " +
                ppdef.getDef().Opcode());
      break;
    }
  }
    
  private void _setHostRun_module(XobjectDef pdef,
                                  ArrayList<XMPtransCoarrayRun> pastRuns)
  {
    for (XMPtransCoarrayRun run: pastRuns) {
      if (pdef == run.def) {
        hostModuleRun = run;
        break;
      }
    }

    if (hostModuleRun == null)
      XMP.fatal("INTERNAL: Host module \'" + hostModuleName +
                "\' is expected in pastRuns before \'" + name + "\'");
  }


  /*  set coarrays declared in the current procedure as localCoarrays
   */
  private void _setLocalCoarrays() {
    localCoarrays = new ArrayList<XMPcoarray>();
    useAssociatedCoarrays = new ArrayList<XMPcoarray>();

    Xobject idList = def.getFuncIdList();
    for (Xobject obj: (XobjList)idList) {
      Ident ident = (Ident)obj;
      if (ident.wasCoarray()) {
        // found it is a coarray or a variable converted from a coarray
        XMPcoarray coarray = new XMPcoarray(ident, def, fblock, env);
        if (coarray.isUseAssociated())
          useAssociatedCoarrays.add(coarray);
        else
          localCoarrays.add(coarray);
      }
    }

    // resolve the problem on static coarrays in a module
    // - move all coarrays to use-associating procedures
    // - move static coarrays to directly host-associated procedures
    for (XMPcoarray coarray: useAssociatedCoarrays) {
      XMPcoarray coarray2 = _copyCoarrayToMergeIn(coarray);
      localCoarrays.add(coarray2);
    }
    if (hostModuleRun != null && hostProcedureRun == null) {
      // found a module procedure
      for (XMPcoarray coarray: hostModuleRun.localCoarrays) {
        if (coarray.isExplicitShape()) {
          XMPcoarray coarray2 = _copyCoarrayToMergeIn(coarray);
          localCoarrays.add(coarray2);
        }
      }
    }      

    // divide localCoarrays into four types
    staticLocalCoarrays = new ArrayList<XMPcoarray>();
    allocatableLocalCoarrays = new ArrayList<XMPcoarray>();
    staticDummyCoarrays = new ArrayList<XMPcoarray>();
    allocatableDummyCoarrays = new ArrayList<XMPcoarray>();
    for (XMPcoarray coarray: localCoarrays) {
      if (coarray.isDummyArg()) {
        if (coarray.isAllocatable())
          allocatableDummyCoarrays.add(coarray);
        else
          staticDummyCoarrays.add(coarray);
      } else {
        if (coarray.isExplicitShape())
          staticLocalCoarrays.add(coarray);
        else
          allocatableLocalCoarrays.add(coarray);
      }
    }
  }

  /*
   *  useful to copy host- and use-associcated coarrays into the list of
   *  local coarrays.
   */
  private XMPcoarray _copyCoarrayToMergeIn(XMPcoarray coarray1)
  {
    Ident ident1 = coarray1.getIdent();
    Xtype type2 = ident1.Type().copy();
    String name = coarray1.getName();
    env.removeIdent(name, null);
    Ident ident2 = env.declIdent(name, type2);
    ident2.setFdeclaredModule(null);

    // reset ident, name, isAllocatable, isPointer and _isUseAssociated
    // but not changed homeBlockName, declCommonName and crayCommonName
    XMPcoarray coarray2 = new XMPcoarray(ident2, def, fblock, env,
                                         coarray1.getHomeBlockName());
    coarray2.setWasMovedFromModule(true);
    return coarray2;
  }


  /*  set coarrays declared in used modules as useAssociatedCoarrays
   */
  private void _setVisibleCoarrays() {
    /*  set visible coarrays
     *   1. add coarrays declared in the current procedure,
     *   2. add all use-associated coarrays, and
     *   3. add all visible coarrays of the host-associated procedure
     *  A name of coarray will be searched in this priority.
     */
    visibleCoarrays = new ArrayList<XMPcoarray>();
    _mergeCoarraysByName(visibleCoarrays, localCoarrays);
    _mergeCoarraysByName(visibleCoarrays, useAssociatedCoarrays);

    if (hostProcedureRun != null)
      _mergeCoarraysByName(visibleCoarrays, hostProcedureRun.visibleCoarrays);
    else if (hostModuleRun != null)
      _mergeCoarraysByName(visibleCoarrays, hostModuleRun.visibleCoarrays);
  }

  private void _mergeCoarraysByName(ArrayList<XMPcoarray> coarrays1,
                                    ArrayList<XMPcoarray> coarrays2)
  {
    ArrayList<XMPcoarray> newCoarrays = new ArrayList();
    for (XMPcoarray coarray2: coarrays2) {
      boolean found = false;
      String name2 = coarray2.getName();
      for (XMPcoarray coarray1: coarrays1) {
        if (name2.equals(coarray1.getName())) {
          found = true;
          break;
        }
      }
      if (!found)
        newCoarrays.add(coarray2);
    }

    coarrays1.addAll(newCoarrays);
  }



  //------------------------------------------------------------
  //  TRANSLATION
  //------------------------------------------------------------

  /*
   *  PASS 1: convert for each procedure that is either 
   *            - the main program or
   *            - an external function/subroutine or
   *            - an internal function/subroutine or
   *            - a module function/subroutine
   *          collect coarrays for each procedure and module
   */
  public void run1() {
    // error check for each coarray declaration
    //    for (XMPcoarray coarray: localCoarrays)
    //      coarray.errorCheck();

    if (_isModule())
      run1_module();
    else
      run1_procedure();
  }
        

  private void run1_procedure() {
    set_autoDealloc(false);

    // convert specification and declaration part
    transDeclPart_staticLocal();
    transDeclPart_allocatableLocal();
    transDeclPart_staticDummy();
    transDeclPart_allocatableDummy();

    // To avoid trouble of the shallow/deep copies, visibleCoarrays
    // should be made after execution of transDeclPart_*.
    _setVisibleCoarrays();

    if (!_isModule()) {
      transExecPart_visibleCoarrays();
    }

    // SPECIAL HANDLING (TEMPORARY) to work XMPtransCoarray alone without XMPtranslate
    //  convert main program to soubroutine xmpf_main
    if (_isMainProgram())
      _convMainProgramToSubroutine("xmpf_main");
  }


  private void _convMainProgramToSubroutine(String newName) {
    Xtype ft = def.getFuncType();
    ft.setIsFprogram(false);
    
    String oldName = def.getName();
    Ident nameId = env.getEnv().findVarIdent(oldName);
    if (nameId == null)
      XMP.fatal("main program name \'" + oldName + "\' not found");
    nameId.setName(newName);
    def.setName(newName);
  }


  private void run1_module() {
    // divide localCoarrays into four types
    ArrayList<XMPcoarray> staticLocalCoarrays = new ArrayList<XMPcoarray>();
    ArrayList<XMPcoarray> allocatableLocalCoarrays = new ArrayList<XMPcoarray>();
    for (XMPcoarray coarray: localCoarrays) {
      if (coarray.isExplicitShape())
        staticLocalCoarrays.add(coarray);
      else
        allocatableLocalCoarrays.add(coarray);
    }

    // convert specification and declaration part
    transModule_staticLocal1();
    transModule_allocatableLocal();

    // To avoid trouble of the shallow/deep copies, visibleCoarrays
    // should be made after execution of transDeclPart_*.
    _setVisibleCoarrays();

    // funcDef.Finalize() is not needed.
  }


  /*
   *  PASS 2: for each module 
   *          excluding its module functions and subroutines
   */
  public void run2() {

    if (_isModule()) {
      // convert specification and declaration part
      transModule_staticLocal2();
    }
  }



  /**
    Handling pseudo-local static coarrays in a procedure.
    Pseudo-local static coarrays are:
    - procedure-local static coarrays in procedures,
    - use-associated static coarrays, and
    - host-associated static coarrays in module procedures.
    --------------------------------------------
      subroutine EX1
        use M1   !! contains "real :: V1(10,20)[4,*]"  ! use-associated static
        complex(8), save :: V2[0:*]                    ! static local
        ...
      end subroutine
    --------------------------------------------
    output:
    --------------------------------------------
      subroutine EX1
        use M1

        real :: V1(1:10,1:20)                                        ! f. f1.
        complex(8) :: V2                                             ! f. f1.

        !-- for use-associated static coarray V1
        integer(8) :: DP_V1                                          ! a.
        common /xmpf_DP_M1/ DP_V1                                    ! a1.
        common /xmpf_CP_M1/ CP_V1                                    ! c.
        pointer (CP_V1, V1)                                          ! c.

        !-- for local static coarray V2
        integer(8) :: DP_V2                                          ! a.
        common /xmpf_DP_EX1/ DP_V2                                   ! a1.
        common /xmpf_CP_EX1/ CP_V2                                   ! c.
        pointer (CP_V2, V2)                                          ! c.
        ...
      end subroutine

    !! - In addition, initializaiton subroutines                     ! b.
    !!     * xmpf_traverse_countcoaray_EX1 and
    !!     * xmpf_traverse_initcoaray_EX1
    !!   will be generated into the same output file to initialize 
    !!   DP_V2 and CP_V2 (See XMPcoarrayInitProcedure).
    --------------------------------------------
      DP_Vn: pointer to descriptor of each coarray Vn
      CP_Vn: cray poiter to the coarray object Vn
  */
  private void transDeclPart_staticLocal() {

    // a. declare descriptor pointers
    genDeclOfDescPointer(staticLocalCoarrays);

    // a1. make common association of descriptor pointers
    genCommonStmt(staticLocalCoarrays);

    // c. link cray-pointers with data object
    genDeclOfCrayPointer(staticLocalCoarrays);

    // b. generate allocation into init procedure
    genAllocOfStaticCoarrays(staticLocalCoarrays);

    // f. remove codimensions from declarations of coarrays
    removeCodimensions(staticLocalCoarrays);

    // f1. remove SAVE attributes from declarations of coarrays
    removeSaveAttr(staticLocalCoarrays);
  }


  /**
    Handling local allocatable coarrays in a procedure/module
    --------------------------------------------
      subroutine EX1  or  module EX1
        integer, allocatable :: V3(:,:)[:,:]            ! allocatable local
        ...
      end subroutine  or  end module
    --------------------------------------------
    output:
    --------------------------------------------
      subroutine EX1  or  module EX1
        integer, pointer :: V3(:,:)                                  ! f. f1. h.
        integer(8) :: DP_V3                                          ! a.
        ...
      end subroutine  or  end module
    --------------------------------------------
      DP_Vn: pointer to descriptor of each coarray Vn
  */
  private void transDeclPart_allocatableLocal() {

    // a. declare descriptor pointers
    genDeclOfDescPointer(allocatableLocalCoarrays);

    // f. remove codimensions from declarations of coarrays
    removeCodimensions(allocatableLocalCoarrays);

    // f1. remove SAVE attributes from declarations of coarrays
    removeSaveAttr(allocatableLocalCoarrays);

    // h. replace allocatable attributes with pointer attributes
    replaceAllocatableWithPointer(allocatableLocalCoarrays);
  }

  private void transModule_allocatableLocal() {
    transDeclPart_allocatableLocal();
  }


  /**
    Handling non-allocatable dummy coarrays in a procecure
    --------------------------------------------
      subroutine EX1(V2)
        complex(8) :: V2[0:*]                          ! static dummy
        ...
        return
      end subroutine
    --------------------------------------------
    output:
    --------------------------------------------
      subroutine EX1(V2)
        complex(8) :: V2                                     ! f.
        integer(8) :: DP_V2                                  ! a.

        !-- initialization for procedure EX1
      ( integer(8) :: tag                                    ! i. )
      ( call xmpf_coarray_prolog(tag, "EX1", 3)              ! i. )

        !-- find DP_V2 and set the attributes
        call xmpf_coarray_get_descptr(DP_V2, V2, tag)        ! a2.
        call xmpf_coarray_set_coshape(DP_V2, 1, 0)           ! m.
        call xmpf_coarray_set_varname(DP_V2, "V2", 2)        ! n.

        ...

        !-- finalization for procedure EX1
      ( call xmpf_coarray_epilog(tag)                        ! i. )
        return
      end subroutine
    --------------------------------------------
      DP_Vn: pointer to descriptor of each coarray Vn
  */
  private void transDeclPart_staticDummy() {

    // a. declare descriptor pointers
    genDeclOfDescPointer(staticDummyCoarrays);

    // a2. m. n. generate definition of descriptor pointers (dummy coarrays only)
    genDefinitionOfDescPointer(staticDummyCoarrays);

    // f. remove codimensions from declarations of coarrays
    removeCodimensions(staticDummyCoarrays);
  }


  /**
    Handling allocatable dummy coarrays in a procecure
    --------------------------------------------
      subroutine EX1(V3)
        integer, allocatable :: V3(:,:)[:,:]           ! allocatable dummy
        ...
      end subroutine
    --------------------------------------------
    output:
    --------------------------------------------
      subroutine EX1(V3)
        integer, pointer :: V3(:,:)                          ! f. h.
        integer(8) :: DP_V3                                  ! a.

        // find DP_V3 and set attributes
        call xmpf_coarray_get_descptr(DP_V3, V3, tag)        ! a2.
        call xmpf_coarray_set_varname(DP_V3, "V3", 2)        ! n.

        ...
      end subroutine
    --------------------------------------------
      DP_Vn: pointer to descriptor of each coarray Vn
  */
  private void transDeclPart_allocatableDummy() {

    if (allocatableDummyCoarrays.isEmpty())
      return;

    // a. declare descriptor pointers
    genDeclOfDescPointer(allocatableDummyCoarrays);

    // a2. m. n. generate definition of descriptor pointers (dummy coarrays only)
    genDefinitionOfDescPointer(allocatableDummyCoarrays);

    // f. remove codimensions from declarations of coarrays
    removeCodimensions(allocatableDummyCoarrays);

    // h. replace allocatable attributes with pointer attributes
    replaceAllocatableWithPointer(allocatableDummyCoarrays);
  }


  /**
    Handling static coarrays in a module
    --------------------------------------------
      module EX1
        use M1   !! contains "real :: V1(10,20)[4,*]"  ! use-associated static
        complex(8), save :: V2[0:*]                    ! static local
        ...
      end module
    --------------------------------------------
    output:
    --------------------------------------------
      module EX1
        use M1
        !<DELETE>                complex(8), save :: V2[0:*]         ! o.
        !<GENERATE then DELETE>  integer(8) :: DP_V2                 ! a. o.
        ...
      end module

    !! - In addition, initializaiton subroutines                     ! b.
    !!     * xmpf_traverse_countcoaray_EX1 and
    !!     * xmpf_traverse_initcoaray_EX1
    !!   will be generated into the same output file to initialize 
    !!   DP_V2 and CP_V2 (See XMPcoarrayInitProcedure).
    --------------------------------------------
      DP_Vn: pointer to descriptor of each coarray Vn
      CP_Vn: cray poiter to the coarray object Vn
  */
  private void transModule_staticLocal1() {
    // a. declare descriptor pointers
    genDeclOfDescPointer(staticLocalCoarrays);

    // b. generate allocation into init procedure (static coarrays only)
    genAllocOfStaticCoarrays(staticLocalCoarrays);
  }

  private void transModule_staticLocal2() {
    // o. remove declarations of variables
    removeDeclOfCoarrays(staticLocalCoarrays);
  }


  /**
    Handling coindexed objects/variables in execution part of a procedure
    --------------------------------------------
      subroutine EX1
        use M1 !! contains "real :: V1(10,20)[4,*]"     ! use-associated static
        use M4 !! contains "real,allocatable::V4(:)[:]" ! use-associated allocatable
        complex(8), save :: V2[0:*]                     ! static local
        integer, allocatable :: V3(:,:)[:,:]            ! allocatable local
        ...
        V1(1:3,j)[k1,k2] = (/1.0,2.0,3.0/)              ! put 1D
        z = V2[k]**2                                    ! get 0D
        allocate (V3(1:10,20)[k1:k12,0:*],V4(10)[*])    ! allocate
        deallocate (V4)                                 ! deallocate
        sync all                                        ! stop code motion beyond this line
        if (allocated(V3)) write(*,*) "yes"             ! Fortran intrinsic
        n1 = this_image(V1,1)                           ! coarray intrinsic
        n2(:) = this_image(V3)                          ! coarray intrinsic
        n3 = image_index(V1,(/1,2/))                    ! coarray intrinsic
        ...
        return                                          ! dealloc V3 automatically
        ...
        stop                                            ! finalize program
      end subroutine
    --------------------------------------------
    output:
    --------------------------------------------
      subroutine EX1
        ...
        integer(8) :: tag                                       ! i.
        call xmpf_coarray_prolog(tag, "EX1", 3)                 ! i.
        call xmpf_coarray_put(DP_V1, V1(1,j), 4, &              ! d.
          k1+4*(k2-1), (/1.0,2.0,3.0/), ...)      
        z = xmpf_coarray_get0d(DP_V2, V2, 16, k, 0) ** 2        ! e.
        call xmpf_coarray_alloc2d(DP_V3, V3, tag, 4, 2, 10, 20) ! j.
        call xmpf_coarray_set_coshape(DP_V3, 2, k1, k2, 0)      ! m.
        call xmpf_coarray_set_varname(DP_V3, "V3", 2)           ! n.
        call xmpf_coarray_dealloc(DP_V3)                        ! j.
        call xmpf_syncall(V1,V4,V2,V3)                          ! p.
        if (associated(V3)) write(*,*) "yes"                    ! l.
        n1 = xmpf_this_image(DP_V1,1)                           ! l.
        n2(:) = this_image(DP_V3)                               ! l.
        n3 = xmpf_image_index(DP_V1,(/1,2/))                    ! l.
        ...
        call xmpf_syncall(V1,V4,V2,V3)                          ! i. p.
        call xmpf_coarray_epilog(tag)                           ! i.
        return
        ...
        call xmpf_finalize_all_f()                              ! k.
        stop
      end subroutine

    !! Additionally, two subroutines xmpf_traverse_* will    ! b.
    !! be generated into the same output file which will
    !! initialize DP_V2 and CP_V2.
    !! (See XMPcoarrayInitProcedure.)
    --------------------------------------------
      DP_Vn: pointer to descriptor of each coarray Vn
      CP_Vn: cray poiter to the coarray object Vn
  */
  private void transExecPart_visibleCoarrays() {

    // e. convert coindexed objects to function references
    convCoidxObjsToFuncCalls(visibleCoarrays);

    // d. convert coindexed variable assignment stmts to call stmts
    convCoidxStmtsToSubrCalls(visibleCoarrays);

    // j. convert allocate/deallocate stmts (allocatable coarrays only)
    convAllocateStmts(visibleCoarrays);
    convDellocateStmts(visibleCoarrays);

    // l. fake intrinsic 'allocatable' (allocatable coarrays only)
    //    replace V of coarray intrinsic calls with DP_V
    replaceFunctionCalls(visibleCoarrays);

    // i. initialization/finalization for auto-syncall and auto-deallocate
    if (get_autoDealloc())
      genCallOfPrologAndEpilog();

    // k. insert finalization call before STOP statements
    insertFinalizationCall();

    // p. add visible coarrays as arguments of sync all statements 
    //     to prohibit code motion
    addVisibleCoarraysToSyncall(visibleCoarrays);

    // o. remove declarations for use-associated allocatable coarrays
    for (XMPcoarray coarray: useAssociatedCoarrays) {
      if (!coarray.isExplicitShape()) 
        removeDeclOfCoarray(coarray);
    }

    // finalize fblock in funcDef
    funcDef.Finalize();
  }



  //-----------------------------------------------------
  //  TRANSLATION a.
  //  declare variables of descriptor pointers
  //-----------------------------------------------------
  //
  private void genDeclOfDescPointer(ArrayList<XMPcoarray> localCoarrays) {
    for (XMPcoarray coarray: localCoarrays) {
      coarray.genDecl_descPointer();
    }
  }


  //-----------------------------------------------------
  //  TRANSLATION a1. (static only)
  //  generate common association of cray pointers
  //-----------------------------------------------------
  //
  private void genCommonStmt(ArrayList<XMPcoarray> coarrays) {
    // do nothing if no coarrays are declared.
    if (coarrays.isEmpty())
      return;

    ArrayList<String> cnameList = new ArrayList();
    for (XMPcoarray coarray0: coarrays) {
      String cname = coarray0.getDescCommonName();
      if (cnameList.contains(cname))
        continue;

      // found new common block to be declared
      cnameList.add(cname);

      Xobject cnameObj = Xcons.Symbol(Xcode.IDENT, cname);
      Xobject varList = Xcons.List();
      for (XMPcoarray coarray: coarrays) {
        if (cname.equals(coarray.getDescCommonName())) {
          Ident descPtrId = coarray.getDescPointerId();
          varList.add(Xcons.FvarRef(descPtrId));
        }
      }

      // add declaration 
      Xobject decls = fblock.getBody().getDecls();
      Xobject args = Xcons.List(Xcode.F_COMMON_DECL,
                                Xcons.List(Xcode.F_VAR_LIST, cnameObj, varList));
      decls.add(args);
    }
  }


  //-----------------------------------------------------
  //  TRANSLATION c.
  //  link cray pointers with data objects and
  //  generate their common association
  //-----------------------------------------------------
  //
  private void genDeclOfCrayPointer(ArrayList<XMPcoarray> coarrays) {
    // do nothing if no coarrays are declared.
    if (coarrays.isEmpty())
      return;

    // genDecl_crayPointer
    for (XMPcoarray coarray: coarrays) {
      coarray.genDecl_crayPointer();
    }

    ArrayList<String> cnameList = new ArrayList();
    for (XMPcoarray coarray0: coarrays) {
      String cname = coarray0.getCrayCommonName();
      if (cnameList.contains(cname))
        continue;

      // found new common block to be declared
      cnameList.add(cname);

      Xobject cnameObj = Xcons.Symbol(Xcode.IDENT, cname);
      Xobject varList = Xcons.List();
      for (XMPcoarray coarray: coarrays) {
        if (cname.equals(coarray.getCrayCommonName())) {
          Ident crayPtrId = coarray.getCrayPointerId();
          varList.add(Xcons.FvarRef(crayPtrId));
        }
      }

      // add declaration 
      Xobject decls = fblock.getBody().getDecls();
      Xobject args = Xcons.List(Xcode.F_COMMON_DECL,
                                Xcons.List(Xcode.F_VAR_LIST, cnameObj, varList));
      decls.add(args);
    }
  }


  //-----------------------------------------------------
  //  TRANSLATION p.
  //  add coarrays as actual arguments to syncall library call
  //-----------------------------------------------------
  //
  private void addVisibleCoarraysToSyncall(ArrayList<XMPcoarray> coarrays) {
    BlockIterator bi = new topdownBlockIterator(fblock);
    for (bi.init(); !bi.end(); bi.next()) {
      BasicBlock bb = bi.getBlock().getBasicBlock();
      if (bb == null) continue;
      for (Statement s = bb.getHead(); s != null; s = s.getNext()) {
        Xobject xobj = s.getExpr();
        if (_isCallForSyncall(xobj)) {
          // found
          Xobject args = _getCoarrayNamesIntoArgs(coarrays);
          Xobject callExpr = xobj.getArg(0);
          callExpr.setArg(1, args);
        }
      }
    }
  }

  private Boolean _isCallForSyncall(Xobject xobj) {
    
    if (xobj == null || xobj.Opcode() != Xcode.EXPR_STATEMENT)
      /* EXPR_STATEMENT conatains does not contain call statement */
      /* F_ASSIGN_STATEMENT does not contain call statement */
      return false;

    Xobject callExpr = xobj.getArg(0);
    if (callExpr == null || callExpr.Opcode() != Xcode.FUNCTION_CALL)
      return false;

    String fname = callExpr.getArg(0).getName();
    if (fname == SYNCALL_NAME || fname == AUTO_SYNCALL_NAME)
      return true;

    return false;
  }


  private Xobject _getCoarrayNamesIntoArgs(ArrayList<XMPcoarray> coarrays) {
    Xobject args = Xcons.List();
    for (XMPcoarray coarray: coarrays)
      args.add(Xcons.FvarRef(coarray.getIdent()));
    return args;
  }


  //-----------------------------------------------------
  //  TRANSLATION k. 
  //  insert finalization call before STOP statements
  //  ZANTEI VERSION: 
  //    This function is used until joining caf and xmp translators.
  //-----------------------------------------------------
  //
  private void insertFinalizationCall() {
    // for STOP statement
    BasicBlockIterator bbi = new BasicBlockIterator(fblock);
    for (bbi.init(); !bbi.end(); bbi.next()) {
      StatementIterator si = bbi.getBasicBlock().statements();
      while (si.hasNext()) {
        Statement st = si.next();
        Xobject stmt = st.getExpr();
        if (stmt == null)
          continue;

        switch(stmt.Opcode()) {
        case F_STOP_STATEMENT:
          LineNo lineno = stmt.getLineNo();
          Ident func =                        // find or generate function name
            env.declInternIdent(FINALIZE_PROGRAM_NAME, Xtype.FsubroutineType);
          Xobject call = func.callSubroutine();
          call.setLineNo(lineno);
          st.insert(call);
          break;
        default:
          break;
        }
      }
    }
  }


  //-----------------------------------------------------
  //  TRANSLATION i.
  //  generate procedure prolog and epilog calls
  //-----------------------------------------------------
  //
  private void genCallOfPrologAndEpilog() {
    genCallOfPrologAndEpilog_dealloc();

    // perform prolog/epilog code generations
    genPrologStmts();
    genEpilogStmts();
  }

  private void genPrologStmts() {
    // for the begining of the procedure
    BlockList blist = fblock.getBody().getHead().getBody();
    int nlines = _prologStmts.size();

    for (int i = nlines - 1; i >= 0; i--)
      blist.insert(_prologStmts.get(i));

    // restriction: for the ENTRY statement
    if (nlines > 0 && _findEntryStmtInBlock(fblock)) {
      XMP.error("restriction: An ENTRY statement is not allowed in the " +
                "program including coarray features.");
    }
  }

  private Boolean _findEntryStmtInBlock(Block fblock) {
    BlockIterator bi = new topdownBlockIterator(fblock);
    for (bi.init(); !bi.end(); bi.next()) {
      Block block = bi.getBlock();
      if (block.Opcode() == Xcode.F_ENTRY_DECL)
        return true;
    }
    return false;
  }

  private void genEpilogStmts() {
    // for RETURN statement
    BlockIterator bi = new topdownBlockIterator(fblock);
    for (bi.init(); !bi.end(); bi.next()) {
      Block block = bi.getBlock();
      switch(block.Opcode()) {
      case RETURN_STATEMENT:
        LineNo lineno = block.getLineNo();
        for (Xobject stmt1: _epilogStmts) {
          Xobject stmt2 = stmt1.copy();
          stmt2.setLineNo(lineno);
          block.insert(stmt2);
        }
        break;

      }
    }

    // for the end of the procedure
    BlockList blist = fblock.getBody().getHead().getBody();

    if (blist.getTail().Opcode() == Xcode.RETURN_STATEMENT)
      return;     // to avoid generating unreachable statements

    for (Xobject stmt: _epilogStmts)
      blist.add(stmt);
  }


  /*  NOT USED: all calls of automatic syncalls are moved into runtime functions.
   */
  private void genCallOfPrologAndEpilog_syncall() {
    // generate "call xmpf_sync_all()" and add to the tail
    Xobject args = Xcons.List();
    Ident fname = /*env.findVarIdent(AUTO_SYNCALL_NAME, null);      // to avoid error of tool
    if (fname == null)
    fname = */env.declExternIdent(AUTO_SYNCALL_NAME,
                                  BasicType.FexternalSubroutineType);
    Xobject call = fname.callSubroutine(args);
    addEpilogStmt(call);
  }

  private void genCallOfPrologAndEpilog_dealloc() {
    // generate "call coarray_prolog(tag)" and insert to the top
    Xobject args1 = 
      Xcons.List(Xcons.FvarRef(getResourceTagId()),
                 Xcons.FcharacterConstant(Xtype.FcharacterType, name, null),
                 Xcons.IntConstant(name.length()));

    Ident fname1 = env.declExternIdent(COARRAY_PROLOG_NAME,
                                       BasicType.FexternalSubroutineType);
    if (args1.hasNullArg())
      XMP.fatal("generated null argument " + fname1 +
                "(genCallofPrologAndEpilog args1)");
    Xobject call1 = fname1.callSubroutine(args1);
    insertPrologStmt(call1);

    // generate "call coarray_epilog(tag)" and add to the tail
    Xobject args2 = Xcons.List(Xcons.FvarRef(getResourceTagId()));
    Ident fname2 = env.declExternIdent(COARRAY_EPILOG_NAME,
                                       BasicType.FexternalSubroutineType);
    if (args2.hasNullArg())
      XMP.fatal("generated null argument " + fname2 +
                "(genCallofPrologAndEpilog args2)");

    Xobject call2 = fname2.callSubroutine(args2);
    addEpilogStmt(call2);
  }


  //-----------------------------------------------------
  //  TRANSLATION a2. m. n.
  //  generate definition of descriptor pointers
  //-----------------------------------------------------
  //
  private void genDefinitionOfDescPointer(ArrayList<XMPcoarray> dummyLocalcoarrays) {
    Xobject args, subrCall;
    Ident subr, descPtrId;

    for (XMPcoarray coarray: dummyLocalcoarrays) {
      // a2. call "get_descptr(descPtr, baseAddr, tag)"
      descPtrId = coarray.getDescPointerId();
      args = Xcons.List(descPtrId, coarray.getIdent(),
                        Xcons.FvarRef(getResourceTagId()));
      subr = env.declExternIdent(GET_DESCPOINTER_NAME,
                                 BasicType.FexternalSubroutineType);
      if (args.hasNullArg())
        XMP.fatal("generated null argument " + GET_DESCPOINTER_NAME +
                  "(genDefinitionOfDescPointer)");

      subrCall = subr.callSubroutine(args);
      addPrologStmt(subrCall);

      // m. "CALL set_coshape(descPtr, corank, clb1, clb2, ..., clbr)"
      subrCall = coarray.makeStmt_setCoshape();
      if (subrCall != null)          // if it is allocated
        addPrologStmt(subrCall);

      // n. "CALL set_varname(descPtr, name, namelen)" for runtime message
      subrCall = coarray.makeStmt_setVarName();
      addPrologStmt(subrCall);
    }
  }


  //-----------------------------------------------------
  //  TRANSLATION d. (PUT)
  //  convert statements whose LHS are coindexed variables
  //  to subroutine calls
  //-----------------------------------------------------
  private void convCoidxStmtsToSubrCalls(ArrayList<XMPcoarray> coarrays) {
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
                                         ArrayList<XMPcoarray> coarrays) {
    Xobject lhs = assignExpr.getArg(0);
    Xobject rhs = assignExpr.getArg(1);

    int condition = _getConditionOfCoarrayPut(rhs);

    XMPcoindexObj coindexObj = new XMPcoindexObj(lhs, coarrays);
    return coindexObj.toCallStmt(rhs, Xcons.IntConstant(condition));
  }

  /*
   * condition 1: Expression rts may be addressed in read-only region or
   *    in temporary area allocated by the compiler. 
   *    (In such cases, Fujitsu-RDMA cannot work without buffer.)
   * condition 0: Otherwise.
   */
  private int _getConditionOfCoarrayPut(Xobject rhs) {
    switch (rhs.Opcode()) {
    case F_VAR_REF:
      return 0;

    case F_ARRAY_REF:
      return 1;          // can be 0 if the rank is 0.

    default:
      break;
    }

    return 1;        // for safe
  }

  //-----------------------------------------------------
  //  TRANSLATION e. (GET)
  //  convert coindexed objects to function references
  //-----------------------------------------------------
  private void convCoidxObjsToFuncCalls(ArrayList<XMPcoarray> coarrays) {
    // itaration to solve nested reference of coindexed object.
    while (_convCoidxObjsToFuncCalls1(coarrays));
  }

  private Boolean _convCoidxObjsToFuncCalls1(ArrayList<XMPcoarray> coarrays) {
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
                                      ArrayList<XMPcoarray> coarrays) {
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
  private void genAllocOfStaticCoarrays(ArrayList<XMPcoarray> coarrays0) {

    ArrayList<XMPcoarray> coarrays = new ArrayList();
    for (XMPcoarray coarray: coarrays0) {
      if (coarray.wasMovedFromModule())
        continue;
      coarrays.add(coarray);
    }

    // do nothing if no coarrays are declared.
    if (coarrays.isEmpty())
      return;

    // output init procedure
    XMPcoarrayInitProcedure coarrayInit = 
      new XMPcoarrayInitProcedure(coarrays,
                                  traverseCountName,
                                  traverseInitName,
                                  env);
    coarrayInit.run();
  }


  //-----------------------------------------------------
  //  TRANSLATION j, m, n
  //  convert allocate/deallocate stmts for allocated coarrays
  //-----------------------------------------------------
  //
  private void convAllocateStmts(ArrayList<XMPcoarray> coarrays) {
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
          // xobj.getArg(0): 'stat=' identifier (not supported)
          // xobj.getArg(1): list of variables to be allocated
          // 'errmsg=' identifier is not supported.
          if (_doesListHaveCoarray(xobj.getArg(1), coarrays)) {
            ArrayList<Xobject> fstmts =
              genAllocateStmt(xobj, coarrays);

            LineNo lineno = xobj.getLineNo();
            for (Xobject fstmt: fstmts) {
              fstmt.setLineNo(lineno);
              st.insert(fstmt);
            }
            st.remove();
          }
          break;
        }
      }
    }
  }


  private void convDellocateStmts(ArrayList<XMPcoarray> coarrays) {
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
        case F_DEALLOCATE_STATEMENT:
          if (_doesListHaveCoarray(xobj.getArg(1), coarrays)) {
            ArrayList<Xobject> fstmts =
              genDeallocateStmt(xobj, coarrays);

            LineNo lineno = xobj.getLineNo();
            for (Xobject fstmt: fstmts) {
              fstmt.setLineNo(lineno);
              st.insert(fstmt);
            }
            st.remove();
          }
          break;
        }
      }
    }
  }


  private Boolean _doesListHaveCoarray(Xobject args,
                                       ArrayList<XMPcoarray> coarrays) {
    Boolean allCoarray = true;
    Boolean allNoncoarray = true;
    for (Xobject arg: (XobjList)args) {
      Boolean foundCoarray = false;
      Xobject var = arg.getArg(0);
      String varname;
      switch (var.Opcode()) {
      case VAR:
        varname = arg.getArg(0).getString();

        for (XMPcoarray coarray: coarrays) {
          if (varname.equals(coarray.getName())) {
            // found coarray
            foundCoarray = true;
            break;
          }
        }
        break;

      case MEMBER_REF:        // allocation of structure component
        /* restriction: coarray structure component is not supported
         */
        break;

      default:
        XMP.fatal("internal error: unexpected code of Xobject in ALLOCATE stmt");
      }        

      // error check for each arg
      if (foundCoarray && allCoarray)
        allNoncoarray = false;
      else if (!foundCoarray && allNoncoarray)
        allCoarray = false;
      else {
        // found both coarray and non-coarray
        XMP.error("current restriction: An ALLOCATE/DEALLOCATE statement "
                  + "cannnot have both coarrays and noncoarrays.");
      }
    }

    return allCoarray;
  }

  private Xobject _getFirstArgOfCall(Xobject fcall) {
    Xobject args = fcall.getArg(1);
    if (args == null)
      return null;
    return args.getArg(0);
  }

  private Boolean _isCoarrayInCoarrays(Xobject var,
                                       ArrayList<XMPcoarray> coarrays) {
    return _findCoarrayInCoarrays(var, coarrays) != null;
  }

  private XMPcoarray _findCoarrayInCoarrays(Xobject var,
                                            ArrayList<XMPcoarray> coarrays) {
    if (var == null)
      return null;
    String name = var.getName();

    for (XMPcoarray coarray: coarrays) {
      if (name.equals(coarray.getName()))
        return coarray;
    }
    return null;
  }

  // assumed that name is a name of intrinsic function.
  //
  private Boolean _isIntrinsic(Xobject obj) {

    // TEMPORARY JUDGEMENT: If name is registered as an ident, it 
    // is not a name of intrinsic function. Else, it is regarded
    // as a name of intrinsic function.
    //  ... found this is not correct judgement for host association: #
    //
    Ident id = env.findVarIdent(obj.getName(), fblock);
    if (id == null)
      return true;          // regarded as intrinsic 

    return id.Type().isFintrinsic();
  }


  private ArrayList<Xobject> genAllocateStmt(Xobject x,
                                             ArrayList<XMPcoarray> coarrays) {

    ArrayList<Xobject> newStmts = new ArrayList<Xobject>();

    for (Xobject arg: (XobjList)x.getArg(1)) {
      Xobject varname = arg.getArg(0);
      XMPcoarray coarray = _findCoarrayInCoarrays(varname, coarrays);
      XobjList shape = Xcons.List();
      XobjList coshape;
      int rank;

      // get the rank of the argument in the ALLOCATE stmt
      int n = arg.getArg(1).Nargs();
      if (arg.getArg(1).getArg(n - 1).Opcode() != Xcode.F_CO_SHAPE) {
        XMP.error("lack of coshape in the ALLOCATE stetement");
        // error recovery
        return newStmts;
      }

      rank = n - 1;
      coshape = (XobjList)arg.getArg(1).getArg(rank);

      for (int i = 0; i < rank; i++)
        shape.add(arg.getArg(1).getArg(i));

      // TRANSLATION j.
      newStmts.add(makeStmt_coarrayAlloc(coarray, shape));
      // TRANSLATION m.
      newStmts.add(coarray.makeStmt_setCoshape(coshape));
      // TRANSLATION n.
      newStmts.add(coarray.makeStmt_setVarName());
    }

    return newStmts;
  }



  private ArrayList<Xobject> genDeallocateStmt(Xobject x,
                                               ArrayList<XMPcoarray> coarrays) {

    ArrayList<Xobject> newStmts = new ArrayList<Xobject>();

    for (Xobject arg: (XobjList)x.getArg(1)) {
      Xobject varname = arg.getArg(0);
      XMPcoarray coarray = _findCoarrayInCoarrays(varname, coarrays);

      // TRANSLATION j.
      newStmts.add(makeStmt_coarrayDealloc(coarray));
    }

    return newStmts;
  }



  private Xobject makeStmt_coarrayAlloc(XMPcoarray coarray, XobjList shape) {
    int rank = coarray.getRank();
    if (rank != shape.Nargs()) {
      XMP.fatal("Number of dimensions " + rank + 
                " does not equal to " + shape.Nargs() +
                ", the rank of coarray " + coarray.getName());
    }

    Xobject tag;
    if (coarray.wasMovedFromModule() || coarray.def != def) {
      // coarray is originally defined in a use-associated module or
      // in a different procedure
      // ... do not deallocate automatically at the exit of the procedure
      tag = Xcons.IntConstant(0, Xtype.Fint8Type, "8");
    } else {
      tag = Xcons.FvarRef(getResourceTagId());
    }

    Xobject descId = coarray.getDescPointerId();
    if (descId == null)
      descId = Xcons.IntConstant(0, Xtype.Fint8Type, "8");
    Xobject args = Xcons.List(descId,
                              Xcons.FvarRef(coarray.getIdent()),
                              _buildCountExpr(shape, rank),
                              coarray.getElementLengthExpr(),
                              tag,
                              Xcons.IntConstant(rank));

    for (int i = 0; i < rank; i++) {
      args.add(_getLboundInIndexRange(shape.getArg(i)));
      args.add(_getUboundInIndexRange(shape.getArg(i)));
    }
    String subrName = COARRAY_ALLOC_PREFIX + rank + "d";
    if (args.hasNullArg())
      XMP.fatal("generated null argument for " + subrName +
                "(makeStmt_coarrayAlloc)");

    Ident subr = env.findVarIdent(subrName, null);
    if (subr == null)
      subr = env.declExternIdent(subrName,
                                 BasicType.FexternalSubroutineType);
    Xobject subrCall = subr.callSubroutine(args);
    return subrCall;
  }


  private Xobject makeStmt_coarrayDealloc(XMPcoarray coarray) {
    int rank = coarray.getRank();

    Xobject args = Xcons.List(coarray.getDescPointerId(),
                              Xcons.FvarRef(coarray.getIdent()));
    String subrName = COARRAY_DEALLOC_PREFIX + rank + "d";
    if (args.hasNullArg())
      XMP.fatal("generated null argument for " + subrName +
                "(makeStmt_coarrayDealloc)");

    Ident subr = env.findVarIdent(subrName, null);
    if (subr == null)
      subr = env.declExternIdent(subrName,
                                 BasicType.FexternalSubroutineType);
    Xobject subrCall = subr.callSubroutine(args);
    return subrCall;
  }


  private Xobject _buildCountExpr(XobjList shape, int rank) {
    if (rank == 0)
      return Xcons.IntConstant(1);

    Xobject countExpr = _getExtentInIndexRange(shape.getArg(0));
    for (int i = 1; i < rank; i++) {
      countExpr = Xcons.binaryOp(Xcode.MUL_EXPR,
                                 countExpr,
                                 _getExtentInIndexRange(shape.getArg(i))
                                 ).cfold(fblock);
    }

    return countExpr;
  }


  private Xobject _getLboundInIndexRange(Xobject dimension) {
    Xobject lbound;

    if (dimension == null)
      lbound = null;
    else {
      switch (dimension.Opcode()) {
      case F_INDEX_RANGE:
        lbound = dimension.getArg(0);
        break;
      case F_ARRAY_INDEX:
        lbound = null;
        break;
      default:
        lbound = null;
        break;
      }
    }

    if (lbound == null)
      return Xcons.IntConstant(1);

    return lbound.cfold(fblock);
  }


  private Xobject _getUboundInIndexRange(Xobject dimension) {
    Xobject ubound;

    if (dimension == null)
      ubound = null;
    else {
      switch (dimension.Opcode()) {
      case F_INDEX_RANGE:
        ubound = dimension.getArg(1);
        break;
      case F_ARRAY_INDEX:
        ubound = dimension.getArg(0);
        break;
      default:
        ubound = dimension;
      }
    }

    if (ubound == null)
      XMP.fatal("illegal upper bound specified in ALLOCATE statement");

    return ubound.cfold(fblock);
  }


  private Xobject _getExtentInIndexRange(Xobject dimension) {
    Xobject extent;

    if (dimension == null)
      extent = null;
    else {
      switch (dimension.Opcode()) {
      case F_INDEX_RANGE:
        Xobject lbound = dimension.getArg(0);
        Xobject ubound = dimension.getArg(1);
        if (ubound == null)                     // illegal
          extent = null;
        else if (lbound == null)                // lbound omitted
          extent = ubound;
        else {                                  // (ubound - lbound + 1)
          Xobject tmp = Xcons.binaryOp(Xcode.MINUS_EXPR,
                                       ubound,
                                       lbound);
          extent = Xcons.binaryOp(Xcode.PLUS_EXPR,
                                  tmp,
                                  Xcons.IntConstant(1));
        }
        break;
      case F_ARRAY_INDEX:
        extent = dimension.getArg(0);
        break;
      default:
        extent = dimension;
        break;
      }
    }

    if (extent == null)
      XMP.fatal("illegal extent of a dimension specified in ALLOCATE statement");

    return extent.cfold(fblock);
  }


  //-----------------------------------------------------
  //  TRANSLATION f.
  //  remove codimensions from declaration of coarray
  //-----------------------------------------------------
  //
  private void removeCodimensions(ArrayList<XMPcoarray> coarrays) {
    // remove codimensions form coarray declaration
    for (XMPcoarray coarray: coarrays)
      coarray.hideCodimensions();
  }


  //-----------------------------------------------------
  //  TRANSLATION f1.
  //  remove SAVE attributes from declarations of coarrays
  //-----------------------------------------------------
  //
  private void removeSaveAttr(ArrayList<XMPcoarray> coarrays) {
    for (XMPcoarray coarray: coarrays) {
      coarray.resetSaveAttr();
    }
  }

  //-----------------------------------------------------
  //  TRANSLATION h.
  //  replace allocatable attributes with pointer attributes
  //-----------------------------------------------------
  //
  private void replaceAllocatableWithPointer(ArrayList<XMPcoarray> coarrays) {
    for (XMPcoarray coarray: coarrays) {
      coarray.resetAllocatable();
      coarray.setPointer();
    }
  }

  //-----------------------------------------------------
  //  TRANSLATION l.
  //  - fake intrinsic function 'allocated' with 'associated'
  //  - replace this_image(V, ...) with THIS_IMAGE_NAME(DP_V, ...)
  //    except this_image()
  //  - replace image_index(V, ...) with IMAGE_INDEX_NAME(DP_V, ...)
  //  - replace lcobound(V, ...) with COBOUND_NAME(DP_V, ..., 0, corank)
  //  - replace ucobound(V, ...) with COBOUND_NAME(DP_V, ..., 1, corank)
  //-----------------------------------------------------
  //
  private void replaceFunctionCalls(ArrayList<XMPcoarray> coarrays) {
    XobjectIterator xi = new topdownXobjectIterator(def.getFuncBody());
    for (xi.init(); !xi.end(); xi.next()) {
      Xobject xobj = xi.getXobject();
      if (xobj == null)
        continue;
      if (xobj.Opcode() != Xcode.FUNCTION_CALL)
        continue;

      String fname = xobj.getArg(0).getString();

      if (fname.equalsIgnoreCase("allocated"))
        _replaceAllocatedWithAssociated(xobj, coarrays);
      else if (fname.equalsIgnoreCase("this_image"))
        _replaceThisImage(xobj, coarrays);
      else if (fname.equalsIgnoreCase("image_index"))
        _replaceImageIndex(xobj, coarrays);
      else if (fname.equalsIgnoreCase("lcobound"))
        _replaceCobound(xobj, coarrays, 0);
      else if (fname.equalsIgnoreCase("ucobound"))
        _replaceCobound(xobj, coarrays, 1);
    }
  }

  /* replace "allocated(coarray)" with "associated(coarray)"
   */
  private void _replaceAllocatedWithAssociated(Xobject xobj, ArrayList<XMPcoarray> coarrays) {
    Xobject fname = xobj.getArg(0);
    XobjList actualArgs = (XobjList)xobj.getArg(1);
    Xobject arg1 = actualArgs.getArg(0);
    if (_isIntrinsic(fname) && _isCoarrayInCoarrays(arg1, coarrays)) {
      XobjString associated = Xcons.Symbol(Xcode.IDENT, "associated");
      xobj.setArg(0, associated);
    }
  }

  /* replace intrinsic this_image
   */
  private void _replaceThisImage(Xobject xobj, ArrayList<XMPcoarray> coarrays) {
    Xobject fname = xobj.getArg(0);
    XobjList actualArgs = (XobjList)xobj.getArg(1);
    int nargs = (actualArgs == null) ? 0 : actualArgs.Nargs();

    if (nargs == 0) {
      // no conversion: this_image() with no argument
      return;
    }

    if (nargs > 2) {
      XMP.error("Too many arguments was found in this_image().");
      return;
    }

    Xobject arg1 = actualArgs.getArgWithKeyword("coarray", 0);
    if (arg1 == null) {
      XMP.error("Argument coarray was not found in this_image().");
      return;
    }
    XMPcoarray coarray = _findCoarrayInCoarrays(arg1, coarrays);
    if (coarray == null) {
      XMP.error("The argument must be a coarray in this_image().");
      return;
    }

    // replace function name 'this_image'
    XobjString newFname = Xcons.Symbol(Xcode.IDENT, THIS_IMAGE_NAME);
    xobj.setArg(0, newFname);

    // replace actual arguments
    Ident descPtr = coarray.getDescPointerId();
    Xobject corankExpr = Xcons.IntConstant(coarray.getCorank());
    Xobject newActualArgs = Xcons.List(descPtr, corankExpr);

    // add argument DIM if any
    if (nargs == 2) {
      Xobject arg2 = actualArgs.getArgWithKeyword("dim", 1);
      if (arg2 == null) {
        XMP.error("The second argument is illegal in this_image().");
        return;
      }
      newActualArgs.add(arg2);
    }

    xobj.setArg(1, newActualArgs);
    return;
  }


  /* replace intrinsic lcobound/ucobound
   */
  private void _replaceCobound(Xobject xobj, ArrayList<XMPcoarray> coarrays,
                               int lu) {
    Xobject fname = xobj.getArg(0);
    XobjList actualArgs = (XobjList)xobj.getArg(1);
    int nargs = (actualArgs == null) ? 0 : actualArgs.Nargs();

    if (nargs <= 0) {
      XMP.error("Too few arguments was found in lcobound/ucobound.");
      return;
    }
    if (nargs > 3) {
      XMP.error("Too many arguments was found in lcobound/ucobound.");
      return;
    }

    // get first argument coarray
    Xobject arg1 = actualArgs.getArgWithKeyword("coarray", 0);
    if (arg1 == null) {
      XMP.error("Argument coarray was not found in lcobound/ucobound.");
      return;
    }
    XMPcoarray coarray = _findCoarrayInCoarrays(arg1, coarrays);
    if (coarray == null) {
      XMP.error("The first argument must be a coarray in this_.");
      return;
    }

    // get 2nd argument dim
    Xobject arg2 = actualArgs.getArgWithKeyword("dim", 1);

    // get 3nd argument kind
    Xobject arg3 = actualArgs.getArgWithKeyword("kind", 2);
    if (arg3 == null)
      arg3 = Xcons.IntConstant(4);       // kind=4 as default integer

    // replace the function name and the argument list
    XobjString newFname;
    Xobject newActualArgs;
    Ident descPtr = coarray.getDescPointerId();
    Xobject luExpr = Xcons.IntConstant(lu);
    Xobject corankExpr = Xcons.IntConstant(coarray.getCorank());

    // replace function name lcobound/ucobound to COBOUND_NAME
    newFname = Xcons.Symbol(Xcode.IDENT, COBOUND_NAME);

    if (arg2 != null) {
      // replace actual arguments (descPtr, dim, kind, lu, corank)
      newActualArgs = Xcons.List(descPtr, arg2, arg3, luExpr, corankExpr);
    } else {
      // replace actual arguments (descPtr, kind, lu, corank)
      newActualArgs = Xcons.List(descPtr, arg3, luExpr, corankExpr);
    }
 
    xobj.setArg(0, newFname);
    xobj.setArg(1, newActualArgs);
    return;
  }


  /* replace intrinsic image_index
   */
  private void _replaceImageIndex(Xobject xobj,
                                  ArrayList<XMPcoarray> coarrays) {
    Xobject fname = xobj.getArg(0);
    XobjList actualArgs = (XobjList)xobj.getArg(1);
    int nargs = (actualArgs == null) ? 0 : actualArgs.Nargs();

    // error detection: # of args
    if (nargs != 2) {
      XMP.error("Too few or too many arguments of image_index was found.");
      return;
    }

    // get 1st argument coarray
    Xobject arg1 = actualArgs.getArgWithKeyword("coarray", 0);
    if (arg1 == null) {
      XMP.error("Argument 'coarray' was not found in the reference of 'image_index'.");
      return;
    }
    XMPcoarray coarray = _findCoarrayInCoarrays(arg1, coarrays);
    if (coarray == null) {
      XMP.error("The argument of 'image_index' must be a coarray.");
      return;
    }

    // get 2nd argument sub
    Xobject arg2 = actualArgs.getArgWithKeyword("sub", 1);
    if (arg2 == null) {
      XMP.error("Argument 'sub' was not found in the reference of 'image_index'.");
      return;
    }

    // replace function name 'image_index'
    XobjString newFname = Xcons.Symbol(Xcode.IDENT, IMAGE_INDEX_NAME);
    xobj.setArg(0, newFname);

    // replace actual arguments
    Ident descPtr = coarray.getDescPointerId();
    Xobject newActualArgs = Xcons.List(descPtr, arg2);
    xobj.setArg(1, newActualArgs);

    return;
  }


  //-----------------------------------------------------
  //  TRANSLATION o.
  //  remove declarations of coarray variables
  //-----------------------------------------------------
  //
  private void removeDeclOfCoarrays(ArrayList<XMPcoarray> coarrays) {
    for (XMPcoarray coarray: coarrays)
      removeDeclOfCoarray(coarray);
  }

  private void removeDeclOfCoarray(XMPcoarray coarray) {
    env.removeIdent(coarray.getCrayPointerName(), null);
    env.removeIdent(coarray.getDescPointerName(), null);
    env.removeIdent(coarray.getName(), null);
  }



  //-----------------------------------------------------
  //  parts
  //-----------------------------------------------------
  private String _genNewProcPostfix() {
    return _genNewProcPostfix(_getHostNames());
  }

  private String _genNewProcPostfix(String ... names) { // host-to-guest order
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

  private String[] _getHostNames() {
    ArrayList<String> list = new ArrayList<String>();
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
   * Detect error if a coarray exists and xmp_coarray.h is not included.
   */
  private void _check_ifIncludeXmpLib() {
    
    if (!_isCoarrayReferred() && !_isCoarrayIntrinsicUsed()) {
      /* any coarray features are not used */
      return;
    }

    /* check a typical name defined in xmp_coarray.h */
    //    Ident id = def.findIdent("xmpf_coarray_get0d");
    //    if (id == null) {
    //      /* xmpf_lib.h seems not included. */
    //      XMP.error("current restriction: " + 
    //                "\'xmp_coarray.h\' must be included to use coarray features.");
    //    }
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
  //  inquire
  //------------------------------
  private boolean _isMainProgram() {
    Xtype ft = def.getFuncType();
    return (ft != null && ft.isFprogram());
  }

  private boolean _isModule() {
    return  def.isFmoduleDef();
  }

  //------------------------------
  //  tool
  //------------------------------
  private Ident getResourceTagId() {
    if (_resourceTagId == null) {
      BlockList blist = fblock.getBody();
      _resourceTagId = blist.declLocalIdent(VAR_TAG_NAME,
                                            BasicType.Fint8Type,
                                            StorageClass.FLOCAL,
                                            null);
    }

    // Prolog/Epilog codes are necessary if and only if the resource tag is
    // defined.
    set_autoDealloc(true);

    return _resourceTagId;
  }

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


  // for automatic deallocation at the end of the program
  private Boolean get_autoDealloc() {
    return _autoDealloc;
  }
  private void set_autoDealloc(Boolean sw) {
    _autoDealloc = sw;
  }

}

