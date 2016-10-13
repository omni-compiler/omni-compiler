package exc.xmpF;

import exc.object.*;
import exc.block.*;
import java.util.*;

/*
 * Translate Coarray Fortran (for each procedure)
 */
public class XMPtransCoarrayRun
{
  /* These lists should match with F-FrontEnd/src/F-intrinsics-table.c.
   */
  final static String[] _coarrayIntrinsics = {
    // functions
    "num_images",
    "this_image",
    "image_index",
    "lcobound",
    "ucocound",
    // subroutines
    "co_broadcast",
    "co_sum",
    "co_max",
    "co_min",
  };
  final static String[] _coarrayStmtKeywords = {
    "xmpf_critical",
    "xmpf_end_critical",
    "xmpf_error_stop",
    "xmpf_lock",
    "xmpf_sync_all",
    "xmpf_sync_images", 
    "xmpf_sync_memory",
    "xmpf_unlock",
  };
  final static List _coarrayIntrinsicList = Arrays.asList(_coarrayIntrinsics);
  final static List _coarrayStmtKeywordList = Arrays.asList(_coarrayStmtKeywords);

  // constants -- cf. libxmpf/src/xmpf_coarray_decl.f90 for generic names
  final static String VAR_TAG_NAME = "xmpf_resource_tag";
  final static String TRAV_COUNTCOARRAY_PREFIX = "xmpf_traverse_countcoarray";
  final static String TRAV_INITCOARRAY_PREFIX = "xmpf_traverse_initcoarray";
  final static String FIND_DESCPOINTER_NAME   = "xmpf_coarray_find_descptr";
  final static String COARRAY_ALLOC_NAME     = "xmpf_coarray_alloc_generic";
  final static String COARRAY_DEALLOC_NAME   = "xmpf_coarray_dealloc_generic";
  final static String NUM_IMAGES_NAME        = "xmpf_num_images";
  final static String THIS_IMAGE_NAME        = "xmpf_this_image_generic";
  final static String COBOUND_NAME           = "xmpf_cobound_generic";
  final static String IMAGE_INDEX_NAME       = "xmpf_image_index";
  final static String CO_BROADCAST_NAME      = "xmpf_co_broadcast_generic";
  final static String CO_SUM_NAME            = "xmpf_co_sum_generic";
  final static String CO_MAX_NAME            = "xmpf_co_max_generic";
  final static String CO_MIN_NAME            = "xmpf_co_min_generic";
  final static String COARRAY_PROLOG_NAME    = "xmpf_coarray_prolog";
  final static String COARRAY_EPILOG_NAME    = "xmpf_coarray_epilog";
  final static String SYNCALL_NAME           = "xmpf_sync_all";
  final static String SYNCIMAGES_NAME        = "xmpf_sync_images";
  final static String AUTO_SYNCALL_NAME      = "xmpf_sync_all_auto";  // another entry of syncall
  final static String FINALIZE_PROGRAM_NAME  = XMP.finalize_all_f;

  // generic intrinsic names that will be renamed in pass1 and pass2
  final static List<String> intrinsicProcedureNames = 
    Arrays.asList( "this_image",
                   "lcobound", "ucobound",
                   "image_index",
                   "co_broadcast",
                   "co_sum", "co_max", "co_min" );


  /** Available Versions currently:
   *
   *  3 Static coarray variables are allocated inside the comunincation library 
   *    and registered with the communication library by the initializer, which
   *    is automatically generated at compile time corresponding to the program
   *    and executed just before the execution of the program.
   *    The initializer informs the program of the address of the coarrays via
   *    common-associated cray ponters.
   *
   *  4 (Supported only for FJ-RDMA and MPI3.) Static coarray variables are allocated
   *    allocated statically by the Fortran system as usual. The initializer accepts 
   *    the addresses of the coarrays from the user program by common-association,
   *    and registers the addresses with the commmunication library. Cray pointers 
   *    are not used.
   *
   *  6 (Supported only For FJ-RDMA and MPI3 with some restriction.) Procedure-local
   *    static coarrays are allocated statically by the Fortran system as usual.
   *    Instead of using the initializer for the procedure-local coarrays, the 
   *    registration of coarrays with the communication library is coused at the
   *    entry point of the first call of the procedure. Cray pointers are not used.
   *
   *  7 The initializer is generated as a part of the corresponding procedure 
   *    starting with the ENTRY statement. So, coarrays declared in the procedure are 
   *    commonly visible to the procedure and to the initializer as procedure-local 
   *    variables. 
   *    For FJ-RDMA and MPI3, procedure-local static coarrays are allocated by the
   *    Fortran system as usual and registered with the communication library by
   *    the initializer. Cray pointers or common accociation are not used.
   *    For GASNet, procedure-local static coarrays are allocated in GASnet and 
   *    and registered with GASNet by the initializer. The initializer informs the 
   *    program of the address of a coarray via a Cray pointer. Common association
   *    is not used.
   */

  private int version;
  private Boolean useGASNet;
  private Boolean onlyCafMode;

  private Boolean DEBUG = true;       // change me in debugger

  private XMPenv env;
  private String name;
  private XobjectDef def;
  FuncDefBlock funcDef;

  // to handle host- and use-associations
  static ArrayList<XMPtransCoarrayRun> ancestors
    = new ArrayList<XMPtransCoarrayRun>();

  // coarrays in the Ident List are divied into two:
  private ArrayList<XMPcoarray> localCoarrays;         // procedire-local coarrays
  private ArrayList<XMPcoarray> useAssociatedCoarrays; // use-associated coarrays
                                                       // (COPIED-IN to this procedure)
  // localCoarrays are divided into four:
  private ArrayList<XMPcoarray> staticLocalCoarrays;
  private ArrayList<XMPcoarray> allocatableLocalCoarrays;
  private ArrayList<XMPcoarray> staticDummyCoarrays;
  private ArrayList<XMPcoarray> allocatableDummyCoarrays;

  // joint list of static coarrays in useAssociatedCoarrays and
  //               host-associated coarrays from the host module
  private ArrayList<XMPcoarray> staticAssociatedCoarrays;    // (COPIED-IN)
  // allocatable coarrays in useAssociatedCoarrays
  private ArrayList<XMPcoarray> allocatableAssociatedCoarrays; // (COPIED-IN)


  // the host module and the host procedure
  private String hostModuleName, hostProcedureName;
  private XMPtransCoarrayRun hostModuleRun, hostProcedureRun;

  // localCoarrays + static/allocatableAssociatedCoarrays (COPIED-IN)
  // + visibleCoarrays of the host (not COPIED-IN)
  private ArrayList<XMPcoarray> visibleCoarrays;    // available after run1()

  //private XMPinitProcedure initProcedure;
  private String initProcTextForFile;

  private String traverseCountName, traverseInitName;
  private Ident _resourceTagId = null;

  // statements to be added at the top of the execution part
  private ArrayList<Xobject> _prologStmts = new ArrayList<Xobject>();
  // statements to be added before all RETURN statements
  private ArrayList<Xobject> _epilogStmts = new ArrayList<Xobject>();
  // statements to be added between the last RETURN and END statements
  private ArrayList<Xobject> _extraStmts = new ArrayList<Xobject>();

  private Boolean _autoDealloc;


  //------------------------------------------------------------
  //  CONSTRUCTOR
  //------------------------------------------------------------
  public XMPtransCoarrayRun(XobjectDef def, XMPenv env,
                            ArrayList<XMPtransCoarrayRun> pastRuns,
                            int pass, int version,
                            Boolean useGASNet, Boolean onlyCafMode)
  {
    this.def = def;
    this.env = env;
    name = def.getName();
    this.version = version;
    this.useGASNet = useGASNet;
    this.onlyCafMode = onlyCafMode;

    funcDef = new FuncDefBlock(def);
    env.setCurrentDef(funcDef);

    _setHostName();
    _setHostRun(pastRuns);

    String postfix = _genNewProcPostfix();
    traverseCountName = TRAV_COUNTCOARRAY_PREFIX + postfix;
    traverseInitName = TRAV_INITCOARRAY_PREFIX + postfix;

    XMP.exitByError();   // exit if error was found.
  }

  private void disp_version(String opt)
  {
    switch (version) {
    case 3:
      // default and stable version
      break;
    case 4:
      XMP.warning("Coarray Fortran Version 4 (trial version): " + opt);
      break;
    case 6:
      XMP.warning("Coarray Fortran Version 6 (another trial version): " + opt);
      break;
    case 7:
      if (useGASNet)
        XMP.warning("Coarray Fortran Version 7 (latest version) over GASNet: " + opt);
      else
        XMP.warning("Coarray Fortran Version 7 (latest version): " + opt);
      break;
    default:
      XMP.fatal("Wrong version number (" + version +
                ") specified for Coarray Fortran");
      break;
    }
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

    /* divide coarrays into localCoarrays and useAssociatedCoarrays
     */
    Xobject idList = def.getFuncIdList();
    for (Xobject obj: (XobjList)idList) {
      Ident ident = (Ident)obj;
      if (ident.wasCoarray()) {
        // found it is a coarray or a variable converted from a coarray
        XMPcoarray coarray = new XMPcoarray(ident, def, getFblock(), env);
        if (coarray.isUseAssociated())
          useAssociatedCoarrays.add(coarray);
        else
          localCoarrays.add(coarray);
      }
    }

    /* localize coarrays of the using modules into this procedure
     */
    for (int i = 0; i < useAssociatedCoarrays.size(); i++) {
      XMPcoarray coarray = useAssociatedCoarrays.get(i);
      useAssociatedCoarrays.set(i, localizedCopyOfCoarray(coarray));
    }

    /* divide localCoarrays into four sets:
     *  - procedure-local coarrays with save attributes
     *  - procedure-local and allocatable coarrays
     *  - explicit-shape dummy coarrays
     *  - assumed- and deffered-shape (i.e., allocatable) dummy coarrays
     */
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

    /* divide useAssociatedCoarrays into two sets:
     *  - explicit-shape use-associated coarrays
     *  - deffered-shape (i.e., allocatable) use-associated coarrays
     */
    staticAssociatedCoarrays = new ArrayList<XMPcoarray>();
    allocatableAssociatedCoarrays = new ArrayList<XMPcoarray>();
    for (XMPcoarray coarray: useAssociatedCoarrays) {
      if (coarray.isExplicitShape())
        //staticAssociatedCoarrays.add(localizeCopyCoarray(coarray));
        staticAssociatedCoarrays.add(coarray);
      else
        //allocatableAssociatedCoarrays.add(localizeCopyCoarray(coarray));
        allocatableAssociatedCoarrays.add(coarray);
    }

    /* staticAssociatedCoarrays is expanded with such host-associated static coarrays
     * under the following condition:
     *  - The current procedure is a module procedure.
     *  - The coarrays are defined in the host module or in the modules use-associated
     *    in the host module.
     */
    if (hostModuleRun != null && hostProcedureRun == null) {
      // found I am a module procedure

      for (XMPcoarray coarray: hostModuleRun.staticLocalCoarrays)
        staticAssociatedCoarrays.add(localizedCopyOfCoarray(coarray));

      for (XMPcoarray coarray: hostModuleRun.staticAssociatedCoarrays)
        staticAssociatedCoarrays.add(localizedCopyOfCoarray(coarray));
    }
  }

  /*
   *  useful to copy host- and use-associcated coarrays into the list of
   *  local coarrays.
   */
  private XMPcoarray localizedCopyOfCoarray(XMPcoarray coarray1)
  {
    Ident ident1 = coarray1.getIdent();
    Xtype type2 = ident1.Type().copy();
    String name = coarray1.getName();
    env.removeIdent(name, null);
    Ident ident2 = env.declIdent(name, type2);
    ident2.setFdeclaredModule(null);

    // reset ident, name, isAllocatable, isPointer and _isUseAssociated
    // but not changed homeBlockName, declCommonName and crayCommonName
    XMPcoarray coarray2 = new XMPcoarray(ident2, def, getFblock(), env,
                                         coarray1.getHomeBlockName());
    coarray2.setWasMovedFromModule(true);
    return coarray2;
  }


  /*  set visibleCoarrays
   */
  private void _setVisibleCoarrays() {
    /*  set the following coarrays as visibleCoarrays:
     *   1. coarrays declared in the current procedure (localCoarrays),
     *   2. the use- and host- associated coarrays that are copied into this
     *      procedure (static/allocatableAssociatedCoarrays), and
     *   3. visibleCoarrays of the host-associated (parent) procedure/module
     */
    visibleCoarrays = new ArrayList<XMPcoarray>();
    _mergeCoarraysByName(visibleCoarrays, localCoarrays);
    _mergeCoarraysByName(visibleCoarrays, staticAssociatedCoarrays);
    _mergeCoarraysByName(visibleCoarrays, allocatableAssociatedCoarrays);

    if (hostProcedureRun != null)
      _mergeCoarraysByName(visibleCoarrays, hostProcedureRun.visibleCoarrays);
    else if (hostModuleRun != null)
      _mergeCoarraysByName(visibleCoarrays, hostModuleRun.visibleCoarrays);
  }

  private void _mergeCoarraysByName(ArrayList<XMPcoarray> coarrays1,
                                    ArrayList<XMPcoarray> coarrays2)
  {
    ArrayList<XMPcoarray> newCoarrays = new ArrayList<XMPcoarray>();
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

    _setLocalCoarrays();
    /* visibleCoarrays will be set after run1 */

    if (version > 3)
      disp_version("run1, " + getName());

    if (isModule()) {
      run1_module();
    } else {
      run1_procedure();

      if (onlyCafMode) {
        // SPECIAL HANDLING (TEMPORARY) to work XMPtransCoarray alone without XMPtranslate
        //  convert main program to soubroutine xmpf_main
        if (isMainProgram())
          _convMainProgramToSubroutine("xmpf_main");
      }
    }

  }
        

  private void run1_procedure() {
    // flag for automatic deallocation of allocatable coarrays
    set_autoDealloc(false);

    // convert specification and declaration part
    transDeclPart_staticLocal();
    transDeclPart_allocatableLocal();
    transDeclPart_staticDummy();
    transDeclPart_allocatableDummy();
    transDeclPart_staticAssociated();
    transDeclPart_allocatableAssociated();

    // To avoid trouble of the shallow/deep copies, visibleCoarrays
    // should be made after execution of transDeclPart_*.
    _setVisibleCoarrays();
    transExecPart();

    funcDef.Finalize();
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
    // convert specification and declaration part
    transModule_staticLocal1();
    transModule_staticAssociated1();
    transModule_allocatableLocal();
    transModule_allocatableAssociated();

    // To avoid trouble of the shallow/deep copies, visibleCoarrays
    // should be made after execution of transDeclPart_*.
    _setVisibleCoarrays();

    funcDef.Finalize();
  }


  /*
   *  PASS 2: for each module 
   *          excluding its module functions and subroutines
   */
  public void run2() {

    if (!isModule())
      return;                 // do nothing

    _setLocalCoarrays();
    /* visibleCoarrays will be set after run1 */

    if (version > 3)
      disp_version("run2, " + getName());

    // convert specification and declaration part
    transModule_staticLocal2();
  }



  /**
    PROCEDURE-LOCAL STATIC COARRAYS
    --------------------------------------------
      subroutine EX1
        real :: V1(10,20)[4,*]
        ...
      end subroutine
    --------------------------------------------
    output (ver.3):
    --------------------------------------------
      subroutine EX1
        real :: V1(1:10,1:20)                                        ! f. f1.
        common /xmpf_crayptr_M1/ crayptr_V1                          ! c.
        pointer (crayptr_V1, V1)                                     ! c.
        integer(8) :: descptr_V1                                     ! a.
        common /xmpf_descptr_M1/ descptr_V1                          ! a1.
        ...
      end subroutine

    !! Generate subroutines traverse_{count,init}coarrays_EX1        ! b.
    !! into the same file to allocate crayptr_V1 and set descptr_V1
    !! (See XMPcoarrayInitProcedure).
    --------------------------------------------
    output (ver.4):
    --------------------------------------------
      subroutine EX1
        real :: V1(1:10,1:20)                                        ! f. f1.
        common /xmpf_COARRAY_M1/ V1                                  ! f4.
        integer(8) :: descptr_V1                                     ! a.
        common /xmpf_descptr_M1/ descptr_V1                          ! a1.
        ...
      end subroutine

    !! Generate subroutine traverse_initcoarrays_EX1 into the same    ! b.
    !! file to register V1 and set descptr_V1 
    !! (See XMPcoarrayInitProcedure).
    --------------------------------------------
    output (ver.6):
    --------------------------------------------
      subroutine EX1
        real, save :: V1(1:10,1:20)                                  ! f. f6.
        integer(8), save :: descptr_V1 = 0_8                         ! a. a6.

        if (descptr_V1 == 0_8) then                                  ! b6.
          call xmpf_coarray_regmem_static(descptr_V1, LOC(V1), ...)
          call xmpf_coarray_set_coshape(descptr_V1, 2, ,,,)
          ...
        end if
        ...
      end subroutine
    --------------------------------------------
    output (ver.7 for FJ-RDMA and MPI3):
    --------------------------------------------
      subroutine EX1
        real, save :: V1(1:10,1:20)                                  ! f. f6.
        integer(8), save :: descptr_V1                               ! a. a7.
        ...
        return
      entry initcoarrays_EX1                                         ! b7.
        call xmpf_coarray_regmem_static(descptr_V1, LOC(V1), ...)
        call xmpf_coarray_set_coshape(descptr_V1, 2, ,,,)
        return
      end subroutine
    --------------------------------------------
    output (ver.7 for GASNet):
    --------------------------------------------
      subroutine EX1
        real :: V1(1:10,1:20)                                        ! f. f1.
        pointer (crayptr_V1, V1)                                     ! c7.
        save crayptr_V1                                              ! c7.
        integer(8), save :: descptr_V1                               ! a. a7.
        ...
        return
      entry countcoarrays_EX1                                        ! b7g.
        call xmpf_coarray_count_size(1, 16)
        return
      entry initcoarrays_EX1                                         ! b7g.
        call xmpf_coarray_alloc_static(descptr_V1, crayptr_V1, ...)
        call xmpf_coarray_set_coshape(descptr_V1, 2, ,,,)
        return
      end subroutine
    --------------------------------------------      
  */
  private void transDeclPart_staticLocal() {

    /*--- a. DESCRIPTOR corresponding to the coarray ---*/
    // a. declare descriptor pointers
    genDeclOfDescPointer(staticLocalCoarrays);
    if (version == 7) {
      // a7. add SAVE attributes without initialization
      addSaveAttrToDescPointer(staticLocalCoarrays, false);
    } else if (version == 6) {
      // a6. add SAVE attributes and initialization to descriptors
      addSaveAttrToDescPointer(staticLocalCoarrays);
    } else {
      // a1. make common association of descriptors
      genCommonStmt(staticLocalCoarrays);
    }

    /*--- c. Cray pointer ---*/
    if (version == 7 && useGASNet) {
      // c7. generate Cray-POINTER with SAVE attributes
      genDeclOfCrayPointer_withSave(staticLocalCoarrays);
    } else if (version <= 3) {
      // c. generate Cray-POINTER and COMMON statements
      genDeclOfCrayPointer(staticLocalCoarrays);
      genCommonStmtForCrayPointer(staticLocalCoarrays);
    } else { //version 4 or 6 or 7 and !useGASNet
      // nothing
    }

    /*--- b. Execution statements for Initialization ---*/
    if (version == 7) {
      if (useGASNet)
        // b7g. generate ENTRY-block incl. count and alloc call
        readyBlockForStaticCoarrays_alloc(staticLocalCoarrays);
      else
        // b7. generate ENTRY-block incl. regmem call
        readyBlockForStaticCoarrays_regmem(staticLocalCoarrays);
    } else if (version == 6) {
      // b6. generate IF-block incl. regmem-call at the top of body
      genRegmemOfStaticCoarrays(staticLocalCoarrays);
    } else {
      // b. generate allocation into the init procedure
      genAllocOfStaticCoarrays(staticLocalCoarrays);
    }

    /*--- f. Conversion of the COARRAY variable ---*/
    // f. remove codimensions from declarations of coarrays
    removeCodimensions(staticLocalCoarrays);
    if (version == 6) {
      // f6. add SAVE attributes to declarations of coarrays
      addSaveAttr(staticLocalCoarrays);
    } else {
      // f1. remove SAVE attributes from declarations of coarrays
      removeSaveAttr(staticLocalCoarrays);
      if (version == 4) {
        // f4. generate common block for data
        genCommonBlockForStaticCoarrays(staticLocalCoarrays);
      }
    }
  }


  /**
    USE-ASSOCIATED STATIC COARRAYS
    --------------------------------------------
      subroutine EX1
      use M1    !! contains "real :: V1(10,20)[4,*]"
        ...
      end subroutine
    --------------------------------------------
    output (ver.3):
    --------------------------------------------
      subroutine EX1
        use M1
        real :: V1(1:10,1:20)                                        ! f. f1.
        common /xmpf_crayptr_M1/ crayptr_V1                          ! c.
        pointer (crayptr_V1, V1)                                     ! c.
        integer(8) :: descptr_V1                                     ! a.
        common /xmpf_descptr_M1/ descptr_V1                          ! a1.
        ...
      end subroutine

    !! Do not generate initializer here for crayptr_V1 and descptr_V1.
    --------------------------------------------
    output (ver.4 and 6):
    --------------------------------------------
      subroutine EX1
        use M1
        real :: V1(1:10,1:20)                                        ! f. f1.
        common /xmpf_COARRAY_M1/ V1                                  ! c4.
        integer(8) :: descptr_V1                                     ! a.
        common /xmpf_descptr_M1/ descptr_V1                          ! a1.
        ...
      end subroutine

    !! Do not generate initializer here for descptr_V1.
    --------------------------------------------
  */
  private void transDeclPart_staticAssociated() {
    // a. declare descriptor pointers
    genDeclOfDescPointer(staticAssociatedCoarrays);
    // a1. make common association of descriptors
    genCommonStmt(staticAssociatedCoarrays);

    if (version >= 4) {
      // c4. generate common block for data
      genCommonBlockForStaticCoarrays(staticAssociatedCoarrays);
    } else {
      // c. generate Cray-POINTER and COMMON statements
      genDeclOfCrayPointer(staticAssociatedCoarrays);
      genCommonStmtForCrayPointer(staticAssociatedCoarrays);
    }

    // b. generate allocation into the init procedure
    genAllocOfStaticCoarrays(staticAssociatedCoarrays);

    // f. remove codimensions from declarations of coarrays
    removeCodimensions(staticAssociatedCoarrays);

    // f1. remove SAVE attributes from declarations of coarrays
    removeSaveAttr(staticAssociatedCoarrays);
  }


  /**
    LOCAL ALLOCATABLE COARRAYS in a PROCEDURE
    --------------------------------------------
      subroutine EX1
        integer, allocatable [, save] :: V3(:,:)[:,:]       ! allocatable local
        ...
      end subroutine
    --------------------------------------------
    output:
    --------------------------------------------
      subroutine EX1
        integer, pointer :: V3(:,:)                                  ! f. f1. h.
        integer(8) :: descptr_V3 = 0_8                               ! a3.
        ...
      end subroutine
    --------------------------------------------
  */
  private void transDeclPart_allocatableLocal() {
    // a3. declare descriptor pointers with zero-init
    genDeclOfDescPointer(allocatableLocalCoarrays, true);

    // f. remove codimensions from declarations of coarrays
    removeCodimensions(allocatableLocalCoarrays);

    // f1. remove SAVE attributes from declarations of coarrays
    removeSaveAttr(allocatableLocalCoarrays);

    // h. replace allocatable attributes with pointer attributes
    replaceAllocatableWithPointer(allocatableLocalCoarrays);
  }


  /**
    Handling use-associated allocatable coarrays in a procedure
    --------------------------------------------
      subroutine EX1
        use M1   !! contains "integer, allocatable :: V3(:,:)[:,:]"
        ...
      end subroutine
    --------------------------------------------
    output:
    --------------------------------------------
      subroutine EX1
        use M1   !! contains new definition of V3
        integer(8) :: descptr_V3 = 0_8                               ! a3.
        ...
      end subroutine
    --------------------------------------------
  */
  private void transDeclPart_allocatableAssociated() {
    // a3. declare descriptor pointers with zero-init
    genDeclOfDescPointer(allocatableAssociatedCoarrays, true);
  }


  /**
    LOCAL ALLOCATABLE COARRAYS in a MODULE
    --------------------------------------------
      module EX1
        integer, allocatable :: V3(:,:)[:,:]            ! allocatable local
        ...
      end module
    --------------------------------------------
    output:
    --------------------------------------------
      module EX1
        integer, pointer :: V3(:,:)                                  ! f. h.
        integer(8) :: descptr_V3                                     ! a.
        ...
      end module
    --------------------------------------------
  */
  private void transModule_allocatableLocal() {
    // a. declare descriptor pointers
    genDeclOfDescPointer(allocatableLocalCoarrays);

    // f. remove codimensions from declarations of coarrays
    removeCodimensions(allocatableLocalCoarrays);

    // h. replace allocatable attributes with pointer attributes
    replaceAllocatableWithPointer(allocatableLocalCoarrays);
  }

  private void transModule_allocatableAssociated() {
    // a. declare descriptor pointers
    genDeclOfDescPointer(allocatableAssociatedCoarrays);

    // f. remove codimensions from declarations of coarrays
    removeCodimensions(allocatableAssociatedCoarrays);

    // h. replace allocatable attributes with pointer attributes
    replaceAllocatableWithPointer(allocatableAssociatedCoarrays);
  }



  /**
    NON-ALLOCATABLE DUMMY COARRAYS in a procecure
    --------------------------------------------
      subroutine EX1(V2)
        complex(8) :: V2[0:*]                          ! static dummy
        ... body ...
        return
      end subroutine
    --------------------------------------------
    output:
    --------------------------------------------
      subroutine EX1(V2)
        complex(8) :: V2                                          ! f.
        integer(8) :: descptr_V2 = 0_8                            ! a3.

        !-- initialization for procedure EX1
      ( integer(8) :: tag                                         ! i. )
      ( call xmpf_coarray_prolog(tag, "EX1", 3)                   ! i. )

        !-- find descptr_V2 and set the attributes
        call xmpf_coarray_get_descptr(descptr_V2, V2, tag)        ! a2.
        call xmpf_coarray_set_coshape(descptr_V2, 1, 0)           ! m.
        call xmpf_coarray_set_varname(descptr_V2, "V2", 2)        ! n.

        ... body ...

        !-- finalization for procedure EX1
      ( call xmpf_coarray_epilog(tag)                        ! i. )
        return
      end subroutine
    --------------------------------------------
  */
  private void transDeclPart_staticDummy() {

    // a. declare descriptor pointers with zero-init
    genDeclOfDescPointer(staticDummyCoarrays, true);

    // a2. m. n. generate definition of descriptor pointers (dummy coarrays only)
    genDefinitionOfDescPointer(staticDummyCoarrays);

    // f. remove codimensions from declarations of coarrays
    removeCodimensions(staticDummyCoarrays);
  }


  /**
    ALLOCATABLE DUMMY COARRAYS in a procecure
    --------------------------------------------
      subroutine EX1(V3)
        integer, allocatable :: V3(:,:)[:,:]           ! allocatable dummy
        ...
      end subroutine
    --------------------------------------------
    output:
    --------------------------------------------
      subroutine EX1(V3)
        integer, pointer :: V3(:,:)                               ! f. h.
        integer(8) :: descptr_V3 = 0_8                            ! a3.

        // find descptr_V3 and set attributes
        call xmpf_coarray_get_descptr(descptr_V3, V3, tag)        ! a2.
        call xmpf_coarray_set_varname(descptr_V3, "V3", 2)        ! n.

        ...
      end subroutine
    --------------------------------------------
  */
  private void transDeclPart_allocatableDummy() {

    if (allocatableDummyCoarrays.isEmpty())
      return;

    // a3. declare descriptor pointers with zero-init
    genDeclOfDescPointer(allocatableDummyCoarrays, true);

    // a2. m. n. generate definition of descriptor pointers (dummy coarrays only)
    genDefinitionOfDescPointer(allocatableDummyCoarrays);

    // f. remove codimensions from declarations of coarrays
    removeCodimensions(allocatableDummyCoarrays);

    // h. replace allocatable attributes with pointer attributes
    replaceAllocatableWithPointer(allocatableDummyCoarrays);
  }


  /**
    STATIC COARRAYS in MODULE
    --------------------------------------------
      module EX1
        real :: V1(10,20)[4,*]
        ...
      end module
    --------------------------------------------
    output (ver.3, 7g)
    (Pass1 is similar to transDeclPart_staticLocal ver.3)
    --------------------------------------------
      module EX1                                     ! pass1    ! pass2
        real :: V1(1:10,1:20)[*]                     ! (none)   ! o.(delete)
        pointer (crayptr_V1, V1)                     ! c.       ! o.(delete)
        integer(8) :: descptr_V1                     ! a.       ! o.(delete)
        ...
      end module
                                                                     ! pass1
    !! Generate subroutines traverse_{count,init}coarrays_EX1        ! b.
    !! into the same file to allocate crayptr_V1 and set descptr_V1
    !! (See XMPcoarrayInitProcedure).
    --------------------------------------------
    output (ver.4, 6, 7)
    (Pass1 is similar to transDeclPart_staticLocal ver.4)
    --------------------------------------------
      module EX1                                     ! pass1    ! pass2
        real :: V1(1:10,1:20)[*]                     ! (none)   ! o.(delete)
        integer(8) :: descptr_V1                     ! a.       ! o.(delete)
        ...
      end module

    !! Generate subroutine traverse_initcoarrays_EX1 into the same    ! b.
    !! file to register V1 and set descptr_V1 
    !! (See XMPcoarrayInitProcedure).
    --------------------------------------------
  */
  private void transModule_staticLocal1() {
    /*--- a. DESCRIPTOR corresponding to the coarray ---*/
    genDeclOfDescPointer(staticLocalCoarrays);

    if (version == 3 || version == 7 && useGASNet) {
      // c. generate Cray-POINTER
      genDeclOfCrayPointer(staticLocalCoarrays);
    }

    /*--- b. Execution statements for Initialization ---*/
    genAllocOfStaticCoarrays(staticLocalCoarrays);
  }

  private void transModule_staticLocal2() {
    // o. remove declarations of variables
    removeDeclOfCoarrays(staticLocalCoarrays);
  }


  private void transModule_staticAssociated1() {
    /**************************  DO NOTHING HERE
    // a. declare descriptor pointers
    genDeclOfDescPointer(staticAssociatedCoarrays);

    // b. generate allocation into init procedure (static coarrays only)
    genAllocOfStaticCoarrays(staticAssociatedCoarrays);
    ***************************************/
  }

  private void transModule_staticAssociated2() {
    /**************************  DO NOTHING HERE
    // o. remove declarations of variables
    removeDeclOfCoarrays(staticAssociatedCoarrays);
    ***************************************/
  }



  /**
    EXECUTION PART
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
        integer(8) :: tag                                            ! i.
        call xmpf_coarray_prolog(tag, "EX1", 3)                      ! i.
        call xmpf_coarray_put(descptr_V1, V1(1,j), 4, &              ! d.
          k1+4*(k2-1), (/1.0,2.0,3.0/), ...)      
        z = xmpf_coarray_get0d(descptr_V2, V2, 16, k, 0) ** 2        ! e.
        call xmpf_coarray_alloc2d(descptr_V3, V3, tag, 4, 2, 10, 20) ! j.
        call xmpf_coarray_set_coshape(descptr_V3, 2, k1, k2, 0)      ! m.
        call xmpf_coarray_set_varname(descptr_V3, "V3", 2)           ! n.
        call xmpf_coarray_dealloc(descptr_V3)                        ! j.
        call xmpf_syncall(V1,V4,V2,V3)                               ! p.
        if (associated(V3)) write(*,*) "yes"                         ! l.
        n1 = xmpf_this_image(descptr_V1,1)                           ! l.
        n2(:) = this_image(descptr_V3)                               ! l.
        n3 = xmpf_image_index(descptr_V1,(/1,2/))                    ! l.
        ...
        call xmpf_syncall(V1,V4,V2,V3)                               ! i. p.
        call xmpf_coarray_epilog(tag)                                ! i.
        return
        ...
        call xmpf_finalize_all_f()                                   ! k.
        stop
      end subroutine

    !! Additionally, two subroutines xmpf_traverse_* will            ! b.
    !! be generated into the same output file which will
    !! initialize descptr_V2 and crayptr_V2.
    !! (See XMPcoarrayInitProcedure.)
    --------------------------------------------
  */
  private void transExecPart() {

    // e. convert coindexed objects to function references
    convCoidxObjsToFuncCalls(visibleCoarrays);

    // d. convert coindexed variable assignment stmts to call stmts
    convCoidxStmtsToSubrCalls(visibleCoarrays);

    // j. convert allocate/deallocate stmts (allocatable coarrays only)
    convAllocateStmts(visibleCoarrays);
    convDellocateStmts(visibleCoarrays);

    // l. fake intrinsic 'allocatable' (allocatable coarrays only)
    //    replace V of coarray intrinsic calls with descptr_V
    replaceFunctionCalls(visibleCoarrays);

    // i. initialization/finalization for auto-syncall and auto-deallocate
    //    and initialization of descPtr (only Ver.6)
    genCallOfPrologAndEpilog();

    // k. insert finalization call before STOP statements (onlyCafMode only)
    if (onlyCafMode)
      insertFinalizationCall();

    // p. add visible coarrays as arguments of sync all statements 
    //     to prohibit code motion
    addVisibleCoarraysToSyncall(visibleCoarrays);

    // o. remove declarations for use-associated allocatable coarrays
    for (XMPcoarray coarray: useAssociatedCoarrays) {
      if (!coarray.isExplicitShape()) 
        removeDeclOfCoarray(coarray);
    }

    // b7. b7g. expand generated ENTRY block
    expandEntryBlockForStaticCoarrays();
  }



  //-----------------------------------------------------
  //  TRANSLATION a.
  //  declare variables of descriptor pointers
  //-----------------------------------------------------
  //
  private void genDeclOfDescPointer(ArrayList<XMPcoarray> coarrays) {
    genDeclOfDescPointer(coarrays, false);
  }
  private void genDeclOfDescPointer(ArrayList<XMPcoarray> coarrays,
                                    Boolean withZeroInit) {
    for (XMPcoarray coarray: coarrays) {
      coarray.genDecl_descPointer();
      if (withZeroInit)
        coarray.setZeroToDescPointer();
    }
  }

  //-----------------------------------------------------
  //  TRANSLATION a1. (static only)
  //  generate common association of descriptor
  //-----------------------------------------------------
  //
  private void genCommonStmt(ArrayList<XMPcoarray> coarrays) {
    // do nothing if no coarrays are declared.
    if (coarrays.isEmpty())
      return;

    ArrayList<String> cnameList = new ArrayList<String>();
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
      Xobject decls = getFblock().getBody().getDecls();
      Xobject args = Xcons.List(Xcode.F_COMMON_DECL,
                                Xcons.List(Xcode.F_VAR_LIST, cnameObj, varList));
      decls.add(args);
    }
  }


  //-----------------------------------------------------
  //  TRANSLATION a6. a7.
  //  add SAVE attributes (if needed) and optionally set
  //  initial value zero to desc-pointers of coarrays
  //-----------------------------------------------------
  //
  private void addSaveAttrToDescPointer(ArrayList<XMPcoarray> coarrays)
  {
    addSaveAttrToDescPointer(coarrays, true);
  }
  private void addSaveAttrToDescPointer(ArrayList<XMPcoarray> coarrays,
                                        Boolean initialized)
  {
    if (isModule()) {
      XMP.fatal("unexpected situation (XMPtransCoarrayRun.addSaveAttrToDescPointer)");
      return;
    }

    /* This is not necessary because these variables have implicit SAVE
     * attributes bue to the initialization value defined later.
     */
    if (!isMainProgram()) {
      for (XMPcoarray coarray: coarrays) {
        coarray.setSaveAttrToDescPointer();   // add SAVE attr.
      }
    }

    if (initialized) {
      for (XMPcoarray coarray: coarrays)
        coarray.setZeroToDescPointer();
    }
  }

  //-----------------------------------------------------
  //  TRANSLATION c4. for Ver.4
  //    generate common block for coarray variables
  //-----------------------------------------------------
  //
  private void genCommonBlockForStaticCoarrays(ArrayList<XMPcoarray> coarrays) {
    // do nothing if no coarrays are declared.
    if (coarrays.isEmpty())
      return;

    ArrayList<String> cnameList = new ArrayList<String>();
    for (XMPcoarray coarray0: coarrays) {
      String cname = coarray0.getCoarrayCommonName();
      if (cnameList.contains(cname))
        continue;

      // found new common block to be declared
      cnameList.add(cname);

      Xobject cnameObj = Xcons.Symbol(Xcode.IDENT, cname);
      Xobject varList = Xcons.List();
      for (XMPcoarray coarray: coarrays) {
        if (cname.equals(coarray.getCoarrayCommonName())) {
          Ident coarrayId = coarray.getIdent();
          varList.add(Xcons.FvarRef(coarrayId));
        }
      }

      // add declaration 
      Xobject decls = getFblock().getBody().getDecls();
      Xobject args = Xcons.List(Xcode.F_COMMON_DECL,
                                Xcons.List(Xcode.F_VAR_LIST, cnameObj, varList));
      decls.add(args);
    }
  }

  //-----------------------------------------------------
  //  TRANSLATION c. (for Ver.3) and c7. (for Ver.7)
  //    link cray pointers with data objects (Cray-POINTER stmt)
  //    and optionally generate:
  //      - common association to the initial subroutine, or
  //      - save attribute
  //-----------------------------------------------------
  //
  private void genDeclOfCrayPointer(ArrayList<XMPcoarray> coarrays)
  {
    genDeclOfCrayPointer(coarrays, false);
  }
  private void genDeclOfCrayPointer(ArrayList<XMPcoarray> coarrays,
                                    Boolean withSave)
  {
    // do nothing if no coarrays are declared.
    if (coarrays.isEmpty())
      return;

    // genDecl_crayPointer
    for (XMPcoarray coarray: coarrays) {
      coarray.genDecl_crayPointer(withSave);
    }
  }

  private void genDeclOfCrayPointer_withSave(ArrayList<XMPcoarray> coarrays)
  {
    genDeclOfCrayPointer(coarrays, true);
  }


  private void genCommonStmtForCrayPointer(ArrayList<XMPcoarray> coarrays)
  {                                           
    ArrayList<String> cnameList = new ArrayList<String>();
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
      Xobject decls = getFblock().getBody().getDecls();
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
    BlockIterator bi = new topdownBlockIterator(getFblock());
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
      /* Not F_ASSIGN_STATEMENT but EXPR_STATEMENT contains call statement */
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
    BasicBlockIterator bbi = new BasicBlockIterator(getFblock());
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
    genPrologStmts();          // stmts on the top of body
    genEpilogStmts();          // stmts before RETURN- and END-stmts
  }

  private void genPrologStmts() {
    // for the begining of the procedure
    BlockList blist = getFblock().getBody().getHead().getBody();
    int nlines = _prologStmts.size();

    for (int i = nlines - 1; i >= 0; i--)
      blist.insert(_prologStmts.get(i));

    // restriction: for the ENTRY statement
    if (nlines > 0 && _findEntryStmtInBlock()) {
      XMP.error("restriction: An ENTRY statement is not allowed in the " +
                "program including coarray features.");
    }
  }

  private Boolean _findEntryStmtInBlock() {
    BlockIterator bi = new topdownBlockIterator(getFblock());
    for (bi.init(); !bi.end(); bi.next()) {
      Block block = bi.getBlock();
      if (block.Opcode() == Xcode.F_ENTRY_DECL)
        return true;
    }
    return false;
  }

  private void genEpilogStmts() {
    // for RETURN statement
    BlockIterator bi = new topdownBlockIterator(getFblock());
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
    BlockList blist = getFblock().getBody().getHead().getBody();

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
                 Xcons.IntConstant(name.length()),
                 Xcons.FcharacterConstant(Xtype.FcharacterType, name, null));

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
      // a2. call "get_descptr(descPtr, baseAddr, tag, varname, varnamelen)"
      descPtrId = coarray.getDescPointerId();
      String varname = coarray.getName();
      args = Xcons.List(descPtrId, coarray.getIdent(),
                        Xcons.FvarRef(getResourceTagId()),
                        Xcons.IntConstant(coarray.isAllocatable() ? 1 : 0),
                        Xcons.IntConstant(varname.length()),
                        Xcons.FcharacterConstant(Xtype.FcharacterType,
                                                 varname, null));
      subr = env.declExternIdent(FIND_DESCPOINTER_NAME,
                                 BasicType.FexternalSubroutineType);
      if (args.hasNullArg())
        XMP.fatal("generated null argument " + FIND_DESCPOINTER_NAME +
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
    BlockIterator bi = new topdownBlockIterator(getFblock());

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

    ArrayList<XMPcoarray> coarrays = new ArrayList<XMPcoarray>();
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
                                  env, version);
    coarrayInit.run();
  }


  //-----------------------------------------------------
  //  TRANSLATION b7.  (for Ver.7, for FJ-RDMA and MPI3)
  //  generate ENTRY-block with reg-mem 
  //-----------------------------------------------------
  //
  private void readyBlockForStaticCoarrays_regmem(ArrayList<XMPcoarray> coarrays)
  {
    if (coarrays.size() == 0)
      return;

    // "ENTRY traverseInitName"
    addExtraStmt(makeStmt_ENTRY(traverseInitName));

    for (XMPcoarray coarray: coarrays) {
      // "call xmpf_coarray_regmem_static(descptr_V1, LOC(V1), ...)"
      addExtraStmt(coarray.makeStmt_regmemStatic());
      // "call xmpf_coarray_set_coshape(descptr_V1, 2, ,,,)"
      addExtraStmt(coarray.makeStmt_setCoshape(env));
    }

    // "RETURN"
    addExtraStmt(makeStmt_RETURN());
  }

  //-----------------------------------------------------
  //  TRANSLATION b7g.  (for Ver.7, for GASNet)
  //  generate ENTRY-block containing:
  //    - size count
  //    - alloc
  //-----------------------------------------------------
  //
  private void readyBlockForStaticCoarrays_alloc(ArrayList<XMPcoarray> coarrays)
  {
    if (coarrays.size() == 0)
      return;

    // "ENTRY traverseCountName"
    addExtraStmt(makeStmt_ENTRY(traverseCountName));

    for (XMPcoarray coarray: coarrays) {
      // "CALL coarray_count_size(count, elem)"
      addExtraStmt(coarray.makeStmt_countCoarrays());
    }

    // "RETURN"
    addExtraStmt(makeStmt_RETURN());

    // "ENTRY traverseInitName"
    addExtraStmt(makeStmt_ENTRY(traverseInitName));

    for (XMPcoarray coarray: coarrays) {
      // "call xmpf_coarray_alloc_static(descptr_V1, crayptr_V1, ...)"
      addExtraStmt(coarray.makeStmt_allocStatic());
      // "call xmpf_coarray_set_coshape(descptr_V1, 2, ,,,)"
      addExtraStmt(coarray.makeStmt_setCoshape(env));
    }

    // "RETURN"
    addExtraStmt(makeStmt_RETURN());
  }



  private void expandEntryBlockForStaticCoarrays()
  {
    // return if no procedure-local coarrays
    if (_extraStmts.size() == 0)
      return;

    BlockList blist = getFblock().getBody().getHead().getBody();

    // add RETURN stmt if needed
    if (blist.getTail().Opcode() != Xcode.RETURN_STATEMENT) {
      blist.add(makeStmt_RETURN());
    }

    // add all stmts in _extraStmts
    for (Xobject stmt: _extraStmts)
      blist.add(stmt);
  }


  private Xobject makeStmt_ENTRY(String name)
  {
    // add Ident 
    Ident entryNameId = env.declIdent(name,
                                      Xtype.FsubroutineType);
    Xobject entryStmt = Xcons.List(Xcode.F_ENTRY_DECL,
                                   (Xtype)null,
                                   Xcons.Symbol(Xcode.IDENT, name));
    return entryStmt;
  }


  private Xobject makeStmt_RETURN()
  {
    Xobject returnStmt = Xcons.List(Xcode.RETURN_STATEMENT);
    return returnStmt;
  }


  //-----------------------------------------------------
  //  TRANSLATION b6. (for Ver.6)
  //  generate IF-block with reg-mem call of static coarrays
  //  at the entry point of the procedure
  //-----------------------------------------------------
  //
  private void genRegmemOfStaticCoarrays(ArrayList<XMPcoarray> coarrays)
  {
    XobjList thenBlock = Xcons.List();
    for (XMPcoarray coarray: coarrays) {
      if (coarray.wasMovedFromModule())
        continue;

      Xobject stmt = coarray.makeStmt_regmemStatic();
      thenBlock.add(stmt);
    }

    for (XMPcoarray coarray: coarrays) {
      Xobject stmt = coarray.makeStmt_setCoshape(env);
      thenBlock.add(stmt);
    }

    // return if no procedure-local coarrays
    if (thenBlock.Nargs() == 0)
      return;

    // IF-condition expr.
    XMPcoarray firstCoarray = coarrays.get(0);
    Ident descPtr = firstCoarray.getDescPointerId();
    Xobject zero_8 = Xcons.IntConstant(0, Xtype.intType, "8");
    Xobject condExpr = Xcons.binaryOp(Xcode.LOG_EQ_EXPR,
                                      (Xobject)descPtr,
                                      zero_8);

    // IF construct
    XobjList ifBlock = Xcons.List(Xcode.F_IF_STATEMENT, (Xtype)null,
                                  (Xobject)null,
                                  (Xobject)condExpr,     // IF condition
                                  thenBlock,             // THEN block
                                  null);                 // ELSE block
    addPrologStmt(ifBlock);
  }


  /************************************************************************

  //  "call xmpf_coarray_alloc_static(descptr_var, crayptr_var, ...)" 
  //
  private Xobject genStmt_allocStaticCoarrays(XMPcoarray coarray)
  {
    BlockList blist = getFblock().getBody();

    String subrName = XMPcoarrayInitProcedure.ALLOC_STATIC_NAME;
    Ident subrIdent =
      blist.declLocalIdent(subrName, BasicType.FexternalSubroutineType);

    // arg2
    Ident crayPtrId = coarray.getCrayPointerId();

    // get args
    Xobject args = _getCommonArgs(crayPtrId, coarray);

    // CALL stmt
    return subrIdent.callSubroutine(args);
  }


  //  "CALL xmpf_coarray_regmem_static(descPtr_var, LOC(var), ... )"
  //
  private Xobject genStmt_regmemStaticCoarrays(XMPcoarray coarray)
  {
    BlockList blist = getFblock().getBody();

    String subrName = XMPcoarrayInitProcedure.REGMEM_STATIC_NAME;
    Ident subrIdent =
      blist.declLocalIdent(subrName, BasicType.FexternalSubroutineType);

    // arg2
    FunctionType ftype = new FunctionType(Xtype.Fint8Type, Xtype.TQ_FINTRINSIC);
    Ident locId = env.declIntrinsicIdent("loc", ftype);
    Xobject locCall = locId.Call(Xcons.List(coarray.getIdent()));

    // get args
    Xobject args = _getCommonArgs(locCall, coarray);

    // CALL stmt
    return subrIdent.callSubroutine(args);
  }


  // common arguments
  //
  private Xobject _getCommonArgs(Xobject arg2, XMPcoarray coarray)
  {
    // arg1
    Ident descPtr = coarray.getDescPointerId();
    // arg3
    Xobject count = coarray.getTotalArraySizeExpr();
    // arg4
    Xobject elem = coarray.getElementLengthExpr();
    if (elem==null)
      XMP.fatal("elem must not be null.");
    // arg5
    String varName = coarray.getName();
    Xobject varNameObj = 
      Xcons.FcharacterConstant(Xtype.FcharacterType, varName, null);
    // arg6
    Xobject nameLen = Xcons.IntConstant(varName.length());

    // args
    Xobject args = Xcons.List(descPtr,
                              arg2,
                              count,
                              elem,
                              varNameObj,
                              nameLen);
    if (args.hasNullArg())
      XMP.fatal("INTERNAL: contains null argument");

    return args;
  }
  ************************************************************/


  //-----------------------------------------------------
  //  TRANSLATION j, m, n
  //  convert allocate/deallocate stmts for allocated coarrays
  //-----------------------------------------------------
  //
  private void convAllocateStmts(ArrayList<XMPcoarray> coarrays) {
    BasicBlockIterator bbi =
      new BasicBlockIterator(getFblock());    // see XMPrewriteExpr iter3 loop
    
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
      new BasicBlockIterator(getFblock());    // see XMPrewriteExpr iter3 loop
    
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
    Ident id = env.findVarIdent(obj.getName(), getFblock());
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
      descId = Xcons.IntConstant(0, Xtype.Fint8Type, "8");    // descId = 0_8
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
    String subrName = COARRAY_ALLOC_NAME;
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
    String subrName = COARRAY_DEALLOC_NAME;
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
                                 ).cfold(getFblock());
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

    return lbound.cfold(getFblock());
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

    return ubound.cfold(getFblock());
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

    return extent.cfold(getFblock());
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
  //  TRANSLATION f6.
  //  add SAVE attributes to declarations of coarrays
  //-----------------------------------------------------
  //
  private void addSaveAttr(ArrayList<XMPcoarray> coarrays) {
    if (def.isFmoduleDef())  // module
      return;
    Xtype ft = def.getFuncType();
    if (ft != null && ft.isFprogram())  // main program
      return;

    for (XMPcoarray coarray: coarrays) {
      coarray.setSaveAttr();
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
  //  - replace num_images() with NUM_IMAGES_NAME()
  //  - replace this_image() with THIS_IMAGE_NAME()
  //  - replace this_image(V, ...) with THIS_IMAGE_NAME(descptr_V, ...)
  //  - replace image_index(V, ...) with IMAGE_INDEX_NAME(descptr_V, ...)
  //  - replace co_broadcast(V, ...) with CO_BROADCAST_NAME(V, ...)
  //  - replace co_sum/min/max(V, ...) with CO_SUM/MIN/MAX_NAME(V, ...)
  //  - replace lcobound(V, ...) with COBOUND_NAME(descptr_V, ..., 0, corank)
  //  - replace ucobound(V, ...) with COBOUND_NAME(descptr_V, ..., 1, corank)
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

      /* replace Fortran90 intrinsic
       */
      if (fname.equalsIgnoreCase("allocated"))
        _replaceAllocatedWithAssociated(xobj, coarrays);

      /* replace Coarray intrinsics
       */
      else if (fname.equalsIgnoreCase("num_images"))
        _replaceNumImages(xobj, coarrays);
      else if (fname.equalsIgnoreCase("this_image"))
        _replaceThisImage(xobj, coarrays);
      else if (fname.equalsIgnoreCase("image_index"))
        _replaceImageIndex(xobj, coarrays);
      else if (fname.equalsIgnoreCase("lcobound"))
        _replaceCobound(xobj, coarrays, 0);
      else if (fname.equalsIgnoreCase("ucobound"))
        _replaceCobound(xobj, coarrays, 1);
      else if (fname.equalsIgnoreCase("co_broadcast"))
        _replaceCoReduction(xobj, fname, CO_BROADCAST_NAME);
      else if (fname.equalsIgnoreCase("co_sum"))
        _replaceCoReduction(xobj, fname, CO_SUM_NAME);
      else if (fname.equalsIgnoreCase("co_max"))
        _replaceCoReduction(xobj, fname, CO_MAX_NAME);
      else if (fname.equalsIgnoreCase("co_min"))
        _replaceCoReduction(xobj, fname, CO_MIN_NAME);
    }

    /* remove ident of Coarray intrinsics
     */
    /////// Sealed. I don't know why this does not work well. //////
    //XobjArgs idArgs = def.getFuncIdList().getArgs();
    //_removeCoarrayIntrinsicIdents(null, idArgs);
  }


  // not used currently.
  private void _removeCoarrayIntrinsicIdents(XobjArgs prevArgs, XobjArgs args) {
    // termination
    if (args == null)
      return;

    XobjArgs nextArgs = args.nextArgs();

    Ident ident = (Ident)args.getArg();
    String name = ident.getName();
    if (_coarrayIntrinsicList.contains(name)) {
      // Found me a coarray intrinsic name. Remove me.
      prevArgs.setNext(nextArgs);
      args = prevArgs;
    }

    _removeCoarrayIntrinsicIdents(args, nextArgs);
  }



  /* replace "co_broadcast/sum/max/min(source, ...)"
   *  with "xmpf_co_broadcast/sum/max/min<dim>d(source, ...)"
   *  where <dim> is the rank of argument source
   *
   *  To avoid bug478, subroutine name co_xxx is converted into the 
   *  intermedeate name co_xxx<dim>d before the final conversion by 
   *  the compiler into co_xxx<dim>d_<typekind>.
   *  --> challenge!
   */
  private void _replaceCoReduction(Xobject xobj, String fname, String genelicName) {
    XobjList actualArgs = (XobjList)xobj.getArg(1);
    int nargs = (actualArgs == null) ? 0 : actualArgs.Nargs();

    if (nargs != 2) {
      XMP.error("Too few or too many arguments are found in " + fname);
      return;
    }

    // get the first argument 'source'
    Xobject arg1 = actualArgs.getArgWithKeyword("source", 0);
    if (arg1 == null) {
      XMP.error("Argument \'source\' was not found in " + fname);
      return;
    }

    int rank = arg1.getFrank(getFblock());
    XobjString newFname = Xcons.Symbol(Xcode.IDENT, genelicName);
    xobj.setArg(0, newFname);
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
  private void _replaceNumImages(Xobject xobj, ArrayList<XMPcoarray> coarrays) {
    Xobject fname = xobj.getArg(0);
    XobjList actualArgs = (XobjList)xobj.getArg(1);
    int nargs = (actualArgs == null) ? 0 : actualArgs.Nargs();

    if (nargs > 0) {
      XMP.error("No arguments are expected in num_images().");
      return;
    }

    // replace function name 'this_image'
    XobjString newFname = Xcons.Symbol(Xcode.IDENT, NUM_IMAGES_NAME);
    xobj.setArg(0, newFname);
  }

  /* replace intrinsic this_image
   */
  private void _replaceThisImage(Xobject xobj, ArrayList<XMPcoarray> coarrays) {
    Xobject fname = xobj.getArg(0);
    XobjList actualArgs = (XobjList)xobj.getArg(1);
    int nargs = (actualArgs == null) ? 0 : actualArgs.Nargs();

    if (nargs > 2) {
      XMP.error("Too many arguments was found in this_image().");
      return;
    }

    // replace function name 'this_image'
    XobjString newFname = Xcons.Symbol(Xcode.IDENT, THIS_IMAGE_NAME);
    xobj.setArg(0, newFname);

    if (nargs == 0)
      return;

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

    // replace argument COARRAY
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
    
    if (!_isCoarrayReferred() &&
        !_isCoarrayIntrinsicUsed() &&
        !_isCoarrayStatementUsed()) {
      /* any coarray features are not used */
      return;
    }

    /* check whether xmp_coarray.h is included */
    Ident id = def.findIdent("xmpf_image_index");
    if (id == null) {
      /* xmpf_lib.h seems not included. */
      XMP.error("current restriction: " + 
                "\'xmp_coarray.h\' must be included to use any coarray features " +
                "in the procodrure/module: " + getName());
    }
  }

  private boolean _isCoarrayReferred() {
    if (localCoarrays.isEmpty())
      return false;
    return true;
  }


  private boolean _isCoarrayIntrinsicUsed() {
    XobjList identList = def.getDef().getIdentList();
    for (Xobject x: identList) {
      Ident id = (Ident)x;
      if (_coarrayIntrinsicList.contains(id.getName()))
        return true;
    }
    return false;
  }


  private boolean _isCoarrayStatementUsed() {
    XobjList identList = def.getDef().getIdentList();
    for (Xobject x: identList) {
      Ident id = (Ident)x;
      if (_coarrayStmtKeywordList.contains(id.getName()))
        return true;
    }
    return false;
  }


  //------------------------------
  //  inquire
  //------------------------------
  private boolean isMainProgram() {
    Xtype ft = def.getFuncType();
    return (ft != null && ft.isFprogram());
  }

  private boolean isModule() {
    return  def.isFmoduleDef();
  }

  private String getName() {
    return name;
  }


  //------------------------------
  //  semantic analysis:
  //    IMAGE directive
  //------------------------------
  public static void analyzeImageDirective(Xobject imagePragma,
                                           XMPenv env, PragmaBlock pb) {

    String nodesName = imagePragma.getArg(0).getString();

    /*----
     *  error check for the corresponding statement
     */
    Block nextBlock = pb.getNext();
    while (_isSkippableBlockForImageDir(nextBlock))
      nextBlock = nextBlock.getNext();

    if (nextBlock == null || !_isTargetStmtOfImageDir(nextBlock)) {
      XMP.errorAt(pb, "Illegal use of IMAGE directive");
      //XMP.warning("Illegal use of IMAGE directive -- ignored");
      return;
    }

    return;
  }


  private static Boolean _isSkippableBlockForImageDir(Block block) {
    // All empty and comment lines seem to be deleted already...
    return false;
  }


  /* Expected nextStmt is a call statement made by F-Front from
   *    - sync all statement, or
   *    - sync images statement, or
   *    - call statement for co_broadcast, or
   *    - call statement for co_sum or co_max or co_min, or
   *    - critical statement
   */
  private static boolean _isTargetStmtOfImageDir(Block nextBlock) {
    String[] targetNamePrefixes = {
      "xmpf_sync_all",
      "xmpf_sync_images",
      "xmpf_co_",
      "xmpf_critical"};
    //    List targetNamePrefixList = Arrays.asList(targetNamePrefixes);

    BasicBlock bblock = nextBlock.getBasicBlock();
    if (bblock == null)
      return false;

    Statement nextStmt = bblock.getHead();
    if (nextStmt == null)
      return false;

    Xobject xobj1 = nextStmt.getExpr();
    Xcode xcode1 = xobj1.Opcode();
    if (xcode1 != Xcode.EXPR_STATEMENT)
      return false;

    Xobject xobj2 = xobj1.getArg(0);
    Xcode xcode2 = xobj2.Opcode();
    if (xcode2 != Xcode.FUNCTION_CALL)
      return false;

    String fname = xobj2.getArg(0).getName();

    for (String prefix: targetNamePrefixes) {
      if (fname.startsWith(prefix)) {
        if (fname.equals(AUTO_SYNCALL_NAME))    // exception
          return false;
        return true;
      }
    }

    return false;
  }


  //------------------------------
  //  translation:
  //    IMAGE directive
  //------------------------------
  public static Block translateImageDirective(PragmaBlock pb,
                                              XMPinfo info) {
    Block b = Bcons.emptyBlock();
    BasicBlock bb = b.getBasicBlock();

    String nodesName = pb.getClauses().getArg(0).getName();
    Xobject stmt = XMPcoarray.makeStmt_setImageNodes(nodesName, info.env);

    bb.add(stmt);
    return b;
  }


  //------------------------------
  //  tool
  //------------------------------
  public String toString() {
    String s = 
      "\n  int version = " +  version +
      "\n  Boolean DEBUG = " +  DEBUG +
      "\n  XMPenv env = " +  env +
      "\n  String name = " +  name +
      "\n  XobjectDef def = " +  def +
      "\n  FuncDefBlock funcDef = " +
      "\n  ArrayList<XMPcoarray> localCoarrays = " +  _coarrayNames(localCoarrays) +
      "\n    ArrayList<XMPcoarray> staticLocalCoarrays = " +  _coarrayNames(staticLocalCoarrays) +
      "\n    ArrayList<XMPcoarray> allocatableLocalCoarrays = " +  _coarrayNames(allocatableLocalCoarrays) +
      "\n    ArrayList<XMPcoarray> staticDummyCoarrays = " +  _coarrayNames(staticDummyCoarrays) +
      "\n    ArrayList<XMPcoarray> allocatableDummyCoarrays = " +  _coarrayNames(allocatableDummyCoarrays) +
      "\n  ArrayList<XMPcoarray> useAssociatedCoarrays = " +  _coarrayNames(useAssociatedCoarrays) +
      "\n    ArrayList<XMPcoarray> staticAssociatedCoarrays = " +  _coarrayNames(staticAssociatedCoarrays) +
      "\n    ArrayList<XMPcoarray> allocatableAssociatedCoarrays = " +  _coarrayNames(allocatableAssociatedCoarrays) +
      "\n  String hostModuleName = " +  hostModuleName +
      "\n  String hostProcedureName = " +  hostProcedureName +
      "\n  XMPtransCoarrayRun hostModuleRun = " +  hostModuleRun +
      "\n  XMPtransCoarrayRun hostProcedureRun = " +  hostProcedureRun +
      "\n  ArrayList<XMPcoarray> visibleCoarrays = " +  _coarrayNames(visibleCoarrays) +
      "\n  String traverseCountName = " +  traverseCountName +
      "\n  String traverseInitName = " +  traverseInitName +
      "\n  Ident _resourceTagId = " +  _resourceTagId +
      "\n  ArrayList<Xobject> _prologStmts = " +  _prologStmts +
      "\n  ArrayList<Xobject> _epilogStmts = " +  _epilogStmts +
      "\n  ArrayList<Xobject> _extraStmts = " +  _extraStmts +
      "\n  Boolean _autoDealloc = " +  _autoDealloc;
    return s;
  }

  private String _coarrayNames(ArrayList<XMPcoarray> coarrays) {
    String s = "[";
    String sep = "";
    for (XMPcoarray coarray: coarrays) {
      s += sep + coarray.getName();
      sep = ",";
    }
    return s + "]";
  }
        
  private Ident getResourceTagId() {
    if (_resourceTagId == null) {
      BlockList blist = getFblock().getBody();
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

  // add at the top of _extraStmts
  private void addExtraStmt(Xobject stmt) {
    _extraStmts.add(stmt);
  }

  // add at the top of _extraStmts
  private void insertExtraStmt(Xobject stmt) {
    _extraStmts.add(0, stmt);
  }


  // for automatic deallocation at the end of the program
  private Boolean get_autoDealloc() {
    return _autoDealloc;
  }
  private void set_autoDealloc(Boolean sw) {
    _autoDealloc = sw;
  }


  private FunctionBlock getFblock() {
    return funcDef.getBlock();
  }
}

