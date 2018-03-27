package exc.xmpF;

import exc.object.*;
import exc.block.*;
import java.util.*;

/*
 * create an initialization subroutine 
 *  corresponding to the procedure
 */
public class XMPcoarrayInitProcedure {

  private Boolean DEBUG = false;          // switch the value on gdb !!

  /* for all procedures */
  private ArrayList<String> procTexts;
  private ArrayList<XMPcoarray> staticCoarrays;
  private XMPenv env;

  /*  Versions 3, 4 & 6 are available currently.
   *    3: bootup time allocation and registration using cray pointer (default)
   *    4: For static coarrays on FJ-RDMA and MPI3,
   *       static allocation/bootup time registration w/o cray pointer using common
   *    6: For static coarrays on FJ-RDMA and MPI3,
   *       static allocation/first time registration w/o cray pointer w/o common
   */
  private int version;         // defined by the constructor

  /* for each procedure */
  private String sizeProcName, initProcName;  // names of procedures to generate

  /* for all variables of a procedure */
  private ArrayList<String> varNames1, varNames2;
  private ArrayList<String> callSizeStmts, callInitStmts;

  //------------------------------
  //  constructor/finalizer
  //------------------------------
  public XMPcoarrayInitProcedure(ArrayList<XMPcoarray> staticCoarrays,
                                 String sizeProcName, String initProcName,
                                 XMPenv env, int version) {
    _init_forFile();

    this.staticCoarrays = staticCoarrays;
    this.sizeProcName = sizeProcName;
    this.initProcName = initProcName;
    this.env = env;
    this.version = version;

    // assertion
    if (version != 3 && version != 4 && version != 6 && version != 7) {
      XMP.fatal("INTERNAL: extinct or unsupported version (" + version +
                ") found in XMPcoarrayInitProcedure constructor");
    }

    varNames1 = new ArrayList<String>();
    varNames2 = new ArrayList<String>();
    callInitStmts = new ArrayList<String>();
    callSizeStmts = new ArrayList<String>();
  }


  //------------------------------
  //  for each procedure
  //------------------------------

  /**
    an example of source program:
    --------------------------------------------
      subroutine EX1  or  module EX1
        use M1   !! contains "real :: V1(10,20)[4,*]"  ! use-associated static coarray
        complex(8), save :: V2[0:*]                    ! local static coarray
        !! other coarrays, i.e., allocatable coarrays and dummy coarrays are handled by
        !! transDeclPart_allocatableLocal, transDeclPart_staticDummy, and
        !! transDeclPart_allocatableDummy in XMPtransCoarrayRun.java.
        ...
      end subroutine  or  end module
    --------------------------------------------

    converted program and generated subroutines
    Case: useMalloc (Ver.3 and 7g):
    --------------------------------------------
      subroutine xmpf_traverse_countcoarrays_EX1
        call xmpf_coarray_count_size(1, 16)
      end subroutine

      subroutine xmpf_traverse_initcoarrays_EX1
        integer(8) :: DP_V2
        integer(8) :: CP_V2
        common /xmpf_DP_EX1/ DP_V2
        common /xmpf_CP_EX1/ CP_V2

        call xmpf_coarray_alloc_static(DP_V2, CP_V2, 1, 16, "V2", 2)
        call xmpf_coarray_set_coshape(DP_V2, 1, 0)
      end subroutine
    --------------------------------------------
      xmpf_DP_EX1 : name of a common block for procedure EX1
      DP_V2       : pointer to descriptor of coarray V2
      xmpf_CP_EX1 : name of a common block for procedure EX1
      CP_V2       : cray poiter to coarray V2

    Case: useRegMem (Ver.4, 6 and 7 for FJRDMA and MPI3):
    --------------------------------------------
      // no subroutine xmpf_traverse_countcoarrays_EX1

      subroutine xmpf_traverse_initcoarrays_EX1
        common /xmpf_DP_EX1/ DP_V2
        common /xmpf_CO_EX1/ V2

        call xmpf_coarray_regmem_static(DP_V2, LOC(V2), 1, 16, "V2", 2)
        call xmpf_coarray_set_coshape(DP_V2, 1, 0)
      end subroutine
    --------------------------------------------

    Ver.6 shoud be the same as Ver.4.  The following description should 
    be wrong.
    //Ver.6 for FJRDMA and MPI3, only for modules EX1:
    //--------------------------------------------
    //// no subroutine xmpf_traverse_countcoarrays_EX1
    //
    // subroutine xmpf_traverse_initcoarrays_EX1
    //  use EX1
    //
    //  call xmpf_coarray_regmem_static(DP_V2, LOC(V2), 1, 16, "V2", 2)
    //  call xmpf_coarray_set_coshape(DP_V2, 1, 0)
    // end subroutine
    --------------------------------------------
  */

  public void run() {
    buildSubroutine_countcoarrays();
    buildSubroutine_initcoarrays();
  }


  /*   build subroutines as Xobject and
   *   link them at the tail of XMPenv
   */

  private void buildSubroutine_countcoarrays() {
    BlockList body = Bcons.emptyBody();         // new body of the building procedure
    Boolean isEmpty = true;

    for (XMPcoarray coarray: staticCoarrays) {
      if (coarray.usesMalloc()) {
        // "CALL coarray_count_size(count, elem)"
        Xobject callStmt = coarray.makeStmt_countCoarrays(body);
        body.add(callStmt);
        isEmpty = false;
      }
    }

    if (isEmpty)
      return;

    // construct a new procedure
    Xobject decls = Xcons.List();
    Ident procedure = env.declExternIdent(sizeProcName, Xtype.FsubroutineType);
    XobjectDef def2 = XobjectDef.Func(procedure, body.getIdentList(),
                                      decls, body.toXobject());

    // link the new procedure as my sister
    env.getEnv().add(def2);
  }


  private void buildSubroutine_initcoarrays() {
    BlockList body = Bcons.emptyBody();         // new body of the building procedure
    Xobject decls = Xcons.List();

    for (XMPcoarray coarray: staticCoarrays)
      build_initEachCoarray(coarray, body, decls);

    // construct a new procedure
    Ident procedure = env.declExternIdent(initProcName, Xtype.FsubroutineType);
    XobjectDef def2 = XobjectDef.Func(procedure, body.getIdentList(),
                                      decls, body.toXobject());
    // link the new procedure as my sister
    env.getEnv().add(def2);


    FuncDefBlock funcDef1 = env.getCurrentDef();

    FuncDefBlock funcDef2 = new FuncDefBlock(def2);
    FunctionBlock fblock2 = funcDef2.getBlock();
    BlockList blist2 = fblock2.getBody().getHead().getBody();

    env.setCurrentDef(funcDef2);

    for (XMPcoarray coarray: staticCoarrays) {
      /**************************
      Xobject setCoshape = coarray.makeStmt_setCoshape(env);
      blist2.add(setCoshape);
      ******************************/
      coarray.addStmts_setCoshapeAndName(blist2, env);
    }

    funcDef2.finalizeBlock();

    env.setCurrentDef(funcDef1);
  }



  private void build_initEachCoarray(XMPcoarray coarray,
                                     BlockList body, Xobject decls) {
    Xobject elem = coarray.getElementLengthExpr_atmost();
    if (elem==null)
      XMP.fatal("INTERNAL: elem must not be null.");
    int count = coarray.getTotalArraySize();

    /*-------------------------------*\
     * specification part
    \*-------------------------------*/

    // verseions all
    //   "common /xmpf_descptr_foo/ descPtr_var"
    String descPtrName = coarray.getDescPointerName();
    Ident descPtrId = null;
    descPtrId = body.declLocalIdent(descPtrName,
                                    Xtype.Farray(BasicType.Fint8Type),
                                    StorageClass.FCOMMON,
                                    null);
    Xobject commonStmt1 =
      Xcons.List(Xcode.F_COMMON_DECL,
                 Xcons.List(Xcode.F_VAR_LIST,
                            Xcons.Symbol(Xcode.IDENT,
                                         coarray.getDescCommonName()),
                            Xcons.List(Xcons.FvarRef(descPtrId))));
    decls.add(commonStmt1);

    // case useMalloc
    //    "common /xmpf_crayptr_foo/ crayPtr_var"   
    // case useRegMem
    //    "TYPE var(N)"  !! without SAVE attr.
    //    "common /xmpf_coarray_foo/ var"           
    Ident ident1 = coarray.getIdent();
    Xtype type1 = ident1.Type();
    String commonName;
    XobjList commonMember;
    if (coarray.usesMalloc()) {
      // Ver.3 or STRUCT type: generate common block for cray-pointers
      String crayPtrName = coarray.getCrayPointerName();
      Ident ident2 = body.declLocalIdent(crayPtrName,
                                         Xtype.Farray(BasicType.Fint8Type),
                                         StorageClass.FCOMMON,  //or StorageClass.FLOCAL
                                         null);
      commonName = coarray.getCrayCommonName();
      commonMember = Xcons.List(Xcons.FvarRef(ident2));
    }
    else {
      // Ver.4(or6) and not STRUCT type : generate common block for coarrays themselves
      Ident ident2 = body.declLocalIdent(ident1.getName(),
                                         ident1.Type().copy());
      ident2.Type().removeCodimensions();       // remove codimensions
      _resetSaveAttrInType(ident2.Type());         // reset SAVE attribute
      ident2.setStorageClass(StorageClass.FCOMMON);   // reset SAVE attribute again
      commonName = coarray.getCoarrayCommonName();
      commonMember = Xcons.List(Xcons.FvarRef(ident2));
    }

    Xobject commonStmt2 =
      Xcons.List(Xcode.F_COMMON_DECL,
                 Xcons.List(Xcode.F_VAR_LIST,
                            Xcons.Symbol(Xcode.IDENT, commonName),
                            commonMember));
    decls.add(commonStmt2);

    /*-------------------------------*\
     * execution part
    \*-------------------------------*/
    // case useRegMem
    //   "CALL coarray_regmem_static(descPtr_var, LOC(var), ... )"
    // case useMalloc
    //   "CALL coarray_alloc_static(descPtr_var, crayPtr_var, ... )"
    if (coarray.usesMalloc()) {
      Xobject subrCall = coarray.makeStmt_allocStatic(body);
      body.add(subrCall);
    } else {
      Xobject subrCall = coarray.makeStmt_regmemStatic(body);
      body.add(subrCall);
    }
  }


  /*
   * suspended
   *  This method could be better but not completed...
   */
  public void genInitRoutine__NOT_USED__(ArrayList<XMPcoarray> staticCoarrays,
                                         String newProcName, String commonName,
                                         FunctionBlock fblock, XMPenv env) {
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



  //------------------------------
  //  parts
  //------------------------------
  private void _init_forFile() {
    procTexts = new ArrayList<String>();
  }

  private void _resetSaveAttrInType(Xtype type) {
    type.setIsFsave(false);

    if (type.copied != null) 
      _resetSaveAttrInType(type.copied);

    if (type.isArray() || type.isFarray())
      _resetSaveAttrInType(type.getRef());
  }
}


