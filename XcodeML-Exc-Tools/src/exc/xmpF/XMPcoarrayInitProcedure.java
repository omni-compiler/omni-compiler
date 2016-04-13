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
  private int version;    // defined by the constructor

  /* for each procedure */
  private String sizeProcName, initProcName;  // names of procedures to generate
  private String commonName1 = null;    // common block name for descptr
  private String commonName2 = null;    // common block name for crayptr (Ver.3)
                                        //                   or coarrays (Ver.4)

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
    Ver.3:
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

    Ver.4 for FJRDMA and MPI3:
    --------------------------------------------
      // no subroutine xmpf_traverse_countcoarrays_EX1

      subroutine xmpf_traverse_initcoarrays_EX1
        common /xmpf_DP_EX1/ DP_V2
        common /xmpf_CO_EX1/ V2

        call xmpf_coarray_regmem_static(DP_V2, LOC(V2), 1, 16, "V2", 2)
        call xmpf_coarray_set_coshape(DP_V2, 1, 0)
      end subroutine
    --------------------------------------------

    Ver.6 for FJRDMA and MPI3, only for modules EX1:
    --------------------------------------------
      // no subroutine xmpf_traverse_countcoarrays_EX1

      subroutine xmpf_traverse_initcoarrays_EX1
        use EX1

        call xmpf_coarray_regmem_static(DP_V2, LOC(V2), 1, 16, "V2", 2)
        call xmpf_coarray_set_coshape(DP_V2, 1, 0)
      end subroutine
    --------------------------------------------
  */

  public void run() {

    /* generate the two subroutines in the same file
     */
    switch(version) {
    case 1:   // generate as Fortran program text
      XMP.fatal("INTERNAL: extinct version #" + version +
                "specified in XMPcoarrayInitProcedure");
      /*------
      fillinSizeProcText();
      fillinInitProcText();

      for (String text: procTexts)
        env.addTailText(text);
      -------*/
      break;

    case 2:   // build and link it at the tail of XMPenv
      XMP.fatal("INTERNAL: extinct version #" + version +
                "specified in XMPcoarrayInitProcedure");
      /*--------
      buildSubroutine_countcoarrays();
      buildSubroutine_initcoarrays();
      --------*/
      break;

    case 3:   // similar to case 2, with changing descr-ID of serno to pointer
      buildSubroutine_countcoarrays();
      buildSubroutine_initcoarrays();
      break;

    case 4:   // temporary version for FJ-RDMA and MPI3
      //buildSubroutine_countcoarrays();   // unnecessary?
      buildSubroutine_initcoarrays();
      break;

    case 6:   // temporary version for FJ-RDMA and MPI3
      // case: module
      buildSubroutine_initcoarrays();
      break;

    default:
      XMP.fatal("INTERNAL: unexpected version number (" + version +
                ") specified in XMPcoarrayInitProcedure");
      break;
    }
  }


  /*
   *  Version 3:
   *    buildSubroutine_countcoarrays()
   *    buildSubroutine_initcoarrays()
   *  Version 4:
   *    buildSubroutine_initcoarrays()
   *  Version 6 (module):
   *    buildSubroutine_initcoarrays()
   *
   *   build subroutines as Xobject and
   *   link them at the tail of XMPenv
   */

  private void buildSubroutine_countcoarrays() {
    BlockList body = Bcons.emptyBody();         // new body of the building procedure

    if (version != 3) {
      XMP.fatal("unexpected version " + version + 
                " in buildSubroutine_countcoarrays");
      return;
    }

    for (XMPcoarray coarray: staticCoarrays) {
      // "CALL coarray_count_size(count, elem)"
      Xobject callStmt = coarray.makeStmt_countCoarrays(body);

      body.add(callStmt);
    }

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
      Xobject setCoshape = coarray.makeStmt_setCoshape(env);
      blist2.add(setCoshape);
    }

    funcDef2.Finalize();

    env.setCurrentDef(funcDef1);
  }


  private void set_commonName1(XMPcoarray coarray) {
    String name = coarray.getDescCommonName();

    if (commonName1 == null)
      commonName1 = name;
    else if (!commonName1.equals(name))
      XMP.fatal("INTERNAL: inconsistent descptr common block names " +
                commonName1 + " and " + name);
  }

  private void set_commonName2(XMPcoarray coarray) {
    String name;
    if (version >= 4)
      name = coarray.getCoarrayCommonName();
    else
      name = coarray.getCrayCommonName();
    
    if (commonName2 == null)
      commonName2 = name;
    else if (!commonName2.equals(name)) 
      XMP.fatal("INTERNAL: inconsistent second common block names " +
                commonName2 + " and " + name);
  }


  private void build_initEachCoarray(XMPcoarray coarray,
                                     BlockList body, Xobject decls) {
    Xobject elem = coarray.getElementLengthExpr();
    if (elem==null)
      XMP.fatal("elem must not be null.");
    int count = coarray.getTotalArraySize();

    /*-------------------------------*\
     * specification part
    \*-------------------------------*/

    // verseions all
    //   "common /xmpf_descptr_foo/ descPtr_var"
    String descPtrName = coarray.getDescPointerName();
    Ident descPtrId = null;
    if (version >= 3) {
      set_commonName1(coarray);
      descPtrId = body.declLocalIdent(descPtrName,
                                      Xtype.Farray(BasicType.Fint8Type),
                                      StorageClass.FCOMMON,
                                      null);
      Xobject commonStmt1 =
        Xcons.List(Xcode.F_COMMON_DECL,
                   Xcons.List(Xcode.F_VAR_LIST,
                              Xcons.Symbol(Xcode.IDENT, commonName1),
                              Xcons.List(Xcons.FvarRef(descPtrId))));
      decls.add(commonStmt1);
    } else {
      XMP.fatal("unexpected version");
    }

    // version 3
    //    "common /xmpf_crayptr_foo/ crayPtr_var"   
    // versions 4 & 6
    //    "TYPE var(N)"  !! without SAVE attr.
    //    "common /xmpf_coarray_foo/ var"           
    set_commonName2(coarray);
    Ident ident2;
    if (version >= 4) {
      // Version 4 & 6: generate common block for coarrays themselves
      //coarray.resetSaveAttr();   // This seems not correct due to the side effect.
      Xtype type1 = coarray.getIdent().Type();
      Xtype type2;
      switch (type1.getKind()) {
      case Xtype.BASIC:
        type2 = type1.copy();
        type2.removeCodimensions();
        _resetSaveAttrInType(type2);      // reset SAVE attribute
        break;
      case Xtype.F_ARRAY:
        Xobject[] sizeExprs2 = new Xobject[1];
        sizeExprs2[0] = Xcons.FindexRange(Xcons.IntConstant(count));
        type2 = new FarrayType(coarray.getName(),
                               type1.getRef().copy(),
                               type1.getTypeQualFlags(),
                               sizeExprs2);
        _resetSaveAttrInType(type2);      // reset SAVE attribute
        break;
      default:
        XMP.fatal("unexpected kind of Xtype of coarray " + coarray.getName());
        return;   // to avoid warning message from javac
      }
      ident2 = body.declLocalIdent(coarray.getName(), type2);
      ident2.setStorageClass(StorageClass.FCOMMON);   // reset SAVE attribute again

    } else {  
      // Version 3: generate common block for cray-pointers
      String crayPtrName = coarray.getCrayPointerName();
      ident2 = body.declLocalIdent(crayPtrName,
                                   Xtype.Farray(BasicType.Fint8Type),
                                   StorageClass.FCOMMON,  //or StorageClass.FLOCAL
                                   null);
    }

    Xobject commonStmt2 =
      Xcons.List(Xcode.F_COMMON_DECL,
                 Xcons.List(Xcode.F_VAR_LIST,
                            Xcons.Symbol(Xcode.IDENT, commonName2),
                            Xcons.List(Xcons.FvarRef(ident2))));
    decls.add(commonStmt2);

    /*-------------------------------*\
     * execution part
    \*-------------------------------*/
    // versions 4 & 6
    //   "CALL coarray_regmem_static(descPtr_var, LOC(var), ... )"
    // version 3
    //   "CALL coarray_alloc_static(descPtr_var, crayPtr_var, ... )"
    if (version == 4 || version == 6) {
      Xobject subrCall = coarray.makeStmt_regmemStatic(body);
      body.add(subrCall);
    } else if (version == 3) {
      Xobject subrCall = coarray.makeStmt_allocStatic(body);
      body.add(subrCall);
    } else {
      XMP.fatal("INTERNAL: unexpected version number");
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


