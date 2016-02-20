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

  final static String COUNT_SIZE_NAME = "xmpf_coarray_count_size";
  final static String ALLOC_STATIC_NAME = "xmpf_coarray_alloc_static";
  final static String REGMEM_STATIC_NAME = "xmpf_coarray_regmem_static";

  private Boolean DEBUG = false;          // switch the value on gdb !!


  /* for all procedures */
  private ArrayList<String> procTexts;
  private ArrayList<XMPcoarray> staticCoarrays;
  private XMPenv env;

  /*  Versions 3 and 4 are available currently.
   *    3: stable version
   *    4: challenging optimization only if using FJRDMA
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

    Ver.4:
    For GASNet, Ver.4 is equivalent to Ver.3
    For FJRDMA:
    --------------------------------------------
      // no subroutine xmpf_traverse_countcoarrays_EX1

      subroutine xmpf_traverse_initcoarrays_EX1
        common /xmpf_DP_EX1/ DP_V2
        common /xmpf_CO_EX1/ V2

        call xmpf_coarray_regmem_static(DP_V2, LOC(V2), 1, 16, "V2", 2)
        call xmpf_coarray_set_coshape(DP_V2, 1, 0)
      end subroutine
    --------------------------------------------
      xmpf_DP_EX1 : name of a common block for procedure EX1
      DP_V2       : pointer to descriptor of coarray V2
      xmpf_CO_EX1 : name of a common block for procedure EX1
      V2          : name of a coarray
  */

  public void run() {
    for (XMPcoarray coarray: staticCoarrays) {
      String descPtrName = coarray.getDescPointerName();
      String crayPtrName = coarray.getCrayPointerName();

      /**************
      int elem = coarray.getElementLength();
      int count = coarray.getTotalArraySize();
      addForVarText(descPtrName, crayPtrName, count, elem);
      *******************/
    }

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

    case 4:   // new version to avoid using cray pointers
      buildSubroutine_countcoarrays();
      buildSubroutine_initcoarrays();
      break;

    default:
      XMP.fatal("INTERNAL: illegal version #" + version +
                "specified in XMPcoarrayInitProcedure");
      break;
    }
  }


  /*
   *  Version 3:
   *    buildSubroutine_countcoarrays()
   *    buildSubroutine_initcoarrays()
   *  Version 4:
   *    buildSubroutine_countcoarrays()
   *    buildSubroutine_initcoarrays()
   *
   *   build subroutines as Xobject and
   *   link them at the tail of XMPenv
   */

  private void buildSubroutine_countcoarrays() {
    BlockList body = Bcons.emptyBody();         // new body of the building procedure
    Xobject decls = Xcons.List();

    for (XMPcoarray coarray: staticCoarrays) {
      // "CALL coarray_count_size(count, elem)"
      Xobject elem = coarray.getElementLengthExpr();

      if (elem == null) {
        XMP.error("current restriction: " + 
                  "could not find the element length of: "+coarray.getName());
      }

      int count = coarray.getTotalArraySize();
      Xobject args = Xcons.List(Xcons.IntConstant(count), elem);
      Ident subr = body.declLocalIdent(COUNT_SIZE_NAME,
                                       BasicType.FexternalSubroutineType);
      body.add(subr.callSubroutine(args));
    }

    // construct a new procedure
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

      // no longer needed. coarray_alloc_static includes setting of the name.
      //Xobject setVarName = coarray.makeStmt_setVarName(env);
      //blist2.add(setVarName);
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
    if (version == 4)
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

    // "common /xmpf_descptr_foo/ descPtr_var"
    set_commonName1(coarray);
    String descPtrName = coarray.getDescPointerName();
    Ident descPtrId =
      body.declLocalIdent(descPtrName,
                          Xtype.Farray(BasicType.Fint8Type),
                          StorageClass.FCOMMON,   //or StorageClass.FLOCAL
                          null);
    Xobject commonStmt1 =
      Xcons.List(Xcode.F_COMMON_DECL,
                 Xcons.List(Xcode.F_VAR_LIST,
                            Xcons.Symbol(Xcode.IDENT, commonName1),
                            Xcons.List(Xcons.FvarRef(descPtrId))));
    decls.add(commonStmt1);

    // (case: Ver.3)
    //    "common /xmpf_crayptr_foo/ crayPtr_var"   
    // (case: Ver.4)
    //    TYPE var(N)
    //    "common /xmpf_coarray_foo/ var"           
    set_commonName2(coarray);
    Ident ident2;
    if (version == 4) {
      Xtype type1 = coarray.getIdent().Type();
      Xtype type2;
      switch (type1.getKind()) {
      case Xtype.BASIC:
        type2 = type1.copy();
        type2.removeCodimensions();
        break;
      case Xtype.F_ARRAY:
        Xobject[] sizeExprs2 = new Xobject[1];
        sizeExprs2[0] = Xcons.FindexRange(Xcons.IntConstant(count));
        type2 = new FarrayType(coarray.getName(),
                               type1.getRef().copy(),
                               type1.getTypeQualFlags(),
                               sizeExprs2);
        break;
      default:
        XMP.fatal("unexpected kind of Xtype of coarray " + coarray.getName());
        return;   // to avoid warning message from javac
      }
      ident2 = body.declLocalIdent(coarray.getName(), type2);
      ident2.setStorageClass(StorageClass.FCOMMON);
    } else {
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

    // "CALL coarray_alloc_static(descPtr_var, crayPtr_var, ... )"  (case: Ver.3)
    // "CALL coarray_regmem_static(descPtr_var, LOC(var), ... )"    (case: Ver.4)
    Xobject arg2;
    String fname;
    if (version == 4) {
      FunctionType ftype = new FunctionType(Xtype.Fint8Type, Xtype.TQ_FINTRINSIC);
      Ident locId = env.declIntrinsicIdent("loc", ftype);
      arg2 = locId.Call(Xcons.List(ident2));
      fname = REGMEM_STATIC_NAME;
    } else {
      arg2 = (Xobject)ident2;
      fname = ALLOC_STATIC_NAME;
    }

    String varName = coarray.getName();
    Xobject varNameObj = 
      Xcons.FcharacterConstant(Xtype.FcharacterType, varName, null);
    Xobject args = Xcons.List(descPtrId,
                              arg2,
                              Xcons.IntConstant(count),
                              elem,
                              varNameObj,
                              Xcons.IntConstant(varName.length()));
    if (args.hasNullArg())
      XMP.fatal("INTERNAL: generated null argument (buildSubroutine_initcoarrays)");
    Ident subr = body.declLocalIdent(fname, BasicType.FexternalSubroutineType);
    body.add(subr.callSubroutine(args));
  }


  /*
   *  Version 1 (incomplete handling of allocatable coarray)
   */
  /**************************
  private void addForVarText(String varName1, String varName2, 
                             int count, int elem) {
    varNames1.add(varName1);
    varNames2.add(varName2);
    callSizeStmts.add(" CALL " + COUNT_SIZE_NAME + " ( " + 
                      count + " , " + elem + " )");
    callInitStmts.add(" CALL " + ALLOC_STATIC_NAME + " ( " + 
                      varName1 + " , " +varName2 + " , " +
                      count + " , " + elem + " )");
  }
  ****************************/

  /**************************
  private void fillinSizeProcText() {
    if (varNames1.size() == 0)
      return;

    String text = "\n";
    text += "SUBROUTINE " + sizeProcName + "\n";

    // call sizecount stmts
    for (String stmt: callSizeStmts)
      text += stmt +"\n";

    text += "END SUBROUTINE " + sizeProcName + "\n";
    procTexts.add(text);
  }
  ****************************/

  /**************************
  private void fillinInitProcText() {
    if (varNames1.size() == 0)
      return;

    String text = "\n";
    text += "SUBROUTINE " + initProcName + "\n";

    // type specification stmt for varNames1
    for (String name: varNames1)
      text += " INTEGER :: " + name + "\n";

    // type specification stmt for varNames2
    for (String name: varNames2)
      text += " INTEGER(8) :: " + name + "\n";

    // common stmt for varNames1
    for (String name: varNames1)
      text += " COMMON / " + commonName1 + " / " + name + "\n";

    // common stmt for varNames2
    for (String name: varNames2)
      text += " COMMON / " + commonName2 + " / " + name + "\n";

    if (DEBUG) {
      text += " WRITE(*,*) \"[XMPcoarrayInitProcedure] start SUBROUTINE " +
        initProcName + "\"\n";
    }

    // call initialization stmts
    for (String stmt: callInitStmts) {
      if (DEBUG) {
        text += " WRITE(*,*) \" calling " + stmt + "\"\n";
      }
      text += stmt +"\n";
    }

    if (DEBUG) {
      text += " WRITE(*,*) \"[XMPcoarrayInitProcedure] end SUBROUTINE " +
        initProcName + "\"\n";
    }
    text += "END SUBROUTINE " + initProcName + "\n";
    procTexts.add(text);
  }
  *************************/


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


}


