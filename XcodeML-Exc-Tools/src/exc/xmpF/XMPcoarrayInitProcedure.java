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

  final static String COUNT_SIZE_LIB_NAME = "xmpf_coarray_count_size";
  final static String SHARE_LIB_NAME = "xmpf_coarray_share";
  final static String SET_COSHAPE_LIB_NAME = "xmpf_coarray_set_coshape";

  private Boolean DEBUG = false;          // switch the value in gdb !!


  /* for all procedures */
  private Vector<String> procTexts;
  Vector<XMPcoarray> staticCoarrays;
  XMPenv env;

  /* for each procedure */
  private String sizeProcName, initProcName;  // names of procedures to generate
  private String commonName1, commonName2;    // common block names

  /* for all variables of a procedure */
  private Vector<String> varNames1, varNames2;
  private Vector<String> callSizeStmts, callInitStmts;

  //------------------------------
  //  constructor/finalizer
  //------------------------------
  public XMPcoarrayInitProcedure(Vector<XMPcoarray> staticCoarrays,
                                 String sizeProcName, String initProcName,
                                 String commonName1, String commonName2,
                                 XMPenv env) {
    _init_forFile();

    this.staticCoarrays = staticCoarrays;
    this.sizeProcName = sizeProcName;
    this.initProcName = initProcName;
    this.commonName1 = commonName1;
    this.commonName2 = commonName2;
    this.env = env;
    varNames1 = new Vector<String>();
    varNames2 = new Vector<String>();
    callInitStmts = new Vector<String>();
    callSizeStmts = new Vector<String>();
  }


  //------------------------------
  //  for each procedure (text version)
  //------------------------------

  /*
    from a source:
    --------------------------------------------
      subroutine EX1
        real :: V1(10,20)[4,*]
        complex(8) :: V2[0:*]
        integer, allocatable :: V3(:)[:,:]
        ...
        V1(1:3,j)[k1,k2] = (/1.0,2.0,3.0/)
        z = V2[k]**2
        allocate (V3(1:10))
        n(1:5) = V3(2:10:2)[k1,k2]
        return
      end subroutine
    --------------------------------------------
    generate two subroutines:
    --------------------------------------------
      subroutine xmpf_traverse_coarraysize_ex1
        call xmpf_coarray_count_size(200, 4)
        call xmpf_coarray_count_size(1, 16)
      end subroutine

      subroutine xmpf_traverse_initcoarray_ex1
        integer :: CD_V1
        integer :: CD_V2
        integer(8) :: CP_V1
        integer(8) :: CP_V2
        common /xmpf_CD_EX1/ CD_V1, CD_V2
        common /xmpf_CP_EX1/ CP_V1, CP_V2
        call xmpf_coarray_share(CD_V1, CP_V1, 200, 4)
        call xmpf_coarray_set_coshape(CD_V1, 2, 1, 4, 1)
        call xmpf_coarray_share(CD_V2, CP_V2, 1, 16)
        call xmpf_coarray_set_coshape(CD_V2, 1, 0)
      end subroutine
    --------------------------------------------
      CD_Vn: serial number for descriptor of Vn
      CP_Vn: cray poiter pointing to Vn
  */

  public void run() {
    run(2);
  }

  public void run(int version) {
    for (XMPcoarray coarray: staticCoarrays) {
      int elem = coarray.getElementLength();
      int count = coarray.getTotalArraySize();
      String descrName = coarray.getDescriptorName();
      String cptrName = coarray.getCrayPointerName();

      addForVarText(descrName, cptrName, count, elem);
    }

    /* generate the two subroutines in the same file
     */
    switch(version) {
    case 1:   // generate as Fortran program text
      fillinSizeProcText();
      fillinInitProcText();

      for (String text: procTexts)
        env.addTailText(text);
      break;

    case 2:   // build and link it at the tail of XMPenv
      buildSubroutine_coarraysize();
      buildSubroutine_initcoarray();
      break;
    }
  }


  /*
   *  Version 2
   */

  private void buildSubroutine_coarraysize() {
    BlockList body = Bcons.emptyBody();

    for (XMPcoarray coarray: staticCoarrays) {
      int elem = coarray.getElementLength();
      int count = coarray.getTotalArraySize();

      Xobject args = Xcons.List(Xcons.IntConstant(count),
                                Xcons.IntConstant(elem));
      Ident subr = env.declExternIdent(COUNT_SIZE_LIB_NAME,
                                       Xtype.FsubroutineType);
      body.add(subr.callSubroutine(args));
    }

    Ident procedure = env.declExternIdent(sizeProcName,
                                          Xtype.FsubroutineType);
    XobjectDef procDef = XobjectDef.Func(procedure, null, null,
                                         body.toXobject());
    env.getEnv().add(procDef);
  }


  private void buildSubroutine_initcoarray() {
    BlockList body = Bcons.emptyBody();
    Xobject decls = Xcons.List();

    for (XMPcoarray coarray: staticCoarrays) {
      int elem = coarray.getElementLength();
      int count = coarray.getTotalArraySize();
      String serno = coarray.getDescriptorName();
      String crayptr = coarray.getCrayPointerName();

      //varNames1.add(serno);
      //varNames2.add(crayptr);

      // build "integer, save :: serno, crayptr"
      Ident sernoId =
        body.declLocalIdent(serno, Xtype.FintType, StorageClass.FCOMMON, null);
      Ident crayptrId = 
        body.declLocalIdent(crayptr, Xtype.Fint8Type, StorageClass.FCOMMON, null);
      /*------
        for 32-bit pointer envirionment
        Ident crayptrId = 
        body.declLocalIdent(serno, Xtype.Fint4Type, StorageClass.FSAVE, null);
        ------*/

      // build "common /codescr_foo/ serno" and "common /crayptr_foo/ crayptr"
      Xobject commonStmt1 = Xcons.List(Xcode.F_COMMON_DECL,
                                       Xcons.List(Xcode.F_VAR_LIST,
                                                  Xcons.Symbol(Xcode.IDENT, commonName1),
                                                  Xcons.List(Xcons.FvarRef(sernoId))));
      Xobject commonStmt2 = Xcons.List(Xcode.F_COMMON_DECL,
                                       Xcons.List(Xcode.F_VAR_LIST,
                                                  Xcons.Symbol(Xcode.IDENT, commonName2),
                                                  Xcons.List(Xcons.FvarRef(crayptrId))));
      decls.add(commonStmt1);
      decls.add(commonStmt2);

      // build "call coarray_share(serno, crayptr, count, elem)"
      Xobject args = Xcons.List(sernoId, crayptrId,
                                Xcons.IntConstant(count),
                                Xcons.IntConstant(elem));
      Ident subr = env.declExternIdent(SHARE_LIB_NAME,
                                    Xtype.FsubroutineType);
      body.add(subr.callSubroutine(args));
    }

    Ident procedure = env.declExternIdent(initProcName, Xtype.FsubroutineType);
    XobjectDef procDef = XobjectDef.Func(procedure, null, decls, body.toXobject());
    env.getEnv().add(procDef);
  }



  /*
   *  Version 1 (incomplete handling of allocatable coarray)
   */

  private void addForVarText(String varName1, String varName2, 
                             int count, int elem) {
    varNames1.add(varName1);
    varNames2.add(varName2);
    callSizeStmts.add(" CALL " + COUNT_SIZE_LIB_NAME + " ( " + 
                      count + " , " + elem + " )");
    callInitStmts.add(" CALL " + SHARE_LIB_NAME + " ( " + 
                      varName1 + " , " +varName2 + " , " +
                      count + " , " + elem + " )");
  }


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


  /*
   * suspended
   *  This method could be better but not completed...
   */
  public void genInitRoutine__NOT_USED__(Vector<XMPcoarray> staticCoarrays,
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
    procTexts = new Vector<String>();
  }


}


