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

  final static String SIZE_LIB_NAME = "xmpf_coarray_count_size";
  final static String INIT_LIB_NAME = "xmpf_coarray_get_share";

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

  // generate initialization subroutine corresponding to 
  // the user program EX1 and coarrays V1 and V2:
  // -------------------------------------------------------
  //     subroutine <sizeProcName>      ! xmpf_traverse_coarraysize_EX1
  //       call <SIZE_LIB_NAME>(200,4)
  //       call <SIZE_LIB_NAME>(1,16)
  //     end subroutine
  //
  //     subroutine <initProcName>      ! xmpf_traverse_initcoarray_EX1
  //       integer :: desc_V1
  //       integer :: desc_V2
  //       common /<commonName1>/desc_V1,desc_V2
  //       common /<commonName2>/ptr_V1,ptr_V2
  //       call <INIT_LIB_NAME>(desc_V1,ptr_V1,200,4)
  //       call <INIT_LIB_NAME>(desc_V2,ptr_V2,1,16)
  //     end subroutine
  // -------------------------------------------------------

  public void run() {
    for (XMPcoarray coarray: staticCoarrays) {
      int elem = coarray.getElementLength();
      int count = coarray.getTotalArraySize();
      String descrName = coarray.getDescriptorName();
      String cptrName = coarray.getCrayPointerName();

      addForVarText(descrName, cptrName, count, elem);
    }

    // fill in program text
    fillinSizeProcText();
    fillinInitProcText();

    //env.clearTailText();
    for (String text: procTexts)
      env.addTailText(text);
  }


  private void addForVarText(String varName1, String varName2, 
                             int count, int elem) {
    varNames1.add(varName1);
    varNames2.add(varName2);
    callSizeStmts.add(" CALL " + SIZE_LIB_NAME + " ( " + 
                      count + " , " + elem + " )");
    callInitStmts.add(" CALL " + INIT_LIB_NAME + " ( " + 
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


