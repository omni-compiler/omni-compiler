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

  final static String MALLOC_LIB_NAME = "xmpf_coarray_malloc";

  private Boolean DEBUG = false;          // switch the value in gdb !!


  /* for all procedures */
  private Vector<String> procTexts;

  /* for each procedure */
  private String procName;      // name of the current procedure
  private String commonName1, commonName2;    // common block names

  /* for all variables of a procedure */
  private Vector<String> varNames1, varNames2;
  private Vector<String> callStmts;

  //------------------------------
  //  constructor/finalizer
  //------------------------------
  public XMPcoarrayInitProcedure() {
    _init_forFile();
  }

  public void finalize(XMPenv env) {
    //env.clearTailText();
    for (String text: procTexts)
      env.addTailText(text);
  }

  //------------------------------
  //  for each procedure (text version)
  //------------------------------

  // generate initialization subroutine corresponding to 
  // the user program EX1 and coarrays V1 and V2:
  // -------------------------------------------------------
  //     subroutine xmpf_traverse_wwww_EX1
  //       integer :: desc_V1
  //       integer :: desc_V2
  //       common /xmpf_desc_EX1/desc_V1,desc_V2
  //       common /xmpf_ptr_EX1/ptr_V1,ptr_V2
  //       call xmp_coarray_malloc(desc_V1,ptr_V1,200,4)
  //       call xmp_coarray_malloc(desc_V2,ptr_V2,1,16)
  //     end subroutine
  // -------------------------------------------------------

  public void genInitRoutine(Vector<XMPcoarray> staticCoarrays,
                             String procName, 
                             String commonName1, String commonName2) {
    // init for each init routine
    openProcText(procName, commonName1, commonName2);

    // malloc call stmts
    for (XMPcoarray coarray: staticCoarrays) {
      int elem = coarray.getElementLength();
      int count = coarray.getTotalArraySize();
      String descrName = coarray.getDescriptorName();
      String cptrName = coarray.getCrayPointerName();
      addForVarText(descrName, cptrName, count, elem);
    }

    // finalize for each init routine
    closeProcText();
  }


  private void openProcText(String procName,
                            String commonName1, String commonName2) {
    this.procName = procName;
    this.commonName1 = commonName1;
    this.commonName2 = commonName2;
    varNames1 = new Vector<String>();
    varNames2 = new Vector<String>();
    callStmts = new Vector<String>();
  }

  private void addForVarText(String varName1, String varName2, 
                             int count, int elem) {
    varNames1.add(varName1);
    varNames2.add(varName2);
    callStmts.add(" CALL " + MALLOC_LIB_NAME + " ( " + 
                  varName1 + " , " +varName2 + " , " +
                  count + " , " + elem + " )");
  }

  private void closeProcText() {
    if (varNames1.size() == 0)
      return;

    String text = "\n";
    text += "SUBROUTINE " + procName + "\n";

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
        procName + "\"\n";
    }

    // call stmts
    for (String stmt: callStmts) {
      if (DEBUG) {
        text += " WRITE(*,*) \" calling " + stmt + "\"\n";
      }
      text += stmt +"\n";
    }

    if (DEBUG) {
      text += " WRITE(*,*) \"[XMPcoarrayInitProcedure] end SUBROUTINE " +
        procName + "\"\n";
    }
    text += "END SUBROUTINE " + procName + "\n";
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


