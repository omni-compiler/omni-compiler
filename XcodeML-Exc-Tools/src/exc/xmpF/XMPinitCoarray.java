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
public class XMPinitCoarray {

  final static String MALLOC_LIB_NAME = "xmpf_coarray_malloc";
  final static String ProcTextSeparator = "\n";

  private Boolean DEBUG = false;          // switch the value in gdb !!


  /* for all procedures */
  private Vector<String> procTexts;

  /* for each procedure */
  private String procName;      // name of the current procedure
  private String commonName;    // common block name

  /* for all variables of a procedure */
  private Vector<String> varNames;
  private Vector<String> callStmts;

  //------------------------------
  //  constructor/finalizer
  //------------------------------
  public XMPinitCoarray() {
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
  //       common /xmpf_yyyy_zzzz_EX1/xxxx_V1,xxxx_V2
  //       call xmp_coarray_malloc(xxxx_V1,200,4)
  //       call xmp_coarray_malloc(xxxx_V2,1,16)
  //     end subroutine
  // -------------------------------------------------------

  public void genInitRoutine(Vector<XMPcoarray> staticCoarrays,
                             String procName, String commonName) {
    // init for each init routine
    openProcText(procName, commonName);

    // malloc call stmts
    for (XMPcoarray coarray: staticCoarrays) {
      int elem = coarray.getElementLength();
      int count = coarray.getTotalArraySize();
      String cptrName = coarray.getCrayPointerName();
      addForVarText(cptrName, count, elem);
    }

    // finalize for each init routine
    closeProcText();
  }


  private void openProcText(String procName, String commonName) {
    _init_forProcedure();
    this.procName = procName;
    this.commonName = commonName;
  }

  private void addForVarText(String varName, int count, int elem) {
    varNames.add(varName);
    callStmts.add(" CALL " + MALLOC_LIB_NAME + "(" + 
                  varName + ", " + count + ", " + elem + ")");
  }

  private void closeProcText() {
    if (varNames.size() == 0)
      return;

    String text = ProcTextSeparator;
    text += "SUBROUTINE " + procName + "\n";

    // type specification stmt
    text += " INTEGER(8) ::";
    String delim = " ";
    for (String name: varNames) {
      text += delim + name;
      delim = ", ";
    }
    text += "\n";

    // common stmt
    text += " COMMON / " + commonName + " /";
    delim = " ";
    for (String name: varNames) {
      text += delim + name;
      delim = ", ";
    }
    text += "\n";

    if (DEBUG) {
      text += " WRITE(*,*) \"[XMPinitCoarray] start SUBROUTINE " +
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
      text += " WRITE(*,*) \"[XMPinitCoarray] end SUBROUTINE " +
        procName + "\"\n";
    }
    text += "END SUBROUTINE " + procName + "\n";
    procTexts.add(text);
  }


  /*
   * suspended
   *  This method could be better but not used...
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

  private void _init_forProcedure() {
    varNames = new Vector<String>();
    callStmts = new Vector<String>();
  }

}


