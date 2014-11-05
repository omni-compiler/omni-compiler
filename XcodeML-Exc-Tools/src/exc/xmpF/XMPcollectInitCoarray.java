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
 * create subroutine xmpf_init_xxx
 */
public class XMPcollectInitCoarray {

  final String ProcTextSeparator = "\n";

  /* for all procedures */
  private Vector<String> procTexts;
  private String mallocName;    // malloc library name

  /* for each procedure */
  private String procName;      // name of the current procedure
  private String commonName;    // common block name

  /* for all variables of a procedure */
  private Vector<String> varNames;
  private Vector<String> stmts;

  //------------------------------
  //  constructor/finalizer
  //------------------------------
  public XMPcollectInitCoarray(String mallocName) {
    _init_forFile();
    this.mallocName = mallocName;
  }

  public void finalize(XMPenv env) {
    //env.clearTailText();
    for (String text: procTexts)
      env.addTailText(text);
  }

  //------------------------------
  //  for each procedure (text version)
  //------------------------------
  public void openProcText(String procName, String commonName) {
    _init_forProcedure();
    this.procName = procName;
    this.commonName = commonName;
  }

  public void addForVarText(String varName, int count, int elem) {
    varNames.add(varName);
    stmts.add(" CALL " + mallocName + "(" + 
              varName + ", " + count + ", " + elem + ")" + "\n");
  }

  public void closeProcText() {
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

    // call stmts
    for (String stmt: stmts) {
      text += stmt;
    }

    text += "END SUBROUTINE " + procName + "\n";
    procTexts.add(text);
  }


  //------------------------------
  //  for each procedure (Xobject version)
  //  NOT IMPLEMENTED
  //------------------------------




  //------------------------------
  //  parts
  //------------------------------
  private void _init_forFile() {
    procTexts = new Vector<String>();
  }

  private void _init_forProcedure() {
    varNames = new Vector<String>();
    stmts = new Vector<String>();
  }

}


