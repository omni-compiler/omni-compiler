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
public class XMPinitProcedure {

  private FuncDefBlock def;
  private XMPenv env;

  private String name;
  private BlockList decls;

  //------------------------------
  //  constructor/destructor
  //------------------------------
  public XMPinitProcedure(FuncDefBlock def, XMPenv env) {
    this.def = def;
    this.env = env;

    //    name = genProcedureName(def);
    decls = new BlockList();




    Xobject xobj = new Xobject(Xcode.FUNCTION_DEFINITION,
                               Xtype.FsubroutineType);

    XobjectDef d = new XobjectDef(xobj);
    this.def = new FuncDefBlock(d);

    //this.def = new FuncDefBlock(nameObj, null, null, null, null, env.getEnv());
    this.env = env;

  }

  public void finalize() {
    finalTest();
  }

  //------------------------------
  //  parts
  //------------------------------
  private void addProcedureIdent() {
    decls.addIdent(Ident.Fident(name, Xtype.FsubroutineType));
  }

  private void finalTest() {

  }

  //------------------------------
  //  interface
  //------------------------------
  public void setName(String name) {
    this.name = name;
  }

  public void addIdent(Ident ident) {
    decls.addIdent(ident);
  }

  public void addStmt(Xobject stmt) {
    decls.add(stmt);
  }

  public void insertStmt(Xobject stmt) {
    decls.insert(stmt);
  }

  public FuncDefBlock getDef() {
    return def;
  }

  public BlockList getDecls() {
    return decls;
  }

  /***********
Statment s;
BasicBlock bb = s.getParent();
Block b = bb.getParent();
for(b_list = getParentk(); b_list != null; 
             b_list = b_list.getParentList()){
	b_list.findLocalIdent(name);
}
  **********/

}

