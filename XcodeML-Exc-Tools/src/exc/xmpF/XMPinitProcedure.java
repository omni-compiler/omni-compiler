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

  final String INITPROC_PREFIX = "xmpf_init";

  private String name;
  private Vector<Ident> idents = new Vector();
  private Vector<Xobject> decls = new Vector();
  private Vector<Ident> commonVars = new Vector();
  private Vector<Xobject> stmts = new Vector();

  //------------------------------
  //  constructor/destructor
  //------------------------------
  public XMPinitProcedure(String... names) {  // host-to-guest order
    name = genProcedureName(names);
    idents.add(Ident.Fident(name, Xtype.FsubroutineType));
  }

  public XMPinitProcedure(FuncDefBlock def) {
    String[] names = getHostNames(def);
    name = genProcedureName(names);
    idents.add(Ident.Fident(name, Xtype.FsubroutineType));
  }

  public void finalize() {
    finalTest();
  }

  //------------------------------
  //  set/get/add
  //------------------------------
  public String getName() {
    return name;
  }

  public void setName(String name) {
    this.name = name;
  }

  public void addIdent(Ident ident) {
    idents.add(ident);
  }

  public void addCommonVar(Ident ident) {
    commonVars.add(ident);
  }

  public void addStmt(Xobject stmt) {
    stmts.add(stmt);
  }

  public void insertStmt(Xobject stmt) {
    stmts.add(0, stmt);
  }


  //------------------------------
  //  parts
  //------------------------------
  private void finalTest() {
    System.out.println("[XMPinitProc.finalize]");
    System.out.println("name:");
    System.out.println(name);
    System.out.println("idents:");
    for (Ident id: idents)
      System.out.println(id);
    System.out.println("decls:");
    for (Xobject decl: decls)
      System.out.println(decl);
    System.out.println("commonVars:");
    for (Xobject cvar: commonVars)
      System.out.println(cvar);
    System.out.println("stmts:");
    for (Xobject stmt: stmts)
      System.out.println(stmts);
  }


  private String[] getHostNames(FuncDefBlock procDef) {
    Vector<String> list = new Vector();
    XobjectDef def = procDef.getDef();
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

  private String genProcedureName(String... names) { // host-to-guest order
    int n = names.length;
    String initProcName = INITPROC_PREFIX;
    for (int i = 0; i < n; i++) {
      initProcName += "_";
      StringTokenizer st = new StringTokenizer(names[i], "_");
      int n_underscore = st.countTokens() - 1;
      if (n_underscore > 0)   // '_' was found in names[i]
        initProcName += String.valueOf(n_underscore);
      initProcName += names[i];
    }
    return initProcName;
  }

}
