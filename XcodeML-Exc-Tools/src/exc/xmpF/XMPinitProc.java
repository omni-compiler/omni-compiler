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
public class XMPinitProc {

  final String NamePrefix = "xmpf_init";

  private String name;
  private BlockList blockList;

  //------------------------------
  //  constructor/destructor
  //------------------------------
  public XMPinitProc(String... names) {  // host-to-guest order
    setName(makeInitProcName(names));
    initBlockList();
  }

  public XMPinitProc(FuncDefBlock def) {
    String[] names = getHostNames(def);
    setName(makeInitProcName(names));
    initBlockList();
  }

  public void finalize() {
    ///////
    System.out.println("XMPinitProc.finalize");
    System.out.println("name: "+name);
    System.out.println("blockList.id_list: ");
    System.out.println("---------------------------------------------------------------------------");
    System.out.println(blockList.getIdentList());
    System.out.println("---------------------------------------------------------------------------");
    System.out.println("blockList.decls: ");
    System.out.println("---------------------------------------------------------------------------");
    System.out.println(blockList.getDecls());
    System.out.println("---------------------------------------------------------------------------");
    ///////
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

  public void add(Ident ident) {
    blockList.addIdent(ident);
  }

  public void add(Xobject stmt) {
    blockList.add(stmt);
  }

  public void insert(Xobject stmt) {
    blockList.insert(stmt);
  }


  //------------------------------
  //  parts
  //------------------------------
  private void initBlockList() {
    blockList = new BlockList();
    Ident ident = Ident.Fident(name, Xtype.FsubroutineType);
    blockList.addIdent(ident);
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

  private String makeInitProcName(String... names) { // host-to-guest order
    int n = names.length;
    String initProcName = NamePrefix;
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
