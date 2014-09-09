package exc.xcalablemp;

import exc.object.*;
import exc.block.*;

public class XACCglobalDecl {
  private XobjectFile           _env;
  private XACCsymbolTable       _globalObjectTable;
  private XobjList              _globalConstructorFuncBody;
  private XobjList              _globalDestructorFuncBody;
  public static final String    CONSTRUCTOR_FUNC_NAME = "_XACC_init";
  public static final String    DESTRUCTOR_FUNC_NAME = "_XACC_finalize";
  private static final String   XACC_SYMBOL_TABLE = "xacc_symbol_table";

  public XACCglobalDecl(XobjectFile env){
    _env = env;
    _globalObjectTable = new XACCsymbolTable();
    _globalConstructorFuncBody = Xcons.List();
    _globalDestructorFuncBody = Xcons.List();
  }
  
  public void setupGlobalConstructor(){
    Xtype funcType = Xtype.Function(Xtype.voidType);
    Ident funcId = _env.declStaticIdent(CONSTRUCTOR_FUNC_NAME, funcType);
    _env.add(XobjectDef.Func(funcId, null, null, Xcons.List(Xcode.COMPOUND_STATEMENT,
                             (Xobject)null, null, _globalConstructorFuncBody)));
  }
  
  public void setupGlobalDestructor(){
    Xtype funcType = Xtype.Function(Xtype.voidType);
    Ident funcId = _env.declStaticIdent(DESTRUCTOR_FUNC_NAME, funcType);
    _env.add(XobjectDef.Func(funcId, null, null, Xcons.List(Xcode.COMPOUND_STATEMENT,
                             (Xobject)null, null, _globalDestructorFuncBody)));
  }
  
  public XACCdeviceArray getXACCdeviceArray(String name) {
    return _globalObjectTable.getXACCdeviceArray(name);
  }

  public XACCdeviceArray getXACCdeviceArray(String name, Block block) {
    XACCdeviceArray a = null;

    // local
    for (Block b = block; b != null; b = b.getParentBlock()){
      XACCsymbolTable symTab = declXACClocalSymbolTable(b);
      if (symTab != null) a = symTab.getXACCdeviceArray(name);
      if (a != null) return a;
    }

//    // parameter
//    XACCsymbolTable symTab = XMPlocalDecl.getXMPsymbolTable(block);
//    if (symTab != null) a = symTab.getXACCdeviceArray(name);
//    if (a != null) return a;

    // global
    a = getXACCdeviceArray(name);
    if (a != null) return a;

    return null;
  }
  public void putXACCdeviceArray(XACCdeviceArray array) {
    _globalObjectTable.putXACCdeviceArray(array);
  }
  
  public XACCsymbolTable declXACClocalSymbolTable(Block block){
    if (block == null) return null;

    XACCsymbolTable table = (XACCsymbolTable)block.getProp(XACC_SYMBOL_TABLE);
    if (table == null) {
      table = new XACCsymbolTable();
      block.setProp(XACC_SYMBOL_TABLE, (Object)table);
    }

    return table;
  }
}
