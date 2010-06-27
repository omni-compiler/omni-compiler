package exc.xcalablemp;

import exc.object.*;

public class XMPglobalDecl {
  private XobjectFile		_env;
  private XMPobjectTable	_globalObjectTable;
  private XobjList		_globalInitFuncBody;

  public XMPglobalDecl(XobjectFile env) {
    _env = env;
    _globalObjectTable = new XMPobjectTable();
    _globalInitFuncBody = Xcons.List();
  }

  public Ident getWorldDescId() {
    return _env.declExternIdent("_XCALABLEMP_world_nodes", Xtype.voidPtrType);
  }

  public Ident getWorldSizeId() {
    return _env.declExternIdent("_XCALABLEMP_world_size", Xtype.intType);
  }

  public Ident getWorldRankId() {
    return _env.declExternIdent("_XCALABLEMP_world_rank", Xtype.intType);
  }

  public void setupGlobalInit() {
    //_globalInitFuncBody.cons(Xcons.List(Xcode.EXPR_STATEMENT,
    //                                    env.declExternIdent(XMP.PREFIX_ + "init_coarray_windows",
    //                                                        Xtype.Function(Xtype.voidType)).Call(Xcons.List(Xcons.IntConstant(globalObjectTable.getCoarrayCount())))));

    _globalInitFuncBody.cons(Xcons.List(Xcode.EXPR_STATEMENT,
                                        _env.declExternIdent("_XCALABLEMP_init_world_NULL",
                                                             Xtype.Function(Xtype.voidType)).Call(null)));

    Xtype consType = Xtype.Function(Xtype.voidType);
    consType.setGccAttributes(Xcons.List(Xcode.GCC_ATTRIBUTES,
                                         Xcons.List(Xcode.GCC_ATTRIBUTE,
                                                    new Ident("constructor", null, null, null, null),
                                                    Xcons.List())));
    Ident consId = _env.declStaticIdent("_XCALABLEMP_constructor", consType);
    _env.add(XobjectDef.Func(consId, null, null, Xcons.List(Xcode.COMPOUND_STATEMENT, (Xobject)null, null, _globalInitFuncBody)));
  }

  public XobjectFile getEnv() {
    return _env;
  }

  public XMPobjectTable getGlobalObjectTable() {
    return _globalObjectTable;
  }

  public Ident declExternFunc(String funcName) {
    return _env.declExternIdent(funcName, Xtype.Function(Xtype.voidType));
  }

  public void addGlobalInitFuncCall(String funcName, Xobject args) {
    Ident funcId = declExternFunc(funcName);
    _globalInitFuncBody.add(Xcons.List(Xcode.EXPR_STATEMENT, funcId.Call(args)));
  }
}
