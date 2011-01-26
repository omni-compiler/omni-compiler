/*
 * $TSUKUBA_Release: $
 * $TSUKUBA_Copyright:
 *  $
 */

package exc.xcalablemp;

import exc.object.*;

public class XMPglobalDecl {
  private XobjectFile		_env;
  private XMPsymbolTable	_globalObjectTable;
  private XobjList		_globalInitFuncBody;

  public XMPglobalDecl(XobjectFile env) {
    _env = env;
    _globalObjectTable = new XMPsymbolTable();
    _globalInitFuncBody = Xcons.List();
  }

  public void checkObjectNameCollision(String name) throws XMPexception {
    // check name collision - global variables
    if (_env.findVarIdent(name) != null) {
      throw new XMPexception("'" + name + "' is already declared");
    }

    // check name collision - global object table
    if (_globalObjectTable.getXMPobject(name) != null) {
      throw new XMPexception("'" + name + "' is already declared");
    }

    // check name collision - descriptor name
    if (_env.findVarIdent(XMP.DESC_PREFIX_ + name) != null) {
      // FIXME generate unique name
      throw new XMPexception("cannot declare desciptor, '" + XMP.DESC_PREFIX_ + name + "' is already declared");
    }
  }

  public Ident getWorldDescId() {
    return _env.declExternIdent("_XMP_world_nodes", Xtype.voidPtrType);
  }

  public Ident getWorldSizeId() {
    return _env.declExternIdent("_XMP_world_size", Xtype.intType);
  }

  public Ident getWorldRankId() {
    return _env.declExternIdent("_XMP_world_rank", Xtype.intType);
  }

  public void setupGlobalInit() {
    // _globalInitFuncBody.cons(Xcons.List(Xcode.EXPR_STATEMENT,
    // env.declExternIdent(XMP.PREFIX_ + "init_coarray_windows",
    // Xtype.Function(Xtype.voidType)).Call(Xcons.List(Xcons.IntConstant(globalObjectTable.getCoarrayCount())))));

    _globalInitFuncBody.cons(Xcons.List(Xcode.EXPR_STATEMENT,
                                        _env.declExternIdent("_XMP_init_in_constructor",
                                                             Xtype.Function(Xtype.voidType)).Call(null)));

    Xtype consType = Xtype.Function(Xtype.voidType);
    consType.setGccAttributes(Xcons.List(Xcode.GCC_ATTRIBUTES,
                                         Xcons.List(Xcode.GCC_ATTRIBUTE,
                                                    new Ident("constructor", null, null, null, null),
                                                    Xcons.List())));
    Ident consId = _env.declStaticIdent("_XMP_constructor", consType);
    _env.add(XobjectDef.Func(consId, null, null, Xcons.List(Xcode.COMPOUND_STATEMENT, (Xobject)null, null, _globalInitFuncBody)));
  }

  public Ident declExternFunc(String funcName) {
    return _env.declExternIdent(funcName, Xtype.Function(Xtype.voidType));
  }

  public Ident declExternFunc(String funcName, Xtype type) {
    return _env.declExternIdent(funcName, Xtype.Function(type));
  }

  public void addGlobalInitFuncCall(String funcName, Xobject args) {
    Ident funcId = declExternFunc(funcName);
    _globalInitFuncBody.add(Xcons.List(Xcode.EXPR_STATEMENT, funcId.Call(args)));
  }

  public Ident declGlobalIdent(String name, Xtype t) {
    return _env.declGlobalIdent(name, t);
  }

  public Ident declStaticIdent(String name, Xtype t) {
    return _env.declStaticIdent(name, t);
  }

  public Ident declExternIdent(String name, Xtype t) {
    return _env.declExternIdent(name, t);
  }

  public Ident findVarIdent(String name) {
    return _env.findVarIdent(name);
  }

  public void putXMPobject(XMPobject obj) {
    _globalObjectTable.putXMPobject(obj);
  }

  public XMPobject getXMPobject(String name) {
    return _globalObjectTable.getXMPobject(name);
  }

  public XMPnodes getXMPnodes(String name) {
    return _globalObjectTable.getXMPnodes(name);
  }

  public XMPtemplate getXMPtemplate(String name) {
    return _globalObjectTable.getXMPtemplate(name);
  }

  public void putXMPalignedArray(XMPalignedArray array) {
    _globalObjectTable.putXMPalignedArray(array);
  }

  public XMPalignedArray getXMPalignedArray(String name) {
    return _globalObjectTable.getXMPalignedArray(name);
  }

  public void finalize() {
    _env.collectAllTypes();
    _env.fixupTypeRef();
  }
}
