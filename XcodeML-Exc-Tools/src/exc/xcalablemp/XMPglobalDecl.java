/*
 * $TSUKUBA_Release: $
 * $TSUKUBA_Copyright:
 *  $
 */

package exc.xcalablemp;

import exc.object.*;
import exc.block.*;
import xcodeml.util.XmOption;

public class XMPglobalDecl {
  private XobjectFile		_env;
  private XMPsymbolTable	_globalObjectTable;
  private XobjList		_globalConstructorFuncBody;
  private XobjList		_globalDestructorFuncBody;

  public XMPglobalDecl(XobjectFile env) {
    _env = env;
    _globalObjectTable = new XMPsymbolTable();
    _globalConstructorFuncBody = Xcons.List();
    _globalDestructorFuncBody = Xcons.List();
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

  public XobjectFile getEnv() {
    return _env;
  }

  public String genSym(String prefix) {
    return _env.genSym(prefix);
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

  public void setupGlobalConstructor() {
    if (XmOption.tlogMPIisEnable()) {
      _globalConstructorFuncBody.cons(Xcons.List(Xcode.EXPR_STATEMENT,
                                                 declExternFunc("_XMP_tlog_init").Call(null)));
    }

    if (XmOption.isXcalableMPGPU()) {
      _globalConstructorFuncBody.cons(Xcons.List(Xcode.EXPR_STATEMENT,
                                                 declExternFunc("_XMP_gpu_init").Call(null)));
    }

    if (XmOption.isXcalableMPthreads()) {
      _globalConstructorFuncBody.cons(Xcons.List(Xcode.EXPR_STATEMENT,
                                                 declExternFunc("_XMP_threads_init").Call(null)));
    }

    _globalConstructorFuncBody.cons(Xcons.List(Xcode.EXPR_STATEMENT,
                                               declExternFunc("_XMP_init").Call(null)));

    Xtype funcType = Xtype.Function(Xtype.voidType);
    funcType.setGccAttributes(Xcons.List(Xcode.GCC_ATTRIBUTES,
                                         Xcons.List(Xcode.GCC_ATTRIBUTE,
                                                    new Ident("constructor", null, null, null, null),
                                                    Xcons.List())));
    Ident funcId = _env.declStaticIdent("_XMP_constructor", funcType);
    _env.add(XobjectDef.Func(funcId, null, null, Xcons.List(Xcode.COMPOUND_STATEMENT,
                             (Xobject)null, null, _globalConstructorFuncBody)));
  }

  public void setupGlobalDestructor() {
    if (XmOption.tlogMPIisEnable()) {
      _globalDestructorFuncBody.add(Xcons.List(Xcode.EXPR_STATEMENT,
                                               declExternFunc("_XMP_tlog_finalize").Call(null)));
    }

    if (XmOption.isXcalableMPGPU()) {
      _globalDestructorFuncBody.add(Xcons.List(Xcode.EXPR_STATEMENT,
                                               declExternFunc("_XMP_gpu_finalize").Call(null)));
    }

    if (XmOption.isXcalableMPthreads()) {
      _globalDestructorFuncBody.add(Xcons.List(Xcode.EXPR_STATEMENT,
                                               declExternFunc("_XMP_threads_finalize").Call(null)));
    }

    _globalDestructorFuncBody.add(Xcons.List(Xcode.EXPR_STATEMENT,
                                             declExternFunc("_XMP_finalize").Call(null)));

    Xtype funcType = Xtype.Function(Xtype.voidType);
    funcType.setGccAttributes(Xcons.List(Xcode.GCC_ATTRIBUTES,
                                         Xcons.List(Xcode.GCC_ATTRIBUTE,
                                                    new Ident("destructor", null, null, null, null),
                                                    Xcons.List())));
    Ident funcId = _env.declStaticIdent("_XMP_destructor", funcType);
    _env.add(XobjectDef.Func(funcId, null, null, Xcons.List(Xcode.COMPOUND_STATEMENT,
                             (Xobject)null, null, _globalDestructorFuncBody)));
  }

  public Ident declExternFunc(String funcName) {
    return XMP.getMacroId(funcName, Xtype.voidType);
  }

  public Ident declExternFunc(String funcName, Xtype type) {
    return XMP.getMacroId(funcName, type);
  }

  public void addGlobalInitFuncCall(String funcName, Xobject args) {
    Ident funcId = declExternFunc(funcName);
    _globalConstructorFuncBody.add(Xcons.List(Xcode.EXPR_STATEMENT, funcId.Call(args)));
  }

  public void addGlobalFinalizeFuncCall(String funcName, Xobject args) {
    Ident funcId = declExternFunc(funcName);
    _globalDestructorFuncBody.add(Xcons.List(Xcode.EXPR_STATEMENT, funcId.Call(args)));
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

  public Block createFuncCallBlock(String funcName, XobjList funcArgs) {
    Ident funcId = declExternFunc(funcName);
    return Bcons.Statement(funcId.Call(funcArgs));
  }

  public void putXMPobject(XMPobject obj) {
    _globalObjectTable.putXMPobject(obj);
  }

  public XMPobject getXMPobject(String name) {
    return _globalObjectTable.getXMPobject(name);
  }

  public XMPobject getXMPobject(String name, XMPsymbolTable localXMPsymbolTable) {
    XMPobject o = null;

    if (localXMPsymbolTable != null) {
      o = localXMPsymbolTable.getXMPobject(name);
    }

    if (o == null) {
      o = getXMPobject(name);
    }

    return o;
  }

  public XMPnodes getXMPnodes(String name) {
    return _globalObjectTable.getXMPnodes(name);
  }

  public XMPnodes getXMPnodes(String name, XMPsymbolTable localXMPsymbolTable) {
    XMPnodes n = null;

    if (localXMPsymbolTable != null) {
      n = localXMPsymbolTable.getXMPnodes(name);
    }

    if (n == null) {
      n = getXMPnodes(name);
    }

    return n;
  }

  public XMPtemplate getXMPtemplate(String name) {
    return _globalObjectTable.getXMPtemplate(name);
  }

  public XMPtemplate getXMPtemplate(String name, XMPsymbolTable localXMPsymbolTable) {
    XMPtemplate t = null;

    if (localXMPsymbolTable != null) {
      t = localXMPsymbolTable.getXMPtemplate(name);
    }

    if (t == null) {
      t = getXMPtemplate(name);
    }

    return t;
  }

  public void putXMPalignedArray(XMPalignedArray array) {
    _globalObjectTable.putXMPalignedArray(array);
  }

  public XMPalignedArray getXMPalignedArray(String name) {
    return _globalObjectTable.getXMPalignedArray(name);
  }

  public XMPalignedArray getXMPalignedArray(String name, XMPsymbolTable localXMPsymbolTable) {
    XMPalignedArray a = null;

    if (localXMPsymbolTable != null) {
      a = localXMPsymbolTable.getXMPalignedArray(name);
    }

    if (a == null) {
      a = getXMPalignedArray(name);
    }

    return a;
  }

  public void putXMPcoarray(XMPcoarray array) {
    _globalObjectTable.putXMPcoarray(array);
  }

  public XMPcoarray getXMPcoarray(String name) {
    return _globalObjectTable.getXMPcoarray(name);
  }

  public XMPcoarray getXMPcoarray(String name, XMPsymbolTable localXMPsymbolTable) {
    XMPcoarray c = null;

    if (localXMPsymbolTable != null) {
      c = localXMPsymbolTable.getXMPcoarray(name);
    }

    if (c == null) {
      c = getXMPcoarray(name);
    }

    return c;
  }

  public void finalize() {
    _env.collectAllTypes();
    _env.fixupTypeRef();
  }
}
