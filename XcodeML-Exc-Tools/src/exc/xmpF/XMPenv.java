/*
 * $TSUKUBA_Release: $
 * $TSUKUBA_Copyright:
 *  $
 */

package exc.xmpF;

import exc.object.*;
import exc.block.*;

import xcodeml.util.XmOption;

public class XMPenv {
  private XobjectFile	_env;
  private XMPsymbolTable _globalObjectTable; // file scope, or module of F

  private final static String SYMBOL_TABLE	= "XMP_PROP_XMP_SYMBOL_TABLE";

  public XMPenv(XobjectFile env) {
    _env = env;
    _globalObjectTable = new XMPsymbolTable();
  }

  public XobjectFile getEnv() {
    return _env;
  }

  /* 
   * external func 
   */
  public static Ident declExternFunc(String funcName) {
    return null; //XMP.getMacroId(funcName, Xtype.voidType);
  }

  public Ident declExternFunc(String funcName, Xtype type) {
    return null; //XMP.getMacroId(funcName, type);
  }

  public Ident declGlobalIdent(String name, Xtype t) {
    return _env.declGlobalIdent(name, t);
  }

  public Ident declStaticIdent(String name, Xtype t) {
    // return _env.declStaticIdent(name, t);
    return Ident.Local(name, t);
  }

  public Ident declExternIdent(String name, Xtype t) {
    return _env.declExternIdent(name, t);
  }

  public Ident findVarIdent(String name) {
    return _env.findVarIdent(name);
  }

  static FunctionBlock parentFunctionBlock(Block bp){
    if(bp == null) return null;
    for (Block b = bp; b != null; b = b.getParentBlock())
      if (b.Opcode() == Xcode.FUNCTION_DEFINITION){
	return  (FunctionBlock) b;
      }
    return null;
  }

  public Ident findVarIdent(String name, Block pb) {
    Ident id;
    BlockList funcBlockList;
    if(pb != null){
      funcBlockList = parentFunctionBlock(pb).getBody();
      id = funcBlockList.findLocalIdent(name);
    } else {
      id = _env.findVarIdent(name);
    }
    return id;
  }

  /* 
   * check name
   */
  public void checkObjectNameCollision(String name) {
    checkObjectNameCollision(name,null);
  }

  public void checkObjectNameCollision(String name, Block block){
    if(block != null){
      // check name collision - parameters
      BlockList body = block.getBody();
      if (body != null && body.findLocalIdent(name) != null){
	XMP.error("'" + name + "' is already declared");
	return;
      }

      XMPsymbolTable symbolTable = getXMPsymbolTable(block);
      // check name collision - local object table
      if (symbolTable.getXMPobject(name) != null){
	XMP.error("'" + name + "' is already declared");
	return;
      }

      // check name collision - descriptor name
      if (body.findLocalIdent(XMP.DESC_PREFIX_ + name) != null) {
	// FIXME generate unique name
	XMP.error("cannot declare template desciptor, '" + 
		  XMP.DESC_PREFIX_ + name + "' is already declared");
	return;
      }
    } else {
      // check name collision - global variables
      if (_env.findVarIdent(name) != null) {
	XMP.error("'" + name + "' is already declared");
	return;
      }

      // check name collision - global object table
      if (_globalObjectTable.getXMPobject(name) != null) {
	XMP.error("'" + name + "' is already declared");
	return;
      }

      // check name collision - descriptor name
      if (_env.findVarIdent(XMP.DESC_PREFIX_ + name) != null) {
	// FIXME generate unique name
	XMP.error("cannot declare desciptor, '" + 
		  XMP.DESC_PREFIX_ + name + "' is already declared");
	return;
      }
    }
  }

  /*
   * put/get XMPobject
   */
  public void putXMPobject(XMPobject obj,Block block) {
    if(block != null){
      XMPsymbolTable localXMPsymbolTable = getXMPsymbolTable(block);
      localXMPsymbolTable.putXMPobject(obj);
    } else 
      putXMPobject(obj);
  }

  public void putXMPobject(XMPobject obj) {
    _globalObjectTable.putXMPobject(obj);
  }

  public XMPobject getXMPobject(String name) {
    return getXMPobject(name, null);
  }

  public XMPobject getXMPobject(String name, Block block){
    XMPobject o = null;
    if(block != null){
      XMPsymbolTable localXMPsymbolTable = getXMPsymbolTable(block);
      if (localXMPsymbolTable != null) {
	o = localXMPsymbolTable.getXMPobject(name);
      }
      if(o != null) return o;
    }
    return _globalObjectTable.getXMPobject(name);
  }

  /*
   * put/get XMPnodes
   */
  public XMPnodes getXMPnodes(String name) {
    return getXMPnodes(name,null);
  }

  public XMPnodes getXMPnodes(String name, Block block){
    XMPnodes n;
    if(block != null){
      XMPsymbolTable localXMPsymbolTable = getXMPsymbolTable(block);
      n = localXMPsymbolTable.getXMPnodes(name);
      if (n != null) return n;
    }
    return _globalObjectTable.getXMPnodes(name);
  }

  /*
   * put/get XMPtemplate
   */
  public XMPtemplate getXMPtemplate(String name) {
    return getXMPtemplate(name,null);
  }

  public XMPtemplate getXMPtemplate(String name, PragmaBlock pb){
    XMPtemplate t;
    if(pb != null){
      XMPsymbolTable localXMPsymbolTable = getXMPsymbolTable(pb);
      t = localXMPsymbolTable.getXMPtemplate(name);
      if(t != null) return t;
    }
    return _globalObjectTable.getXMPtemplate(name);
  }

  /*
   * put/get XMParray
   */
  public void putXMParray(XMParray array) {
    putXMParray(array,null);
  }

  public void putXMParray(XMParray array, Block block) {
    if(block != null){
      XMPsymbolTable localXMPsymbolTable = getXMPsymbolTable(block);
      localXMPsymbolTable.putXMParray(array);
    } else
      _globalObjectTable.putXMParray(array);
  }

  public XMParray getXMParray(String name) {
    return getXMParray(name,null);
  }

  public XMParray getXMParray(String name, Block bp){
    XMParray a;
    if(bp != null){
      XMPsymbolTable localXMPsymbolTable = getXMPsymbolTable(bp);
      a = localXMPsymbolTable.getXMParray(name);
      if(a != null) return a;
    }
    return _globalObjectTable.getXMParray(name);
  }

  /*
   * put/get XMPcorray
   */
  public void putXMPcoarray(XMPcoarray array) {
    _globalObjectTable.putXMPcoarray(array);
  }

  public XMPcoarray getXMPcoarray(String name) {
    return _globalObjectTable.getXMPcoarray(name);
  }

  public void finalize() {
    _env.collectAllTypes();
    _env.fixupTypeRef();
  }

  public Ident declObjectId(String objectName, Block block) {
    return declObjectId(objectName, Xtype.Fint8Type, block);
  }

  public Ident declObjectId(String objectName, Xtype type, Block block) {
    if(block != null)
      return Ident.Local(objectName, type);
    else
      return declStaticIdent(objectName, type);
  }

  public static XMPsymbolTable getXMPsymbolTable(Block block) {
    FunctionBlock fb = parentFunctionBlock(block);

    XMPsymbolTable table = (XMPsymbolTable)fb.getProp(SYMBOL_TABLE);
    if (table == null) {
      table = new XMPsymbolTable();
      fb.setProp(SYMBOL_TABLE, (Object)table);
    }
    return table;
  }

  public static XMPsymbolTable localXMPsymbolTable(Block block) {
    FunctionBlock fb = parentFunctionBlock(block);
    XMPsymbolTable table = (XMPsymbolTable)fb.getProp(SYMBOL_TABLE);
    return table;
  }
}
