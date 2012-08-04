/*
 * $TSUKUBA_Release: $
 * $TSUKUBA_Copyright:
 *  $
 */

package exc.xmpF;

import exc.object.*;
import exc.block.*;

import xcodeml.util.XmOption;

/* this is for Fortran environment:
 *   no global declaration
 */

public class XMPenv {
  private XobjectFile env;
  private FuncDefBlock current_def;
  private boolean is_module = false;

  private final static String SYMBOL_TABLE = "XMP_PROP_XMP_SYMBOL_TABLE";

  public XMPenv(XobjectFile env) {
    this.env = env;
  }

  public XobjectFile getEnv() {
    return env;
  }

  public void setCurrentDef(FuncDefBlock def){
    current_def = def;
    XobjectDef d = def.getDef();
    if(d.getProp(SYMBOL_TABLE) == null){
      XMPsymbolTable table = new XMPsymbolTable();
      d.setProp(SYMBOL_TABLE, (Object)table);
    }
    is_module = def.getDef().isFmoduleDef();
  }

  FuncDefBlock getCurrentDef() { return current_def; }
  
  public boolean currentDefIsModule() { return is_module; }

  // get symbol table bind to XobjectDef def
  public static XMPsymbolTable getXMPsymbolTable(XobjectDef def) {
    return (XMPsymbolTable)def.getProp(SYMBOL_TABLE);
  }

  public XMPsymbolTable getXMPsymbolTable() {
    return getXMPsymbolTable(current_def.getDef());
  }

  /* 
   * Symbol management: external func
   */
  public Ident declExternIdent(String name, Xtype type) {
    return declIdent(name,type,true,null);
  }

  // this is local
  public Ident declIdent(String name, Xtype type, Block block) {
    return declIdent(name,type,false,block);
  }

  // this is static
  public Ident declIdent(String name, Xtype type) {
    return declIdent(name,type,false,null);
  }

  public Ident declIdent(String name, Xtype type, 
			 boolean is_external, Block block) {
    BlockList body= current_def.getBlock().getBody();
    Xobject id_list = body.getIdentList();
    if(id_list != null){
      for(Xobject o : (XobjList)id_list){
	if(name.equals(o.getName())){
	  if(!type.equals(o.Type()))
	    XMP.fatal("declIdent: duplicated declaration: "+name);
	  return (Ident)o;
	}
      }
    }

    Ident id;
    if(block == null && is_external)
      id = Ident.Fident(name,type,env);
    else
      id = Ident.FidentNotExternal(name,type);
    body.addIdent(id);
    return id;
  }

  // Id is Fint8type
  public Ident declObjectId(String objectName, Block block) {
    Xtype t = Xtype.Fint8Type;
//     if(is_module){
//       t = t.copy();
//       t.setIsFsave(true);
//     }
    return declIdent(objectName, t, block);
  }

  public void finalize() {
    env.collectAllTypes();
    env.fixupTypeRef();
  }

  // intrinsic
  public Ident FintrinsicIdent(Xtype t, String name){
    Xtype tt;
    tt = t.copy();
    tt.setIsFintrinsic(true);
    return Ident.Fident(name, tt ,false, false, env);
  }

  /*
   *  Serch symbols nested definitions in Fortran
   */
  public Ident findVarIdent(String name, Block b){
    for(XobjectDef def = current_def.getDef(); def != null; 
	def = def.getParent()){
      Xobject id_list = def.getDef().getArg(1);
      for(Xobject id: (XobjList)id_list){
	if(id.getName().equals(name)){
	  return (Ident) id;
	}
      }
    }
    return null;
  }

  /*
   * put/get XMPobject (nodes and template)
   */
  public void declXMPobject(XMPobject obj, Block block) {
    // in case of fortran, block is ingored
    declXMPobject(obj);
  }

  public void declXMPobject(XMPobject obj) {
    XMPsymbolTable table = getXMPsymbolTable();
    table.putXMPobject(obj);
  }

  public XMPobject findXMPobject(String name, Block block){
    // in case of fortran, block is ingored
    return findXMPobject(name);
  }

  public XMPobject findXMPobject(String name) {
    for(XobjectDef def = current_def.getDef(); 
	def != null; def = def.getParent()){
      XMPsymbolTable table = getXMPsymbolTable(def);
      if(table == null) break;
      XMPobject o = table.getXMPobject(name);
      if(o != null) return o;
    }
    return null;
  }

  /*
   * decl/find XMPnodes
   */
  public XMPnodes findXMPnodes(String name, Block block){
    return findXMPnodes(name);
  }

  public XMPnodes findXMPnodes(String name) {
    for(XobjectDef def = current_def.getDef(); 
	def != null; def = def.getParent()){
      XMPsymbolTable table = getXMPsymbolTable(def);
      if(table == null) break;
      XMPnodes o = table.getXMPnodes(name);
      if(o != null) return o;
    }
    return null;
  }

  /*
   * decl/find XMPtemplate
   */
  public XMPtemplate getXMPtemplate(String name, Block block){
    return getXMPtemplate(name);
  }

  public XMPtemplate getXMPtemplate(String name) {
    for(XobjectDef def = current_def.getDef(); 
	def != null; def = def.getParent()){
      XMPsymbolTable table = getXMPsymbolTable(def);
      if(table == null) break;
      XMPtemplate t = table.getXMPtemplate(name);
      if(t != null) return t;
    }
    return null;
  }

  /*
   * decl/find XMParray
   */
  public void declXMParray(XMParray array, Block block) {
    declXMParray(array);
  }

  public void declXMParray(XMParray array) {
    XMPsymbolTable table = getXMPsymbolTable();
    table.putXMParray(array);
  }

//   public XMParray findXMParray(String name, Block bp){
//     return findXMParray(name);
//   }

//   public XMParray findXMParray(String name) {
//     for(XobjectDef def = current_def.getDef(); def != null; def = def.getParent()){
//       XMPsymbolTable table = getXMPsymbolTable(def);
//       XMParray a = table.getXMParray(name);
//       if(a != null) return a;
//     }
//     return null;
//   }

  /*
   * put/get XMPcorray (not yet ...)
   */
//   public void putXMPcoarray(XMPcoarray array) {
//     _globalObjectTable.putXMPcoarray(array);
//   }

//   public XMPcoarray getXMPcoarray(String name) {
//     return _globalObjectTable.getXMPcoarray(name);
//   }

}
