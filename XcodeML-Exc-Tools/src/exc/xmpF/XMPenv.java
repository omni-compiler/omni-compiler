/*
 * $TSUKUBA_Release: $
 * $TSUKUBA_Copyright:
 *  $
 */

package exc.xmpF;

import java.io.*;
import java.util.Vector;

import exc.object.*;
import exc.block.*;

import xcodeml.util.XmOption;

/* this is for Fortran environment:
 *   no global declaration
 */

public class XMPenv {
  protected XobjectFile env;
  protected boolean is_module = false;

  private Vector<XMPmodule> modules = new Vector<XMPmodule>();
  private FuncDefBlock current_def;

  private final static String SYMBOL_TABLE = "XMP_PROP_XMP_SYMBOL_TABLE";
  
  public XMPenv() { }

  public XMPenv(XobjectFile env) {
    this.env = env;
  }

  public XobjectFile getEnv() {
    return env;
  }

  public XMPmodule findModule(String module_name){
    for(XMPmodule mod : modules){
      if(module_name.equals(mod.getModuleName())) return mod;
    }
    XMPmodule module = new XMPmodule(this);
    module.inputFile(module_name);
    modules.add(module);
    return module;
  }

  public void useModule(String module_name){
    XMPsymbolTable table = getXMPsymbolTable();
    table.addUseModule(module_name);
  }

  public Vector<XMPmodule> getModules(){
    return modules;
  }

  // set current definition and set symbol table each DEF
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

  public String currentDefName() { return current_def.getDef().getName(); }

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

  // Internal ident is used in the same way as static
  public Ident declInternIdent(String name, Xtype type) {
    return declIdent(name,type,false,null);
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

  public void removeIdent(String name, Block block){

    BlockList body= current_def.getBlock().getBody();

    Xobject id_list = body.getIdentList();
    if (id_list != null){
      for (XobjArgs r = id_list.getArgs(); r != null; r = r.nextArgs()){
	if (name.equals(r.getArg().getName())){
	  id_list.removeArgs(r);
	}
      }
    }

    Xobject decls = body.getDecls();
    if (decls != null){
      for (XobjArgs r = decls.getArgs(); r != null; r = r.nextArgs()){
	if (r.getArg().Opcode() != Xcode.VAR_DECL) continue;
	if (name.equals(r.getArg().getArg(0).getName())){
	  decls.removeArgs(r);
	}
      }
    }

  }    

  public Ident declIntrinsicIdent(String name, Xtype type) {
    return Ident.Fident(name,type,false,false,null);
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

  public Ident declObjectId(String objectName, Block block, Xobject init) {
    BlockList body = current_def.getBlock().getBody();
    return body.declLocalIdent(objectName, Xtype.Fint8Type, StorageClass.FLOCAL, init);
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
      for(Xobject i: (XobjList)id_list){
	if(i.getName().equals(name)){
	  Ident id = (Ident)i;
	  String mod_name = id.getFdeclaredModule();
	  if(mod_name == null) return id;

	  /* check module */
	  XMPmodule mod = findModule(mod_name);
	  return mod.findVarIdent(name,null);
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
    XMPobject o;

    for(XobjectDef def = current_def.getDef(); 
	def != null; def = def.getParent()){
      XMPsymbolTable table = getXMPsymbolTable(def);
      if(table == null) break;
      o = table.getXMPobject(name);
      if(o != null) return o;

      // search used module
      for(String module_name: table.getUsedModules()){
	XMPmodule mod = findModule(module_name);
	o = mod.findXMPobject(name);
	if(o != null) return o;
      }
    }
    return null;
  }

  /*
   * find XMPnodes
   */
  public XMPnodes findXMPnodes(String name, Block block){
    return findXMPnodes(name);
  }

  public XMPnodes findXMPnodes(String name) {
    XMPobject o = findXMPobject(name);
    if (o != null && o.getKind() == XMPobject.NODES) 
      return (XMPnodes)o;
    return null;
  }

  /*
   * find XMPtemplate
   */
   public XMPtemplate findXMPtemplate(String name, Block block){
     return findXMPtemplate(name);
   }

   public XMPtemplate findXMPtemplate(String name) {
     XMPobject o = findXMPobject(name);
     if (o != null && o.getKind() == XMPobject.TEMPLATE) 
       return (XMPtemplate)o;
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

  /*
   * put/get XMPcorray (not yet ...)
   */
//   public void putXMPcoarray(XMPcoarray array) {
//     _globalObjectTable.putXMPcoarray(array);
//   }

//   public XMPcoarray getXMPcoarray(String name) {
//     return _globalObjectTable.getXMPcoarray(name);
//   }


  public Ident declOrGetSizeArray(Block b){

    Ident sizeArray = findVarIdent(XMP.SIZE_ARRAY_NAME, b);

    if (sizeArray == null){

      Xobject sizeExprs[] = new Xobject[2];
      sizeExprs[0] = Xcons.FindexRange(Xcons.IntConstant(0), Xcons.IntConstant(XMP.MAX_ASSUMED_SHAPE - 1));
      sizeExprs[1] = Xcons.FindexRange(Xcons.IntConstant(0), Xcons.IntConstant(XMP.MAX_DIM - 1));
      Xtype sizeArrayType = Xtype.Farray(Xtype.FintType, sizeExprs);
      sizeArray = declIdent(XMP.SIZE_ARRAY_NAME, sizeArrayType, false, b);
      sizeArray.setStorageClass(StorageClass.FCOMMON);

      Xobject decls = current_def.getBlock().getBody().getDecls();
      decls.add(Xcons.List(Xcode.F_COMMON_DECL,
  			   Xcons.List(Xcode.F_VAR_LIST,
  				      Xcons.Symbol(Xcode.IDENT, XMP.XMP_COMMON_NAME),
  				      Xcons.List(Xcons.FvarRef(sizeArray)))));
    }

    return sizeArray;
  }

  Ident getNullIdent(Block b){
    Ident xmp_null = findVarIdent("XMP_NULL", b);
    if (xmp_null == null){
      xmp_null = declObjectId("XMP_NULL", null,
			      Xcons.Cast(Xtype.voidPtrType, Xcons.IntConstant(0)));
    }
    return xmp_null;
  }

  /*
   *  wrapper -- for collect init
   */
  /***********************************
  public String getTailText() {
    return env.getTailText();
  }
  public void clearTailText() {
    env.clearTailText();
  }
  public void addTailText(String text) {
    env.addTailText(text);
  }
  ***********************************/
}
