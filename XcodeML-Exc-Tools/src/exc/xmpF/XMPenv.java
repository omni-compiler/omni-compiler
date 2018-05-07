package exc.xmpF;

import java.io.*;
import java.util.Vector;
import exc.object.*;
import exc.block.*;
import xcodeml.util.XmOption;

/** 
 * XMPenv represents the environment structure
 * this is for Fortran environment:
 *   no global declaration
 */

public class XMPenv {
  protected XobjectFile env;
  protected boolean is_module = false;

  private Vector<XMPmodule> modules = new Vector<XMPmodule>();
  private FuncDefBlock current_def;

  private final static String SYMBOL_TABLE = "XMP_PROP_XMP_SYMBOL_TABLE";
  
  public XMPenv() { }

  /**
   * Constructor with linked XobjectFile environment.
   */
  public XMPenv(XobjectFile env) {
    this.env = env;
  }

  /**
   * Return the XobjectFile which this XMPenv belongs to.
   */
  public XobjectFile getEnv() {
    return env;
  }

  /**
   * Find XMPmodule object of the give module name in this XMPenv environment.
   *  This method reads .xmod file to create the XMPmodule object for it.
   *  If the module is already read, then return it.
   */
  public XMPmodule findModule(String module_name){
    for(XMPmodule mod : modules){
      if(module_name.equals(mod.getModuleName())) return mod;
    }
    XMPmodule module = new XMPmodule(this);
    module.inputFile(module_name);
    modules.add(module);
    return module;
  }

  /**
   * add the given name of the module to 
   */
  public void useModule(String module_name){
    XMPsymbolTable table = getXMPsymbolTable();
    table.addUseModule(module_name);
  }

  /**
   * return the found modules as a vector of XMPmodule.
   */
  public Vector<XMPmodule> getModules(){
    return modules;
  }

  /**
   * Set the given definition as the current definition in this XMPenv.
   *  When setting the current definition, set XMP sybmol table associcated
   *  to the current definition.
   */
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

  /**
   * Return the crrent definition setted by "setCurrentDef".
   */
  FuncDefBlock getCurrentDef() { return current_def; }
  
  /**
   * Return wheter the current definition is a module.
   */
  public boolean currentDefIsModule() { return is_module; }

  /**
   * Return the name of the current definition.
   */
  public String currentDefName() { return current_def.getDef().getName(); }

  /**
   * Static method to get the XMP sybmol table asscoated with 
   *  the give definition.
   */
  // get symbol table bind to XobjectDef def
  public static XMPsymbolTable getXMPsymbolTable(XobjectDef def) {
    return (XMPsymbolTable)def.getProp(SYMBOL_TABLE);
  }

  /**
   *  Return XMP symbol table assocated with the current defintion.
   */
  public XMPsymbolTable getXMPsymbolTable() {
    return getXMPsymbolTable(current_def.getDef());
  }

  /* 
   * Symbol management: external func
   */
  /**
   * Declare an external identifier with the given name and type 
   *  in this environment and return the Ident object of the identifier.
   */
  public Ident declExternIdent(String name, Xtype type) {
    return declIdent(name,type,true,null);
  }

  // Internal ident is used in the same way as static
  /**
   * Declare a local (internal) identifier with the given name and type 
   *  in this environment and return the Ident object of the identifier.
   */
  public Ident declInternIdent(String name, Xtype type) {
    return declIdent(name,type,false,null);
  }

  public Ident declInternIdent(String name, Xtype type, Block b) {
    return declIdent(name,type,false,b);
  }

  // this is local
  /**
   * Declare a local (internal) identifier with the given name and type
   *  in this environment and return the Ident object of the identifier.
   *  The parameter "block" specifies the block where the identifier 
   *  is declared. 
   */
  public Ident declIdent(String name, Xtype type, Block block) {
    return declIdent(name,type,false,block);
  }

  /**
   * Declare a local (internal) identifier with the given name and type
   *  in this environment and return the Ident object of the identifier.
   *  The parameter "block" is not specified, then it is static.
   */
  // this is static
  public Ident declIdent(String name, Xtype type) {
    return declIdent(name,type,false,null);
  }

  /**
   * Declare an identifier with the given name and type.
   * the arguemnt is_external specifies whther it is declared external or not.
   * If it is declared as local identiier (is_external == false), the block
   * where the identifier is declared can be specified.
   */
  public Ident declIdent(String name, Xtype type, 
			 boolean is_external, Block block) {
    BlockList body = null;
    if (block != null)
      block = block.findParentBlockStmt();
    body = (block != null) ? block.getBody() : current_def.getBlock().getBody();

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

  /**
   * Remove the identifier specified with the name and block 
   *  where the identifier is decalred, in this XMPenv.
   */
  public void removeIdent(String name, Block block){

    if (block != null && block.removeVarIdent(name))
      return;

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

  /**
   * Declare the identifier as intrinsic function with the name and type.
   *  (use 
   */
  public Ident declIntrinsicIdent(String name, Xtype type) {
    return Ident.Fident(name,type,false,false,null);
  }

  // Id is Fint8type
  /**
   * Declare the identifer of Fortran integer 8 with the name and the block 
   *  where the identifer is declared. 
   *  This identifier is used to store object ID.
   */
  public Ident declObjectId(String objectName, Block block) {
    Xtype t = Xtype.Fint8Type;
//     if(is_module){
//       t = t.copy();
//       t.setIsFsave(true);
//     }
    return declIdent(objectName, t, block);
  }

  /**
   * Declare the identifer of Fortran integer 8 with the name and the block 
   *  where the identifer is declared. 
   *  This identifier is used to store object ID.
   *  The initial value is specified as a paramter "init".
   */
  public Ident declObjectId(String objectName, Block block, Xobject init) {
    if (block != null)
      block = block.findParentBlockStmt();
    BlockList body = ((block != null) ? block : current_def.getBlock()).getBody();
    return body.declLocalIdent(objectName, Xtype.Fint8Type, StorageClass.FLOCAL, init);
  }

  /**
   * Finalize this XMPenv. 
   */
  public void finalizeEnv() {
    env.collectAllTypes();
    env.fixupTypeRef();
  }

  // intrinsic
  /**
   * Create an identifier for intrinsic function with the type and name.
   *  (use declIntrinsicIdent ???)
   */
  public Ident FintrinsicIdent(Xtype t, String name){
    Xtype tt;
    tt = t.copy();
    tt.setIsFintrinsic(true);
    return Ident.Fident(name, tt ,false, false, env);
  }

  /*
   *  Serch symbols nested definitions in Fortran
   */
  /**
   * Search the identifier specified by the given name from the scope 
   *  of the given block.
   */
  public Ident findVarIdent(String name, Block b){
    Ident id = null;  // find ident, in this block(b) or parent
    if (b == null || (id = b.findVarIdent(name)) == null)
out:{
      for(XobjectDef def = current_def.getDef(); def != null;
	  def = def.getParent()){
        Xobject id_list = def.getDef().getArg(1);
        for(Xobject i: (XobjList)id_list){
	  if(i.getName().equals(name)){
	    id = (Ident)i;
            break out;
	  }
        }
      }
      return null;
    } /* out */

    String mod_name = id.getFdeclaredModule();
    if(mod_name == null) return id;

    /* check module */
    XMPmodule mod = findModule(mod_name);
    return mod.findVarIdent(name,null);
  }

  public Block findVarIdentBlock(String name, Block b){
    return (b != null) ? b.findVarIdentBlock(name) : null;
  }

  /*
   * put/get XMPobject (nodes and template)
   */
  /**
   * 
   */
  public void declXMPobject(XMPobject obj, Block block) {
    XMPsymbolTable table = null;
    if (block != null) {
      block = block.findParentBlockStmt();
      if (block != null) {
        table = block.getXMPsymbolTable();
      }
    }
    if (table == null)
      table = getXMPsymbolTable();
    table.putXMPobject(obj);
  }

  public XMPobject findXMPobject(String name, Block block){
    XMPobject o;

    if (block != null) {
        o = block.findXMPobject(name);
        if(o != null)
          return o;
    }

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
    XMPobject o = findXMPobject(name, block);
    if (o != null && o.getKind() == XMPobject.NODES) 
      return (XMPnodes)o;
    return null;
  }

  /*
   * find XMPtemplate
   */
   public XMPtemplate findXMPtemplate(String name, Block block){
     XMPobject o = findXMPobject(name, block);
     if (o != null && o.getKind() == XMPobject.TEMPLATE) 
       return (XMPtemplate)o;
     return null;
   }

  /*
   * decl/find XMParray
   */
  public void declXMParray(XMParray array, Block block) {
    XMPsymbolTable table = null;
    if (block != null) {
      block = block.findParentBlockStmt();
      if (block != null) {
        table = block.getXMPsymbolTable();
      }
    }
    if (table == null)
      table = getXMPsymbolTable();
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
      xmp_null = declObjectId("XMP_NULL", b,
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
