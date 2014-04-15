/* 
 * $TSUKUBA_Release: Omni XMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
package exc.xmpF;

import java.io.*;
import java.util.Vector;

import exc.object.*;
import exc.block.*;
import exc.xcodeml.*;

import xcodeml.util.XmOption;

/**
 * module structure 
 */
public class XMPmodule extends XMPenv {
  XMPenv parent;
  private String module_name;
  private Vector<Xobject> aux_info;

  private XMPsymbolTable table;

  public XMPmodule(XMPenv parent){ 
    this.parent = parent;
    is_module = true; 
    table = new XMPsymbolTable();
  }

  public String getModuleName() { return module_name; }

  public void inputFile(String module_name){
    if(XMP.debugFlag) System.out.println("module read begin: "+module_name);
    String mod_file_name = module_name+".xmod";
    Reader reader = null;

    String mod_file_name_with_path = "";
    boolean found = false;
    File mod_file;

    for (String spath: XcodeMLtools_Fmod.getSearchPath()){
      mod_file_name_with_path = spath + "/" + mod_file_name;
      mod_file = new File(mod_file_name_with_path);
      if (mod_file.exists()){
	found = true;
	break;
      }
    }

    if (!found){
      mod_file_name_with_path = mod_file_name;
      mod_file = new File(mod_file_name_with_path);
      if (mod_file.exists()){
	found = true;
      }
    }

    if (!found){
      XMP.error("module file '"+mod_file_name+"' not found");
      return;
    }

    try {
      reader = new BufferedReader(new FileReader(mod_file_name_with_path));
    } catch(Exception e){
      XMP.error("cannot open module file '"+mod_file_name+"'");
      return;
    }
    XcodeMLtools_Fmod tools = new XcodeMLtools_Fmod();
    env = tools.read(reader);
    this.module_name = tools.getModuleName();
    // xobjFile.Output(new PrintWriter(System.out));

    // process xmp pragma
    aux_info = tools.getAuxInfo();
    for(Xobject x: aux_info){
      if(x.Opcode() != Xcode.XMP_PRAGMA) continue;
      XMPpragma pragma = XMPpragma.valueOf(x.getArg(0));
      if(XMP.debugFlag) System.out.println("module pragma="+x);

      switch(pragma){
      case NODES:
	{
	  Xobject clauses = x.getArg(1);
	  XMPnodes.analyzePragma(clauses, this, null);
	}
	break;

      case TEMPLATE:
	{
	  Xobject templateDecl = x.getArg(1);
	  XobjList templateNameList = (XobjList)templateDecl.getArg(0);
	  
	  for(Xobject xx:templateNameList){
	    XMPtemplate.analyzeTemplate(xx,templateDecl.getArg(1),this,null);
	  }
	}
	break;

      case DISTRIBUTE:
	{
	  Xobject distributeDecl = x.getArg(1);
	  XobjList distributeNameList = (XobjList)distributeDecl.getArg(0);
	  Xobject distributeDeclCopy = distributeDecl.copy();

	  for(Xobject xx:distributeNameList){
	    XMPtemplate.analyzeDistribute(xx,distributeDecl.getArg(1),
					  distributeDecl.getArg(2),this,null);
	  }
	}
      break;

      case ALIGN:
	{
	  Xobject alignDecl = x.getArg(1);
	  XobjList alignNameList = (XobjList)alignDecl.getArg(0);

	  for(Xobject xx: alignNameList){
	    XMParray.analyzeAlign(xx, alignDecl.getArg(1),
				  alignDecl.getArg(2),
				  alignDecl.getArg(3),
				  this, null);
	    if(XMP.hasError()) break;
	  }
	}
      break;

      case SHADOW:
	{
	  Xobject shadowDecl = x.getArg(1);
	  XobjList shadowNameList = (XobjList) shadowDecl.getArg(0);
	  Xobject shadow_w_list = shadowDecl.getArg(1);

	  for(Xobject xx: shadowNameList){
	    XMParray.analyzeShadow(xx,shadow_w_list,this,null);
	    if(XMP.hasError()) break;
	  }
	}
	break;

      case LOCAL_ALIAS:
	XMPanalyzePragma.analyzeLocalAlias(x.getArg(1), this, null);
	break;

      default:
	XMP.error("directive '"+x.getArg(0).getName()+"' appears in module");
      }
    }
    try {
      reader.close();
    } catch(Exception e){
      XMP.error("close failed, module file '"+mod_file_name+"'");
      return;
    }
    if(XMP.debugFlag) {
      System.out.println("module read end: "+module_name);
      table.dump("module symbol");
    }
  }

  public void useModule(String module_name){
    table.addUseModule(module_name);
  }

  public XMPsymbolTable getXMPsymbolTable() {
    return table;
  }

  public void declXMPobject(XMPobject obj, Block block) {
    // in case of fortran, block is ingored
    declXMPobject(obj);
  }

  public void declXMPobject(XMPobject obj) {
    table.putXMPobject(obj);
  }

  public XMPobject findXMPobject(String name, Block block){
    // in case of fortran, block is ingored
    return findXMPobject(name);
  }

  public XMPobject findXMPobject(String name) {
    XMPobject o;

    o = table.getXMPobject(name);
    if(o != null) return o;
      
    // search used module
    for(String module_name: table.getUsedModules()){
      XMPmodule mod = parent.findModule(module_name);
      o = mod.findXMPobject(name);
      if(o != null) return o;
    }
    return null;
  }

  public Ident declIdent(String name, Xtype type, 
			 boolean is_external, Block block) {
    Xobject id_list = env.getGlobalIdentList();
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
    id_list.add(id);
    return id;
  }

  public void removeIdent(String name, Block block){

    Xobject id_list = env.getGlobalIdentList();
    if (id_list != null){
      for (XobjArgs r = id_list.getArgs(); r != null; r = r.nextArgs()){
	if (name.equals(r.getArg().getName())){
	  id_list.removeArgs(r);
	}
      }
    }

    // Xobject decls = body.getDecls();
    // if (decls != null){
    //   for (XobjArgs r = decls.getArgs(); r != null; r = r.nextArgs()){
    // 	if (r.getArg().Opcode() != Xcode.VAR_DECL) continue;
    // 	if (name.equals(r.getArg().getArg(0).getName())){
    // 	  decls.removeArgs(r);
    // 	}
    //   }
    // }

  }    

  public Ident findVarIdent(String name, Block b){
    Xobject id_list = env.getGlobalIdentList();
    for(Xobject id: (XobjList)id_list){
      if(id.getName().equals(name)){
	return (Ident) id;
      }
    }
    return null;
  }

}
