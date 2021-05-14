package exc.block;

import exc.object.*;

import java.lang.reflect.Method;
import java.lang.reflect.InvocationTargetException;

public class METAXblock extends PragmaBlock {

  Class<?> metaxClass;
  Object metaxObj;

  Method run;
  Method runDecl;

  METAXblock(Xcode code, String pragma, Xobject args, BlockList body){
    super(code, pragma, args, body);
    try {
      metaxClass = Class.forName(pragma);
    } catch (ClassNotFoundException e){
      e.printStackTrace();
    }
    try {
      metaxObj = metaxClass.getDeclaredConstructor().newInstance();
    } catch (InstantiationException e){
      e.printStackTrace();
    } catch (IllegalAccessException e){
      e.printStackTrace();
    } catch (NoSuchMethodException e){
      e.printStackTrace();
    } catch (InvocationTargetException e){
      e.printStackTrace();
    }
    try {
      run = metaxClass.getMethod("run", BlockList.class, XobjList.class, METAXblock.class);
    } catch (NoSuchMethodException e){
      e.printStackTrace();
    }
    try {
      runDecl = metaxClass.getMethod("runDecl", BlockList.class, XobjList.class, METAXblock.class);
    } catch (NoSuchMethodException e){
      e.printStackTrace();
    }

  }

  public void run(){
    try {
      run.invoke(metaxObj, body, args, this);
    } catch (IllegalAccessException e){
      e.printStackTrace();
    } catch (InvocationTargetException e){
      e.printStackTrace();
    }
  }

  public void runDecl(){
    try {
      runDecl.invoke(metaxObj, body, args, this);
    } catch (IllegalAccessException e){
      e.printStackTrace();
    } catch (InvocationTargetException e){
      e.printStackTrace();
    }
  }

  public Ident declIdent(String name, Xtype type){

    Block block_decl = this.findParentDeclBlock();
    //BlockList body_decl = (block != null) ? block.getBody() : current_def.getBlock().getBody();
    BlockList body_decl = block_decl.getBody();

    // Xobject id_list = body.getIdentList();
    // if (id_list != null){
    //   for (Xobject o : (XobjList)id_list){
    // 	if (name.equals(o.getName())){
    // 	  if (!type.equals(o.Type()))
    // 	    XMP.fatal("declIdent: duplicated declaration: "+name);
    // 	  return (Ident)o;
    // 	}
    //   }
    // }

    Ident id = Ident.FidentNotExternal(name, type);
    body_decl.addIdent(id);

    id.setIsDeclared(true);
    XobjList declList = (XobjList)body_decl.getDecls();
    if (declList == null){
      declList = Xcons.List();
      body_decl.setDecls(declList);
    }
    declList.add(Xcons.List(Xcode.VAR_DECL, id, null, null));
    
    return id;

  }
    
}
