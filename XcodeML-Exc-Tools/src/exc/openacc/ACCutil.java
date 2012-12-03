package exc.openacc;

import exc.block.*;
import exc.object.*;
import java.util.*;


public class ACCutil {
  public static ACCinfo getACCinfo(Block b){
    return (ACCinfo)b.getProp(ACC.prop);
  }
  public static void setACCinfo(Block b, ACCinfo info){
    b.setProp(ACC.prop, info);
  }
  
  public static long getArrayElmtCount(Xtype type) throws ACCexception {
    if (type.isArray()) {
      ArrayType arrayType = (ArrayType)type;
      long arraySize = arrayType.getArraySize();
      if ((arraySize == 0) || (arraySize == -1)) {
        throw new ACCexception("array size should be declared statically");
      } else {
        return arraySize * getArrayElmtCount(arrayType.getRef());
      }
    } else {
      return 1;
    }
  }
  
  public static XobjList getVarDeclList(List<Ident> varIdList){
    XobjList varDeclList = Xcons.List();
    for(Ident id : varIdList){
      varDeclList.add(Xcons.List(Xcode.VAR_DECL, id, null, null));
    }
    return varDeclList;
  }
  public static XobjList getVarIdList(List<Ident> varIdList){
    XobjList objVarIdList = Xcons.IDList();
    for(Ident id : varIdList) objVarIdList.add(id);
    return objVarIdList;
  }
  
  public static Ident getMacroFuncId(String name, Xtype type) {
    return new Ident(name, StorageClass.EXTERN, Xtype.Function(type),
                     Xcons.Symbol(Xcode.FUNC_ADDR, Xtype.Pointer(Xtype.Function(type)), name), VarScope.GLOBAL);
  }
  
//  public static Ident getMacroId(String name, Xtype type) {
//    return new Ident(name, StorageClass.EXTDEF, type,
//        Xcons.Symbol(Xcode.VAR, Xtype.Pointer(type), name), VarScope.GLOBAL);
//  }

  public static boolean hasElmt(XobjList list, String string) {
    Iterator<Xobject> it = list.iterator();
    while (it.hasNext()) {
      Xobject x = it.next();
      if (x == null) continue;

      if (x.Opcode() == Xcode.STRING) {
        if (x.getString().equals(string))
          return true;
      }
    }

    return false;
  }
  
  public static boolean hasIdent(XobjList list, String string) {
    Iterator<Xobject> it = list.iterator();
    while (it.hasNext()) {
      Ident id = (Ident)it.next();
      if (id == null) {
        continue;
      }
      
      if (id.getName().equals(string)) {
        return true;
      }
    }
    
    return false;
  }
  public static int getIdentNum(XobjList list, String string){
    int count = 0;
    for(Iterator<Xobject> it = list.iterator(); it.hasNext(); count++){
      Ident id = (Ident)it.next();
      if(id == null) continue;
      if(id.getName().equals(string)){
        return count;
      }
    }
    return -1;
  }
  public static Block createFuncCallBlock(String funcName, XobjList funcArgs) {
    Ident funcId = ACCutil.getMacroFuncId(funcName, Xtype.voidType);
    return Bcons.Statement(funcId.Call(funcArgs));
  }
  
  public static XobjList getDecls(XobjList ids){
    XobjList decls = Xcons.List();
    for(Xobject x : ids){
      Ident id = (Ident)x;
      decls.add(Xcons.List(Xcode.VAR_DECL, id, null, null));
    }
    return decls;
  }
  public static XobjList getRefs(XobjList ids){
    XobjList refs = Xcons.List();
    for(Xobject x : ids){
      Ident id = (Ident)x;
      refs.add(id.Ref());
    }
    return refs;
  }
  public static XobjList getAddrs(XobjList ids){
    XobjList addrs = Xcons.List();
    for(Xobject x : ids){
      Ident id = (Ident)x;
      addrs.add(id.getAddr());
    }
    return addrs;
  }
}

