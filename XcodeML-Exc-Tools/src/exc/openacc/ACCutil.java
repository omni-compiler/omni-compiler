package exc.openacc;

import exc.block.*;
import exc.object.*;


class ACCutil {
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
  
  public static Xobject getArrayElmtCountObj(Xtype type) throws ACCexception {
    if (type.isArray()) {
      ArrayType arrayType = (ArrayType)type;
      long arraySize = arrayType.getArraySize();
      Xobject arraySizeObj;
      if ((arraySize == 0)){// || (arraySize == -1)) {
        throw new ACCexception("array size should be declared statically");
      }else if(arraySize==-1){
        arraySizeObj = arrayType.getArraySizeExpr();
      }else{
        arraySizeObj = Xcons.LongLongConstant(0, arraySize);
      }
      return Xcons.binaryOp(Xcode.MUL_EXPR, arraySizeObj, getArrayElmtCountObj(arrayType.getRef()));
    } else {
      return Xcons.IntConstant(1);
    }
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
    for (Xobject x : list) {
      if (x == null) continue;

      if (x.Opcode() == Xcode.STRING) {
        if (x.getString().equals(string))
          return true;
      }
    }

    return false;
  }
  
  public static boolean hasIdent(XobjList list, String string) {
    for (Xobject aList : list) {
      Ident id = (Ident) aList;
      if (id == null) {
        continue;
      }

      if (id.getName().equals(string)) {
        return true;
      }
    }
    
    return false;
  }
  
  public static Ident getIdent(XobjList list, String string){
    for (Xobject aList : list) {
      Ident id = (Ident) aList;
      if (id == null) {
        continue;
      }

      if (id.getName().equals(string)) {
        return id;
      }
    }
    return null;
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
  public static Xobject foldIntConstant(Xobject exp){
    if(exp.isBinaryOp()){
      Xcode code = exp.Opcode();
      Xobject left = foldIntConstant(exp.left());
      Xobject right = foldIntConstant(exp.right());
      if(left.isIntConstant() && right.isIntConstant()){
        switch(code){
        case PLUS_EXPR:
          return Xcons.IntConstant(left.getInt() + right.getInt());
        case MINUS_EXPR:
          return Xcons.IntConstant(left.getInt() - right.getInt());
        case MUL_EXPR:
          return Xcons.IntConstant(left.getInt() * right.getInt());
        case DIV_EXPR:
          return Xcons.IntConstant(left.getInt() / right.getInt());
        }
      }//else if(left.Opcode()==Xcode.LONGLONG_CONSTANT && right.getLong())
    }
    return exp;
  }
  
  public static Xobject foldIntConstant_mod(Xobject exp){
    if(exp.isBinaryOp()){
      Xcode code = exp.Opcode();
      Xobject lhs = foldIntConstant(exp.left());
      Xobject rhs = foldIntConstant(exp.right());
      boolean isLhsConstant = lhs.isIntConstant() || lhs.Opcode() == Xcode.LONGLONG_CONSTANT;
      boolean isRhsConstant = rhs.isIntConstant() || rhs.Opcode() == Xcode.LONGLONG_CONSTANT;
      if(isLhsConstant && isRhsConstant){
        long lhsValue = lhs.isIntConstant()? lhs.getInt() : getLong(lhs);
        long rhsValue = rhs.isIntConstant()? rhs.getInt() : getLong(rhs);
        long value;
        
        switch(code){
        case PLUS_EXPR:
          value = lhsValue + rhsValue; break;
        case MINUS_EXPR:
          value = lhsValue - rhsValue; break;
        case MUL_EXPR:
          value = lhsValue * rhsValue; break;
        case DIV_EXPR:
          value = lhsValue / rhsValue; break;
        case MOD_EXPR:
          value = lhsValue % rhsValue; break;
        default:
          
        }
        
      }//else if(left.Opcode()==Xcode.LONGLONG_CONSTANT && right.getLong())
    }
    return exp;
  }
  public static long getLong(Xobject longObj){
    long low = longObj.getLongLow();
    long high = longObj.getLongHigh();
    
    
    return high << 32 + low;
  }
  
  public static String removeExtension(String name){
	int dotPos = name.lastIndexOf(".");
	  if(dotPos != -1){
	    return name.substring(0, dotPos);
	  }
    return name; 
  }
  
  public static Block createFuncCallBlockWithArrayRange(String funcName, XobjList args, XobjList arrayArgs)
  {
    XobjList funcArgs = (args != null)? (XobjList)args.copy() : Xcons.List();
    int i = 0;
    BlockList body = Bcons.emptyBody();
    XobjList declList = Xcons.List();
    for(Xobject x : arrayArgs){
      XobjList arrayElements = (XobjList)x;
      Ident id = body.declLocalIdent("_ACC_funcarg_"+i, Xtype.Array(Xtype.unsignedlonglongType, null));
      id.setIsDeclared(true);
      declList.add(Xcons.List(Xcode.VAR_DECL, id, arrayElements, null));
      funcArgs.add(id.Ref());
      i++;
    }
    
    body.setDecls(declList);
    body.add(ACCutil.createFuncCallBlock(funcName, funcArgs));
    return Bcons.COMPOUND(body);
  }
}

