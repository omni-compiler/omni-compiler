package exc.xcalablemp;

import exc.block.*;
import exc.object.*;
import java.util.*;

public class XMPutil {
  private static final String LOOP_ITER = "XCALABLEMP_LOOP_ITER_PROP";

  public static XobjList castList(Xtype type, XobjList list) {
    XobjList newList = Xcons.List();
    Iterator<Xobject> it = list.iterator();
    while (it.hasNext()) {
      Xobject x = it.next();
      newList.add(Xcons.Cast(type, x));
    }

    return newList;
  }

  public static Xobject getArrayElmtsObj(Xtype type) throws XMPexception {
    if (type.isArray()) {
      ArrayType arrayType = (ArrayType)type;

      long arraySize = arrayType.getArraySize();
      if ((arraySize == 0) || (arraySize == -1)) {
        throw new XMPexception("array size should be declared statically");
      } else {
        return Xcons.binaryOp(Xcode.MUL_EXPR, Xcons.LongLongConstant(0, arraySize), getArrayElmtsObj(arrayType.getRef()));
      }
    } else {
      return Xcons.LongLongConstant(0, 1);
    }
  }

  public static String getXobjSymbolName(Xobject x) throws XMPexception {
    switch (x.Opcode()) {
    case VAR:
      return x.getSym();
    case ARRAY_REF:
    case SUB_ARRAY_REF:
      return x.getArg(0).getSym();
    default:
      throw new XMPexception("cannot get the symbol name of " + x.toString());
    }
  }

  public static XMPalignedArray findXMPalignedArray(String arrayName, XMPglobalDecl globalDecl, XMPsymbolTable localXMPsymbolTable) {
    XMPalignedArray alignedArray = null;

    if (localXMPsymbolTable != null) {
      alignedArray = localXMPsymbolTable.getXMPalignedArray(arrayName);
    }

    if (alignedArray == null) {
      alignedArray = globalDecl.getXMPalignedArray(arrayName);
    }

    return alignedArray;
  }

  public static void putLoopIter(CforBlock b, String indVarName, XobjList loopIter) {
    //    HashMap<String, XobjList> loopIterTable = (HashMap<String, XobjList>)b.getProp(LOOP_ITER);
    HashMap<String, XobjList> loopIterTable = new HashMap<String, XobjList>();
    loopIterTable.put(indVarName, (XobjList)b.getProp(LOOP_ITER));

    if (loopIterTable == null) {
      loopIterTable = new HashMap<String, XobjList>();
      b.setProp(LOOP_ITER, (Object)loopIterTable);
    }

    loopIterTable.put(indVarName, loopIter);
  }

  public static XobjList getLoopIter(CforBlock b, String indVarName) {
    //    HashMap<String, XobjList> loopIterTable = (HashMap<String, XobjList>)b.getProp(LOOP_ITER);
    HashMap<String, XobjList> loopIterTable = new HashMap<String, XobjList>();
    loopIterTable.put(indVarName, (XobjList)b.getProp(LOOP_ITER));

    if (loopIterTable == null) {
      return null;
    }
    else {
      return loopIterTable.get(indVarName);
    }
  }

  public static long getArrayElmtCount(Xtype type) throws XMPexception {
    if (type.isArray()) {
      ArrayType arrayType = (ArrayType)type;
      long arraySize = arrayType.getArraySize();
      if ((arraySize == 0) || (arraySize == -1)) {
        throw new XMPexception("array size should be declared statically");
      } else {
        return arraySize * getArrayElmtCount(arrayType.getRef());
      }
    }
    else {
      return 1;
    }
  }

  public static Xobject getArrayElmt(Xtype type, int dim) throws XMPexception {
    if(dim < 0)
      throw new XMPexception("int dim must be more than 0");

    if(!type.isArray())
      return Xcons.IntConstant(1);
    
    ArrayType arrayType = (ArrayType)type;
    for(int i=0;i<dim;i++)
      arrayType = (ArrayType)arrayType.getRef();

    long arraySize = arrayType.getArraySize();
    if(arraySize == 0 || arraySize == -1)
      return arrayType.getArraySizeExpr();
    else
      return Xcons.IntConstant((int)arrayType.getArraySize());
  }

  public static XMPpair<Ident, Xtype> findTypedVar(String name, PragmaBlock pb) throws XMPexception {
    Ident id = pb.findVarIdent(name);
    if (id == null)
      throw new XMPexception("cannot find '" + name + "'");

    Xtype type = id.Type();
    if (type == null)
      throw new XMPexception("'" + name + "' has no type");

    return new XMPpair<Ident, Xtype>(id, type);
  }

  public static boolean isIntegerType(Xtype type) {
    if (type.getKind() == Xtype.BASIC) {
      switch (type.getBasicType()) {
        case BasicType.CHAR:
        case BasicType.UNSIGNED_CHAR:
        case BasicType.SHORT:
        case BasicType.UNSIGNED_SHORT:
        case BasicType.INT:
        case BasicType.UNSIGNED_INT:
        case BasicType.LONG:
        case BasicType.UNSIGNED_LONG:
        case BasicType.LONGLONG:
        case BasicType.UNSIGNED_LONGLONG:
          return true;
        default:
          return false;
      }
    }
    else return false;
  }

  public static String getTypeName(Xtype type) {
    if (type.getKind() == Xtype.BASIC) {
      switch (type.getBasicType()) {
        case BasicType.BOOL:			return new String("BOOL");
        case BasicType.CHAR:			return new String("CHAR");
        case BasicType.UNSIGNED_CHAR:		return new String("UNSIGNED_CHAR");
        case BasicType.SHORT:			return new String("SHORT");
        case BasicType.UNSIGNED_SHORT:		return new String("UNSIGNED_SHORT");
        case BasicType.INT:			return new String("INT");
        case BasicType.UNSIGNED_INT:		return new String("UNSIGNED_INT");
        case BasicType.LONG:			return new String("LONG");
        case BasicType.UNSIGNED_LONG:		return new String("UNSIGNED_LONG");
        case BasicType.LONGLONG:		return new String("LONGLONG");
        case BasicType.UNSIGNED_LONGLONG:	return new String("UNSIGNED_LONGLONG");
        case BasicType.FLOAT:			return new String("FLOAT");
        case BasicType.DOUBLE:			return new String("DOUBLE");
        case BasicType.LONG_DOUBLE:		return new String("LONG_DOUBLE");
        case BasicType.FLOAT_IMAGINARY:		return new String("FLOAT_IMAGINARY");
        case BasicType.DOUBLE_IMAGINARY:	return new String("DOUBLE_IMAGINARY");
        case BasicType.LONG_DOUBLE_IMAGINARY:	return new String("LONG_DOUBLE_IMAGINARY");
        case BasicType.FLOAT_COMPLEX:		return new String("FLOAT_COMPLEX");
        case BasicType.DOUBLE_COMPLEX:		return new String("DOUBLE_COMPLEX");
        case BasicType.LONG_DOUBLE_COMPLEX:	return new String("LONG_DOUBLE_COMPLEX");
        default:				return null;
      }
    }
    else {
      return null;
    }
  }

  public static int countElmts(XobjList list) {
    int count = 0;
    Iterator<Xobject> it = list.iterator();
    while (it.hasNext()) {
      it.next();
      count++;
    }

    return count;
  }

  public static int countElmts(XobjList list, int constant) {
    int count = 0;
    Iterator<Xobject> it = list.iterator();
    while (it.hasNext()) {
      Xobject x = it.next();
      if (x == null) continue;
      if (x.Opcode() == Xcode.INT_CONSTANT) {
        if (x.getInt() == constant)
          count++;
      }
    }

    return count;
  }

  public static int countElmts(XobjList list, String string) {
    int count = 0;
    Iterator<Xobject> it = list.iterator();
    while (it.hasNext()) {
      Xobject x = it.next();
      if (x == null) continue;

      if (x.Opcode() == Xcode.STRING) {
        if (x.getString().equals(string))
          count++;
      }
      else if (x.Opcode() == Xcode.VAR) {
        if (x.getString().equals(string))
          count++;
      }
    }

    return count;
  }

  public static boolean hasElmt(XobjList list, int constant) {
    Iterator<Xobject> it = list.iterator();
    while (it.hasNext()) {
      Xobject x = it.next();
      if (x == null) continue;

      if (x.Opcode() == Xcode.INT_CONSTANT) {
        if (x.getInt() == constant)
          return true;
      }
    }

    return false;
  }

  public static boolean hasElmt(XobjList list, String string) {
    Iterator<Xobject> it = list.iterator();
    while (it.hasNext()) {
      Xobject x = it.next();
      if (x == null) continue;

      if (x.Opcode() == Xcode.STRING) {
        if (x.getString().equals(string))
          return true;
      }
      else if (x.Opcode() == Xcode.VAR) {
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

  public static int getFirstIndex(XobjList list, int constant) {
    int index = 0;
    Iterator<Xobject> it = list.iterator();
    while (it.hasNext()) {
      Xobject x = it.next();
      if (x != null) {
        if (x.Opcode() == Xcode.INT_CONSTANT) {
          if (x.getInt() == constant)
            return index;
        }
      }

      index++;
    }

    XMP.fatal("exception in exc.xcalablemp.XMPutil.getFirstIndex(), element does not exist");
    return -1;
  }

  public static int getFirstIndex(XobjList list, String string) {
    int index = 0;
    Iterator<Xobject> it = list.iterator();
    while (it.hasNext()) {
      Xobject x = it.next();
      if (x != null) {
        if (x.Opcode() == Xcode.STRING) {
          if (x.getString().equals(string))
            return index;
        }
	else if (x.Opcode() == Xcode.VAR) {
	  if (x.getString().equals(string))
	    return index;
	}
      }

      index++;
    }

    XMP.fatal("exception in exc.xcalablemp.XMPutil.getFirstIndex(), element does not exist");
    return -1;
  }

  public static int getLastIndex(XobjList list, int constant) {
    int elmtIndex = 0;
    boolean hasFound = false;

    int index = 0;
    Iterator<Xobject> it = list.iterator();
    while (it.hasNext()) {
      Xobject x = it.next();
      if (x != null) {
        if (x.Opcode() == Xcode.INT_CONSTANT) {
          if (x.getInt() == constant) {
            hasFound = true;
            elmtIndex = index;
          }
        }
      }

      index++;
    }

    if (hasFound) return elmtIndex;
    else{
      XMP.fatal("exception in exc.xcalablemp.XMPutil.getLastIndex(), element does not exist");
      return -1;
    }
  }

  public static int getLastIndex(XobjList list, String string) {
    int elmtIndex = 0;
    boolean hasFound = false;

    int index = 0;
    Iterator<Xobject> it = list.iterator();
    while (it.hasNext()) {
      Xobject x = it.next();
      if (x != null) {
        if (x.Opcode() == Xcode.STRING) {
          if (x.getString().equals(string)) {
            hasFound = true;
            elmtIndex = index;
          }
        }
      }

      index++;
    }

    if (hasFound) return elmtIndex;
    else{
      XMP.fatal("exception in exc.xcalablemp.XMPutil.getLastIndex(), element does not exist");
      return -1;
    }
  }

  public static Xobject foldIntConstant(Xobject expr){
    if(expr.isBinaryOp()){
      Xcode code    = expr.Opcode();
      Xobject left  = foldIntConstant(expr.left());
      Xobject right = foldIntConstant(expr.right());
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
      }
    }
    return expr;
  }

  public static Boolean is_Constant(Xobject obj, int index)
  {
    topdownXobjectIterator iter = obj.topdownIterator();
    for(iter.init(); !iter.end(); iter.next()){
      Xobject expr = iter.getXobject();
      Xcode code   = expr.Opcode();
      if(code == Xcode.PLUS_EXPR || code == Xcode.MINUS_EXPR ||
         code == Xcode.MUL_EXPR  || code == Xcode.DIV_EXPR)
        continue;
      else if(! expr.isConstant())
        return false;
    }

    return true;
  }

  public static Xobject getVar(Xobject obj)
  {
    topdownXobjectIterator iter = obj.topdownIterator();
    for(iter.init(); !iter.end(); iter.next()){
      Xobject expr = iter.getXobject();
      Xcode code   = expr.Opcode();
      if(code == Xcode.VAR){
	return expr;
      }
    }
    
    return obj;
  }

  private static boolean is_F_colon(Xobject obj)
  {
    if(!(obj instanceof XobjList))
      return false;

    if(obj.Nargs() != 3)
      return false;

    if(obj.getArg(0) == null && obj.getArg(1) == null && obj.getArg(2) == null)
      return true;
    else
      return false;
  }
  
  public static XobjList getLoopIterListFromOnRef(XobjList loopIterList)
  {
    XobjList newLoopIterList = Xcons.List();
    for(int i=0;i<loopIterList.Nargs();i++){
      Xobject obj = loopIterList.getArg(i);
      if(obj != null)
        if(!obj.equals(Xcons.StringConstant(XMP.COLON)) && !obj.equals(Xcons.StringConstant(XMP.ASTERISK)) && !is_F_colon(obj))
	  newLoopIterList.add(getVar(obj));
    }

    return newLoopIterList;
  }
  
  public static Boolean is_AllConstant(XMPnodes n)
  {
    for(int i=0;i<n.getDim();i++)
      if(is_Constant(n.getSizeAt(i), i) == false)
        return false;
          
    return true;
  }

  public static Boolean is_AllConstant(XMPtemplate t)
  {
    for(int i=0;i<t.getDim();i++)
      if(is_Constant(t.getSizeAt(i), i) == false)
        return false;

    return true;
  }

  public static String getArrayName(Xobject expr) {
    if ((expr.Opcode() == Xcode.ARRAY_REF) ||
        (expr.Opcode() == Xcode.SUB_ARRAY_REF)) {
      Xobject arrayAddr = expr.getArg(0);
      Boolean isStructure = (arrayAddr.Opcode() == Xcode.MEMBER_ARRAY_REF);
      Boolean isArray = (arrayAddr.Opcode() == Xcode.ARRAY_ADDR);
      Boolean isPointer = (arrayAddr.Opcode() == Xcode.POINTER_REF);
      Boolean isVar = (arrayAddr.Opcode() == Xcode.VAR);
      if (isStructure){

	if (arrayAddr.getArg(0).Opcode() != Xcode.ADDR_OF ||
	    arrayAddr.getArg(0).getArg(0).Opcode() != Xcode.VAR) return null;

	String structName = arrayAddr.getArg(0).getArg(0).getSym();
	String arrayName  = arrayAddr.getArg(1).getSym();
	return XMP.STRUCT + structName + "_" + arrayName;

      }
      else if (isArray || isPointer || isVar){
        return arrayAddr.getSym();
      }
      else {
	return null;
      }
    }
    else {
      XMP.fatal("cannot find array ref");
      return null;
    }
  }
}
  
