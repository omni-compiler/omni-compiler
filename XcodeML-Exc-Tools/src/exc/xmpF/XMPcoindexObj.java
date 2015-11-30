/*
 * $TSUKUBA_Release: $
 * $TSUKUBA_Copyright:
 *  $
 */

package exc.xmpF;

import exc.object.*;
import exc.block.*;
import java.util.*;

/*
 * Madiator for coindexed object such as a(i,j)[k]
 */
public class XMPcoindexObj {
  // constants
  final static String COARRAYPUT_PREFIX = "xmpf_coarray_put";
  final static String COARRAYGET_PREFIX = "xmpf_coarray_get";

  // attributes
  String name;
  Xobject obj;               // Xcode.CO_ARRAY_REF
  Xobject subscripts;
  Xobject cosubscripts;
  int exprRank;                  // rank of the reference

  // mediator
  XMPcoarray coarray;

  //------------------------------
  //  CONSTRUCTOR
  //------------------------------
  public XMPcoindexObj(Xobject obj, XMPcoarray coarray) {
    this.obj = obj;
    name = _getName(obj);
    this.coarray = coarray;
    _initOthers();
  }

  public XMPcoindexObj(Xobject obj, ArrayList<XMPcoarray> coarrays) {
    this.obj = obj;
    name = _getName(obj);
    coarray = _findCoarrayInCoarrays(name, coarrays);
    _initOthers();
  }

  private void _initOthers() {
    subscripts = _getSubscripts(obj);
    cosubscripts = _getCosubscripts(obj);
    exprRank = _getExprRank();
    if (subscripts == null && exprRank > 0)
      subscripts = _wholeArraySubscripts(exprRank);
  }

  private String _getName(Xobject obj) {
    assert (obj.Opcode() == Xcode.CO_ARRAY_REF);
    Xobject varRef = obj.getArg(0).getArg(0);
    if (varRef.Opcode() == Xcode.F_ARRAY_REF)        // subarray
      name = varRef.getArg(0).getArg(0).getName();
    else if (varRef.Opcode() == Xcode.VAR)    // scalar or whole array
      name = varRef.getName();
    else
      XMP.fatal("broken Xcode to describe a coindexed object");
    return name;
  }

  private Xobject _getSubscripts(Xobject obj) {
    Xobject varRef = obj.getArg(0).getArg(0);
    if (varRef.Opcode() == Xcode.F_ARRAY_REF)        // subarray
      return varRef.getArg(1);
    else if (varRef.Opcode() == Xcode.VAR)    // scalar or whole array
      return null;
    else
      XMP.fatal("broken Xcode to describe a coindexed object");
    return null;
  }

  private Xobject _getCosubscripts(Xobject obj) {
    Xobject list = obj.getArg(1);
    if (list.Opcode() != Xcode.LIST)
      XMP.fatal("broken Xcode to describe a coindexed object");
    return list;
  }


  // make whole-array shape (:,:,...,:)
  //
  private Xobject _wholeArraySubscripts(int exprRank) {
    Xobject list = Xcons.List();
    for (int i = 0; i < exprRank; i++)
      list.add(Xcons.FindexRangeOfAssumedShape());
    if (list.hasNullArg())
      XMP.fatal("INTERNAL: generated null argument (_wholeArraySubscripts)");

    return list;
  }


  /*
   *  convert expr to int(expr) if expr is not surely int*4.
   */
  private Xobject _convInt4(Xobject expr) {
    if (expr.Type().isBasic() &&
        expr.Type().getBasicType() == BasicType.INT) {
      if (expr.isIntConstant()) {
        if ("4".equals(((XobjConst)expr).getFkind()))
          // found it seems a 4-byte integer literal constant
          return expr;
      }
      Xobject fkind = expr.Type().getFkind();
      if (fkind != null && fkind.getInt() == 4)
        // found it is a 4-byte integer expression
        return expr;
    }

    Ident intId = declIntIntrinsicIdent("int");
    return intId.Call(Xcons.List(expr));
  }    


  /*  Implicit type conversion if necessary
   *  ex.  RHS -> int(RHS)
   *  ex.  RHS -> real(RHS,8)
   */
  private Xobject _convRhsType(Xobject rhs)
  {
    String lhsTypeStr = coarray.getFtypeString();
    Xobject lhsKind = coarray.getFkind();

    if (lhsTypeStr == null) {
      XMP.warning("No cast function for RHS was generated because of " +
                  "unknown type of coarray LHS: " + coarray);
      return rhs;
    }

    Xtype rhsType = rhs.Type();
    if (rhsType.getKind() == Xtype.F_ARRAY)
      rhsType = rhsType.getRef();
    String rhsTypeStr = coarray.getFtypeString(rhsType.getBasicType());
    Xobject rhsKind = rhsType.getFkind();

    if (lhsKind == null) {      // case default kind
      if (rhsKind == null && lhsTypeStr.equals(rhsTypeStr)) {
        // same type and same default kind
        return rhs;
      }

      //XMP.warning("Invoked automatic type conversion ("+lhsTypeStr+")", block);
      return _convTypeByIntrinsicCall(lhsTypeStr, rhs);
    }

    if (lhsTypeStr.equals(rhsTypeStr) && rhsKind != null) {
      if (lhsKind.canGetInt() && rhsKind.canGetInt() &&
          lhsKind.getInt() == rhsKind.getInt())
        // same type and same kind
        return rhs;
    }

    //XMP.warning("Invoked automatic type conversion ("+lhsTypeStr+" with kind)", block);
    return _convTypeByIntrinsicCall(lhsTypeStr, rhs, lhsKind);
  }

  private Xobject _convTypeByIntrinsicCall(String fname, Xobject expr)
  {
    //FunctionType ftype = new FunctionType(Xtype.FunspecifiedType, Xtype.TQ_FINTRINSIC);
    FunctionType ftype = new FunctionType(Xtype.FintType, Xtype.TQ_FINTRINSIC);
    Ident fident = getEnv().declIntrinsicIdent(fname, ftype);
    return fident.Call(Xcons.List(expr));
  }

  private Xobject _convTypeByIntrinsicCall(String fname, Xobject expr, Xobject kind)
  {
    //FunctionType ftype = new FunctionType(Xtype.FunspecifiedType, Xtype.TQ_FINTRINSIC);
    FunctionType ftype = new FunctionType(Xtype.FintType, Xtype.TQ_FINTRINSIC);
    Ident fident = getEnv().declIntrinsicIdent(fname, ftype);
    return fident.Call(Xcons.List(expr, kind));
  }


  private int _getExprRank() {
    int hostRank = coarray.getRank();

    if (subscripts == null)
      return hostRank;

    int count = 0;
    for (int i = 0; i < hostRank; i++)
      if (isTripletIndex(i))
        ++count;
    return count;
  }

  private XMPcoarray _findCoarrayInCoarrays(String name,
                                            ArrayList<XMPcoarray> coarrays) {
    for (XMPcoarray coarray: coarrays) {
      if (coarray.getName() == name) {
        return coarray;
      }
    }

    if (coarray == null)
      XMP.fatal("INTERNAL: could not find coarray in coarrays. name=" + name);
    return null;
  }


  //------------------------------
  //  run
  //------------------------------
  public Xobject toFuncRef() {
    // type6 used
    Xobject mold = getObj().getArg(0).getArg(0);   // object w/o coindex
    Xobject actualArgs = _makeActualArgs_type6(mold);

    Xtype xtype = getType().copy();
    xtype.removeCodimensions();

    String funcName = COARRAYGET_PREFIX + exprRank + "d";
    Ident funcIdent = getEnv().findVarIdent(funcName, null);
    if (funcIdent == null) {
      Xtype baseType = _getBasicType(xtype);   // regard type as its basic type
                                               //// RETHINK! It might be neutral type?
      Xtype funcType = Xtype.Function(baseType);
      funcIdent = getEnv().declExternIdent(funcName, funcType);
    }                                           

    Xobject funcRef = funcIdent.Call(actualArgs);
    return funcRef;
  }

  public Xobject toCallStmt(Xobject rhs, Xobject condition) {
    // type7 used
    Xobject actualArgs =
      _makeActualArgs_type7(_convRhsType(rhs),
                            condition);

    // "scalar" or "array" or "spread" will be selected.
    String pattern = _selectCoarrayPutPattern(rhs);

    String subrName = COARRAYPUT_PREFIX + "_" + pattern;
    //// I'm not clear why this is OK and the case xmpf_coarray_proc_init is 
    //// not OK with the similar interface blocks.
    Ident subrIdent = getEnv().findVarIdent(subrName, null);
    if (subrIdent == null)
      subrIdent = getEnv().declExternIdent(subrName,
                                           BasicType.FexternalSubroutineType);
    Xobject subrCall = subrIdent.callSubroutine(actualArgs);

    return subrCall;
  }


  /***
      restriction: pattern spread is not supported yet.
   ***/
  private String _selectCoarrayPutPattern(Xobject rhs) {
    String pattern;

    if (exprRank == 0)                      // scalar = scalar
      pattern = "scalar";
    /*******************
    else if (rhs.getFrank() == 0)       // array = scalar
      pattern = "spread";
    *******************/
    else                                // array = array
      pattern = "array";

    return pattern;
  }


  /* generate actual arguments
   * cf. libxmpf/src/xmpf_coarray_put.c
   *
   * Type-5:
   *       (void *descptr, void* baseAddr, int element, int image,
   *        [void* rhs, int scheme,] int exprRank,
   *        void* nextAddr1, int count1,
   *        ...
   *        void* nextAddrN, int countN )
   *   where N is rank of the reference (0<=N<=15 in Fortran 2008).
   */
  private Xobject _makeActualArgs_type5() {
    return _makeActualArgs_type5(null, null);
  }

  private Xobject _makeActualArgs_type5(Xobject rhs, Xobject condition) {
    XMPcoarray coarray = getCoarray();

    Xobject baseAddr = getBaseAddr();
    Xobject descPtr = coarray.getDescPointerIdExpr(baseAddr);
    Xobject element = coarray.getElementLengthExpr();
    Xobject image = coarray.getImageIndex(baseAddr, cosubscripts);
    Xobject actualArgs = Xcons.List(descPtr, baseAddr, element, image);

    if (rhs != null)
      actualArgs.add(_convRhsType(rhs));
    if (condition != null)
      actualArgs.add(condition);

    actualArgs.add(Xcons.IntConstant(exprRank));

    int hostRank = coarray.getRank();
    for (int i = 0; i < hostRank; i++) {
      if (isTripletIndex(i)) {
        actualArgs.add(getNeighboringAddr(i));
        actualArgs.add(getSizeFromTriplet(i));
      }
    }

    if (actualArgs.hasNullArg())
      XMP.fatal("INTERNAL: generated null argument (_makeActualArgs_type5)");

    return actualArgs;
  }


  /* generate actual arguments
   * cf. libxmpf/src/xmpf_coarray_put.c
   *
   * Type-7:
   *       add baseAddr to Type-6 as an extra argument 
   *       in order to tell the optimization compiler the data will be referred.
   * Type-6:
   *       (void *descPtr, void* baseAddr, int element, int image,
   *        [void* rhs, int scheme,] int exprRank,
   *        void* nextAddr1, int count1,
   *        ...
   *        void* nextAddrN, int countN )
   *   where N is rank of the reference (0<=N<=15 in Fortran 2008).
   */
  private Xobject _makeActualArgs_type7(Xobject addArg1) {
    return _makeActualArgs_type7(addArg1, null);
  }
  private Xobject _makeActualArgs_type7(Xobject addArg1, Xobject addArg2) {
    Xobject actualArgs = _makeActualArgs_type6(addArg1, addArg2);
    Xobject coarrayName =  Xcons.FvarRef(coarray.getIdent());
    actualArgs.add(coarrayName);
    return actualArgs;
  }

  private Xobject _makeActualArgs_type6(Xobject addArg1) {
    return _makeActualArgs_type6(addArg1, null);
  }
  private Xobject _makeActualArgs_type6(Xobject addArg1, Xobject addArg2) {
    XMPcoarray coarray = getCoarray();

    Xobject baseAddr = getBaseAddr();
    Xobject descPtr = coarray.getDescPointerIdExpr(baseAddr);
    Xobject locBaseAddr = getBaseAddr_type6();
    Xobject element = coarray.getElementLengthExpr();
    Xobject image = coarray.getImageIndex(baseAddr, cosubscripts);
    Xobject actualArgs =
      Xcons.List(descPtr, locBaseAddr, element, image);

    if (addArg1 != null)
      actualArgs.add(addArg1);
    if (addArg2 != null)
      actualArgs.add(addArg2);

    actualArgs.add(Xcons.IntConstant(exprRank));

    int hostRank = coarray.getRank();
    for (int i = 0; i < hostRank; i++) {
      if (isTripletIndex(i)) {
        actualArgs.add(getNeighboringAddr_type6(i));
        actualArgs.add(getSizeFromTriplet(i));
      }
    }

    if (actualArgs.hasNullArg())
      XMP.fatal("INTERNAL: generated null argument (_makeActualArgs_type6)");

    return actualArgs;
  }


  // TEMPORARY VERSION
  private Xtype _getBasicType(Xtype xtype) {

    int baseTypeCode = 0;
    switch (xtype.getKind()) {
    case Xtype.F_ARRAY:
      baseTypeCode = xtype.getRef().getBasicType();
      break;
    case Xtype.BASIC:
      baseTypeCode = xtype.getBasicType();
      break;
    case Xtype.STRUCT:
      XMP.fatal("internal error: STRUCT unsupported in _getTypeSuffix()");
      break;
    default:
      XMP.fatal("internal error: unexpected kind in _getTypeSuffix(): xtype.getKind()");
      break;
    }

    return new BasicType(baseTypeCode);
  }


  //------------------------------
  //  inquirement and evaluation
  //------------------------------
  public Boolean isScalarIndex(int i) {
    Xobject subscr = subscripts.getArg(i);
    return (subscr.Opcode() == Xcode.F_ARRAY_INDEX);
  }

  public Boolean isTripletIndex(int i) {
    Xobject subscr = subscripts.getArg(i);
    return (subscr.Opcode() == Xcode.F_INDEX_RANGE);
  }

  public Xobject getBaseAddr() {
    return getNeighboringAddr(-1);
  }

  public Xobject getBaseAddr_type6() {
    return getNeighboringAddr_type6(-1);
  }

  /* get address of
      a(i[0], ..., i[hostRank-1])  for rank=-1
      a(i[0], ..., i[rank]+stride[axis], ..., i[hostRank-1]) for rank>=0
  */
  public Xobject getNeighboringAddr(int axis) {
    int hostRank = coarray.getRank();

    if (hostRank == 0) {
      // if host variable is a scalar
      return Xcons.FvarRef(coarray.getIdent());
    }

    Xobject arrElemRef = Xcons.FarrayRef(coarray.getIdent().Ref());

    for (int i = 0; i < hostRank; i++) {
      Xobject start = getStart(i);
      if (i == axis) {
        start = Xcons.binaryOp(Xcode.PLUS_EXPR,
                               start, 
                               getStride(i));
      }
      Xobject subscr = Xcons.FarrayIndex(start);
      arrElemRef.getArg(1).setArg(i, subscr);
    }

    return arrElemRef;
  }

  public Xobject getNeighboringAddr_type6(int axis) {
    Xobject arrElem = getNeighboringAddr(axis);

    Ident locId = declInt8IntrinsicIdent("loc");
    return locId.Call(Xcons.List(arrElem));
  }


  public Xobject getStart(int i) {
    Xobject subscr = subscripts.getArg(i);
    Xobject start;
    switch(subscr.Opcode()) {
    case F_ARRAY_INDEX:         // scalar
      start = subscr.getArg(0);
      break;
    case F_INDEX_RANGE:         // triplet
      start = subscr.getArg(0);
      if (start == null) 
        start = coarray.getLbound(i);
      break;
    default:        // vector subscript is not supported
      XMP.fatal("internal error: unexpected Xcode: "+subscr.Opcode());
      start = null;
      break;
    }
    return start;
  }


  public Xobject getEnd(int i) {
    Xobject subscr = subscripts.getArg(i);
    Xobject end;
    switch(subscr.Opcode()) {
    case F_ARRAY_INDEX:         // scalar
      end = subscr.getArg(0);
      break;
    case F_INDEX_RANGE:         // triplet
      end = subscr.getArg(1);
      if (end == null) 
        end = coarray.getUbound(i);
      break;
    default:        // vector subscript is not supported
      XMP.fatal("internal error: unexpected Xcode: "+subscr.Opcode());
      end = null;
      break;
    }
    return end;
  }

  public Xobject getStride(int i) {
    Xobject subscr = subscripts.getArg(i);
    Xobject stride;
    switch(subscr.Opcode()) {
    case F_ARRAY_INDEX:         // scalar
      stride = Xcons.IntConstant(1);
      break;
    case F_INDEX_RANGE:         // triplet
      stride = subscr.getArg(2);
      if (stride == null)
        stride = Xcons.IntConstant(1);
      break;
    default:        // vector subscript is not supported
      XMP.fatal("internal error: unexpected Xcode: "+subscr.Opcode());
      stride = null;
      break;
    }
    return stride;
  }


  public Xobject getSizeFromTriplet(int i) {
    Xobject subscr = subscripts.getArg(i);
    Xobject size;
    switch(subscr.Opcode()) {
    case F_ARRAY_INDEX:         // scalar
      size = Xcons.IntConstant(1);
      break;

    case F_INDEX_RANGE:         // triplet
      Xobject i1 = subscr.getArg(0);  // can be null
      Xobject i2 = subscr.getArg(1);  // can be null
      Xobject i3 = subscr.getArg(2);  // can be null
      size = coarray.getSizeFromTriplet(i, i1, i2, i3);

      if (size != null) {            // success
        // cast function for safe
        if (!size.Type().isBasic() &&
            size.Type().getBasicType() != BasicType.INT) {
          Ident intId = declIntIntrinsicIdent("int");
          size = intId.Call(Xcons.List(size));
        }
        break;
      }

      // The following codes until the break statement are not needed
      // if you could guarantee the value of size is always non-null
      // even for allocatable arrays.

      // for case a(:), 
      // use a Fortran intrinsic function size.
      if (i1 == null && i2 == null && i3 == null) {
        Xobject arg1 = Xcons.Symbol(Xcode.VAR, name);
        Xobject arg2 = Xcons.IntConstant(i + 1);
        Ident sizeId = declIntIntrinsicIdent("size");
        size = sizeId.Call(Xcons.List(arg1, arg2));
        break;
      }

      // for case a(i1:), a(:i2), a(::i3), a(i1::i3) and a(:i2:i3),
      // retry coarray.getSizeFromTriplet(i, i1, i2, i3) with
      // intrinsic functions lbound and ubound.
      if (i1 == null) i1 = coarray.getLbound(i);
      if (i2 == null) i2 = coarray.getUbound(i);
      size = coarray.getSizeFromTriplet(i, i1, i2, i3);

      if (size != null) {
        // cast function for safe
        if (!size.Type().isBasic() &&
            size.Type().getBasicType() != BasicType.INT) {
          Ident intId = declIntIntrinsicIdent("int");
          size = intId.Call(Xcons.List(size));
        }
      }
      break;

    default:        // vector subscript is not supported
      XMP.fatal("internal error: maybe vector subscript. Xcode: "
                + subscr.Opcode());
      size = null;
      break;
    }
    return size;
  }    

  public int getElementLength() {
    return coarray.getElementLength();
  }

  public Xobject getElementLengthExpr() {
    return coarray.getElementLengthExpr();
  }

  public Xobject getElementLengthExpr(Block block) {
    return coarray.getElementLengthExpr(block);
  }

  public int getTotalArraySize() {
    return getTotalArraySize(getBlock());
  }
  public int getTotalArraySize(Block block) {
    Xobject size = getTotalArraySizeExpr(block);
    if (size != null || ! size.isIntConstant())
      return -1;
    return size.getInt();
  }

  public Xobject getTotalArraySizeExpr() {
    return getTotalArraySizeExpr(getBlock());
  }
  public Xobject getTotalArraySizeExpr(Block block) {
    return obj.Type().getTotalArraySizeExpr(block);
  }


  //------------------------------
  //  tool
  //------------------------------
  private Ident declIntIntrinsicIdent(String name) { 
    FunctionType ftype = new FunctionType(Xtype.FintType, Xtype.TQ_FINTRINSIC);
    Ident ident = getEnv().declIntrinsicIdent(name, ftype);
    return ident;
  }

  private Ident declInt8IntrinsicIdent(String name) { 
    FunctionType ftype = new FunctionType(Xtype.Fint8Type, Xtype.TQ_FINTRINSIC);
    Ident ident = getEnv().declIntrinsicIdent(name, ftype);
    return ident;
  }


  //------------------------------
  //  inquire
  //------------------------------
  public Xobject getObj() {
    return obj;
  }

  public XMPcoarray getCoarray() {
    return coarray;
  }

  public XMPenv getEnv() {
    return coarray.getEnv();
  }

  public Block getBlock() {
    return coarray.fblock;
  }

  public BlockList getBlockList() {
    return coarray.fblock.getBody();
  }

  public Xobject getDecls() {
    return getBlockList().getDecls();
  }

  public String getName() {
    return getName();
  }

  public Xtype getType() {
    return coarray.getType();
  }

  public Ident getIdent() {
    return coarray.getIdent();
  }

  public String toString() {
    return "XMPcoindexObj(" + obj.toString() + ")";
  }


}
