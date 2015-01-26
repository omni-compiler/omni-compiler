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
  /* Type-4 */
  final static String COARRAYPUT_NAME = "xmpf_coarray_put_array";    // selected
  final static String COARRAYGET_NAME = "xmpf_coarray_get_array";

  /* Type-3 */
  final static String COARRAYPUT_PREFIX = "xmpf_coarray_put";
  final static String COARRAYGET_PREFIX = "xmpf_coarray_get";    // selected

  final static String PUT77_LIB_PREFIX = "xmpf_coarray_put77";
  final static String GET77_LIB_PREFIX = "xmpf_coarray_get77";
  final static String PUT90_LIB_PREFIX = "xmpf_coarray_put90";
  final static String GET90_LIB_PREFIX = "xmpf_coarray_get90";

  // attributes
  String name;
  Xobject obj;          // Xcode.CO_ARRAY_REF
  Xobject subscripts;
  Xobject coindex;
  int rank;            // rank of the reference

  // mediator
  XMPcoarray coarray;

  //------------------------------
  //  CONSTRUCTOR
  //------------------------------
  public XMPcoindexObj(Xobject obj, XMPcoarray coarray) {
    assert (obj.Opcode() == Xcode.CO_ARRAY_REF);
    this.obj = obj;
    this.coarray = coarray;
    _constructor_sub();
  }

  /* find XMPcoarray and construct
   */
  public XMPcoindexObj(Xobject obj, Vector<XMPcoarray> coarrays) {
    assert (obj.Opcode() == Xcode.CO_ARRAY_REF);
    this.obj = obj;
    Xobject varRef = obj.getArg(0).getArg(0);
    if (varRef.Opcode() == Xcode.F_ARRAY_REF) {    // array
      name = varRef.getArg(0).getArg(0).getName();
    } else {                                       // scalar
      name = varRef.getName();
    }
    coarray = _findCoarrayInCoarrays(name, coarrays);
    _constructor_sub();
  }

  private void _constructor_sub() {
    name = coarray.getName();
    coindex = _getCoindex();
    Xobject varRef = obj.getArg(0).getArg(0);
    if (varRef.Opcode() == Xcode.F_ARRAY_REF) {    // array
      subscripts = varRef.getArg(1);
    } else {                                       // scalar
      subscripts = null;
    }
    rank = _getRank();
  }    

  private Xobject _getCoindex() {
    Xobject cosubList = obj.getArg(1);
    Xobject[] codims = coarray.getCodimensions();
    int corank = coarray.getCorank();

    // coindex(1) = c[0]                                     for 1-dimensional
    // coindex(d) = coindex(d-1) + factor(d-1) * (c[d-1]-1)  for d-dimensional
    //   where factor(i) = cosize[0] * ... * cosize[i-1]  (i>0)

    Xobject c = cosubList.getArg(0).getArg(0);         // =c[0]
    Xobject coindex = c;                               // =coindex(1)
    if (corank == 1)
      return _genInt(coindex);

    Xobject cosize = coarray.getSizeFromIndexRange(codims[0]);   // =cosize[0]
    Xobject factor = cosize;                               // =factor(1)

    for (int i = 2; ; i++) {
      // coindex(i) = coindex(i-1) + factor(i-1) * (c[i-1]-1)
      c = cosubList.getArg(i-1).getArg(0);                   // =c[i-1]
      Xobject tmp1 = Xcons.binaryOp(Xcode.MINUS_EXPR, c, Xcons.IntConstant(1));
      Xobject tmp2 = Xcons.binaryOp(Xcode.MUL_EXPR, factor, tmp1);
      coindex = Xcons.binaryOp(Xcode.PLUS_EXPR, coindex, tmp2);
      if (i == corank)
        break;

      // factor(i) = factor(i-1) * cosize[i-1]
      cosize = coarray.getSizeFromIndexRange(codims[i-1]);
      factor = Xcons.binaryOp(Xcode.MUL_EXPR, factor, cosize);
    }

    return _genInt(coindex);
  }

  private Xobject _genInt(Xobject expr) {
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

    Ident intrinsicInt =
      getEnv().declIntrinsicIdent("int", Xtype.FintFunctionType);

    return intrinsicInt.Call(Xcons.List(expr));
  }    

  private int _getRank() {
    int count = 0;
    for (int i = 0; i < coarray.getRank(); i++)
      if (isTripletIndex(i)) ++count;
    return count;
  }

  private XMPcoarray _findCoarrayInCoarrays(String name,
                                            Vector<XMPcoarray> coarrays) {
    for (XMPcoarray coarray: coarrays) {
      if (coarray.getName() == name) {
        return coarray;
      }
    }
    return null;
  }


  //------------------------------
  //  run
  //------------------------------
  public Xobject toFuncRef() {
    // type-5 used
    Xobject actualArgs = _makeActualArgs_type5();

    Xtype xtype = getType().copy();
    xtype.removeCodimensions();

    String funcName = COARRAYGET_PREFIX + rank + "d";
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

  public Xobject toCallStmt(Xobject rhs, Xobject scheme) {
    // type-5 used
    Xobject actualArgs = _makeActualArgs_type5(rhs, scheme);

    String subrName = COARRAYPUT_NAME;
    Ident subrIdent = getEnv().findVarIdent(subrName, null);
    if (subrIdent == null)
      subrIdent = getEnv().declExternIdent(subrName,
                                           BasicType.FexternalSubroutineType);
    Xobject subrCall = subrIdent.callSubroutine(actualArgs);

    return subrCall;
  }


  /* generate actual arguments Type-4 and Type-5
   * cf. libxmpf/src/xmpf_coarray_put.c
   *
   * Type-4: (not used now)
   *       (int serno, void* baseAddr, int coindex,
   *        [void* rhs,] int rank,
   *        void* nextAddr1, int count1,
   *        ...
   *        void* nextAddrN, int countN )
   *
   * Type-5:
   *       (int serno, void* baseAddr, int element, int coindex,
   *        [void* rhs, int scheme,] int rank,
   *        void* nextAddr1, int count1,
   *        ...
   *        void* nextAddrN, int countN )
   *   where N is rank of the reference (0<=N<=15 in Fortran 2008).
   */
  private Xobject _makeActualArgs_type5() {
    return _makeActualArgs_type5(null, null);
  }

  private Xobject _makeActualArgs_type5(Xobject rhs, Xobject scheme) {
    XMPcoarray coarray = getCoarray();

    Xobject baseAddr = getBaseAddr();
    Xobject serno = coarray.getDescriptorIdExpr(baseAddr);
    Xobject element = coarray.getElementLengthExpr();
    Xobject coindex = getCoindex();

    Xobject actualArgs = Xcons.List(serno, baseAddr, element, coindex);

    if (rhs != null)
      actualArgs.add(rhs);
    if (scheme != null)
      actualArgs.add(scheme);

    actualArgs.add(Xcons.IntConstant(rank));

    int hostRank = coarray.getRank();
    for (int i = 0; i < hostRank; i++) {
      if (isTripletIndex(i)) {
        actualArgs.add(getNeighboringAddr(i));
        actualArgs.add(getSizeFromTriplet(i));
      }
    }

    return actualArgs;
  }

  /* generate actual arguments of F77-style interface 
   *  ( serno, elemLen, baseAddr &a(0,0,..,0),
   *    size[0],   stride[0],   neighborAddr &a(1,0,..,0),
   *    size[1],   stride[1],   neighborAddr &a(0,1,..,0),
   *    ...
   *    size[r-1], stride[r-1], neighborAddr &a(0,0,..,1),
   *    coindex )
   */
  private class _GetF77styleActualArgs {
    public Xobject args;
    public int rank;

    public _GetF77styleActualArgs() {
      XMPcoarray coarray = getCoarray();
      BlockList blist = getBlockList();

      Xobject serno = coarray.getDescriptorId();
      Xobject elemLen = coarray.getElementLengthExpr();
      Xobject baseAddr = getBaseAddr();
      args = Xcons.List(serno, elemLen, baseAddr);

      int rank = coarray.getRank();
      int count = 0;
      for (int i = 0; i < rank; i++) {
        if (isTripletIndex(i)) {
          ++count;
          args.add(getSizeFromTriplet(i));
          args.add(getStride(i));
          args.add(getNeighboringAddr(i));
        }
      }

      rank = count;
      args.add(getCoindex());
    }
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
      XMP.error("internal error: STRUCT unsupported in _getTypeSuffix()");
      break;
    default:
      XMP.error("internal error: unexpected kind in _getTypeSuffix(): xtype.getKind()");
      break;
    }

    return new BasicType(baseTypeCode);
  }



  private String _getTypeSuffix(Xtype xtype) {
    String key = null;
    switch (xtype.getKind()) {
    case Xtype.F_ARRAY:
      key = _getTypeSuffixKey(xtype.getRef());
      break;
    case Xtype.BASIC:
      key = _getTypeSuffixKey(xtype);
      break;
    case Xtype.STRUCT:
      XMP.error("internal error: STRUCT unsupported in _getTypeSuffix()");
      break;
    default:
      XMP.error("internal error: unexpected kind in _getTypeSuffix(): xtype.getKind()");
      break;
    }

    int bytes = xtype.getElementLength(getBlock());
    return key + bytes;
  }

  /// see also BasicType.getElementLength
  private String _getTypeSuffixKey(Xtype xtype) {
    String key = null;
    switch(xtype.getBasicType()) {
    case BasicType.BOOL:
      key = "l";
      break;
    case BasicType.SHORT:
    case BasicType.UNSIGNED_SHORT:
    case BasicType.INT:
    case BasicType.UNSIGNED_INT:
    case BasicType.LONG:
    case BasicType.UNSIGNED_LONG:
    case BasicType.LONGLONG:
    case BasicType.UNSIGNED_LONGLONG:
      key = "i";
      break;
    case BasicType.FLOAT:
    case BasicType.DOUBLE:
    case BasicType.LONG_DOUBLE:
      key = "r";
      break;
    case BasicType.FLOAT_COMPLEX:
    case BasicType.DOUBLE_COMPLEX:
    case BasicType.LONG_DOUBLE_COMPLEX:
      key = "z";
      break;
    case BasicType.CHAR:
    case BasicType.UNSIGNED_CHAR:
    case BasicType.F_CHARACTER:
    default:
      XMP.error("[XMPcoindexObj] unsupported type of coarray");
      break;
    }

    return key;
  }


  //------------------------------
  //  evaluation
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

  /* get address of
      a(i[0], ..., i[hostRank-1])  for rank=-1
      a(i[0], ..., i[rank]+stride[rank], ..., i[hostRank-1]) for rank>=0
  */
  public Xobject getNeighboringAddr(int rank) {
    Xobject baseAddr;
    if (obj.Opcode() == Xcode.CO_ARRAY_REF)
      baseAddr = obj.getArg(0).copy();
    else
      baseAddr = obj.copy();

    int hostRank = coarray.getRank();
    if (hostRank == 0)    // host variable is scalar
      return baseAddr;

    Xobject obj2 = baseAddr.getArg(0);
    assert (obj2.Opcode() == Xcode.F_ARRAY_REF);
    Xobject subscripts2 = obj2.getArg(1);

    for (int i = 0; i < hostRank; i++) {
      Xobject start = getStart(i);
      if (i == rank) {
        start = Xcons.binaryOp(Xcode.PLUS_EXPR,
                               start, 
                               getStride(i));
      }
      Xobject subscr2 = Xcons.FarrayIndex(start);
      subscripts2.setArg(i, subscr2);
    }

    return baseAddr;
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
      XMP.error("internal error: unexpected Xcode: "+subscr.Opcode());
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
      XMP.error("internal error: unexpected Xcode: "+subscr.Opcode());
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
      XMP.error("internal error: unexpected Xcode: "+subscr.Opcode());
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
      if (!size.Type().isBasic() &&
          size.Type().getBasicType() != BasicType.INT) {
        Ident intrinsicInt =
          getEnv().declIntrinsicIdent("int", Xtype.FintFunctionType);
        size = intrinsicInt.Call(Xcons.List(size));
      }
      break;
    default:        // vector subscript is not supported
      XMP.error("internal error: unexpected Xcode: "+subscr.Opcode());
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
  //  inquire
  //------------------------------
  public Xobject getObj() {
    return obj;
  }

  public XMPcoarray getCoarray() {
    return coarray;
  }

  public Xobject getCoindex() {
    return coindex;
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
