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
    _setNameAndSubscripts(obj);
    this.obj = obj;
    this.coarray = coarray;
    coindex = _getCoindex();
    rank = _getRank();
    if (subscripts == null && rank > 0)
      subscripts = _wholeArraySubscripts(rank);
  }

  /* construct and find the instance of XMPcoarray
   */
  public XMPcoindexObj(Xobject obj, Vector<XMPcoarray> coarrays) {
    _setNameAndSubscripts(obj);
    this.obj = obj;
    coarray = _findCoarrayInCoarrays(name, coarrays);
    coindex = _getCoindex();
    rank = _getRank();
    if (subscripts == null && rank > 0)
      subscripts = _wholeArraySubscripts(rank);
  }

  private void _setNameAndSubscripts(Xobject obj) {
    assert (obj.Opcode() == Xcode.CO_ARRAY_REF);
    Xobject varRef = obj.getArg(0).getArg(0);
    if (varRef.Opcode() == Xcode.F_ARRAY_REF) {        // subarray
      name = varRef.getArg(0).getArg(0).getName();
      subscripts = varRef.getArg(1);
    } else if (varRef.Opcode() == Xcode.VAR) {    // scalar or whole array
      name = varRef.getName();
      subscripts = null;
    } else {
      XMP.fatal("unexpected Xcode describing coindexed object");
    }
  }

  private Xobject _wholeArraySubscripts(int rank) {
    Xobject list = Xcons.List();
    for (int i = 0; i < rank; i++)
      list.add(Xcons.FindexRangeOfAssumedShape());

    return list;
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
      return _convInt4(coindex);

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

    return _convInt4(coindex);
  }

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

    Ident intrinsicInt =
      getEnv().declIntrinsicIdent("int", Xtype.FintFunctionType);

    return intrinsicInt.Call(Xcons.List(expr));
  }    

  /* TEMPORARY VERSION
   *  not conversion but only error checking
   */
  private Xobject _convRhsType(Xobject rhs) {
    Xtype lhsType = coarray.getFtype();
    Xobject lhsKind = coarray.getFkind();

    Xtype rhsType;
    Xobject rhsKind;

    rhsType = rhs.Type();
    if (rhsType.getKind() == Xtype.F_ARRAY)
      rhsType = rhsType.getRef();
    rhsKind = rhsType.getFkind();

    int ltype = (lhsType == null) ? 0 : (lhsType.getKind());
    int rtype = (rhsType == null) ? 0 : (rhsType.getKind());

    int lkind = (lhsKind == null) ? 0 : (lhsKind.getInt());
    int rkind = (rhsKind == null) ? 0 : (rhsKind.getInt());

    if (ltype == 0 || rtype == 0) {
      // doubtful
      XMP.warning("Automatic type conversion will not be generated " +
                  "in this coindexed assignment statement.");
      return rhs;
    }

    if (ltype != rtype) {
      // error
      XMP.error("current restriction: found coindexed assignment statement " +
                "with implicit type conversion");
      return rhs;
    }

    if (lkind != 0 && rkind != 0 && lkind != rkind) {
      // error
      XMP.error("current restriction: found coindexed assignment statement " +
                "with implicit type-kind conversion");
      return rhs;
    }

    // Though it is still doubtful in the case (lkind == 0 || rkind == ).

    return rhs;
  }


  private int _getRank() {
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

  public Xobject toCallStmt(Xobject rhs, Xobject condition) {
    // type-5 used
    Xobject actualArgs = _makeActualArgs_type5(rhs, condition);

    // "scalar" or "array" or "spread" will be selected.
    String pattern = _selectCoarrayPutPattern(rhs);

    String subrName = COARRAYPUT_PREFIX + "_" + pattern;
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

    if (rank == 0)                      // scalar = scalar
      pattern = "scalar";
    /*******************  getRank() is not supported enough.
    else if (rhs.getFrank() == 0)       // array = scalar
      pattern = "spread";
    *******************/
    else                                // array = array
      pattern = "array";

    return pattern;
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

  private Xobject _makeActualArgs_type5(Xobject rhs, Xobject condition) {
    XMPcoarray coarray = getCoarray();

    Xobject baseAddr = getBaseAddr();
    Xobject serno = coarray.getDescriptorIdExpr(baseAddr);
    Xobject element = coarray.getElementLengthExpr();
    Xobject coindex = getCoindex();

    Xobject actualArgs = Xcons.List(serno, baseAddr, element, coindex);

    if (rhs != null)
      actualArgs.add(_convRhsType(rhs));
    if (condition != null)
      actualArgs.add(condition);

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
      a(i[0], ..., i[rank]+stride[axis], ..., i[hostRank-1]) for rank>=0
  */
  public Xobject getNeighboringAddr(int axis) {
    int hostRank = coarray.getRank();

    if (hostRank == 0) {
      // host variable is scalar
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
