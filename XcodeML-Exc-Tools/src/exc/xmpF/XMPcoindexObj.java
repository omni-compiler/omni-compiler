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
  final static String COMM_PUT_LIB_PREFIX = "xmpf_coarray_put";
  final static String COMM_GET_LIB_PREFIX = "xmpf_coarray_get";

  // attributes
  String name;
  Xobject obj;          // Xcode.CO_ARRAY_REF
  Xobject args;
  Xobject coindex;

  // mediator
  XMPcoarray coarray;


  //------------------------------
  //  CONSTRUCTOR
  //------------------------------
  public XMPcoindexObj(Xobject obj, XMPcoarray coarray) {
    assert (obj.Opcode() == Xcode.CO_ARRAY_REF);
    this.obj = obj;
    this.coarray = coarray;
    this.name = coarray.getName();
    this.coindex = _getCoindex();
  }

  /* find XMPcoarray and construct
   */
  public XMPcoindexObj(Xobject obj, Vector<XMPcoarray> coarrays) {
    assert (obj.Opcode() == Xcode.CO_ARRAY_REF);
    this.obj = obj;

    Xobject varRef = obj.getArg(0).getArg(0);
    if (varRef.Opcode() == Xcode.F_ARRAY_REF) {
      name = varRef.getArg(0).getArg(0).getName();
      args = varRef.getArg(1);
    } else {
      name = varRef.getName();
      args = null;
    }
    this.coarray = _findCoarrayInCoarrays(name, coarrays);
    this.coindex = _getCoindex();
  }


  private Xobject _getCoindex() {
    Xobject cosubs = obj.getArg(1).cfold(getBlock());
    Xobject[] codims = coarray.getCodimensions();
    int corank = coarray.getCorank();

    Xobject coindex = cosubs.getArg(0).cfold(getBlock());
    Xobject factor = codims[0];
    for (int i = 1; i < corank; i++) {
      if (i > 1)
        factor = Xcons.binaryOp(Xcode.MUL_EXPR,
                                factor, codims[i-1]);
      Xobject nextDim = Xcons.binaryOp(Xcode.MUL_EXPR,
                                       factor,
                                       cosubs.getArg(i));
      coindex = Xcons.binaryOp(Xcode.PLUS_EXPR,
                               coindex, nextDim);
    }
    return coindex;
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
  public Xobject genCommGetStmt() {
    Xobject funcRef = genCommProcCall(COMM_GET_LIB_PREFIX, null);
    return funcRef;
  }

  public Xobject genCommPutStmt(Xobject rhs) {
    Xobject callStmt = genCommProcCall(COMM_PUT_LIB_PREFIX, rhs);
    return callStmt;
  }

  public Xobject genCommProcCall(String prefix, Xobject rhs) {
    XMPcoarray coarray = getCoarray();
    BlockList blist = getBlockList();

    Xobject desc = coarray.getDescriptorId();
    Xobject elemLen = coarray.getElementLengthExpr();
    Xobject baseAddr = getBaseAddr();
    Xobject actualArgs = Xcons.List(desc, elemLen, baseAddr);

    int rank = coarray.getRank();
    int rankCount = 0;
    for (int i = 0; i < rank; i++) {
      if (isTripletIndex(i)) {
        ++rankCount;
        actualArgs.add(getSizeFromTriplet(i));
        actualArgs.add(getStride(i));
        actualArgs.add(getNeighboringAddr(i));
      }
    }

    actualArgs.add(getCoindex());
    if (rhs != null)                // case: put
      actualArgs.add(rhs);

    Xobject procId = blist.declLocalIdent(prefix + "_" + rankCount,
                                          BasicType.FexternalSubroutineType);

    Xobject call = Xcons.functionCall(procId, actualArgs);
    return call;
  }


  //------------------------------
  //  evaluation
  //------------------------------
  public Boolean isScalarIndex(int i) {
    Xobject arg = args.getArg(i);
    return (arg.Opcode() == Xcode.F_ARRAY_INDEX);
  }

  public Boolean isTripletIndex(int i) {
    Xobject arg = args.getArg(i);
    return (arg.Opcode() == Xcode.F_INDEX_RANGE);
  }

  public Xobject getBaseAddr() {
    return getNeighboringAddr(-1);
  }

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
    Xobject args2 = obj2.getArg(1);

    for (int i = 0; i < hostRank; i++) {
      Xobject start = getStart(i);
      if (i == rank) {
        start = Xcons.binaryOp(Xcode.PLUS_EXPR,
                               start, 
                               Xcons.IntConstant(1));
      }
      Xobject arg2 = Xcons.FarrayIndex(start);
      args2.setArg(i, arg2);
    }

    return baseAddr;
  }

  public Xobject getStart(int i) {
    Xobject arg = args.getArg(i);
    Xobject start;
    switch(arg.Opcode()) {
    case F_ARRAY_INDEX:         // scalar
      start = arg.getArg(0);
      break;
    case F_INDEX_RANGE:         // triplet
      start = arg.getArg(0);
      if (start == null) 
        start = coarray.getLbound(i);
      break;
    default:        // vector subscript is not supported
      XMP.error("internal error: unexpected Xcode: "+arg.Opcode());
      start = null;
      break;
    }
    return start;
  }

  public Xobject getEnd(int i) {
    Xobject arg = args.getArg(i);
    Xobject end;
    switch(arg.Opcode()) {
    case F_ARRAY_INDEX:         // scalar
      end = arg.getArg(0);
      break;
    case F_INDEX_RANGE:         // triplet
      end = arg.getArg(1);
      if (end == null) 
        end = coarray.getUbound(i);
      break;
    default:        // vector subscript is not supported
      XMP.error("internal error: unexpected Xcode: "+arg.Opcode());
      end = null;
      break;
    }
    return end;
  }

  public Xobject getStride(int i) {
    Xobject arg = args.getArg(i);
    Xobject stride;
    switch(arg.Opcode()) {
    case F_ARRAY_INDEX:         // scalar
      stride = Xcons.IntConstant(1);
      break;
    case F_INDEX_RANGE:         // triplet
      stride = arg.getArg(2);
      if (stride == null)
        stride = Xcons.IntConstant(1);
      break;
    default:        // vector subscript is not supported
      XMP.error("internal error: unexpected Xcode: "+arg.Opcode());
      stride = null;
      break;
    }
    return stride;
  }

  public Xobject getSizeFromTriplet(int i) {
    Xobject arg = args.getArg(i);
    Xobject size;
    switch(arg.Opcode()) {
    case F_ARRAY_INDEX:         // scalar
      size = Xcons.IntConstant(1);
      break;
    case F_INDEX_RANGE:         // triplet
      Xobject i1 = arg.getArg(0);  // can be null
      Xobject i2 = arg.getArg(1);  // can be null
      Xobject i3 = arg.getArg(2);  // can be null
      size = coarray.getSizeFromTriplet(i, i1, i2, i3);
      break;
    default:        // vector subscript is not supported
      XMP.error("internal error: unexpected Xcode: "+arg.Opcode());
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

  public String toString() {
    return "XMPcoindexObj(" + obj.toString() + ")";
  }


}
