/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
package exc.object;
import exc.block.Block;
import exc.xmpF.XMPenv;

/**
 * methods for Fortran index range both for:
 *   - subarray indexes
 *   - coindexes
 */
public class FindexRange
{
  private int rank;
  private Xobject[] subscripts;
  private Block block = null;
  private XMPenv env = null;

  /*
   *  constructor
   */
  public FindexRange(Xobject[] subscripts) {
    rank = subscripts.length;
    this.subscripts = subscripts;
  }
  public FindexRange(Xobject[] subscripts, Block block) {
    this(subscripts);
    this.block = block;
  }
  public FindexRange(Xobject[] subscripts, Block block, XMPenv env) {
    this(subscripts, block);
    this.env = env;
  }

  public FindexRange(Xobject subscript) {
    rank = 1;
    subscripts = new Xobject[1];
    subscripts[0] = subscript;
  }
  public FindexRange(Xobject subscript, Block block) {
    this(subscript);
    this.block = block;
  }
  public FindexRange(Xobject subscript, Block block, XMPenv env) {
    this(subscript, block);
    this.env = env;
  }


  /*
   *  get lower and upper bounds of subscripts
   */
  public Xobject getLbound(int i) {
    Xobject lbound;
    if (subscripts.length <= i || subscripts[i] == null)
      return Xcons.IntConstant(1);
    if (subscripts[i].code == Xcode.F_INDEX_RANGE)
      lbound = subscripts[i].getArg(0);
    else
      return Xcons.IntConstant(1);

    if (lbound == null)
      return Xcons.IntConstant(1);
    return lbound.cfold(block);
  }

  public Xobject getUbound(int i) {
    Xobject ubound;
    if (subscripts.length <= i || subscripts[i] == null)
      return null;
    if (subscripts[i].code == Xcode.F_INDEX_RANGE)
      ubound = subscripts[i].getArg(1);
    else
      ubound = subscripts[i];

    if (ubound == null)
      return null;
    return ubound.cfold(block);
  }

  public Xobject[] getLbounds() {
    Xobject[] lbounds = new Xobject[rank];
    for (int i = 0; i < rank; i++)
      lbounds[i] = getLbound(i);
    return lbounds;
  }

  public Xobject[] getUbounds() {
    Xobject[] ubounds = new Xobject[rank];
    for (int i = 0; i < rank; i++)
      ubounds[i] = getUbound(i);
    return ubounds;
  }


  /*
   *  get extents of array specification
   */
  public Xobject getExtent(int i) {
    Xobject extent;
    if (subscripts[i].code == Xcode.F_INDEX_RANGE)
      extent = getSizeFromLbUb(subscripts[i].getArg(0),
                               subscripts[i].getArg(1));
    else
      extent = Xcons.IntConstant(1);
    return extent;
  }

  public Xobject[] getExtents() {
    Xobject[] extents = new Xobject[rank];
    for (int i = 0; i < rank; i++)
      extents[i] = getExtent(i);
    return extents;
  }


  public Xobject getTotalArraySizeExpr() {
    Xobject totalSize = Xcons.IntConstant(1);
    for (int i = 0; i < rank; i++) {
      Xobject extent = getExtent(i);
      if (extent.isZeroConstant())
        return Xcons.IntConstant(0);
      else if (extent.isIntConstant() && totalSize.isIntConstant())
        totalSize = Xcons.IntConstant(totalSize.getInt() * extent.getInt());
      else
        totalSize = Xcons.binaryOp(Xcode.MUL_EXPR, totalSize, extent);
    }
    return totalSize;
  }


  /*
   *  get size from lower and upper bounds
   */
  public Xobject getSizeFromIndexRange(Xobject range) {
    if (range == null)    // illegal
      throw new UnsupportedOperationException
        ("internal error: index range is null");

    Xobject lb = range.getArg(0);
    Xobject ub = range.getArg(1);
    return getSizeFromLbUb(lb, ub);
  }

  public Xobject getSizeFromLbUb(int i, Xobject lb, Xobject ub) {
    if (lb == null)
      lb = getLbound(i);
    if (ub == null)
      ub = getUbound(i);
    return getSizeFromLbUb(lb, ub);
  }

  public Xobject getSizeFromLbUb(Xobject lb, Xobject ub) {
    if (ub == null)    // illegal
      return null;
    ub = ub.cfold(block);

    if (lb == null)
      return ub;
    lb = lb.cfold(block);

    if (ub.equals(lb))     // it's scalar
      return Xcons.IntConstant(1);

    Xobject arg1;
    if (lb.isIntConstant()) {
      if (ub.isIntConstant()) {
        // max((ub-lb+1),0)
        int extent = ub.getInt() - lb.getInt() + 1;
        if (extent < 0)
          extent = 0;
        return Xcons.IntConstant(extent);
      } else {
        // max(ub-(lb-1),0)
        Xobject tmp = Xcons.IntConstant(lb.getInt() - 1);
        arg1 = Xcons.binaryOp(Xcode.MINUS_EXPR, ub, tmp);
      }
    } else {
      if (ub.isIntConstant()) {
        // max((ub+1)-lb,0)
        Xobject tmp = Xcons.IntConstant(ub.getInt() + 1);
        arg1 = Xcons.binaryOp(Xcode.MINUS_EXPR, tmp, lb);
      } else {
        // max(ub-lb+1,0)
        Xobject tmp = Xcons.binaryOp(Xcode.MINUS_EXPR, ub, lb);
        arg1 = Xcons.binaryOp(Xcode.PLUS_EXPR, tmp, Xcons.IntConstant(1));
      }
    }
    Xobject arg2 = Xcons.IntConstant(0);
    Ident max = env.declIntrinsicIdent("max", Xtype.FintFunctionType);
    Xobject result = max.Call(Xcons.List(arg1, arg2));
    return result.cfold(block);
  }


  /*
   *  get size from triplet
   */

  // case: get number with array specification
  //       thus, i1 or i2 can be null
  public Xobject getSizeFromTriplet(int i, Xobject i1, Xobject i2, Xobject i3) {
    if (i1 == null)
      i1 = getLbound(i);
    if (i2 == null)
      i2 = getUbound(i);
    return getSizeFromTriplet(i1, i2, i3);
  }

  // case: get number w/o array specification
  public Xobject getSizeFromTriplet(Xobject i1, Xobject i2, Xobject i3) {
    if (i3 == null)
      return getSizeFromLbUb(i1, i2);

    i3 = i3.cfold(block);
    if (i3.isIntConstant()) {
      if (i3.getInt() == 1)
        return getSizeFromLbUb(i1, i2);
      else if (i3.getInt() == -1)
        return getSizeFromLbUb(i2, i1);
    }

    if (i2 == null)    // illegal
      return null;
    i2 = i2.cfold(block);

    if (i1 == null)
      i1 = Xcons.IntConstant(1);
    else 
      i1 = i1.cfold(block);

    if (i2.equals(i1))     // it's scalar
      return Xcons.IntConstant(1);

    if (i1.isIntConstant() && i2.isIntConstant() && i3.isIntConstant()) {
      // max((i2-i1+i3)/i3,0)       if i3 > 0
      // max((i1-i2+(-i3))/(-i3),0) if i3 < 0
      int ist = i3.getInt();
      int extent = (ist > 0) ?
        (i2.getInt() - i1.getInt() + ist) / ist :
        (i1.getInt() - i2.getInt() - ist) / (-ist);
      if (extent < 0)
        extent = 0;
      return Xcons.IntConstant(extent);
    }

    // max( int((i2-i1+i3)/i3), 0 )
    Xobject e1 = Xcons.binaryOp(Xcode.MINUS_EXPR, i2, i1);
    Xobject e2 = Xcons.binaryOp(Xcode.PLUS_EXPR, e1, i3);
    Xobject arg1 = Xcons.binaryOp(Xcode.DIV_EXPR, e2, i3);
    Xobject arg2 = Xcons.IntConstant(0);
    Ident max = env.declIntrinsicIdent("max", Xtype.FintFunctionType);
    Xobject result = max.Call(Xcons.List(arg1, arg2));
    return result.cfold(block);
  }


  public int getFrank()
  {
    return rank;
  }
}
