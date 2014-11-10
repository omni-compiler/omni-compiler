/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
package exc.object;
import exc.block.Block;

/**
 * Represents Fortran array type.
 */
public class FarrayType extends Xtype
{
    /** basic type */
    private Xtype ref;
    
    /** size expressions (F_INDEX_RANGE) */
    private Xobject[] sizeExprs;
    
    /** constructor */
    public FarrayType(String id, Xtype ref, int typeQualFlags, Xobject[] sizeExprs,
                      Xobject[] codimensions)
    {
        super(Xtype.F_ARRAY, id, typeQualFlags, null, codimensions);
        this.ref = ref;
        this.sizeExprs = sizeExprs;
    }

    public FarrayType(String id, Xtype ref, int typeQualFlags, Xobject[] sizeExprs)
    {
        this(id, ref, typeQualFlags, sizeExprs, null);
    }
    
    @Override
    public Xtype getRef()
    {
        return ref;
    }
    
    @Override
    public void setRef(Xtype ref)
    {
        this.ref = ref;
    }
    
    @Override
    public Xobject[] getFarraySizeExpr()
    {
        return sizeExprs;
    }

    @Override
    public Xobject getTotalArraySizeExpr(Block block)
    {
      Xobject[] sizes = getFarraySizeExpr();
      if (sizes == null)
        throw new UnsupportedOperationException
          ("internal error: getFarraySizeExpr() failed");

      Xobject totalSize = Xcons.IntConstant(1);
      for (int i = 0; i < sizes.length; i++) {
        Xobject extent = getExtent(i, block);
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
     *  get lower-bound of array specification
     */
    public Xobject getLbound(int i, Block block) {
      Xobject[] sizes = getFarraySizeExpr();

      Xobject lbound;
      if (sizes[i].code == Xcode.F_INDEX_RANGE)
        lbound = sizes[i].getArg(0);
      else
        lbound = sizes[i];

      return lbound.cfold(block);
    }

    public Xobject[] getLbounds(Block block) {
      Xobject[] sizes = getFarraySizeExpr();
      int n = sizes.length;
      Xobject[] lbounds = new Xobject[n];

      for (int i = 0; i < n; i++)
        lbounds[i] = getLbound(i, block);

      return lbounds;
    }

    /*
     *  get upper-bound of array specification
     */
    public Xobject getUbound(int i, Block block) {
      Xobject[] sizes = getFarraySizeExpr();

      Xobject ubound;
      if (sizes[i].code == Xcode.F_INDEX_RANGE)
        ubound = sizes[i].getArg(1);
      else
        ubound = sizes[i];

      return ubound.cfold(block);
    }

    public Xobject[] getUbounds(Block block) {
      Xobject[] sizes = getFarraySizeExpr();
      int n = sizes.length;
      Xobject[] ubounds = new Xobject[n];

      for (int i = 0; i < n; i++)
        ubounds[i] = getUbound(i, block);

      return ubounds;
    }

    /*
     *  get extents of array specification
     */
    public Xobject getExtent(int i, Block block) {
      Xobject[] sizes = getFarraySizeExpr();

      Xobject extent;
      if (sizes[i].code == Xcode.F_INDEX_RANGE)
        extent = getSizeFromLbUb(sizes[i].getArg(0),
                                 sizes[i].getArg(1), block);
      else
        extent = Xcons.IntConstant(1);

      return extent;
    }

    public Xobject[] getExtents(Block block) {
      Xobject[] sizes = getFarraySizeExpr();
      int n = sizes.length;
      Xobject[] extents = new Xobject[n];

      for (int i = 0; i < n; i++)
        extents[i] = getExtent(i, block);

      return extents;
    }


    // case: get size optionally with subscript range
    public Xobject getSizeFromLbUb(int i, Xobject lb, Xobject ub, Block block)
    {
      if (lb == null)
        lb = getLbound(i, block);
      if (ub == null)
        ub = getUbound(i, block);

      return getSizeFromLbUb(lb, ub, block);
    }


    // case: get size of the range [lb, ub]
    public Xobject getSizeFromLbUb(Xobject lb, Xobject ub, Block block)
    {
      if (ub == null)    // illegal
        throw new UnsupportedOperationException
          ("internal error: upper-bound is null");
      ub = ub.cfold(block);

      if (lb == null)
        return ub;
      lb = lb.cfold(block);

      if (ub.equals(lb))     // it's scalar
        return Xcons.IntConstant(1);

      if (lb.isIntConstant()) {
        if (lb.getInt() == 1) {
          return ub;
        } 
        if (ub.isIntConstant()) {
          // max(ub-lb+1,0)
          int extent = ub.getInt() - lb.getInt() + 1;
          if (extent < 0)
            extent = 0;
          return Xcons.IntConstant(extent);
        } 
      }

      // max(ub-lb+1,0)
      Xobject e1 = Xcons.binaryOp(Xcode.MINUS_EXPR, ub, lb);
      Xobject arg1 = Xcons.binaryOp(Xcode.PLUS_EXPR, e1, Xcons.IntConstant(1));
      Xobject arg2 = Xcons.IntConstant(0);
      Xobject max = block.getBody().declLocalIdent("max", Xtype.FintFunctionType);
      Xobject result = Xcons.functionCall(max, Xcons.List(arg1, arg2));
      return result.cfold(block);
    }


    /*
     *  get number from triplet i1:i2:i3
     */

    // case: get number with array specification
    //       thus, i1 or i2 can be null
    public Xobject getSizeFromTriplet(int i, Xobject i1, Xobject i2, Xobject i3,
                                      Block block)
    {
      if (i1 == null)
        i1 = getLbound(i, block);
      if (i2 == null)
        i2 = getUbound(i, block);
      return getSizeFromTriplet(i1, i2, i3, block);
    }

    // case: get number w/o array specification
    public Xobject getSizeFromTriplet(Xobject i1, Xobject i2, Xobject i3,
                                      Block block)
    {
      if (i3 == null)
        return getSizeFromLbUb(i1, i2, block);

      i3 = i3.cfold(block);
      if (i3.isIntConstant()) {
        if (i3.getInt() == 1)
          return getSizeFromLbUb(i1, i2, block);
        else if (i3.getInt() == -1)
          return getSizeFromLbUb(i2, i1, block);
      }

      if (i2 == null)    // illegal
        throw new UnsupportedOperationException
          ("internal error: upper-bound absent in array specification");
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
      Xobject max = block.getBody().declLocalIdent("max", Xtype.FintFunctionType);
      Xobject result = Xcons.functionCall(max, Xcons.List(arg1, arg2));
      return result.cfold(block);
    }


    @Override
    public Xobject getElementLengthExpr(Block block)
    {
      return getRef().getElementLengthExpr(block);
    }

    @Override
    public int getElementLength(Block block)
    {
      return getRef().getElementLength(block);
    }


    @Override
    public void convertFindexRange(boolean extendsLowerBound,
                                   boolean extendsUpperBound, Block b)
    {
        if(isFassumedShape()) {
            for(int i = 0; i < sizeExprs.length; ++i) {
                Xobject s = Xcons.FindexRangeOfAssumedShape();
                if(extendsLowerBound)
                    s.setArg(0, sizeExprs[i].getArg(0));
                sizeExprs[i] = s;
            }
        } else if(isFassumedSize()) {
            for(int i = 0; i < sizeExprs.length; ++i) {
                Xobject s, s0 = sizeExprs[i];
                if(i == sizeExprs.length - 1) {
                    s = Xcons.FindexRangeOfAssumedSize();
                } else {
                    Xobject lb = Xcons.Reduce(s0.getArg(0), b);
                    Xobject ub = Xcons.Reduce(s0.getArg(1), b);
                    Xobject st = Xcons.Reduce(s0.getArgOrNull(2), b);
                    s = Xcons.FindexRange(lb, ub, st);
                }
                sizeExprs[i] = s;
            }
        } else {
            for(int i = 0; i < sizeExprs.length; ++i) {
                Xobject s0 = sizeExprs[i];
                Xobject lb = Xcons.Reduce(s0.getArg(0), b);
                Xobject ub = Xcons.Reduce(s0.getArg(1), b);
                Xobject st = Xcons.Reduce(s0.getArgOrNull(2), b);
                sizeExprs[i] = Xcons.FindexRange(lb, ub, st);
            }
        }
    }
    
    /** convert to assumed shape array. */
    public void convertToAssumedShape()
    {
        for(int i = 0; i < sizeExprs.length; ++i) {
            Xobject s = Xcons.FindexRangeOfAssumedShape();
            sizeExprs[i] = s;
        }
    }
    
    @Override
    public boolean isFassumedSize()
    {
        for(Xobject s : sizeExprs) {
            Xobject x = s.getArgOrNull(4);
            if(x != null && x.getInt() == 1)
                return true;
        }
        return false;
    }

    @Override
    public boolean isFassumedShape()
    {
        for(Xobject s : sizeExprs) {
            Xobject x = s.getArgOrNull(3);
            if(x != null && x.getInt() == 1)
                return true;
        }
        return false;
    }

    @Override
    public boolean isFfixedShape()
    {
        return !isFassumedShape() && !isFassumedSize();
    }
    
    @Override
    public int getNumDimensions()
    {
        return sizeExprs.length;
    }
    
    @Override
    public Xtype copy(String id)
    {
        Xobject[] sizeExprs1 = new Xobject[sizeExprs.length];
        System.arraycopy(sizeExprs, 0, sizeExprs1, 0, sizeExprs.length);
        return new FarrayType(id, ref, getTypeQualFlags(), sizeExprs1,
                              copyCodimensions());
    }
    
    @Override
    public Xtype getBaseRefType()
    {
        return ref.getBaseRefType();
    }
}
