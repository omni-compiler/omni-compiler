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

    public void setFarraySizeExpr(Xobject[] sizeEexprs)
    {
        this.sizeExprs = sizeExprs;
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
     *  handling subscripts
     */
    public FindexRange getFindexRange() {
      return getFindexRange(null);
    }
    public FindexRange getFindexRange(Block block) {
      Xobject[] sizes = getFarraySizeExpr();
      FindexRange range = new FindexRange(sizes, block);
      return range;
    }

    /*
     *  get lower and upper bounds and extent
     */
    public Xobject getLbound(int i, Block block) {
      return getFindexRange(block).getLbound(i);
    }

    public Xobject[] getLbounds(Block block) {
      return getFindexRange(block).getLbounds();
    }

    public Xobject getUbound(int i, Block block) {
      return getFindexRange(block).getUbound(i);
    }

    public Xobject[] getUbounds(Block block) {
      return getFindexRange(block).getUbounds();
    }

    public Xobject getExtent(int i, Block block) {
      return getFindexRange(block).getExtent(i);
    }

    public Xobject[] getExtents(Block block) {
      return getFindexRange(block).getExtents();
    }


    /*
     *  get sizes
     */
    public Xobject getSizeFromIndexRange(Xobject range, Block block) {
      return getFindexRange(block).getSizeFromIndexRange(range);
    }

    // case: get size optionally with subscript range
    //   ex: (lb:ub)  (lb:)  (:ub)  (:)
    public Xobject getSizeFromLbUb(int i, Xobject lb, Xobject ub, Block block) {
      return getFindexRange(block).getSizeFromLbUb(i,lb,ub);
    }

    // case: get size of the range [lb, ub]
    public Xobject getSizeFromLbUb(Xobject lb, Xobject ub, Block block) {
      return getFindexRange(block).getSizeFromLbUb(lb,ub);
    }

    /*
     *  get number from triplet i1:i2:i3
     */

    // case: get number with array specification
    //       thus, i1 or i2 can be null
    public Xobject getSizeFromTriplet(int i, Xobject i1, Xobject i2, Xobject i3, Block block) {
      return getFindexRange(block).getSizeFromTriplet(i, i1, i2, i3);
    }

    // case: get number w/o array specification
    public Xobject getSizeFromTriplet(Xobject i1, Xobject i2, Xobject i3, Block block) {
      return getFindexRange(block).getSizeFromTriplet(i1, i2, i3);
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
