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
    public Xobject getFtotalSizeExpr()
    {
        Xobject[] sizes = getFarraySizeExpr();
        Xobject totalSize = null;
        for (Xobject size1: sizes) {
            if (size1 == null)
                return null;
            if (totalSize == null)
                totalSize = size1;
            else
                totalSize = Xcons.binaryOp(Xcode.MUL_EXPR,
                                           totalSize, size1);
        }
        if (totalSize == null)
            return Xcons.IntConstant(1);
        return totalSize;
    }

    @Override
    public void convertFindexRange(boolean extendsLowerBound, boolean extendsUpperBound, Block b)
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
