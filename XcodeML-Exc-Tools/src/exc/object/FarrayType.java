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
    public Xobject getTotalArraySizeExpr(XobjectDef def, Block block)
    {
      Xobject[] sizes = getFarraySizeExpr();
      if (sizes == null)
        throw new UnsupportedOperationException
          ("internal error: getFarraySizeExpr() failed");

      Xobject totalSize = Xcons.IntConstant(1);
      for (int i = 0; i < sizes.length; i++) {
        Xobject size1 = sizes[i];
        if (size1 == null)
          throw new UnsupportedOperationException
            ("internal error: null size of a dimension of array");

        if (size1.code == Xcode.F_INDEX_RANGE) {
          // evaluate size from lb:ub 

          Xobject lb = size1.getArg(0);
          Xobject ub = size1.getArg(1);
          Xobject st = size1.getArg(2);
          if (st != null)    // illegal in array declaration
            throw new UnsupportedOperationException
              ("internal error: stride specified in array specification");

          size1 = _getSizeFromLbUb(lb, ub, def, block);
        }

        if (size1.isZeroConstant())
          return Xcons.IntConstant(0);
        else if (size1.isIntConstant() && totalSize.isIntConstant())
          totalSize = Xcons.IntConstant(totalSize.getInt() * size1.getInt());
        else
          totalSize = Xcons.binaryOp(Xcode.MUL_EXPR, totalSize, size1);
      }
      return totalSize;
    }
  

    private Xobject _getSizeFromLbUb(Xobject lb, Xobject ub,
                                     XobjectDef def, Block block)
    {
      if (ub == null)    // illegal
        throw new UnsupportedOperationException
          ("internal error: upper-bound absent in array specification");
      ub = ub.cfold(def, block);

      if (lb == null)
        return ub;
      lb = lb.cfold(def, block);

      if (lb.isIntConstant()) {
        if (lb.getInt() == 1) {
          return ub;
        } 
        if (ub.isIntConstant()) {
          // max(ub-lb+1,0)
          int extent = ub.getInt() - lb.getInt() + 1;
          if (extent < 0) extent = 0;
          return Xcons.IntConstant(extent);
        } 
      }

      // max(ub-lb+1,0)
      Xobject e1 = Xcons.binaryOp(Xcode.MINUS_EXPR, ub, lb);
      Xobject arg1 = Xcons.binaryOp(Xcode.PLUS_EXPR, e1, Xcons.IntConstant(1));
      Xobject arg2 = Xcons.IntConstant(0);
      Xobject max = block.getBody().declLocalIdent("max", Xtype.FintFunctionType);
      Xobject result = Xcons.functionCall(max, Xcons.List(arg1, arg2));
      return result.cfold(def, block);
    }

    // not used so far
    private Xobject _getSizeFromLbUbSt(Xobject lb, Xobject ub, Xobject st,
                                       XobjectDef def, Block block)
    {
      if (st == null)
        return _getSizeFromLbUb(lb, ub, def, block);
      st = st.cfold(def, block);
      if (st.isIntConstant() && st.getInt() == 1)
        return _getSizeFromLbUb(lb, ub, def, block);

      if (ub == null)    // illegal
        throw new UnsupportedOperationException
          ("internal error: upper-bound absent in array specification");
      ub = ub.cfold(def, block);

      if (lb == null)
        lb = Xcons.IntConstant(1);
      else 
        lb = lb.cfold(def, block);

      if (lb.isIntConstant() && ub.isIntConstant() && st.isIntConstant()) {
        // max((ub-lb+st)/st,0)       if st > 0
        // max((lb-ub+(-st))/(-st),0) if st < 0
        int ist = st.getInt();
        int extent = (ist > 0) ?
          (ub.getInt() - lb.getInt() + ist) / ist :
          (lb.getInt() - ub.getInt() - ist) / (-ist);
        if (extent < 0)
          extent = 0;
        return Xcons.IntConstant(extent);
      }

      // max( int((ub-lb+st)/st), 0 )
      Xobject e1 = Xcons.binaryOp(Xcode.MINUS_EXPR, ub, lb);
      Xobject e2 = Xcons.binaryOp(Xcode.PLUS_EXPR, e1, st);
      Xobject arg1 = Xcons.binaryOp(Xcode.DIV_EXPR, e2, st);
      Xobject arg2 = Xcons.IntConstant(0);
      Xobject max = block.getBody().declLocalIdent("max", Xtype.FintFunctionType);
      Xobject result = Xcons.functionCall(max, Xcons.List(arg1, arg2));
      return result.cfold(def, block);
    }


    @Override
    public Xobject getElementLengthExpr(XobjectDef def, Block block)
    {
      return getRef().getElementLengthExpr(def, block);
    }

    @Override
    public int getElementLength(XobjectDef def, Block block)
    {
      return getRef().getElementLength(def, block);
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
