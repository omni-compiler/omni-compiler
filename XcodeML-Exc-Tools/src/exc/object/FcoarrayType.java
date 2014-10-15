/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 * ID=060
 */

package exc.object;
import static xcodeml.util.XmLog.fatal;
import exc.block.Block;

/**
 * Represents Fortran and XMP/Fortran coarray type.
 */
public class FcoarrayType extends FarrayType
{
    /** cosize expressions for coarray declaration */
    private Xobject[] cosizeExprs;
    
    public FcoarrayType(String id, Xtype ref, int typeQualFlags,
                        Xobject[] sizeExprs, Xobject[] cosizeExprs)
    {
        super(id, ref, typeQualFlags, sizeExprs);
        if (cosizeExprs != null)
            this.cosizeExprs = cosizeExprs;
        else
            this.cosizeExprs = new Xobject[0];
    }
    
    @Override
    public Xobject[] getFcoarraySizeExpr()
    {
        return getFarraySizeExpr();
    }

    @Override
    public Xobject[] getFcoarrayCosizeExpr()
    {
        return cosizeExprs;
    }

    public int getNumCoimensions()
    {
        return cosizeExprs.length;
    }

    @Override
    public Xtype copy(String id)
    {
        Xobject[] sizeExprs = getFarraySizeExpr();
        Xobject[] sizeExprs1 = new Xobject[sizeExprs.length];
        System.arraycopy(sizeExprs, 0, sizeExprs1, 0, sizeExprs.length);
        Xobject[] cosizeExprs1 = new Xobject[cosizeExprs.length];
        System.arraycopy(cosizeExprs, 0, cosizeExprs1, 0, cosizeExprs.length);
        return new FcoarrayType(id, getRef(), getTypeQualFlags(),
                                sizeExprs1, cosizeExprs1);
    }
}
