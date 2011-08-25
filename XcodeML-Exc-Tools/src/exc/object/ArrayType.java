/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
package exc.object;

/**
 * Represents C array type.
 */
public class ArrayType extends Xtype
{
    /** basic type or array type of next dimension */
    Xtype ref;
    /** fixed array size (when -1, size is represented by sizeExpr) */
    private long size;
    /** array size expression */
    private Xobject sizeExpr;

    protected ArrayType(int type_kind, String id, Xtype ref,
        int typeQualFlags, long size, Xobject sizeExpr, Xobject gccAttrs)
    {
        super(type_kind, id, typeQualFlags, gccAttrs);
        this.ref = ref;
        this.size = size;
        this.sizeExpr = sizeExpr;
    }

    public ArrayType(String id, Xtype ref,
        int typeQualFlags, long size, Xobject sizeExpr, Xobject gccAttrs)
    {
        this(Xtype.ARRAY, id, ref, typeQualFlags, size, sizeExpr, gccAttrs);
    }

    public ArrayType(Xtype ref, long size)
    {
        this(null, ref, 0, size, null, null);
    }

    public ArrayType(Xtype ref, Xobject sizeExpr)
    {
        this(null, ref, 0, -1, sizeExpr, null);
    }

    @Override
    public Xtype getRef()
    {
        return ref;
    }
    
    @Override
    public long getArraySize()
    {
        return size;
    }

    @Override
    public Xobject getArraySizeExpr()
    {
        return sizeExpr;
    }
   
    public void setArraySize(long s)
    {
        size = s;
    }
 
    public void setArraySizeExpr(Xobject x)
    {
        sizeExpr = x;
    }

    @Override
    public int getNumDimensions()
    {
        return ref.getNumDimensions() + 1;
    }

    @Override
    public Xtype getArrayElementType()
    {
        if(ref.isArray())
            return ref.getArrayElementType();
        else
            return ref;
    }

    @Override
    public Xtype copy(String id)
    {
        return new ArrayType(id, ref, getTypeQualFlags(),
            size, sizeExpr, getGccAttributes());
    }
}
