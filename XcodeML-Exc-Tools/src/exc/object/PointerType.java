/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
package exc.object;

/**
 * Represents C pointer type.
 */
public class PointerType extends Xtype
{
    Xtype ref;

    public PointerType(String id, Xtype ref, int typeQualFlags, Xobject gccAttrs)
    {
        super(Xtype.POINTER, id, typeQualFlags, gccAttrs);
        this.ref = ref;
    }

    public PointerType(Xtype ref)
    {
        super(Xtype.POINTER);
        this.ref = ref;
    }

    @Override
    public Xtype getRef()
    {
        return ref;
    }

    @Override
    public Xtype copy(String id)
    {
        return new PointerType(id, ref, getTypeQualFlags(), getGccAttributes());
    }
}
