/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
package exc.object;

/**
 * Represents enum type.
 */
public class EnumType extends Xtype
{
    /** member id list */
    private XobjList moe_list;
    /** original type (to suppress output same type) */
    private Xtype original;

    public EnumType(String id, XobjList moe_list, int typeQualFlags, Xobject gccAttrs)
    {
        super(Xtype.ENUM, id, typeQualFlags, gccAttrs);
        this.moe_list = moe_list;
    }

    @Override
    public boolean isIntegral()
    {
        return true;
    }

    @Override
    public XobjList getMemberList()
    {
        return moe_list;
    }

    @Override
    public Xtype copy(String id)
    {
        EnumType t = new EnumType(
            id, moe_list, getTypeQualFlags(), getGccAttributes());
        t.original = (original != null ? original : this);
        t.tag = tag;
        return t;
    }

    @Override
    public Xtype getOriginal()
    {
    	return original;
    }
}
