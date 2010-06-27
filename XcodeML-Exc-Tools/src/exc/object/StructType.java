/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
package exc.object;

/**
 * Represents C-struct / Fortran-type type.
 */
public class StructType extends CompositeType
{
    public StructType(String id, XobjList id_list, int typeQualFlags, Xobject gccAttrs)
    {
        super(Xtype.STRUCT, id, id_list, typeQualFlags, gccAttrs);
    }

    @Override
    public Xtype copy(String id)
    {
        StructType t = new StructType(id, getMemberList(),
            getTypeQualFlags(), getGccAttributes());
        t.original = (original != null) ? original : this;
        t.tag = (tag != null) ? tag : ((original != null) ? original.tag : null);
        return t;
    }
}
