/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
package exc.object;

public class UnionType extends CompositeType
{
    public UnionType(String id, XobjList id_list, int typeQualFlags, Xobject gccAttrs)
    {
        super(Xtype.UNION, id, id_list, typeQualFlags, gccAttrs);
    }

    @Override
    public Xtype copy(String id)
    {
        UnionType t = new UnionType(id, getMemberList(), getTypeQualFlags(), getGccAttributes());
        t.original = (original != null ? original : this);
        t.tag = tag;
        return t;
    }
}
