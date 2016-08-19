package exc.object;
import exc.block.Block;

/**
 * Represents C-struct / Fortran-type / C++-class type.
 */
public class StructType extends CompositeType
{
    private boolean is_class = false;

    public StructType(String id, XobjList id_list, int typeQualFlags, Xobject gccAttrs,
                      Xobject[] codimensions)
    {
        super(Xtype.STRUCT, id, id_list, typeQualFlags, gccAttrs, codimensions);
    }

    public StructType(String id, XobjList id_list, int typeQualFlags, Xobject gccAttrs)
    {
        this(id, id_list, typeQualFlags, gccAttrs, null);
    }

    public StructType(String id, boolean is_class, XobjList tag_names, 
                                 XobjList id_list, int typeQualFlags, Xobject gccAttrs)
    {
        super(Xtype.STRUCT, id, tag_names, id_list, typeQualFlags, gccAttrs, null);
        this.is_class = is_class;
    }

    public boolean isClass()
    {
        return is_class;
    }

    @Override
    public Xobject getTotalArraySizeExpr(Block block)
    {
        return Xcons.IntConstant(1);
    }

    @Override
    public Xobject getElementLengthExpr(Block block)
    {
      //throw new UnsupportedOperationException
      //("Restriction: could not get size of a structure");
      return null;
    }

    @Override
    public int getElementLength(Block block)
    {
        throw new UnsupportedOperationException
          ("Restriction: could not get size of a structure as integer");
    }

    @Override
    public Xtype copy(String id)
    {
        StructType t = new StructType(id, getMemberList(), getTypeQualFlags(),
                                      getGccAttributes(), copyCodimensions());
        t.original = (original != null) ? original : this;
        t.tag = (tag != null) ? tag : ((original != null) ? original.tag : null);
        return t;
    }
}
