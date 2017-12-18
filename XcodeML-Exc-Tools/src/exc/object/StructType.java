package exc.object;
import exc.block.Block;

/**
 * Represents C-struct / Fortran-type / C++-class type.
 */
public class StructType extends CompositeType
{
    private boolean is_class = false;
    protected XobjList fTypeParams;
    protected Xobject finalProcedure;

    public StructType(String id, String parent_id, XobjList id_list, long typeQualFlags, Xobject gccAttrs,
                      Xobject[] codimensions)
    {
        super(Xtype.STRUCT, id, parent_id, id_list, typeQualFlags, gccAttrs, codimensions);
    }

    public StructType(String id, String parent_id, XobjList id_list, XobjList proc_list, long typeQualFlags,
		      Xobject gccAttrs, XobjList typeParams, Xobject finalProcedure)
    {
        this(id, parent_id, id_list, typeQualFlags, gccAttrs, (Xobject[])null);
        this.proc_list = proc_list;
        this.fTypeParams = typeParams;
	this.finalProcedure = finalProcedure;
    }

    public StructType(String id, boolean is_class, XobjString tag_names, 
                                 XobjList id_list, long typeQualFlags, Xobject gccAttrs)
    {
        super(Xtype.STRUCT, id, null, tag_names, id_list, typeQualFlags, gccAttrs, null);
        this.is_class = is_class;
    }

    public boolean isClass()
    {
        return is_class;
    }

    public XobjList getFTypeParams()
    {
        return fTypeParams;
    }

    public Xobject getFinalProcedure()
    {
        return finalProcedure;
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
        StructType t = new StructType(id, null, getMemberList(), getTypeQualFlags(),
                                      getGccAttributes(), copyCodimensions());
        t.original = (original != null) ? original : this;
        t.tag = (tag != null) ? tag : ((original != null) ? original.tag : null);
        return t;
    }
}
