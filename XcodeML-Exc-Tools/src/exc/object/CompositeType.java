package exc.object;

/**
 * Represents struct/union or Fortran 'type' type.
 */
public abstract class CompositeType extends Xtype
{
    private XobjString tagNames;
    /** member id list */
    private XobjList id_list;
    /** member function list */
    protected XobjList proc_list;
    /** original type (to suppress output same type) */
    protected CompositeType original;
    /** Fortran2003 : type extension. */
    protected String parent_type_id;

    protected CompositeType(int type_kind, String id, String parent_id, XobjString tag_names, XobjList id_list, XobjList proc_list,
                            long typeQualFlags, Xobject gccAttrs, Xobject[] codimensions)
    {
        super(type_kind, id, typeQualFlags, gccAttrs, codimensions);
        if(id_list == null)
            id_list = Xcons.List();
        this.id_list = id_list;
        this.proc_list = proc_list;
        this.tagNames = tag_names;
        this.parent_type_id = parent_id;
    }

    protected CompositeType(int type_kind, String id, String parent_id, XobjString tag_names, XobjList id_list,
                            long typeQualFlags, Xobject gccAttrs, Xobject[] codimensions)
    {
        this(type_kind, id, parent_id, tag_names, id_list, null, typeQualFlags, gccAttrs, codimensions);
    }

    protected CompositeType(int type_kind, String id, String parent_id, XobjList id_list, long typeQualFlags,
                            Xobject gccAttrs, Xobject[] codimensions)
    {
        this(type_kind, id, parent_id, null, id_list, typeQualFlags, gccAttrs, codimensions);
    }

    protected CompositeType(int type_kind, String id, XobjList id_list, long typeQualFlags,
                            Xobject gccAttrs)
    {
        this(type_kind, id, null, id_list, typeQualFlags, gccAttrs, null);
    }


    /** return parent type id */
    public final String parentId()
    {
        return  parent_type_id;
    }

    public XobjString getTagNames()
    {
        return tagNames;
    }

    @Override
    public final XobjList getMemberList()
    {
        return id_list;
    }

    @Override
    public final XobjList getProcList()
    {
        return proc_list;
    }

    @Override
    public Xtype getMemberType(String member)
    {
        for(Xobject a : id_list) {
            Ident ident = (Ident)a;
            if(member.equals(ident.getName())) {
                return ident.Type();
            }
        }
        
        return null;
    }

    @Override
    public Xtype getBaseRefType()
    {
    	if(original != null)
        	return original.getBaseRefType();
    	if(copied != null)
    		return copied.getBaseRefType();
    	return this;
    }
    
    @Override
    public Xtype getOriginal()
    {
    	return original;
    }

    @Override
    /** Fortran: return if the type extends parent type */
    public boolean isExtended()
    {
        return parent_type_id != null;
    }

}
