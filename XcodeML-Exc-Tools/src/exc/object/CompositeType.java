package exc.object;

/**
 * Represents struct/union or Fortran 'type' type.
 */
public abstract class CompositeType extends Xtype
{
    /** member id list */
    private XobjList id_list;
    /** original type (to suppress output same type) */
    protected CompositeType original;

    protected CompositeType(int type_kind, String id, XobjList id_list, int typeQualFlags,
                            Xobject gccAttrs, Xobject[] codimensions)
    {
        super(type_kind, id, typeQualFlags, gccAttrs, codimensions);
        if(id_list == null)
            id_list = Xcons.List();
        this.id_list = id_list;
    }

    protected CompositeType(int type_kind, String id, XobjList id_list, int typeQualFlags,
                            Xobject gccAttrs)
    {
        this(type_kind, id, id_list, typeQualFlags, gccAttrs, null);
    }


    @Override
    public final XobjList getMemberList()
    {
        return id_list;
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
}
