/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
package exc.object;

/**
 * Represents function type.
 */
public class FunctionType extends Xtype
{
    Xtype ref; // return type
    Xobject param_list;
    boolean is_function_proto;
    private String fresult_name;

    public FunctionType(String id, Xtype ref, Xobject param_list, int typeQualFlags,
        boolean is_func_proto, Xobject gccAttrs, String fresult_name)
    {
        super(Xtype.FUNCTION, id, typeQualFlags, gccAttrs);
        this.ref = ref;
        this.param_list = param_list;
        this.is_function_proto = is_func_proto;
        this.fresult_name = fresult_name;
    }

    public FunctionType(Xtype ref, int typeQualFlags)
    {
        this(null, ref, null, typeQualFlags, false, null, null);
    }

    public FunctionType(Xtype ref)
    {
        this(ref, 0);
    }
    
    @Override
    public Xtype getRef()
    {
        return ref;
    }

    @Override
    public Xobject getFuncParam()
    {
        return param_list;
    }
    
    @Override
    public boolean isFuncProto()
    {
        return is_function_proto;
    }

    @Override
    public String getFuncResultName()
    {
        return fresult_name;
    }

    public void setFuncResultName(String fresult_name)
    {
        this.fresult_name = fresult_name;
    }

    public void setFuncParam(Xobject param_list)
    {
        this.param_list = param_list;
        this.is_function_proto = true;
    }

    public void setFuncParamIdList(Xobject id_list)
    {
        Xobject param_list = Xcons.List();
        for(Xobject a : (XobjList)id_list) {
            Ident id = (Ident)a;
            param_list.add(Xcons.Symbol(Xcode.IDENT, id.Type(), id.getName()));
        }
        setFuncParam(param_list);
    }
    
    @Override
    public Xtype copy(String id)
    {
        return new FunctionType(id, ref, param_list != null ? param_list.copy() : null,
            getTypeQualFlags(), is_function_proto, getGccAttributes(), fresult_name);
    }

    @Override
    public Xtype getBaseRefType()
    {
        return ref.getBaseRefType();
    }
}
