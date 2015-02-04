/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
package exc.object;

/**
 * represents constant elements.
 */
public abstract class XobjConst extends Xobject
{
    private String fkind;
    
    public XobjConst(Xcode code, Xtype type, String fkind)
    {
        super(code, type);
        this.fkind = fkind;
    }
    
    public final String getFkind()
    {
        return fkind;
    }

    @Override
    public int getFrank()
    {
        return 0;
    }
}
