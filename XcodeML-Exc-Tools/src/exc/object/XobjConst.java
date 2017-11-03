package exc.object;

import exc.block.Block;

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
    public int getFrank(Block block)
    {
        return 0;
    }
}
