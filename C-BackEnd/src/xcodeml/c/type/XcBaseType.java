/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
package xcodeml.c.type;

import xcodeml.c.obj.XcNode;

/**
 * base types
 */
public abstract class XcBaseType extends XcType
{
    private XcBaseTypeEnum _baseTypeEnum;
    
    public XcBaseType()
    {
    }
    
    public XcBaseType(XcBaseTypeEnum baseTypeEnum, String typeId)
    {
        super(XcTypeEnum.BASETYPE, typeId);
        _baseTypeEnum = baseTypeEnum;
    }
    
    public XcBaseTypeEnum getBaseTypeEnum()
    {
        return _baseTypeEnum;
    }

    @Override
    public final void addChild(XcNode child)
    {
        throw new IllegalArgumentException(child.getClass().getName());
    }

    @Override
    public final void checkChild()
    {
    }

    @Override
    public final XcNode[] getChild()
    {
        return null;
    }
    
    @Override
    public final void setChild(int index, XcNode child)
    {
        throw new IllegalArgumentException(index + ":" + child.getClass().getName());
    }

    @Override
    protected final void resolveOverride(XcIdentTableStack itStack)
    {
    }

    @Override
    public String toString()
    {
        StringBuilder b = new StringBuilder(128);
        b.append("[");
        commonToString(b);
        b.append(",bt=").append(_baseTypeEnum);
        b.append("]");
        
        return b.toString();
    }
}
