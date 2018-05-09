package xcodeml.c.type;

import xcodeml.c.obj.XcNode;

/**
 * type represent 'basicType' tag.
 */
public class XcBasicType extends XcExtType
{
    public XcBasicType(String typeId)
    {
        super(XcTypeEnum.BASICTYPE, typeId);
    }

    public XcBasicType(String typeId, XcType refType)
    {
        super(XcTypeEnum.BASICTYPE, typeId);
        setRefType(refType);
    }

    @Override
    protected void resolveOverride(XcIdentTableStack itStack)
    {
    }

    @Override
    public void addChild(XcNode child)
    {
    }

    @Override
    public void checkChild()
    {
    }

    @Override
    public XcNode[] getChild()
    {
        return toNodeArray(getGccAttribute());
    }

    @Override
    public void setChild(int index, XcNode child)
    {
        throw new IllegalArgumentException(index + ":" + child.getClass().getName());
    }

    @Override
    public String toString()
    {
        StringBuilder b = new StringBuilder(128);
        b.append("[");
        commonToString(b);
        b.append(",bt=").append(XcTypeEnum.BASICTYPE);
        b.append("]");
        return b.toString();
    }
}
