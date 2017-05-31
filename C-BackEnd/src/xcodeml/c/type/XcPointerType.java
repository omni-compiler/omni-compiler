/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
package xcodeml.c.type;

import xcodeml.util.XmException;
import xcodeml.c.obj.XcNode;

/**
 * type of pointer
 */
public final class XcPointerType extends XcExtType
{
    public XcPointerType()
    {
        this(null);
    }

    public XcPointerType(String typeId)
    {
        super(XcTypeEnum.POINTER, typeId);
    }

    public XcPointerType(String typeId, XcType type)
    {
        super(XcTypeEnum.POINTER, typeId);
        setRefType(type);
    }

    @Override
    public void addChild(XcNode child)
    {
        throw new IllegalArgumentException(child.getClass().getName());
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
    public final void setChild(int index, XcNode child)
    {
        throw new IllegalArgumentException(index + ":" + child.getClass().getName());
    }

    @Override
    protected final void resolveOverride(XcIdentTableStack itStack) throws XmException
    {
    }
}
