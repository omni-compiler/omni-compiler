/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
package xcodeml.c.decompile;

import xcodeml.util.XmException;
import xcodeml.c.obj.XcNode;
import xcodeml.c.util.XmcWriter;

/**
 * Internal object represents name.
 */
public class XcNameObj extends XcObj
{
    private String _symbol;
    
    /**
     * Creates XcNameObj.
     * 
     * @param symbol a content of the object.
     */
    public XcNameObj(String symbol)
    {
        _symbol = symbol;
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
        return null;
    }

    @Override
    public final void setChild(int index, XcNode child)
    {
        throw new IllegalArgumentException(index + ":" + child.getClass().getName());
    }

    @Override
    public void appendCode(XmcWriter w) throws XmException
    {
        w.add(_symbol);
    }
}
