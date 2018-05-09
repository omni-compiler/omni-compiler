package xcodeml.c.decompile;

import xcodeml.util.XmException;
import xcodeml.c.obj.XcNode;
import xcodeml.c.util.XmcWriter;

/**
 * Represents expression.
 */
public interface XcExprObj extends XcAppendable, XcNode
{
    public void appendCode(XmcWriter w) throws XmException;
}
