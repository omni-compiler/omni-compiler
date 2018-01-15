package xcodeml.c.decompile;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

import xcodeml.util.XmException;
import xcodeml.c.obj.XcNode;
import xcodeml.c.util.XmcWriter;

/**
 * Internal object represents value.
 */
public class XcValueObj extends XcObj implements XcExprObj
{
    private List<XcExprObj> _exprList = new ArrayList<XcExprObj>();

    @Override
    public final void addChild(XcNode child)
    {
        if(child instanceof XcExprObj)
            _exprList.add((XcExprObj)child);
        else
            throw new IllegalArgumentException(child.getClass().getName());
    }

    @Override
    public final void checkChild()
    {
        if(_exprList == null || _exprList.isEmpty())
            throw new IllegalArgumentException("no expression");
    }

    @Override
    public XcNode[] getChild()
    {
        return null;
    }

    @Override
    public final void setChild(int index, XcNode child)
    {
    }

    @Override
    public final void appendCode(XmcWriter w) throws XmException
    {
        if(_exprList.isEmpty()) {
            throw new IllegalArgumentException("no expression");
        } else if (_exprList.size() == 1) {
            w.add(_exprList.get(0));
        } else {
            w.add("{");
            for(Iterator<XcExprObj> iter = _exprList.iterator();iter.hasNext();) {
                if(iter.hasNext())
                    w.add(iter.next()).add(",").lf();
                else
                    w.add(iter.next()).lf();
            }
            w.add("}");
        }
    }
}
