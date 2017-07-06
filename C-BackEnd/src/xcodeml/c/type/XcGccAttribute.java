package xcodeml.c.type;

import java.util.List;
import java.util.ArrayList;
import java.util.Iterator;

import xcodeml.util.XmException;
import xcodeml.c.decompile.XcExprObj;
import xcodeml.c.decompile.XcObj;
import xcodeml.c.obj.XcNode;
import xcodeml.c.util.XmcWriter;

/**
 * represent gccAttribute.
 */
public class XcGccAttribute extends XcObj
{
    private String _name;

    private List<XcExprObj> _exprList = new ArrayList<XcExprObj>();

    public XcGccAttribute()
    {
    }

    public XcGccAttribute(String name)
    {
         _name = name;
    }

    @Override
    public void appendCode(XmcWriter w) throws XmException
    {
        w.add(_name);

        if(_exprList.isEmpty() == false) {
            Iterator<XcExprObj> iter = _exprList.iterator();

            w.add("(");
            while(iter.hasNext()) {
                w.add(iter.next());
                if(iter.hasNext())
                    w.add(", ");
            }
            w.add(")");
        }
    }

    @Override
    public void addChild(XcNode child)
    {
        if(child instanceof XcExprObj) {
            _exprList.add((XcExprObj)child);
        } else {
            throw new IllegalArgumentException();
        }
    }

    @Override
    public void checkChild()
    {
    }

    @Override
    public XcNode[] getChild()
    {
        return _exprList.toArray(new XcNode[_exprList.size()]);
    }

    @Override
    public void setChild(int index, XcNode child)
    {
    }

    public boolean containsAttrAlias()
    {
        if(_name.startsWith("alias") || _name.startsWith("__alias__"))
            return true;

        return false;
    }

    public void setName(String name)
    {
        _name = name;
    }
}
