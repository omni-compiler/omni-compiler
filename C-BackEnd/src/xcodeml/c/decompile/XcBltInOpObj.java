package xcodeml.c.decompile;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

import xcodeml.util.XmException;
import xcodeml.c.obj.XcNode;
import xcodeml.c.type.XcType;
import xcodeml.c.util.XmcWriter;

/**
 * Internal object represents following elements:
 *   builtin_op
 */
public class XcBltInOpObj extends XcObj implements XcExprObj
{
    private String _name;

    private List<XcBltInArgObj> _arguments = new ArrayList<XcBltInArgObj>();

    private boolean _isId = false;

    private boolean _isAddrOf = false;

    /**
     * Creates a XcBltInOpObj.
     */
    public XcBltInOpObj()
    {
    }

    /**
     * Sets a name of builtin_op.
     * 
     * @param name a name of builtin_op.
     */
    public void setName(String name)
    {
        _name = name;
    }

    /**
     * Gets a name of builtin_op.
     * 
     * @return a name of builtin_op.
     */
    public String getName()
    {
        return _name;
    }

    /**
     * Sets is builtin_op id.
     * 
     * @param enable true if builtin_op object is an id.
     */
    public void setIsId(boolean enable)
    {
        _isId = enable;
    }

    /**
     * Tests is builtin_op id.
     * 
     * @return true if builtin_op object is an id.
     */
    public boolean isId()
    {
        return _isId;
    }

    /**
     * Sets is builtin_op address.
     * 
     * @param enable true if builtin_op is an address.
     */
    public void setIsAddrOf(boolean enable)
    {
        _isAddrOf = enable;
    }

    /**
     * Tests is builtin_op address.
     * 
     * @return true if builtin_op is an address.
     */
    public boolean isAddrOf()
    {
        return _isAddrOf;
    }

    @Override
    public final void addChild(XcNode child)
    {
        if(child instanceof XcExprObj)
            _arguments.add(new XcBltInArgObj((XcExprObj)child));
        else if(child instanceof XcType)
            _arguments.add(new XcBltInArgObj((XcType)child));
        else if(child instanceof XcMemberDesignator)
            _arguments.add(new XcBltInArgObj((XcMemberDesignator)child));
        else
            throw new IllegalArgumentException(child.getClass().getName());
    }

    @Override
    public final void checkChild()
    {
        if(_name == null)
            throw new IllegalArgumentException("no name");
    }

    @Override
    public XcNode[] getChild()
    {
        return _arguments.toArray(new XcNode[_arguments.size()]);
    }

    @Override
    public final void setChild(int index, XcNode child)
    {
        XcBltInArgObj arg = null;

        if(child instanceof XcExprObj)
            arg = new XcBltInArgObj((XcExprObj)child);
        else if(child instanceof XcType)
            arg = new XcBltInArgObj((XcType)child);
        else if(child instanceof XcMemberDesignator)
            arg = new XcBltInArgObj((XcMemberDesignator)child);
        else if(child instanceof XcBltInArgObj)
            arg = (XcBltInArgObj)child;

        if(arg == null)
            throw new IllegalArgumentException(child.getClass().getName());

        if(index < _arguments.size())
            _arguments.set(index, arg);
        else
            throw new IllegalArgumentException(index + ":" + child.getClass().getName());
    }

    @Override
    public final void appendCode(XmcWriter w) throws XmException
    {
        boolean brace = !(_isId || _isAddrOf);

        Iterator<XcBltInArgObj> iter = _arguments.iterator();

        w.add(_name);

        if(brace)
            w.add("(");
        else {
            if(iter.hasNext())
                w.spc();
        }

        while(iter.hasNext()) {
            XcBltInArgObj arg = iter.next();

            w.add(arg);

            if(iter.hasNext())
                w.add(", ");
        }

        if(brace) w.add(")");
    }
}
