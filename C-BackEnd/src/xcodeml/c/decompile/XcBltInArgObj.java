/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
package xcodeml.c.decompile;

import xcodeml.util.XmException;
import xcodeml.c.obj.XcNode;
import xcodeml.c.type.XcType;
import xcodeml.c.util.XmcWriter;

/**
 * Internal object represents built_in operator argument {type, memberDesignator, expression}.
 */
public class XcBltInArgObj implements XcAppendable, XcNode
{
    private XcExprObj                _expr;

    private XcType                   _type;

    private XcMemberDesignator _designator;

    /**
     * Creates a XcBltInArgObj and Sets its content to an expression object.
     * 
     * @param expr a content of the object.
     */
    public XcBltInArgObj(XcExprObj expr)
    {
        _expr = expr;
        _type = null;
        _designator = null;
    }

    /**
     * Creates a XcBltInArgObj and Sets its content to a type object.
     * 
     * @param type a content of the object.
     */
    public XcBltInArgObj(XcType type)
    {
        _expr = null;
        _type = type;
        _designator = null;
    }

    /**
     * Creates a XcBltInArgObj and Sets its content to a memberDesignator object.
     * 
     * @param designator a content of the object.
     */
    public XcBltInArgObj(XcMemberDesignator designator)
    {
        _expr = null;
        _type = null;
        _designator = designator;
    }

    @Override
    public final void addChild(XcNode child)
    {
        throw new IllegalArgumentException();
    }

    @Override
    public final void checkChild()
    {
        if((_expr == null && _type == null && _designator == null) ||
           (_expr != null && _type != null) ||
           (_type != null && _designator != null) ||
           (_expr != null &&  _designator != null))
            throw new IllegalArgumentException("content is empty");
    }

    @Override
    public final XcNode[] getChild()
    {
        if(_expr != null)
            return new XcNode[] { _expr };
        else if(_type != null)
            return new XcNode[] { _type };
        else if(_designator != null)
            return new XcNode[] { _designator };
        else
            return null;
    }

    @Override
    public final void setChild(int index, XcNode child)
    {
        if(index != 0)
            throw new IllegalArgumentException(index + ":" + child.getClass().getName());

        if(child instanceof XcExprObj) {
            _expr = (XcExprObj)child;
            return;
        }

        if(child instanceof XcType) {
            _type = (XcType)child;
            return;
        }

        if(child instanceof XcMemberDesignator) {
            _designator = (XcMemberDesignator)child;
            return;
        }

        throw new IllegalArgumentException(index + ":" + child.getClass().getName());
    }

    @Override
    public final void appendCode(XmcWriter w) throws XmException
    {
        w.add(_expr);
        w.add(_type);
        w.add(_designator);
    }
}
