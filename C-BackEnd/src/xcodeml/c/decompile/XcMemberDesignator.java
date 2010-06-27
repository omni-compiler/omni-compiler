/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
package xcodeml.c.decompile;

import xcodeml.XmException;
import xcodeml.c.obj.XcNode;
import xcodeml.c.type.XcType;
import xcodeml.c.util.XmcWriter;

/**
 * Internal object represents built_in operator argument.
 */
public class XcMemberDesignator extends XcObj
{
    private String _member;

    private XcExprObj _expr;

    private XcMemberDesignator _memberDesignator;

    private XcType _type;

    public void setMember(String member)
    {
        _member = member;
    }

    public String getMember()
    {
        return _member;
    }

    public void setType(XcType type)
    {
        _type = type;
    }

    public XcType getType()
    {
        return _type;
    }

    public XcMemberDesignator()
    {
    }

    @Override
    public final void addChild(XcNode child)
    {
        if(child instanceof XcExprObj)
            _expr = (XcExprObj)child;
        else if(child instanceof XcMemberDesignator)
            _memberDesignator = (XcMemberDesignator)child;
        else
            throw new IllegalArgumentException(child.getClass().getName());
    }

    @Override
    public final void checkChild()
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
    }

    @Override
    public final void appendCode(XmcWriter w) throws XmException
    {
        if(_memberDesignator != null) {
            w.add(_memberDesignator);

        }

        if(_expr != null)
            w.add("[").add(_expr).add("]");

        if(_member != null) {
            if(_memberDesignator != null)
                w.add(".");

            w.add(_member);
        }
    }
}
