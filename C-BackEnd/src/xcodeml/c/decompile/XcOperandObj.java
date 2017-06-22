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
import xcodeml.util.XmStringUtil;

/**
 * Internal object represents gccAsmOperand.
 */
public class XcOperandObj extends XcObj {

    private XcExprObj _expression;

    private String _match;

    private String _constraint;

    public void setMatch(String match)
    {
        _match = XmStringUtil.trim(match);
    }

    public String getMatch()
    {
        return _match;
    }

    public void setConstraint(String constraint)
    {
        _constraint = XmStringUtil.trim(constraint);
    }

    public String getConstraint()
    {
        return _constraint;
    }

    @Override
    public void addChild(XcNode child)
    {
        if(child instanceof XcExprObj) {
            _expression = (XcExprObj)child;
        } else
            throw new IllegalArgumentException(child.getClass().getName());
    }

    @Override
    public void checkChild()
    {
        if(_constraint == null || _expression == null)
            throw new IllegalArgumentException();
    }

    @Override
    public XcNode[] getChild()
    {
        return toNodeArray(_expression);
    }

    @Override
    public void setChild(int index, XcNode child)
    {
        if(index != 0)
            throw new IllegalArgumentException(index + ":" + child.getClass().getName());

        if((child instanceof XcExprObj) == false)
            throw new IllegalArgumentException(child.getClass().getName());

        _expression = (XcExprObj)child;
    }

    @Override
    public void appendCode(XmcWriter w) throws XmException
    {
        if (_match != null)
            w.add("[").add(_match).addSpc("]");

        w.add("\"").add(_constraint).add("\"").
            add("(").add(_expression).add(")");
    }
}
