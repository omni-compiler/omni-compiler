package xcodeml.c.decompile;

import xcodeml.util.XmException;
import xcodeml.c.obj.XcNode;
import xcodeml.c.util.XmcWriter;

/**
 * Internal object represents gccCompoundExpr.
 */
public class XcGccCompoundExprObj extends XcObj implements XcExprObj
{
    private XcCompStmtObj _compStmt;

    private boolean _isGccExtension;

    /**
     * Creates XcCompoundExprObj.
     */
    public XcGccCompoundExprObj()
    {
    }

    /**
     * Tests if the expression is used with with __extension__.
     * 
     * @return true if the expression is used with with __extension__.
     */
    public final boolean isGccExtension()
    {
        return _isGccExtension;
    }

    /**
     * Sets if is the expression used with __extension__.
     *  
     * @param isGccExtension true if the expression is used with with __extension__.
     */
    public final void setIsGccExtension(boolean enable)
    {
        _isGccExtension = enable;
    }

    /**
     * Sets a compound statement to a value of the expression.
     * 
     * @param compStmt a compound statement of a value of the expression.
     */
    public final void setCompStmt(XcCompStmtObj compStmt)
    {
        _compStmt = compStmt;
    }

    @Override
    public final void addChild(XcNode child)
    {
        if(child instanceof XcCompStmtObj)
            _compStmt = (XcCompStmtObj)child;
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
        return new XcNode[] { _compStmt };
    }

    @Override
    public final void setChild(int index, XcNode child)
    {
        switch(index) {
        case 0:
            _compStmt = (XcCompStmtObj)child;
            break;
        default:
            throw new IllegalArgumentException(index + ":" + child.getClass().getName());
        }
    }

    @Override
    public final void appendCode(XmcWriter w) throws XmException
    {
        boolean insideCompExpr = true;

        if(_isGccExtension)
            w.addSpc("__extension__");

        w.add("({");
        if(_compStmt != null)
            _compStmt.appendContent(w, insideCompExpr);
        w.add("})");
    }
}
