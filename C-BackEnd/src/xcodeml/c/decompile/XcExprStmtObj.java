/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
package xcodeml.c.decompile;

import xcodeml.XmException;
import xcodeml.c.obj.XcNode;
import xcodeml.c.util.XmcWriter;

/**
 * Internal object represents following elements:
 *   exprStatement
 */
public final class XcExprStmtObj extends XcStmtObj
{
    private XcExprObj _expr;

    private boolean _needCoArrayGuard = false;

    /**
     * Creates XcExprStmtObj.
     */
    public XcExprStmtObj()
    {
    }

    /**
     * Creates XcExprStmtObj.
     * 
     * @param expr a value of the expression statement object.
     */
    public XcExprStmtObj(XcExprObj expr)
    {
        _expr = expr;
    }

    /**
     * Sets the statement need to be enclosed with braces<br>
     * by having a temporary variable of coarray.
     * 
     * @param enable true if the object have a temporary variable of coarray.
     */
    public void setIsNeedCoArrayGurad(boolean enable)
    {
        _needCoArrayGuard = enable;
    }

    /**
     * Gets a expression of the expression statement.
     * 
     * @return a expression of the expression statement.
     */
    public XcExprObj getExprObj()
    {
        return _expr;
    }

    @Override
    public void addChild(XcNode child)
    {
        if(child instanceof XcExprObj)
            _expr = (XcExprObj)child;
        else
            throw new IllegalArgumentException(child.getClass().getName());
    }

    @Override
    public void checkChild()
    {
    }

    @Override
    public XcNode[] getChild()
    {
        return toNodeArray(_expr);
    }

    @Override
    public final void setChild(int index, XcNode child)
    {
        if((child instanceof XcExprObj) == false)
            throw new IllegalArgumentException(child.getClass().getName());

        if(index == 0)
            _expr = (XcExprObj)child;
        else
            throw new IllegalArgumentException(index + ":" + child.getClass().getName());
    }

    @Override
    public final void appendCode(XmcWriter w) throws XmException
    {
//         if(_isSubArrayStmt()) {
//             XcExprObj expr = XcXmpFactory.createSuAAsgFuncCall(this);
//             w.add(expr).eos();

//         } else
	 if(_needCoArrayGuard == false) {
             appendNoXmpCode(w);

         } else {
            XcStmtObj _xmpGuardStmt = XcXmpFactory.createXmpStmt(this);

            _needCoArrayGuard = false;

            w.add(_xmpGuardStmt);
         }
    }

    /**
     * Appends code as if there are no code of XcalableMP.
     * 
     * @param w a writer appends codes.
     * @throws XmException thrown if the writer fault to appends codes.
     */
    public final void appendNoXmpCode(XmcWriter w) throws XmException
    {
        super.appendCode(w);
        w.add(_expr).eos();
    }

    private boolean _isSubArrayStmt()
    {
        if(_expr instanceof XcOperatorObj) {
            XcExprObj[] exprs = ((XcOperatorObj)_expr).getExprObjs();

            if(exprs[0] instanceof XcXmpSubArrayRefObj)
                return true;
            else
                return false;
//         } else if(_expr instanceof XcXmpCoArrayAssignObj) {

//             if(((XcXmpCoArrayAssignObj)_expr).getDst() instanceof XcXmpSubArrayRefObj)
//                 return true;
//             else
//                 return false;
        }
        return false;
    }
}
