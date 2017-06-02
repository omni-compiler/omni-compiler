/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
package xcodeml.c.decompile;

import xcodeml.util.XmException;
import xcodeml.c.util.XmcWriter;

/**
 * internal object represents following elements:
 *   declarations,
 *   exprStatement, compoundStatement, ifStatement, whileStatment,
 *   doStatement, forStatement, breakStatement, continueStatement,
 *   returnStatment, gotoStatement, statementLabel, switchStatement,
 *   caseLabel, defaultLabel
 */
public abstract class XcStmtObj extends XcObj implements XcStAndDeclObj
{
    private XcSourcePosObj _srcPos;
    
    /**
     * Creates a XcStmtObj.
     */
    public XcStmtObj()
    {
    }
    
    @Override
    public final XcSourcePosObj getSourcePos()
    {
        return _srcPos;
    }
    
    @Override
    public final void setSourcePos(XcSourcePosObj srcPos)
    {
        _srcPos = srcPos;
    }

    @Override
    public void appendCode(XmcWriter w) throws XmException
    {
        if(_srcPos != null)
            _srcPos.appendCode(w);
    }
}
