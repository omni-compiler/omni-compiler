/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
package xcodeml.c.decompile;

import xcodeml.c.obj.XcNode;
import xcodeml.c.type.XcType;


/**
 * Internal object represents following elements:
 * addrOfExpr
 */
public class XcAddrOfExprObj extends XcOperatorObj
{
    /**
     * Creates a XcAddrOfExprObj.
     * 
     * @param opeEnum an operator code enumerator.
     * @param expr  a term of "&".
     */
    public XcAddrOfExprObj(XcOperatorEnum opeEnum, XcExprObj expr)
    {
        super(opeEnum);
        addChild(expr);
    }

    @Override
    public void checkChild()
    {
        if (super.getExprObjs() == null)
            throw new IllegalArgumentException("number of expression for the operator is invalid : 0");
    }

    @Override
    public XcNode[] getChild()
    {
        XcExprObj[] exprs = super.getExprObjs();

        if (exprs == null) {
	    return null;
        } else {
	    return toNodeArray(exprs[0]);
        }
    }
}
