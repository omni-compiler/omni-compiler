/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
package xcodeml.c.binding;

import xcodeml.c.binding.gen.IXbcExpressionsChoice;

/**
 * Concrete class implement the interface are follows :
 * XbcUnaryMinusExpr, XbcPostDecrExpr, XbcPostIncrExpr
 * XbcPreDecrExpr, XbcPreIncrExpr
 */
public interface IXbcUnaryExpr extends IXbcTypedExpr
{
    /**
     * Gets the single term of the expression.
     *
     * @return the single term of the expression.
     */
    public IXbcExpressionsChoice getExpressions();

    public void setExpressions(IXbcExpressionsChoice choice);
}
