/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
package xcodeml.c.binding;

import xcodeml.c.binding.gen.IXbcExpressionsChoice;

/**
 * Concrete class implement the interface are
 * equivalent to binary expression.
 */
public interface IXbcBinaryExpr extends IXbcTypedExpr
{
    /**
     * Gets the right-hand term of the expression.
     *
     * @return the right-hand term of the expression.
     */
    public IXbcExpressionsChoice getExpressions1();

    public void setExpressions1(IXbcExpressionsChoice choice);
    
    /**
     * Gets the left-hand term of the expression.
     *
     * @return the left-hand term of the expression.
     */
    public IXbcExpressionsChoice getExpressions2();

    public void setExpressions2(IXbcExpressionsChoice choice);
}
