/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
package xcodeml.c.binding;

import xcodeml.c.binding.gen.IXbcExprOrTypeChoice;

/**
 * Concrete classes implement the interface are follows :
 * XbcSizeOfExpr, XbcGccAlignOfExpr.
 */
public interface IXbcSizeOrAlignExpr
{
    /**
     * Gets the single term of the expression.
     *
     * @return the single term of the expression
     */
    public IXbcExprOrTypeChoice getExprOrType();

    public void setExprOrType(IXbcExprOrTypeChoice choice);
}
