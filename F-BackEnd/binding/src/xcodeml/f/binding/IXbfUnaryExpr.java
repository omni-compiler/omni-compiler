/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
package xcodeml.f.binding;

import xcodeml.binding.IXbTypedExpr;
import xcodeml.f.binding.gen.IXbfDefModelExprChoice;

public interface IXbfUnaryExpr extends IXbTypedExpr
{
    public IXbfDefModelExprChoice getDefModelExpr();

    /**
     * Sets the IXbfDefModelExprChoice property <b>defModelExpr</b>.
     *
     * @param defModelExpr
     */
    public void setDefModelExpr(IXbfDefModelExprChoice defModelExpr);
}
