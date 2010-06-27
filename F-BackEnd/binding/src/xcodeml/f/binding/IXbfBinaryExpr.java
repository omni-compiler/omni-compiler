/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
package xcodeml.f.binding;

import xcodeml.binding.IXbTypedExpr;
import xcodeml.f.binding.gen.IXbfDefModelExprChoice;

public interface IXbfBinaryExpr extends IXbTypedExpr
{
    public IXbfDefModelExprChoice getDefModelExpr1();

    public void setDefModelExpr1(IXbfDefModelExprChoice defModelExpr1);

    public IXbfDefModelExprChoice getDefModelExpr2();

    public void setDefModelExpr2(IXbfDefModelExprChoice defModelExpr2);
}
