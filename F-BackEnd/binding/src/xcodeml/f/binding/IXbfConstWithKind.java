/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
package xcodeml.f.binding;

import xcodeml.binding.IXbStringContent;
import xcodeml.binding.IXbTypedExpr;

public interface IXbfConstWithKind extends IXbTypedExpr, IXbStringContent
{
    public String getKind();
    public void setKind(String kind);
}
