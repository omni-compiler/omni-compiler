/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
package xcodeml.binding;

public interface IXbVarRef extends IXbTypedExpr, IXbStringContent
{
    public String getScope();
    
    public void setScope(String scope);
}
