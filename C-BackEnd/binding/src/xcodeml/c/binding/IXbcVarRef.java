/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
package xcodeml.c.binding;

import xcodeml.binding.IXbStringContent;
import xcodeml.binding.IXbVarRef;


/**
 * Represents variable reference (ref/addr).
 */
public interface IXbcVarRef extends IXbVarRef, IXbcTypedExpr
{
    public String getScope();
    
    public void setScope(String scope);
}
