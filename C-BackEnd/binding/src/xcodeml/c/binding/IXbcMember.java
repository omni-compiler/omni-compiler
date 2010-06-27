/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
package xcodeml.c.binding;

import xcodeml.c.binding.gen.IXbcExpressionsChoice;

/**
 * Concretes class implement the interface are follows: 
 * XbcMemberRef, XbcMemberAddr, XbcMmberArrayRef, XbcMemberArrayAddr.
 */
public interface IXbcMember extends IXbcTypedExpr
{
    /**
     * Gets content of a member attribute.
     * 
     * @return content of a member attribute.
     */
    public String getMember();

    public void setMember(String member);

    /**
     * Gets the single term of this expression interface.
     *
     * @return the single term of this expression
     */
    public IXbcExpressionsChoice getExpressions();

    public void setExpressions(IXbcExpressionsChoice choice);
}
