/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
package xcodeml.c.binding;

public interface IXbcAnnotation
{
    public String getIsGccSyntax();
    public void setIsGccSyntax(String flag);
    
    public String getIsModified();
    public void setIsModified(String flag);
}
