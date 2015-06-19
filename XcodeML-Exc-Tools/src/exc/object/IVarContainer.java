/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
package exc.object;

public interface IVarContainer
{
    public Ident findVarIdent(String name);
    public Ident findCommonIdent(String name);
}
