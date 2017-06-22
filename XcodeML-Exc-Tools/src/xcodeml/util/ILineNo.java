/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
package xcodeml.util;

/**
 * Represents line number
 */
public interface ILineNo
{
    /**
     * Get line number
     */
    public int lineNo();
    
    /**
     * Get filename
     */
    public String fileName();
}
