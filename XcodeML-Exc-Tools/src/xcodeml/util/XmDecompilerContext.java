/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
package xcodeml.util;

/**
 * Decompiler Context.
 */
public interface XmDecompilerContext
{
    public static final String KEY_MAX_COLUMNS = "MAX_COLUMNS";
    
    /**
     * set property value.
     */
    public void setProperty(String key, Object value);
}
