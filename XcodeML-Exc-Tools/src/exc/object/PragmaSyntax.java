/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
package exc.object;

/**
 * pragma syntax code.
 */
public enum PragmaSyntax
{
    SYN_NONE,
    SYN_DECL,       // declaration pragma
    SYN_EXEC,       // execution pragma
    SYN_SECTION,    // prefix of section
    SYN_PREFIX,     // prefix of block
    SYN_POSTFIX,    // end block
    SYN_START,      // start block
    ;
    
    public static PragmaSyntax valueOrNullOf(String key)
    {
        try {
            return valueOf(key);
        } catch(IllegalArgumentException e) {
            return null;
        }
    }
}
