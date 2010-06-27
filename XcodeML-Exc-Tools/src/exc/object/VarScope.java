/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
package exc.object;

/**
 * Represents variable scope.
 */
public enum VarScope
{
    GLOBAL,
    LOCAL,
    PARAM,
    ;
    
    public String toXcodeString()
    {
        return toString();
    }
    
    @Override
    public String toString()
    {
        return super.toString().toLowerCase();
    }
    
    public static VarScope get(String scope)
    {
        for(VarScope v : values()) {
            if(v.toString().equalsIgnoreCase(scope))
                return v;
        }
        return null;
    }
}
