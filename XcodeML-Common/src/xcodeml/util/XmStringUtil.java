package xcodeml.util;

/**
 * Utility of String for XcodeML decompiler.
 */
public final class XmStringUtil
{
    public enum Radix
    {
        ALL,
        HEX,
        DEC,
    }
    
    private XmStringUtil()
    {
    }

    /**
     * Wrapper method of String's trim method.
     *
     * @param s string.
     */
    public static final String trim(String s)
    {
        if(s == null)
            return null;
        s = s.trim();
        if(s.length() == 0)
            return null;
        return s;
    }

    /**
     * Gets integer from string represents decimal/hex number.
     *
     * @param str represents decimal/hex number.
     * @return integer repereseted by the string
     */
    public static int getAsCInt(String str)
    {
        return (int)getAsCLong(str, Radix.ALL);
    }

    /**
     * Gets long integer from string represents some radix number.
     *
     * @param str represents decimal/hex number.
     * @return long integer repereseted by the string.
     */
    public static long getAsCLong(String str)
    {
        return getAsCLong(str, Radix.ALL);
    }
    
    /**
     * Gets long integer from string represents some radix number.
     *
     * @param str represents decimal/hex number.
     * @param radix indicates a number base.
     * @return long integer repereseted by the string.
     */
    public static long getAsCLong(String str, Radix radix)
    {
        str = trim(str);
        if(str == null)
            throw new XmBindingException("not integer value '" + str + "'");

        do {
            if(str.startsWith("0x") && radix == Radix.ALL || radix == Radix.HEX) {
                
                if(str.length() <= 2)
                    break;
                
                try {
                    return Long.parseLong(str.substring(2, str.length()), 16);
                } catch(Exception e) {
                    break;
                }
                
            }
            
            if(radix == Radix.ALL || radix == Radix.DEC) {
                try {
                    return Long.parseLong(str);
                } catch(Exception e) {
                    break;
                }
            }
        } while(false);
        
        switch(radix) {
        case HEX:
            throw new XmBindingException("invalid hex integer value '" + str + "'");
        case DEC:
            throw new XmBindingException("invalid integer value '" + str + "'");
        default:
            /* not reachable */
            throw new IllegalArgumentException();
        }
    }

    /**
     * Gets boolean from string represents boolean.
     *
     * @param str represents boolean.
     * @return boolean repereseted by str.
     */
    public static boolean getAsBool(String str)
    {
        if(str == null)

        str = trim(str);
        if(str == null)
            throw new XmBindingException("invalid bool value");
        if(str.equals("1") || str.equals("true"))
            return true;
        if(str.equals("0") || str.equals("false"))
            return false;
        throw new XmBindingException("invalid bool value '" + str + "'");
    }
}
