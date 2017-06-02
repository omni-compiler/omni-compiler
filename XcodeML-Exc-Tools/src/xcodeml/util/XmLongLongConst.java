package xcodeml.util;

public class XmLongLongConst
{
    private long _high, _low;
    
    private String _highStr, _lowStr;
    
    /**
     * Creates LongLongConst
     * 
     * @param high a forward 32 bit value of long long constant.
     * @param low a backward 32 bit value of long long constant.
     * @param typeEnum indicates type of value.
     */
    public XmLongLongConst(long high, long low)
    {
        _high = high;
        _low = low;
    }

    /**
     * Creates XmLongLongConst
     * 
     * @param high a forward 32 bit value of long long constant.
     * @param low a backward 32 bit value of long long constant.
     * @param typeEnum indicates type of value.
     * @throws XmException thrown if a value does not represent integer.
     */
    public XmLongLongConst(String hvalue, String lvalue) throws XmException
    {
        this(XmStringUtil.getAsCLong(hvalue, XmStringUtil.Radix.HEX),
             XmStringUtil.getAsCLong(lvalue, XmStringUtil.Radix.HEX));
        
        if(hvalue.startsWith("0x")) {
            _highStr = hvalue.substring(2, hvalue.length());
        } else {
            throw new XmBindingException("invalid hex integer value '" + hvalue + "'");
        }

        if(lvalue.startsWith("0x")) {
            _lowStr = "00000000" + lvalue.substring(2, lvalue.length());
            _lowStr = _lowStr.substring(_lowStr.length() - 8, _lowStr.length());
        } else {
            throw new XmBindingException("invalid hex integer value '" + hvalue + "'");
        }
    }

    public static XmLongLongConst createLongLongConst(String valuesText)
    {
        XmLongLongConst obj;

        String[] values = XmStringUtil.trim(valuesText).split(" ");
        if (values.length != 2)
	return null;

        try {
            obj = new XmLongLongConst(values[0], values[1]);
        } catch(XmException e) {
            return null;
        }
        return obj;
    }

    /**
     * get high bits value
     */
    public long getHigh()
    {
        return _high;
    }
        
    /**
     * get low bits value
     */
    public long getLow()
    {
        return _low;
    }
}
