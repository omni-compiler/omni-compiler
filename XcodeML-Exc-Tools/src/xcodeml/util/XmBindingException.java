package xcodeml.util;

/**
 * Exception about XcodeML description.
 */
public class XmBindingException extends RuntimeException
{
    /**
     * Creates XbcBindingException.
     *
     * @param msg the detail message
     */
    public XmBindingException(String msg)
    {
        super(msg);
    }

    /**
     * Creates XbcBindingException.
     *
     * @param e the cause
     */
    public XmBindingException(Exception e)
    {
        super(e);
    }
}

