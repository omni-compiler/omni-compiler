package xcodeml.util;

/**
 * Exception of internal objects.
 */
public class XmException extends Exception
{
    private static final long serialVersionUID = -1407566211340512575L;

    /**
     * Creates XmException.
     */
    public XmException()
    {
    }

    /**
     * Creates XmException.
     *
     * @param msg the detail message.
     */
    public XmException(String msg)
    {
        super(msg);
    }
    
    /**
     * Creates XmException.
     *
     * @param cause the cause.
     */
    public XmException(Throwable cause)
    {
        super(cause);
    }

    /**
     * Creates XmException.
     *
     * @param msg the detail message.
     * @param cause the cause.
     */
    public XmException(String msg, Throwable cause)
    {
        super(msg, cause);
    }
} 


