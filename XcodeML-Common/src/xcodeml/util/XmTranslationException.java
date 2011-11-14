/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
package xcodeml.util;

import org.w3c.dom.Node;

/**
 * Exception about XcodeML translation.
 */
public class XmTranslationException extends RuntimeException {
    /**
     * 
     */
    private static final long serialVersionUID = 1L;

    /**
     * Creates XmTranslationException.
     *
     * @param node a XcodeML node object which has fault description.
     */
    public XmTranslationException(Node node, String msg)
    {
        super(msg);
    }

    /**
     * Creates XmTranslationException.
     *
     * @param e the cause
     */
    public XmTranslationException(Node node, Exception e)
    {
        super(e);
    }

    /**
     * Creates XmTranslationException.
     *
     * @param e the cause
     */
    public XmTranslationException(Exception e)
    {
        super(e);
    }
}
