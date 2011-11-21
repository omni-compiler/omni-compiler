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
    private static final long serialVersionUID = 1L;

    /**
     * Creates XmTranslationException.
     *
     * @param node    a XcodeML node object which has fault description.
     * @param case    the cause
     */
    public XmTranslationException(Node node, String msg, Throwable cause) {
        super(String.format("<%s> %s", node.getNodeName(), msg), cause);
    }

    /**
     * Creates XmTranslationException.
     *
     * @param node    a XcodeML node object which has fault description.
     */
    public XmTranslationException(Node node, String msg) {
        super(String.format("<%s> %s", node.getNodeName(), msg));
    }

    /**
     * Creates XmTranslationException.
     *
     * @param node    a XcodeML node object which has fault description.
     * @param case    the cause
     */
    public XmTranslationException(Node node, Throwable cause) {
        super(String.format("<%s>", node.getNodeName()), cause);
    }

    /**
     * Creates XmTranslationException.
     *
     * @param case    the cause
     */
    public XmTranslationException(Throwable cause) {
        super(cause);
    }
}
