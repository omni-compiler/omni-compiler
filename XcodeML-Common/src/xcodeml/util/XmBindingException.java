/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
package xcodeml.util;

import xcodeml.XmObj;

/**
 * Exception about XcodeML description.
 */
public class XmBindingException extends RuntimeException
{
    private static final long serialVersionUID = 1482673969593630141L;
    
    private XmObj _xobj;

    /**
     * Creates XbcBindingException.
     *
     * @param msg the detail message
     * @param xobj a object binding to XcodeML which has fault description.
     */
    public XmBindingException(XmObj xobj, String msg)
    {
        super(msg);
        _xobj = xobj;
    }

    /**
     * Creates XbcBindingException.
     *
     * @param e the cause
     * @param xobj a object binding to XcodeML which has fault description
     */
    public XmBindingException(XmObj xobj, Exception e)
    {
        super(e);
        _xobj = xobj;
    }

    /**
     * Gets a binding object.
     *
     * @return a binding object.
     */
    public XmObj getXbObj()
    {
        return _xobj;
    }

    /**
     * Gets a XcodeML description of a binding object.
     * 
     * @return a XcodeML description
     */
    public String getElementDesc()
    {
        if(_xobj == null)
            return null;

        StringBuffer b = new StringBuffer(128);

        _xobj.makeTextElement(b);
        if(b.length() > 256) {
            b.delete(253, b.length());
            b.append("...");
        }

        return b.toString();
    }
}
