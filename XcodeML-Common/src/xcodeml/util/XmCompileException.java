/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
package xcodeml.util;

import xcodeml.XmObj;

public class XmCompileException extends RuntimeException
{
    private static final long serialVersionUID = -6715840674962886606L;
    
    private XmObj _xmobj;
    
    public XmCompileException(XmObj xmobj, String msg)
    {
        super(msg);
        _xmobj = xmobj;
    }
    
    public XmObj getXmObj()
    {
        return _xmobj;
    }
}
