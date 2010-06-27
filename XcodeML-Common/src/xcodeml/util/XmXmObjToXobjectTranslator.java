/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
package xcodeml.util;

import xcodeml.IXobject;
import xcodeml.XmException;
import xcodeml.XmObj;

/**
 * Represents XmObj to Xobject translator
 */
public interface XmXmObjToXobjectTranslator
{
    /**
     * Translate XmObj to Xobject/XobjectFile.
     * 
     * @param xmobj
     *      target XcodeML object
     * @return
     *      if translation is successed, translated Xobject/XobjectFile.
     *      otherwise null.
     * @throws XmException
     *      XcodeML syntax error or other error
     */
    public IXobject translate(XmObj xmobj) throws XmException;
}
