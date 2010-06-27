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
 * Represents Xobject to XmObj translator
 */
public interface XmXobjectToXmObjTranslator
{
    public XmObj translate(IXobject xobj) throws XmException;
}
