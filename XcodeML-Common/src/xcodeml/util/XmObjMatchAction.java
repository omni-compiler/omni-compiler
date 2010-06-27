/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
package xcodeml.util;

import xcodeml.XmObj;

/**
 * The interface for some action for XmNode.
 */
public interface XmObjMatchAction
{
    public XmObj execute(XmObj n);
}
