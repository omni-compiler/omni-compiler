/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
package exc.object;

import xcodeml.XmException;

/**
 * Pragma parser which processes pragma statement in XcodeML.
 */
public interface ExternalPragmaParser
{
    /**
     * Parse pragma element.
     * 
     * @param x
     *      pragma element
     * @return
     *      parsed pragma element
     */
    public Xobject parse(Xobject x) throws XmException;
}
