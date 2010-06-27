/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
package xcodeml.c.util;

import xcodeml.XmException;
import xcodeml.c.binding.gen.XbcXcodeProgram;
import xcodeml.util.XmValidator;

/**
 * Validator of XcodeML for C.
 */

public class XmcValidator extends XmValidator {
    public XmcValidator() throws XmException {
        super(XbcXcodeProgram.class.getResource("XcodeML_C.xsd"));
    }
}
