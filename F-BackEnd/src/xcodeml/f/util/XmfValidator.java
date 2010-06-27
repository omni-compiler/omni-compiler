/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
package xcodeml.f.util;

import xcodeml.XmException;
import xcodeml.f.binding.gen.XbfXcodeProgram;
import xcodeml.util.XmValidator;

/**
 * Validator of XcodeML for F.
 */
public class XmfValidator extends XmValidator
{
    public XmfValidator() throws XmException
    {
        super(XbfXcodeProgram.class.getResource("XcodeML_F.xsd"));
    }
}
