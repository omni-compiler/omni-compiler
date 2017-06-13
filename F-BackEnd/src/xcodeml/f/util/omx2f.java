/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
package xcodeml.f.util;

import xcodeml.util.XmException;
import xcodeml.util.XmBackEnd;

/**
 * Run XcodeML/Fortran decompiler.
 */
public class omx2f
{
    public static void main(String[] args) throws XmException
    {
        System.exit(new XmBackEnd("F", "F_Back").run(args));
    }
}
