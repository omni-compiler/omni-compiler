/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
package xcodeml.c.decompile;

import xcodeml.XmException;
import xcodeml.c.util.XmcWriter;

/**
 * Implementing this interface allows an object to be appended by XmWriter.
 */
public interface XcAppendable
{
    /**
     * Allows writer to append the object.
     *
     * @param w writer.
     */
    public void appendCode(XmcWriter w) throws XmException;
}
