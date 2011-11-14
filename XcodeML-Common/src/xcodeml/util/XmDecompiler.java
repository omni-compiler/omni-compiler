/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
package xcodeml.util;

import java.io.Reader;
import java.io.Writer;
import org.w3c.dom.Document;

import xcodeml.XmException;
import xcodeml.binding.XmXcodeProgram;

/**
 * The interface for XcodeML decompiler.
 */
public interface XmDecompiler
{
    /**
     * Decompile XcodeProgram document from specified reader.
     * 
     * @param context
     *      decompiler context (optional).
     * @param reader
     *      reader of XcodeProgram document.
     * @param writer
     *      writer of decompiled code.
     * @returns
     *      false if xml validating failed, otherwise true.
     */
    public boolean decompile(XmDecompilerContext context, Reader reader, Writer writer) throws XmException;

    /**
     * Decompile XcodeProgram document from specified XmXcodeProgram object.
     * 
     * @param xmprog
     *      XcodeProgram document object.
     * @param writer
     *      writer of decompiled code.
     */
    public void decompile(XmDecompilerContext context, XmXcodeProgram xmprog, Writer writer) throws XmException;

    /**
     * Decompile XcodeProgram document from specified DOM document object.
     * 
     * @param xcode
     *      XcodeML document object.
     * @param writer
     *      writer of decompiled code.
     */
    public void decompile(XmDecompilerContext context, Document xcode, Writer writer) throws XmException;
}
