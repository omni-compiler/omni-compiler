/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
package xcodeml.c.util;

import java.io.Reader;
import java.io.Writer;
import java.util.List;
import java.util.ArrayList;

import xcodeml.XmException;
import xcodeml.binding.XmXcodeProgram;
import xcodeml.c.binding.gen.XbcXcodeProgram;
import xcodeml.c.binding.gen.XcodeML_CFactory;
import xcodeml.c.decompile.XcBindingVisitor;
import xcodeml.c.decompile.XcProgramObj;
import xcodeml.util.XmDecompiler;
import xcodeml.util.XmDecompilerContext;
import xcodeml.util.XmLog;

/**
 * XcodeML decompiler for C language.
 */
public class XmcDecompiler implements XmDecompiler
{
    /**
     * Creates XmcDecompiler.
     */
    public XmcDecompiler()
    {
    }

    @Override
    public boolean decompile(XmDecompilerContext context, Reader reader, Writer writer) throws XmException
    {
        XbcXcodeProgram xmprog = XcodeML_CFactory.getFactory().createXbcXcodeProgram();
        List<String> errorList = new ArrayList<String>();
        XmcValidator validator = new XmcValidator();

        if(!validator.read(reader, xmprog, errorList)) {
            for (String error : errorList) {
                XmLog.error(error);
            }
            return false;
        }
        
        decompile(context, xmprog, writer);
        return true;
    }

    @Override
    public void decompile(XmDecompilerContext context, XmXcodeProgram xmprog, Writer writer) throws XmException
    {
        XcProgramObj prog = XcBindingVisitor.createXcProgramObj((XbcXcodeProgram)xmprog);
        XmcWriter xmcWriter = new XmcWriter(writer);
        prog.writeTo(xmcWriter);
        xmcWriter.flush();
    }
}
