package xcodeml.c.util;

import java.io.Reader;
import java.io.Writer;
import java.util.List;
import java.util.ArrayList;
import org.w3c.dom.Document;

import xcodeml.util.XmException;
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
    public void decompile(XmDecompilerContext context, Document xcode, Writer writer) throws XmException {
        XcProgramObj prog = new XmcXcodeToXcTranslator().trans(xcode);
        XmcWriter xmcWriter = new XmcWriter(writer);
        prog.writeTo(xmcWriter);
        xmcWriter.flush();
    }
}
