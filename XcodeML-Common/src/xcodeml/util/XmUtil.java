/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
package xcodeml.util;

import java.io.PrintWriter;
import java.io.Reader;
import java.io.StringReader;
import java.io.Writer;

import javax.xml.transform.OutputKeys;
import javax.xml.transform.TransformerConfigurationException;
import javax.xml.transform.TransformerException;
import javax.xml.transform.TransformerFactory;
import javax.xml.transform.Transformer;
import javax.xml.transform.stream.StreamResult;
import javax.xml.transform.stream.StreamSource;

import xcodeml.XmException;
import xcodeml.XmObj;

import org.apache.xml.serializer.OutputPropertiesFactory;

public class XmUtil
{
    /**
     * Format xml text to indented one.
     * 
     * @param indentSpaces
     *      number of indent spaces
     * @param reader
     *      input reader
     * @param writer
     *      output writer
     */
    public static void transformToIndentedXml(int indentSpaces, Reader reader, Writer writer)
        throws XmException
    {
        Transformer transformer = null;
        try {
            transformer = TransformerFactory.newInstance().newTransformer();
        } catch(TransformerConfigurationException e) {
            throw new XmException(e);
        }
        
        transformer.setOutputProperty(OutputKeys.INDENT, "yes");
        transformer.setOutputProperty(OutputKeys.METHOD, "xml");
        transformer.setOutputProperty(
            OutputPropertiesFactory.S_KEY_INDENT_AMOUNT, "" + indentSpaces);
        
        try {
            transformer.transform(new StreamSource(reader), new StreamResult(writer));
        } catch(TransformerException e) {
            throw new XmException(e);
        }
    }

    /**
     * Print XmObj as indented xml to stdout, for debugging.
     * 
     * @param indentSpaces
     *      number of indent spaces
     * @param xmobj
     *      XcodeML object
     */
    public static void printIdentedXml(int indentSpaces, XmObj xmobj) throws XmException
    {
        StringBuffer buf = new StringBuffer();
        xmobj.makeTextElement(buf);
        StringReader reader = new StringReader(buf.toString());
        buf = null;
        PrintWriter writer = new PrintWriter(System.out);
        
        transformToIndentedXml(indentSpaces, reader, writer);
    }
}
