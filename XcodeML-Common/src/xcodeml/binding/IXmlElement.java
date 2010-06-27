/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
package xcodeml.binding;

import java.io.IOException;
import java.io.Reader;
import java.io.Writer;

import javax.xml.parsers.ParserConfigurationException;

import org.xml.sax.SAXException;

/**
 * Represents XML element.
 */
public interface IXmlElement
{
    /**
     * Setup by reader.
     */
    public void setup(Reader reader) throws IOException, SAXException, ParserConfigurationException;

    /**
     * Makes an XML text representation.
     *
     * @param buffer output
     */
    public void makeTextElement(StringBuffer buffer);

    /**
     * Makes an XML text representation.
     *
     * @param buffer output
     * @exception IOException
     */
    public void makeTextElement(Writer buffer) throws IOException;
}
