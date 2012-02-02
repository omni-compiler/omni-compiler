/*
 * The Relaxer artifact
 * Copyright (c) 2000-2003, ASAMI Tomoharu, All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject to
 * the following conditions:
 *
 * - Redistributions of source code must retain the above copyright
 *   notice, this list of conditions and the following disclaimer. 
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 * LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 * OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 * WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */
package xcodeml.c.binding.gen;

import xcodeml.binding.*;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.io.PrintWriter;
import java.io.Reader;
import java.io.Writer;
import java.net.URL;
import javax.xml.parsers.ParserConfigurationException;
import org.w3c.dom.Document;
import org.w3c.dom.Element;
import org.w3c.dom.Node;
import org.xml.sax.InputSource;
import org.xml.sax.SAXException;

/**
 * <b>IXbcDeclarationsChoice</b> is generated from XcodeML_C.rng by Relaxer.
 * Concrete classes of the interface are XbcFunctionDefinition, XbcVarDecl and XbcFunctionDecl.
 *
 * @version XcodeML_C.rng (Thu Feb 02 16:55:19 JST 2012)
 * @author  Relaxer 1.0 (http://www.relaxer.org)
 */
public interface IXbcDeclarationsChoice extends IRVisitable, IRNode {
    /**
     * Creates a DOM representation of the object.
     * Result is appended to the Node <code>parent</code>.
     *
     * @param parent
     */
    void makeElement(Node parent);

    /**
     * Makes an XML text representation.
     *
     * @param buffer
     */
    void makeTextElement(StringBuffer buffer);

    /**
     * Makes an XML text representation.
     *
     * @param buffer
     * @exception IOException
     */
    void makeTextElement(Writer buffer) throws IOException;

    /**
     * Makes an XML text representation.
     *
     * @param buffer
     */
    void makeTextElement(PrintWriter buffer);

    /**
     * Makes an XML text representation.
     *
     * @param buffer
     */
    void makeTextAttribute(StringBuffer buffer);

    /**
     * Makes an XML text representation.
     *
     * @param buffer
     * @exception IOException
     */
    void makeTextAttribute(Writer buffer) throws IOException;

    /**
     * Makes an XML text representation.
     *
     * @param buffer
     */
    void makeTextAttribute(PrintWriter buffer);

    /**
     * @param doc
     */
    void setup(Document doc);

    /**
     * @param element
     */
    void setup(Element element);

    /**
     * @param stack
     */
    void setup(RStack stack);

    /**
     * @return Object
     */
    Object clone();

    /**
     * @param file
     * @exception IOException
     * @exception SAXException
     * @exception ParserConfigurationException
     */
    void setup(File file) throws IOException, SAXException, ParserConfigurationException;

    /**
     * @param uri
     * @exception IOException
     * @exception SAXException
     * @exception ParserConfigurationException
     */
    void setup(String uri) throws IOException, SAXException, ParserConfigurationException;

    /**
     * @param url
     * @exception IOException
     * @exception SAXException
     * @exception ParserConfigurationException
     */
    void setup(URL url) throws IOException, SAXException, ParserConfigurationException;

    /**
     * @param in
     * @exception IOException
     * @exception SAXException
     * @exception ParserConfigurationException
     */
    void setup(InputStream in) throws IOException, SAXException, ParserConfigurationException;

    /**
     * @param is
     * @exception IOException
     * @exception SAXException
     * @exception ParserConfigurationException
     */
    void setup(InputSource is) throws IOException, SAXException, ParserConfigurationException;

    /**
     * @param reader
     * @exception IOException
     * @exception SAXException
     * @exception ParserConfigurationException
     */
    void setup(Reader reader) throws IOException, SAXException, ParserConfigurationException;

    /**
     * @exception ParserConfigurationException
     * @return Document
     */
    Document makeDocument() throws ParserConfigurationException;

    /**
     * @return String
     */
    String getLineno();

    /**
     * @param lineno
     */
    void setLineno(String lineno);

    /**
     * @return String
     */
    String getRawlineno();

    /**
     * @param rawlineno
     */
    void setRawlineno(String rawlineno);

    /**
     * @return String
     */
    String getFile();

    /**
     * @param file
     */
    void setFile(String file);

    /**
     * @return String
     */
    String getIsGccSyntax();

    /**
     * @param isGccSyntax
     */
    void setIsGccSyntax(String isGccSyntax);

    /**
     * @return String
     */
    String getIsModified();

    /**
     * @param isModified
     */
    void setIsModified(String isModified);

    /**
     * @return XbcName
     */
    XbcName getName();

    /**
     * @param name
     */
    void setName(XbcName name);

    /**
     * @return String
     */
    String makeTextDocument();

    /**
     * @return String
     */
    String getLinenoAsString();

    /**
     * @return String
     */
    String getRawlinenoAsString();

    /**
     * @return String
     */
    String getFileAsString();

    /**
     * @return String
     */
    String getIsGccSyntaxAsString();

    /**
     * @return String
     */
    String getIsModifiedAsString();

    /**
     * @param string
     */
    void setLinenoByString(String string);

    /**
     * @param string
     */
    void setRawlinenoByString(String string);

    /**
     * @param string
     */
    void setFileByString(String string);

    /**
     * @param string
     */
    void setIsGccSyntaxByString(String string);

    /**
     * @param string
     */
    void setIsModifiedByString(String string);

    /**
     * @return String
     */
    String toString();
}
