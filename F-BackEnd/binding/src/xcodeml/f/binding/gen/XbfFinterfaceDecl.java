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
package xcodeml.f.binding.gen;

import xcodeml.binding.*;

import java.io.*;
import java.io.IOException;
import java.io.PrintWriter;
import java.io.Writer;
import java.net.URL;
import javax.xml.parsers.*;
import org.w3c.dom.*;
import org.xml.sax.*;

/**
 * <b>XbfFinterfaceDecl</b> is generated from XcodeML_F.rng by Relaxer.
 * This class is derived from:
 * 
 * <!-- for programmer
 * <element java:extends="xcodeml.f.XmfObj" java:implements="xcodeml.binding.IXbLineNo" name="FinterfaceDecl">
 *   <ref name="defAttrSourceLine"/>
 *   <ref name="defAttrGenericName"/>
 *   <ref name="defAttrInterface"/>
 *   <zeroOrMore>
 *     <choice>
 *       <ref name="FfunctionDecl"/>
 *       <ref name="FmoduleProcedureDecl"/>
 *     </choice>
 *   </zeroOrMore>
 * </element>
 * -->
 * <!-- for javadoc -->
 * <pre> &lt;element java:extends="xcodeml.f.XmfObj" java:implements="xcodeml.binding.IXbLineNo" name="FinterfaceDecl"&gt;
 *   &lt;ref name="defAttrSourceLine"/&gt;
 *   &lt;ref name="defAttrGenericName"/&gt;
 *   &lt;ref name="defAttrInterface"/&gt;
 *   &lt;zeroOrMore&gt;
 *     &lt;choice&gt;
 *       &lt;ref name="FfunctionDecl"/&gt;
 *       &lt;ref name="FmoduleProcedureDecl"/&gt;
 *     &lt;/choice&gt;
 *   &lt;/zeroOrMore&gt;
 * &lt;/element&gt;
 * </pre>
 *
 * @version XcodeML_F.rng (Fri Dec 04 19:18:15 JST 2009)
 * @author  Relaxer 1.0 (http://www.relaxer.org)
 */
public class XbfFinterfaceDecl extends xcodeml.f.XmfObj implements java.io.Serializable, Cloneable, xcodeml.binding.IXbLineNo, IRVisitable, IRNode, IXbfDeclarationsChoice {
    private String lineno_;
    private String rawlineno_;
    private String file_;
    private String name_;
    private Boolean isOperator_;
    private Boolean isAssignment_;
    // List<IXbfFinterfaceDeclChoice>
    private java.util.List content_ = new java.util.ArrayList();
    private IRNode parentRNode_;

    /**
     * Creates a <code>XbfFinterfaceDecl</code>.
     *
     */
    public XbfFinterfaceDecl() {
    }

    /**
     * Creates a <code>XbfFinterfaceDecl</code>.
     *
     * @param source
     */
    public XbfFinterfaceDecl(XbfFinterfaceDecl source) {
        setup(source);
    }

    /**
     * Creates a <code>XbfFinterfaceDecl</code> by the Stack <code>stack</code>
     * that contains Elements.
     * This constructor is supposed to be used internally
     * by the Relaxer system.
     *
     * @param stack
     */
    public XbfFinterfaceDecl(RStack stack) {
        setup(stack);
    }

    /**
     * Creates a <code>XbfFinterfaceDecl</code> by the Document <code>doc</code>.
     *
     * @param doc
     */
    public XbfFinterfaceDecl(Document doc) {
        setup(doc.getDocumentElement());
    }

    /**
     * Creates a <code>XbfFinterfaceDecl</code> by the Element <code>element</code>.
     *
     * @param element
     */
    public XbfFinterfaceDecl(Element element) {
        setup(element);
    }

    /**
     * Creates a <code>XbfFinterfaceDecl</code> by the File <code>file</code>.
     *
     * @param file
     * @exception IOException
     * @exception SAXException
     * @exception ParserConfigurationException
     */
    public XbfFinterfaceDecl(File file) throws IOException, SAXException, ParserConfigurationException {
        setup(file);
    }

    /**
     * Creates a <code>XbfFinterfaceDecl</code>
     * by the String representation of URI <code>uri</code>.
     *
     * @param uri
     * @exception IOException
     * @exception SAXException
     * @exception ParserConfigurationException
     */
    public XbfFinterfaceDecl(String uri) throws IOException, SAXException, ParserConfigurationException {
        setup(uri);
    }

    /**
     * Creates a <code>XbfFinterfaceDecl</code> by the URL <code>url</code>.
     *
     * @param url
     * @exception IOException
     * @exception SAXException
     * @exception ParserConfigurationException
     */
    public XbfFinterfaceDecl(URL url) throws IOException, SAXException, ParserConfigurationException {
        setup(url);
    }

    /**
     * Creates a <code>XbfFinterfaceDecl</code> by the InputStream <code>in</code>.
     *
     * @param in
     * @exception IOException
     * @exception SAXException
     * @exception ParserConfigurationException
     */
    public XbfFinterfaceDecl(InputStream in) throws IOException, SAXException, ParserConfigurationException {
        setup(in);
    }

    /**
     * Creates a <code>XbfFinterfaceDecl</code> by the InputSource <code>is</code>.
     *
     * @param is
     * @exception IOException
     * @exception SAXException
     * @exception ParserConfigurationException
     */
    public XbfFinterfaceDecl(InputSource is) throws IOException, SAXException, ParserConfigurationException {
        setup(is);
    }

    /**
     * Creates a <code>XbfFinterfaceDecl</code> by the Reader <code>reader</code>.
     *
     * @param reader
     * @exception IOException
     * @exception SAXException
     * @exception ParserConfigurationException
     */
    public XbfFinterfaceDecl(Reader reader) throws IOException, SAXException, ParserConfigurationException {
        setup(reader);
    }

    /**
     * Initializes the <code>XbfFinterfaceDecl</code> by the XbfFinterfaceDecl <code>source</code>.
     *
     * @param source
     */
    public void setup(XbfFinterfaceDecl source) {
        int size;
        setLineno(source.getLineno());
        setRawlineno(source.getRawlineno());
        setFile(source.getFile());
        setName(source.getName());
        setIsOperator(source.getIsOperator());
        setIsAssignment(source.getIsAssignment());
        this.content_.clear();
        size = source.content_.size();
        for (int i = 0;i < size;i++) {
            addContent((IXbfFinterfaceDeclChoice)source.getContent(i).clone());
        }
    }

    /**
     * Initializes the <code>XbfFinterfaceDecl</code> by the Document <code>doc</code>.
     *
     * @param doc
     */
    public void setup(Document doc) {
        setup(doc.getDocumentElement());
    }

    /**
     * Initializes the <code>XbfFinterfaceDecl</code> by the Element <code>element</code>.
     *
     * @param element
     */
    public void setup(Element element) {
        init(element);
    }

    /**
     * Initializes the <code>XbfFinterfaceDecl</code> by the Stack <code>stack</code>
     * that contains Elements.
     * This constructor is supposed to be used internally
     * by the Relaxer system.
     *
     * @param stack
     */
    public void setup(RStack stack) {
        init(stack.popElement());
    }

    /**
     * @param element
     */
    private void init(Element element) {
        IXcodeML_FFactory factory = XcodeML_FFactory.getFactory();
        RStack stack = new RStack(element);
        lineno_ = URelaxer.getAttributePropertyAsString(element, "lineno");
        rawlineno_ = URelaxer.getAttributePropertyAsString(element, "rawlineno");
        file_ = URelaxer.getAttributePropertyAsString(element, "file");
        name_ = URelaxer.getAttributePropertyAsString(element, "name");
        isOperator_ = URelaxer.getAttributePropertyAsBooleanObject(element, "is_operator");
        isAssignment_ = URelaxer.getAttributePropertyAsBooleanObject(element, "is_assignment");
        content_.clear();
        while (true) {
            if (XbfFfunctionDecl.isMatch(stack)) {
                addContent(factory.createXbfFfunctionDecl(stack));
            } else if (XbfFmoduleProcedureDecl.isMatch(stack)) {
                addContent(factory.createXbfFmoduleProcedureDecl(stack));
            } else {
                break;
            }
        }
    }

    /**
     * @return Object
     */
    public Object clone() {
        IXcodeML_FFactory factory = XcodeML_FFactory.getFactory();
        return (factory.createXbfFinterfaceDecl(this));
    }

    /**
     * Creates a DOM representation of the object.
     * Result is appended to the Node <code>parent</code>.
     *
     * @param parent
     */
    public void makeElement(Node parent) {
        Document doc;
        if (parent instanceof Document) {
            doc = (Document)parent;
        } else {
            doc = parent.getOwnerDocument();
        }
        Element element = doc.createElement("FinterfaceDecl");
        int size;
        if (this.lineno_ != null) {
            URelaxer.setAttributePropertyByString(element, "lineno", this.lineno_);
        }
        if (this.rawlineno_ != null) {
            URelaxer.setAttributePropertyByString(element, "rawlineno", this.rawlineno_);
        }
        if (this.file_ != null) {
            URelaxer.setAttributePropertyByString(element, "file", this.file_);
        }
        if (this.name_ != null) {
            URelaxer.setAttributePropertyByString(element, "name", this.name_);
        }
        if (this.isOperator_ != null) {
            URelaxer.setAttributePropertyByBoolean(element, "is_operator", this.isOperator_);
        }
        if (this.isAssignment_ != null) {
            URelaxer.setAttributePropertyByBoolean(element, "is_assignment", this.isAssignment_);
        }
        size = this.content_.size();
        for (int i = 0;i < size;i++) {
            IXbfFinterfaceDeclChoice value = (IXbfFinterfaceDeclChoice)this.content_.get(i);
            value.makeElement(element);
        }
        parent.appendChild(element);
    }

    /**
     * Initializes the <code>XbfFinterfaceDecl</code> by the File <code>file</code>.
     *
     * @param file
     * @exception IOException
     * @exception SAXException
     * @exception ParserConfigurationException
     */
    public void setup(File file) throws IOException, SAXException, ParserConfigurationException {
        setup(file.toURL());
    }

    /**
     * Initializes the <code>XbfFinterfaceDecl</code>
     * by the String representation of URI <code>uri</code>.
     *
     * @param uri
     * @exception IOException
     * @exception SAXException
     * @exception ParserConfigurationException
     */
    public void setup(String uri) throws IOException, SAXException, ParserConfigurationException {
        setup(UJAXP.getDocument(uri, UJAXP.FLAG_NONE));
    }

    /**
     * Initializes the <code>XbfFinterfaceDecl</code> by the URL <code>url</code>.
     *
     * @param url
     * @exception IOException
     * @exception SAXException
     * @exception ParserConfigurationException
     */
    public void setup(URL url) throws IOException, SAXException, ParserConfigurationException {
        setup(UJAXP.getDocument(url, UJAXP.FLAG_NONE));
    }

    /**
     * Initializes the <code>XbfFinterfaceDecl</code> by the InputStream <code>in</code>.
     *
     * @param in
     * @exception IOException
     * @exception SAXException
     * @exception ParserConfigurationException
     */
    public void setup(InputStream in) throws IOException, SAXException, ParserConfigurationException {
        setup(UJAXP.getDocument(in, UJAXP.FLAG_NONE));
    }

    /**
     * Initializes the <code>XbfFinterfaceDecl</code> by the InputSource <code>is</code>.
     *
     * @param is
     * @exception IOException
     * @exception SAXException
     * @exception ParserConfigurationException
     */
    public void setup(InputSource is) throws IOException, SAXException, ParserConfigurationException {
        setup(UJAXP.getDocument(is, UJAXP.FLAG_NONE));
    }

    /**
     * Initializes the <code>XbfFinterfaceDecl</code> by the Reader <code>reader</code>.
     *
     * @param reader
     * @exception IOException
     * @exception SAXException
     * @exception ParserConfigurationException
     */
    public void setup(Reader reader) throws IOException, SAXException, ParserConfigurationException {
        setup(UJAXP.getDocument(reader, UJAXP.FLAG_NONE));
    }

    /**
     * Creates a DOM document representation of the object.
     *
     * @exception ParserConfigurationException
     * @return Document
     */
    public Document makeDocument() throws ParserConfigurationException {
        Document doc = UJAXP.makeDocument();
        makeElement(doc);
        return (doc);
    }

    /**
     * Gets the String property <b>lineno</b>.
     *
     * @return String
     */
    public final String getLineno() {
        return (lineno_);
    }

    /**
     * Sets the String property <b>lineno</b>.
     *
     * @param lineno
     */
    public final void setLineno(String lineno) {
        this.lineno_ = lineno;
    }

    /**
     * Gets the String property <b>rawlineno</b>.
     *
     * @return String
     */
    public final String getRawlineno() {
        return (rawlineno_);
    }

    /**
     * Sets the String property <b>rawlineno</b>.
     *
     * @param rawlineno
     */
    public final void setRawlineno(String rawlineno) {
        this.rawlineno_ = rawlineno;
    }

    /**
     * Gets the String property <b>file</b>.
     *
     * @return String
     */
    public final String getFile() {
        return (file_);
    }

    /**
     * Sets the String property <b>file</b>.
     *
     * @param file
     */
    public final void setFile(String file) {
        this.file_ = file;
    }

    /**
     * Gets the String property <b>name</b>.
     *
     * @return String
     */
    public final String getName() {
        return (name_);
    }

    /**
     * Sets the String property <b>name</b>.
     *
     * @param name
     */
    public final void setName(String name) {
        this.name_ = name;
    }

    /**
     * Gets the boolean property <b>isOperator</b>.
     *
     * @return boolean
     */
    public boolean getIsOperator() {
        if (isOperator_ == null) {
            return(false);
        }
        return (isOperator_.booleanValue());
    }

    /**
     * Gets the boolean property <b>isOperator</b>.
     *
     * @param isOperator
     * @return boolean
     */
    public boolean getIsOperator(boolean isOperator) {
        if (isOperator_ == null) {
            return(isOperator);
        }
        return (this.isOperator_.booleanValue());
    }

    /**
     * Gets the boolean property <b>isOperator</b>.
     *
     * @return Boolean
     */
    public Boolean getIsOperatorAsBoolean() {
        return (isOperator_);
    }

    /**
     * Check the boolean property <b>isOperator</b>.
     *
     * @return boolean
     */
    public boolean checkIsOperator() {
        return (isOperator_ != null);
    }

    /**
     * Sets the boolean property <b>isOperator</b>.
     *
     * @param isOperator
     */
    public void setIsOperator(boolean isOperator) {
        this.isOperator_ = new Boolean(isOperator);
    }

    /**
     * Sets the boolean property <b>isOperator</b>.
     *
     * @param isOperator
     */
    public void setIsOperator(Boolean isOperator) {
        this.isOperator_ = isOperator;
    }

    /**
     * Gets the boolean property <b>isAssignment</b>.
     *
     * @return boolean
     */
    public boolean getIsAssignment() {
        if (isAssignment_ == null) {
            return(false);
        }
        return (isAssignment_.booleanValue());
    }

    /**
     * Gets the boolean property <b>isAssignment</b>.
     *
     * @param isAssignment
     * @return boolean
     */
    public boolean getIsAssignment(boolean isAssignment) {
        if (isAssignment_ == null) {
            return(isAssignment);
        }
        return (this.isAssignment_.booleanValue());
    }

    /**
     * Gets the boolean property <b>isAssignment</b>.
     *
     * @return Boolean
     */
    public Boolean getIsAssignmentAsBoolean() {
        return (isAssignment_);
    }

    /**
     * Check the boolean property <b>isAssignment</b>.
     *
     * @return boolean
     */
    public boolean checkIsAssignment() {
        return (isAssignment_ != null);
    }

    /**
     * Sets the boolean property <b>isAssignment</b>.
     *
     * @param isAssignment
     */
    public void setIsAssignment(boolean isAssignment) {
        this.isAssignment_ = new Boolean(isAssignment);
    }

    /**
     * Sets the boolean property <b>isAssignment</b>.
     *
     * @param isAssignment
     */
    public void setIsAssignment(Boolean isAssignment) {
        this.isAssignment_ = isAssignment;
    }

    /**
     * Gets the IXbfFinterfaceDeclChoice property <b>content</b>.
     *
     * @return IXbfFinterfaceDeclChoice[]
     */
    public final IXbfFinterfaceDeclChoice[] getContent() {
        IXbfFinterfaceDeclChoice[] array = new IXbfFinterfaceDeclChoice[content_.size()];
        return ((IXbfFinterfaceDeclChoice[])content_.toArray(array));
    }

    /**
     * Sets the IXbfFinterfaceDeclChoice property <b>content</b>.
     *
     * @param content
     */
    public final void setContent(IXbfFinterfaceDeclChoice[] content) {
        this.content_.clear();
        for (int i = 0;i < content.length;i++) {
            addContent(content[i]);
        }
        for (int i = 0;i < content.length;i++) {
            content[i].rSetParentRNode(this);
        }
    }

    /**
     * Sets the IXbfFinterfaceDeclChoice property <b>content</b>.
     *
     * @param content
     */
    public final void setContent(IXbfFinterfaceDeclChoice content) {
        this.content_.clear();
        addContent(content);
        if (content != null) {
            content.rSetParentRNode(this);
        }
    }

    /**
     * Adds the IXbfFinterfaceDeclChoice property <b>content</b>.
     *
     * @param content
     */
    public final void addContent(IXbfFinterfaceDeclChoice content) {
        this.content_.add(content);
        if (content != null) {
            content.rSetParentRNode(this);
        }
    }

    /**
     * Adds the IXbfFinterfaceDeclChoice property <b>content</b>.
     *
     * @param content
     */
    public final void addContent(IXbfFinterfaceDeclChoice[] content) {
        for (int i = 0;i < content.length;i++) {
            addContent(content[i]);
        }
        for (int i = 0;i < content.length;i++) {
            content[i].rSetParentRNode(this);
        }
    }

    /**
     * Gets number of the IXbfFinterfaceDeclChoice property <b>content</b>.
     *
     * @return int
     */
    public final int sizeContent() {
        return (content_.size());
    }

    /**
     * Gets the IXbfFinterfaceDeclChoice property <b>content</b> by index.
     *
     * @param index
     * @return IXbfFinterfaceDeclChoice
     */
    public final IXbfFinterfaceDeclChoice getContent(int index) {
        return ((IXbfFinterfaceDeclChoice)content_.get(index));
    }

    /**
     * Sets the IXbfFinterfaceDeclChoice property <b>content</b> by index.
     *
     * @param index
     * @param content
     */
    public final void setContent(int index, IXbfFinterfaceDeclChoice content) {
        this.content_.set(index, content);
        if (content != null) {
            content.rSetParentRNode(this);
        }
    }

    /**
     * Adds the IXbfFinterfaceDeclChoice property <b>content</b> by index.
     *
     * @param index
     * @param content
     */
    public final void addContent(int index, IXbfFinterfaceDeclChoice content) {
        this.content_.add(index, content);
        if (content != null) {
            content.rSetParentRNode(this);
        }
    }

    /**
     * Remove the IXbfFinterfaceDeclChoice property <b>content</b> by index.
     *
     * @param index
     */
    public final void removeContent(int index) {
        this.content_.remove(index);
    }

    /**
     * Remove the IXbfFinterfaceDeclChoice property <b>content</b> by object.
     *
     * @param content
     */
    public final void removeContent(IXbfFinterfaceDeclChoice content) {
        this.content_.remove(content);
    }

    /**
     * Clear the IXbfFinterfaceDeclChoice property <b>content</b>.
     *
     */
    public final void clearContent() {
        this.content_.clear();
    }

    /**
     * Makes an XML text representation.
     *
     * @return String
     */
    public String makeTextDocument() {
        StringBuffer buffer = new StringBuffer();
        makeTextElement(buffer);
        return (new String(buffer));
    }

    /**
     * Makes an XML text representation.
     *
     * @param buffer
     */
    public void makeTextElement(StringBuffer buffer) {
        int size;
        buffer.append("<FinterfaceDecl");
        if (lineno_ != null) {
            buffer.append(" lineno=\"");
            buffer.append(URelaxer.escapeAttrQuot(URelaxer.getString(getLineno())));
            buffer.append("\"");
        }
        if (rawlineno_ != null) {
            buffer.append(" rawlineno=\"");
            buffer.append(URelaxer.escapeAttrQuot(URelaxer.getString(getRawlineno())));
            buffer.append("\"");
        }
        if (file_ != null) {
            buffer.append(" file=\"");
            buffer.append(URelaxer.escapeAttrQuot(URelaxer.getString(getFile())));
            buffer.append("\"");
        }
        if (name_ != null) {
            buffer.append(" name=\"");
            buffer.append(URelaxer.escapeAttrQuot(URelaxer.getString(getName())));
            buffer.append("\"");
        }
        if (isOperator_ != null) {
            buffer.append(" is_operator=\"");
            buffer.append(URelaxer.getString(getIsOperator()));
            buffer.append("\"");
        }
        if (isAssignment_ != null) {
            buffer.append(" is_assignment=\"");
            buffer.append(URelaxer.getString(getIsAssignment()));
            buffer.append("\"");
        }
        size = this.content_.size();
        for (int i = 0;i < size;i++) {
            IXbfFinterfaceDeclChoice value = (IXbfFinterfaceDeclChoice)this.content_.get(i);
            value.makeTextAttribute(buffer);
        }
        buffer.append(">");
        size = this.content_.size();
        for (int i = 0;i < size;i++) {
            IXbfFinterfaceDeclChoice value = (IXbfFinterfaceDeclChoice)this.content_.get(i);
            value.makeTextElement(buffer);
        }
        buffer.append("</FinterfaceDecl>");
    }

    /**
     * Makes an XML text representation.
     *
     * @param buffer
     * @exception IOException
     */
    public void makeTextElement(Writer buffer) throws IOException {
        int size;
        buffer.write("<FinterfaceDecl");
        if (lineno_ != null) {
            buffer.write(" lineno=\"");
            buffer.write(URelaxer.escapeAttrQuot(URelaxer.getString(getLineno())));
            buffer.write("\"");
        }
        if (rawlineno_ != null) {
            buffer.write(" rawlineno=\"");
            buffer.write(URelaxer.escapeAttrQuot(URelaxer.getString(getRawlineno())));
            buffer.write("\"");
        }
        if (file_ != null) {
            buffer.write(" file=\"");
            buffer.write(URelaxer.escapeAttrQuot(URelaxer.getString(getFile())));
            buffer.write("\"");
        }
        if (name_ != null) {
            buffer.write(" name=\"");
            buffer.write(URelaxer.escapeAttrQuot(URelaxer.getString(getName())));
            buffer.write("\"");
        }
        if (isOperator_ != null) {
            buffer.write(" is_operator=\"");
            buffer.write(URelaxer.getString(getIsOperator()));
            buffer.write("\"");
        }
        if (isAssignment_ != null) {
            buffer.write(" is_assignment=\"");
            buffer.write(URelaxer.getString(getIsAssignment()));
            buffer.write("\"");
        }
        size = this.content_.size();
        for (int i = 0;i < size;i++) {
            IXbfFinterfaceDeclChoice value = (IXbfFinterfaceDeclChoice)this.content_.get(i);
            value.makeTextAttribute(buffer);
        }
        buffer.write(">");
        size = this.content_.size();
        for (int i = 0;i < size;i++) {
            IXbfFinterfaceDeclChoice value = (IXbfFinterfaceDeclChoice)this.content_.get(i);
            value.makeTextElement(buffer);
        }
        buffer.write("</FinterfaceDecl>");
    }

    /**
     * Makes an XML text representation.
     *
     * @param buffer
     */
    public void makeTextElement(PrintWriter buffer) {
        int size;
        buffer.print("<FinterfaceDecl");
        if (lineno_ != null) {
            buffer.print(" lineno=\"");
            buffer.print(URelaxer.escapeAttrQuot(URelaxer.getString(getLineno())));
            buffer.print("\"");
        }
        if (rawlineno_ != null) {
            buffer.print(" rawlineno=\"");
            buffer.print(URelaxer.escapeAttrQuot(URelaxer.getString(getRawlineno())));
            buffer.print("\"");
        }
        if (file_ != null) {
            buffer.print(" file=\"");
            buffer.print(URelaxer.escapeAttrQuot(URelaxer.getString(getFile())));
            buffer.print("\"");
        }
        if (name_ != null) {
            buffer.print(" name=\"");
            buffer.print(URelaxer.escapeAttrQuot(URelaxer.getString(getName())));
            buffer.print("\"");
        }
        if (isOperator_ != null) {
            buffer.print(" is_operator=\"");
            buffer.print(URelaxer.getString(getIsOperator()));
            buffer.print("\"");
        }
        if (isAssignment_ != null) {
            buffer.print(" is_assignment=\"");
            buffer.print(URelaxer.getString(getIsAssignment()));
            buffer.print("\"");
        }
        size = this.content_.size();
        for (int i = 0;i < size;i++) {
            IXbfFinterfaceDeclChoice value = (IXbfFinterfaceDeclChoice)this.content_.get(i);
            value.makeTextAttribute(buffer);
        }
        buffer.print(">");
        size = this.content_.size();
        for (int i = 0;i < size;i++) {
            IXbfFinterfaceDeclChoice value = (IXbfFinterfaceDeclChoice)this.content_.get(i);
            value.makeTextElement(buffer);
        }
        buffer.print("</FinterfaceDecl>");
    }

    /**
     * Makes an XML text representation.
     *
     * @param buffer
     */
    public void makeTextAttribute(StringBuffer buffer) {
    }

    /**
     * Makes an XML text representation.
     *
     * @param buffer
     * @exception IOException
     */
    public void makeTextAttribute(Writer buffer) throws IOException {
    }

    /**
     * Makes an XML text representation.
     *
     * @param buffer
     */
    public void makeTextAttribute(PrintWriter buffer) {
    }

    /**
     * Gets the property value as String.
     *
     * @return String
     */
    public String getLinenoAsString() {
        return (URelaxer.getString(getLineno()));
    }

    /**
     * Gets the property value as String.
     *
     * @return String
     */
    public String getRawlinenoAsString() {
        return (URelaxer.getString(getRawlineno()));
    }

    /**
     * Gets the property value as String.
     *
     * @return String
     */
    public String getFileAsString() {
        return (URelaxer.getString(getFile()));
    }

    /**
     * Gets the property value as String.
     *
     * @return String
     */
    public String getNameAsString() {
        return (URelaxer.getString(getName()));
    }

    /**
     * Gets the property value as String.
     *
     * @return String
     */
    public String getIsOperatorAsString() {
        return (URelaxer.getString(getIsOperator()));
    }

    /**
     * Gets the property value as String.
     *
     * @return String
     */
    public String getIsAssignmentAsString() {
        return (URelaxer.getString(getIsAssignment()));
    }

    /**
     * Sets the property value by String.
     *
     * @param string
     */
    public void setLinenoByString(String string) {
        setLineno(string);
    }

    /**
     * Sets the property value by String.
     *
     * @param string
     */
    public void setRawlinenoByString(String string) {
        setRawlineno(string);
    }

    /**
     * Sets the property value by String.
     *
     * @param string
     */
    public void setFileByString(String string) {
        setFile(string);
    }

    /**
     * Sets the property value by String.
     *
     * @param string
     */
    public void setNameByString(String string) {
        setName(string);
    }

    /**
     * Sets the property value by String.
     *
     * @param string
     */
    public void setIsOperatorByString(String string) {
        setIsOperator(new Boolean(string).booleanValue());
    }

    /**
     * Sets the property value by String.
     *
     * @param string
     */
    public void setIsAssignmentByString(String string) {
        setIsAssignment(new Boolean(string).booleanValue());
    }

    /**
     * Returns a String representation of this object.
     * While this method informs as XML format representaion, 
     *  it's purpose is just information, not making 
     * a rigid XML documentation.
     *
     * @return String
     */
    public String toString() {
        try {
            return (makeTextDocument());
        } catch (Exception e) {
            return (super.toString());
        }
    }

    /**
     * Accepts the Visitor for enter behavior.
     *
     * @param visitor
     * @return boolean
     */
    public boolean enter(IRVisitor visitor) {
        return (visitor.enter(this));
    }

    /**
     * Accepts the Visitor for leave behavior.
     *
     * @param visitor
     */
    public void leave(IRVisitor visitor) {
        visitor.leave(this);
    }

    /**
     * Gets the IRNode property <b>parentRNode</b>.
     *
     * @return IRNode
     */
    public final IRNode rGetParentRNode() {
        return (parentRNode_);
    }

    /**
     * Sets the IRNode property <b>parentRNode</b>.
     *
     * @param parentRNode
     */
    public final void rSetParentRNode(IRNode parentRNode) {
        this.parentRNode_ = parentRNode;
    }

    /**
     * Gets child RNodes.
     *
     * @return IRNode[]
     */
    public IRNode[] rGetRNodes() {
        java.util.List classNodes = new java.util.ArrayList();
        classNodes.addAll(content_);
        IRNode[] nodes = new IRNode[classNodes.size()];
        return ((IRNode[])classNodes.toArray(nodes));
    }

    /**
     * Tests if a Element <code>element</code> is valid
     * for the <code>XbfFinterfaceDecl</code>.
     *
     * @param element
     * @return boolean
     */
    public static boolean isMatch(Element element) {
        if (!URelaxer.isTargetElement(element, "FinterfaceDecl")) {
            return (false);
        }
        RStack target = new RStack(element);
        boolean $match$ = false;
        Element child;
        while (true) {
            if (XbfFfunctionDecl.isMatchHungry(target)) {
                $match$ = true;
            } else if (XbfFmoduleProcedureDecl.isMatchHungry(target)) {
                $match$ = true;
            } else {
                break;
            }
        }
        if (!target.isEmptyElement()) {
            return (false);
        }
        return (true);
    }

    /**
     * Tests if elements contained in a Stack <code>stack</code>
     * is valid for the <code>XbfFinterfaceDecl</code>.
     * This mehtod is supposed to be used internally
     * by the Relaxer system.
     *
     * @param stack
     * @return boolean
     */
    public static boolean isMatch(RStack stack) {
        Element element = stack.peekElement();
        if (element == null) {
            return (false);
        }
        return (isMatch(element));
    }

    /**
     * Tests if elements contained in a Stack <code>stack</code>
     * is valid for the <code>XbfFinterfaceDecl</code>.
     * This method consumes the stack contents during matching operation.
     * This mehtod is supposed to be used internally
     * by the Relaxer system.
     *
     * @param stack
     * @return boolean
     */
    public static boolean isMatchHungry(RStack stack) {
        Element element = stack.peekElement();
        if (element == null) {
            return (false);
        }
        if (isMatch(element)) {
            stack.popElement();
            return (true);
        } else {
            return (false);
        }
    }
}
