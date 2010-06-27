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
 * <b>XbfFcontainsStatement</b> is generated from XcodeML_F.rng by Relaxer.
 * This class is derived from:
 * 
 * <!-- for programmer
 * <element java:extends="xcodeml.f.XmfObj" java:implements="xcodeml.binding.IXbStatement" name="FcontainsStatement">
 *   <ref name="defAttrSourceLine"/>
 *   <oneOrMore>
 *     <ref name="FfunctionDefinition"/>
 *   </oneOrMore>
 * </element>
 * -->
 * <!-- for javadoc -->
 * <pre> &lt;element java:extends="xcodeml.f.XmfObj" java:implements="xcodeml.binding.IXbStatement" name="FcontainsStatement"&gt;
 *   &lt;ref name="defAttrSourceLine"/&gt;
 *   &lt;oneOrMore&gt;
 *     &lt;ref name="FfunctionDefinition"/&gt;
 *   &lt;/oneOrMore&gt;
 * &lt;/element&gt;
 * </pre>
 *
 * @version XcodeML_F.rng (Fri Dec 04 19:18:15 JST 2009)
 * @author  Relaxer 1.0 (http://www.relaxer.org)
 */
public class XbfFcontainsStatement extends xcodeml.f.XmfObj implements java.io.Serializable, Cloneable, xcodeml.binding.IXbStatement, IRVisitable, IRNode, IXbfDefModelStatementChoice {
    private String lineno_;
    private String rawlineno_;
    private String file_;
    // List<XbfFfunctionDefinition>
    private java.util.List ffunctionDefinition_ = new java.util.ArrayList();
    private IRNode parentRNode_;

    /**
     * Creates a <code>XbfFcontainsStatement</code>.
     *
     */
    public XbfFcontainsStatement() {
    }

    /**
     * Creates a <code>XbfFcontainsStatement</code>.
     *
     * @param source
     */
    public XbfFcontainsStatement(XbfFcontainsStatement source) {
        setup(source);
    }

    /**
     * Creates a <code>XbfFcontainsStatement</code> by the Stack <code>stack</code>
     * that contains Elements.
     * This constructor is supposed to be used internally
     * by the Relaxer system.
     *
     * @param stack
     */
    public XbfFcontainsStatement(RStack stack) {
        setup(stack);
    }

    /**
     * Creates a <code>XbfFcontainsStatement</code> by the Document <code>doc</code>.
     *
     * @param doc
     */
    public XbfFcontainsStatement(Document doc) {
        setup(doc.getDocumentElement());
    }

    /**
     * Creates a <code>XbfFcontainsStatement</code> by the Element <code>element</code>.
     *
     * @param element
     */
    public XbfFcontainsStatement(Element element) {
        setup(element);
    }

    /**
     * Creates a <code>XbfFcontainsStatement</code> by the File <code>file</code>.
     *
     * @param file
     * @exception IOException
     * @exception SAXException
     * @exception ParserConfigurationException
     */
    public XbfFcontainsStatement(File file) throws IOException, SAXException, ParserConfigurationException {
        setup(file);
    }

    /**
     * Creates a <code>XbfFcontainsStatement</code>
     * by the String representation of URI <code>uri</code>.
     *
     * @param uri
     * @exception IOException
     * @exception SAXException
     * @exception ParserConfigurationException
     */
    public XbfFcontainsStatement(String uri) throws IOException, SAXException, ParserConfigurationException {
        setup(uri);
    }

    /**
     * Creates a <code>XbfFcontainsStatement</code> by the URL <code>url</code>.
     *
     * @param url
     * @exception IOException
     * @exception SAXException
     * @exception ParserConfigurationException
     */
    public XbfFcontainsStatement(URL url) throws IOException, SAXException, ParserConfigurationException {
        setup(url);
    }

    /**
     * Creates a <code>XbfFcontainsStatement</code> by the InputStream <code>in</code>.
     *
     * @param in
     * @exception IOException
     * @exception SAXException
     * @exception ParserConfigurationException
     */
    public XbfFcontainsStatement(InputStream in) throws IOException, SAXException, ParserConfigurationException {
        setup(in);
    }

    /**
     * Creates a <code>XbfFcontainsStatement</code> by the InputSource <code>is</code>.
     *
     * @param is
     * @exception IOException
     * @exception SAXException
     * @exception ParserConfigurationException
     */
    public XbfFcontainsStatement(InputSource is) throws IOException, SAXException, ParserConfigurationException {
        setup(is);
    }

    /**
     * Creates a <code>XbfFcontainsStatement</code> by the Reader <code>reader</code>.
     *
     * @param reader
     * @exception IOException
     * @exception SAXException
     * @exception ParserConfigurationException
     */
    public XbfFcontainsStatement(Reader reader) throws IOException, SAXException, ParserConfigurationException {
        setup(reader);
    }

    /**
     * Initializes the <code>XbfFcontainsStatement</code> by the XbfFcontainsStatement <code>source</code>.
     *
     * @param source
     */
    public void setup(XbfFcontainsStatement source) {
        int size;
        setLineno(source.getLineno());
        setRawlineno(source.getRawlineno());
        setFile(source.getFile());
        this.ffunctionDefinition_.clear();
        size = source.ffunctionDefinition_.size();
        for (int i = 0;i < size;i++) {
            addFfunctionDefinition((XbfFfunctionDefinition)source.getFfunctionDefinition(i).clone());
        }
    }

    /**
     * Initializes the <code>XbfFcontainsStatement</code> by the Document <code>doc</code>.
     *
     * @param doc
     */
    public void setup(Document doc) {
        setup(doc.getDocumentElement());
    }

    /**
     * Initializes the <code>XbfFcontainsStatement</code> by the Element <code>element</code>.
     *
     * @param element
     */
    public void setup(Element element) {
        init(element);
    }

    /**
     * Initializes the <code>XbfFcontainsStatement</code> by the Stack <code>stack</code>
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
        ffunctionDefinition_.clear();
        while (true) {
            if (XbfFfunctionDefinition.isMatch(stack)) {
                addFfunctionDefinition(factory.createXbfFfunctionDefinition(stack));
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
        return (factory.createXbfFcontainsStatement(this));
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
        Element element = doc.createElement("FcontainsStatement");
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
        size = this.ffunctionDefinition_.size();
        for (int i = 0;i < size;i++) {
            XbfFfunctionDefinition value = (XbfFfunctionDefinition)this.ffunctionDefinition_.get(i);
            value.makeElement(element);
        }
        parent.appendChild(element);
    }

    /**
     * Initializes the <code>XbfFcontainsStatement</code> by the File <code>file</code>.
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
     * Initializes the <code>XbfFcontainsStatement</code>
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
     * Initializes the <code>XbfFcontainsStatement</code> by the URL <code>url</code>.
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
     * Initializes the <code>XbfFcontainsStatement</code> by the InputStream <code>in</code>.
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
     * Initializes the <code>XbfFcontainsStatement</code> by the InputSource <code>is</code>.
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
     * Initializes the <code>XbfFcontainsStatement</code> by the Reader <code>reader</code>.
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
     * Gets the XbfFfunctionDefinition property <b>FfunctionDefinition</b>.
     *
     * @return XbfFfunctionDefinition[]
     */
    public final XbfFfunctionDefinition[] getFfunctionDefinition() {
        XbfFfunctionDefinition[] array = new XbfFfunctionDefinition[ffunctionDefinition_.size()];
        return ((XbfFfunctionDefinition[])ffunctionDefinition_.toArray(array));
    }

    /**
     * Sets the XbfFfunctionDefinition property <b>FfunctionDefinition</b>.
     *
     * @param ffunctionDefinition
     */
    public final void setFfunctionDefinition(XbfFfunctionDefinition[] ffunctionDefinition) {
        this.ffunctionDefinition_.clear();
        for (int i = 0;i < ffunctionDefinition.length;i++) {
            addFfunctionDefinition(ffunctionDefinition[i]);
        }
        for (int i = 0;i < ffunctionDefinition.length;i++) {
            ffunctionDefinition[i].rSetParentRNode(this);
        }
    }

    /**
     * Sets the XbfFfunctionDefinition property <b>FfunctionDefinition</b>.
     *
     * @param ffunctionDefinition
     */
    public final void setFfunctionDefinition(XbfFfunctionDefinition ffunctionDefinition) {
        this.ffunctionDefinition_.clear();
        addFfunctionDefinition(ffunctionDefinition);
        if (ffunctionDefinition != null) {
            ffunctionDefinition.rSetParentRNode(this);
        }
    }

    /**
     * Adds the XbfFfunctionDefinition property <b>FfunctionDefinition</b>.
     *
     * @param ffunctionDefinition
     */
    public final void addFfunctionDefinition(XbfFfunctionDefinition ffunctionDefinition) {
        this.ffunctionDefinition_.add(ffunctionDefinition);
        if (ffunctionDefinition != null) {
            ffunctionDefinition.rSetParentRNode(this);
        }
    }

    /**
     * Adds the XbfFfunctionDefinition property <b>FfunctionDefinition</b>.
     *
     * @param ffunctionDefinition
     */
    public final void addFfunctionDefinition(XbfFfunctionDefinition[] ffunctionDefinition) {
        for (int i = 0;i < ffunctionDefinition.length;i++) {
            addFfunctionDefinition(ffunctionDefinition[i]);
        }
        for (int i = 0;i < ffunctionDefinition.length;i++) {
            ffunctionDefinition[i].rSetParentRNode(this);
        }
    }

    /**
     * Gets number of the XbfFfunctionDefinition property <b>FfunctionDefinition</b>.
     *
     * @return int
     */
    public final int sizeFfunctionDefinition() {
        return (ffunctionDefinition_.size());
    }

    /**
     * Gets the XbfFfunctionDefinition property <b>FfunctionDefinition</b> by index.
     *
     * @param index
     * @return XbfFfunctionDefinition
     */
    public final XbfFfunctionDefinition getFfunctionDefinition(int index) {
        return ((XbfFfunctionDefinition)ffunctionDefinition_.get(index));
    }

    /**
     * Sets the XbfFfunctionDefinition property <b>FfunctionDefinition</b> by index.
     *
     * @param index
     * @param ffunctionDefinition
     */
    public final void setFfunctionDefinition(int index, XbfFfunctionDefinition ffunctionDefinition) {
        this.ffunctionDefinition_.set(index, ffunctionDefinition);
        if (ffunctionDefinition != null) {
            ffunctionDefinition.rSetParentRNode(this);
        }
    }

    /**
     * Adds the XbfFfunctionDefinition property <b>FfunctionDefinition</b> by index.
     *
     * @param index
     * @param ffunctionDefinition
     */
    public final void addFfunctionDefinition(int index, XbfFfunctionDefinition ffunctionDefinition) {
        this.ffunctionDefinition_.add(index, ffunctionDefinition);
        if (ffunctionDefinition != null) {
            ffunctionDefinition.rSetParentRNode(this);
        }
    }

    /**
     * Remove the XbfFfunctionDefinition property <b>FfunctionDefinition</b> by index.
     *
     * @param index
     */
    public final void removeFfunctionDefinition(int index) {
        this.ffunctionDefinition_.remove(index);
    }

    /**
     * Remove the XbfFfunctionDefinition property <b>FfunctionDefinition</b> by object.
     *
     * @param ffunctionDefinition
     */
    public final void removeFfunctionDefinition(XbfFfunctionDefinition ffunctionDefinition) {
        this.ffunctionDefinition_.remove(ffunctionDefinition);
    }

    /**
     * Clear the XbfFfunctionDefinition property <b>FfunctionDefinition</b>.
     *
     */
    public final void clearFfunctionDefinition() {
        this.ffunctionDefinition_.clear();
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
        buffer.append("<FcontainsStatement");
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
        buffer.append(">");
        size = this.ffunctionDefinition_.size();
        for (int i = 0;i < size;i++) {
            XbfFfunctionDefinition value = (XbfFfunctionDefinition)this.ffunctionDefinition_.get(i);
            value.makeTextElement(buffer);
        }
        buffer.append("</FcontainsStatement>");
    }

    /**
     * Makes an XML text representation.
     *
     * @param buffer
     * @exception IOException
     */
    public void makeTextElement(Writer buffer) throws IOException {
        int size;
        buffer.write("<FcontainsStatement");
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
        buffer.write(">");
        size = this.ffunctionDefinition_.size();
        for (int i = 0;i < size;i++) {
            XbfFfunctionDefinition value = (XbfFfunctionDefinition)this.ffunctionDefinition_.get(i);
            value.makeTextElement(buffer);
        }
        buffer.write("</FcontainsStatement>");
    }

    /**
     * Makes an XML text representation.
     *
     * @param buffer
     */
    public void makeTextElement(PrintWriter buffer) {
        int size;
        buffer.print("<FcontainsStatement");
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
        buffer.print(">");
        size = this.ffunctionDefinition_.size();
        for (int i = 0;i < size;i++) {
            XbfFfunctionDefinition value = (XbfFfunctionDefinition)this.ffunctionDefinition_.get(i);
            value.makeTextElement(buffer);
        }
        buffer.print("</FcontainsStatement>");
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
        classNodes.addAll(ffunctionDefinition_);
        IRNode[] nodes = new IRNode[classNodes.size()];
        return ((IRNode[])classNodes.toArray(nodes));
    }

    /**
     * Tests if a Element <code>element</code> is valid
     * for the <code>XbfFcontainsStatement</code>.
     *
     * @param element
     * @return boolean
     */
    public static boolean isMatch(Element element) {
        if (!URelaxer.isTargetElement(element, "FcontainsStatement")) {
            return (false);
        }
        RStack target = new RStack(element);
        boolean $match$ = false;
        Element child;
        if (!XbfFfunctionDefinition.isMatchHungry(target)) {
            return (false);
        }
        $match$ = true;
        while (true) {
            if (!XbfFfunctionDefinition.isMatchHungry(target)) {
                break;
            }
            $match$ = true;
        }
        if (!target.isEmptyElement()) {
            return (false);
        }
        return (true);
    }

    /**
     * Tests if elements contained in a Stack <code>stack</code>
     * is valid for the <code>XbfFcontainsStatement</code>.
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
     * is valid for the <code>XbfFcontainsStatement</code>.
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
