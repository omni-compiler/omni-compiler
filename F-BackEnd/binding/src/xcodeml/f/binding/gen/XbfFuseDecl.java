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
 * <b>XbfFuseDecl</b> is generated from XcodeML_F.rng by Relaxer.
 * This class is derived from:
 * 
 * <!-- for programmer
 * <element java:extends="xcodeml.f.XmfObj" java:implements="xcodeml.binding.IXbLineNo" name="FuseDecl">
 *   <ref name="defAttrSourceLine"/>
 *   <ref name="defAttrGenericName"/>
 *   <zeroOrMore>
 *     <ref name="rename"/>
 *   </zeroOrMore>
 * </element>
 * -->
 * <!-- for javadoc -->
 * <pre> &lt;element java:extends="xcodeml.f.XmfObj" java:implements="xcodeml.binding.IXbLineNo" name="FuseDecl"&gt;
 *   &lt;ref name="defAttrSourceLine"/&gt;
 *   &lt;ref name="defAttrGenericName"/&gt;
 *   &lt;zeroOrMore&gt;
 *     &lt;ref name="rename"/&gt;
 *   &lt;/zeroOrMore&gt;
 * &lt;/element&gt;
 * </pre>
 *
 * @version XcodeML_F.rng (Mon Jan 23 20:53:32 JST 2012)
 * @author  Relaxer 1.0 (http://www.relaxer.org)
 */
public class XbfFuseDecl extends xcodeml.f.XmfObj implements java.io.Serializable, Cloneable, xcodeml.binding.IXbLineNo, IRVisitable, IRNode, IXbfDeclarationsChoice {
    private String lineno_;
    private String endlineno_;
    private String rawlineno_;
    private String file_;
    private String name_;
    // List<XbfRename>
    private java.util.List rename_ = new java.util.ArrayList();
    private IRNode parentRNode_;

    /**
     * Creates a <code>XbfFuseDecl</code>.
     *
     */
    public XbfFuseDecl() {
    }

    /**
     * Creates a <code>XbfFuseDecl</code>.
     *
     * @param source
     */
    public XbfFuseDecl(XbfFuseDecl source) {
        setup(source);
    }

    /**
     * Creates a <code>XbfFuseDecl</code> by the Stack <code>stack</code>
     * that contains Elements.
     * This constructor is supposed to be used internally
     * by the Relaxer system.
     *
     * @param stack
     */
    public XbfFuseDecl(RStack stack) {
        setup(stack);
    }

    /**
     * Creates a <code>XbfFuseDecl</code> by the Document <code>doc</code>.
     *
     * @param doc
     */
    public XbfFuseDecl(Document doc) {
        setup(doc.getDocumentElement());
    }

    /**
     * Creates a <code>XbfFuseDecl</code> by the Element <code>element</code>.
     *
     * @param element
     */
    public XbfFuseDecl(Element element) {
        setup(element);
    }

    /**
     * Creates a <code>XbfFuseDecl</code> by the File <code>file</code>.
     *
     * @param file
     * @exception IOException
     * @exception SAXException
     * @exception ParserConfigurationException
     */
    public XbfFuseDecl(File file) throws IOException, SAXException, ParserConfigurationException {
        setup(file);
    }

    /**
     * Creates a <code>XbfFuseDecl</code>
     * by the String representation of URI <code>uri</code>.
     *
     * @param uri
     * @exception IOException
     * @exception SAXException
     * @exception ParserConfigurationException
     */
    public XbfFuseDecl(String uri) throws IOException, SAXException, ParserConfigurationException {
        setup(uri);
    }

    /**
     * Creates a <code>XbfFuseDecl</code> by the URL <code>url</code>.
     *
     * @param url
     * @exception IOException
     * @exception SAXException
     * @exception ParserConfigurationException
     */
    public XbfFuseDecl(URL url) throws IOException, SAXException, ParserConfigurationException {
        setup(url);
    }

    /**
     * Creates a <code>XbfFuseDecl</code> by the InputStream <code>in</code>.
     *
     * @param in
     * @exception IOException
     * @exception SAXException
     * @exception ParserConfigurationException
     */
    public XbfFuseDecl(InputStream in) throws IOException, SAXException, ParserConfigurationException {
        setup(in);
    }

    /**
     * Creates a <code>XbfFuseDecl</code> by the InputSource <code>is</code>.
     *
     * @param is
     * @exception IOException
     * @exception SAXException
     * @exception ParserConfigurationException
     */
    public XbfFuseDecl(InputSource is) throws IOException, SAXException, ParserConfigurationException {
        setup(is);
    }

    /**
     * Creates a <code>XbfFuseDecl</code> by the Reader <code>reader</code>.
     *
     * @param reader
     * @exception IOException
     * @exception SAXException
     * @exception ParserConfigurationException
     */
    public XbfFuseDecl(Reader reader) throws IOException, SAXException, ParserConfigurationException {
        setup(reader);
    }

    /**
     * Initializes the <code>XbfFuseDecl</code> by the XbfFuseDecl <code>source</code>.
     *
     * @param source
     */
    public void setup(XbfFuseDecl source) {
        int size;
        setLineno(source.getLineno());
        setEndlineno(source.getEndlineno());
        setRawlineno(source.getRawlineno());
        setFile(source.getFile());
        setName(source.getName());
        this.rename_.clear();
        size = source.rename_.size();
        for (int i = 0;i < size;i++) {
            addRename((XbfRename)source.getRename(i).clone());
        }
    }

    /**
     * Initializes the <code>XbfFuseDecl</code> by the Document <code>doc</code>.
     *
     * @param doc
     */
    public void setup(Document doc) {
        setup(doc.getDocumentElement());
    }

    /**
     * Initializes the <code>XbfFuseDecl</code> by the Element <code>element</code>.
     *
     * @param element
     */
    public void setup(Element element) {
        init(element);
    }

    /**
     * Initializes the <code>XbfFuseDecl</code> by the Stack <code>stack</code>
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
        endlineno_ = URelaxer.getAttributePropertyAsString(element, "endlineno");
        rawlineno_ = URelaxer.getAttributePropertyAsString(element, "rawlineno");
        file_ = URelaxer.getAttributePropertyAsString(element, "file");
        name_ = URelaxer.getAttributePropertyAsString(element, "name");
        rename_.clear();
        while (true) {
            if (XbfRename.isMatch(stack)) {
                addRename(factory.createXbfRename(stack));
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
        return (factory.createXbfFuseDecl(this));
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
        Element element = doc.createElement("FuseDecl");
        int size;
        if (this.lineno_ != null) {
            URelaxer.setAttributePropertyByString(element, "lineno", this.lineno_);
        }
        if (this.endlineno_ != null) {
            URelaxer.setAttributePropertyByString(element, "endlineno", this.endlineno_);
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
        size = this.rename_.size();
        for (int i = 0;i < size;i++) {
            XbfRename value = (XbfRename)this.rename_.get(i);
            value.makeElement(element);
        }
        parent.appendChild(element);
    }

    /**
     * Initializes the <code>XbfFuseDecl</code> by the File <code>file</code>.
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
     * Initializes the <code>XbfFuseDecl</code>
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
     * Initializes the <code>XbfFuseDecl</code> by the URL <code>url</code>.
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
     * Initializes the <code>XbfFuseDecl</code> by the InputStream <code>in</code>.
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
     * Initializes the <code>XbfFuseDecl</code> by the InputSource <code>is</code>.
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
     * Initializes the <code>XbfFuseDecl</code> by the Reader <code>reader</code>.
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
     * Gets the String property <b>endlineno</b>.
     *
     * @return String
     */
    public final String getEndlineno() {
        return (endlineno_);
    }

    /**
     * Sets the String property <b>endlineno</b>.
     *
     * @param endlineno
     */
    public final void setEndlineno(String endlineno) {
        this.endlineno_ = endlineno;
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
     * Gets the XbfRename property <b>rename</b>.
     *
     * @return XbfRename[]
     */
    public final XbfRename[] getRename() {
        XbfRename[] array = new XbfRename[rename_.size()];
        return ((XbfRename[])rename_.toArray(array));
    }

    /**
     * Sets the XbfRename property <b>rename</b>.
     *
     * @param rename
     */
    public final void setRename(XbfRename[] rename) {
        this.rename_.clear();
        for (int i = 0;i < rename.length;i++) {
            addRename(rename[i]);
        }
        for (int i = 0;i < rename.length;i++) {
            rename[i].rSetParentRNode(this);
        }
    }

    /**
     * Sets the XbfRename property <b>rename</b>.
     *
     * @param rename
     */
    public final void setRename(XbfRename rename) {
        this.rename_.clear();
        addRename(rename);
        if (rename != null) {
            rename.rSetParentRNode(this);
        }
    }

    /**
     * Adds the XbfRename property <b>rename</b>.
     *
     * @param rename
     */
    public final void addRename(XbfRename rename) {
        this.rename_.add(rename);
        if (rename != null) {
            rename.rSetParentRNode(this);
        }
    }

    /**
     * Adds the XbfRename property <b>rename</b>.
     *
     * @param rename
     */
    public final void addRename(XbfRename[] rename) {
        for (int i = 0;i < rename.length;i++) {
            addRename(rename[i]);
        }
        for (int i = 0;i < rename.length;i++) {
            rename[i].rSetParentRNode(this);
        }
    }

    /**
     * Gets number of the XbfRename property <b>rename</b>.
     *
     * @return int
     */
    public final int sizeRename() {
        return (rename_.size());
    }

    /**
     * Gets the XbfRename property <b>rename</b> by index.
     *
     * @param index
     * @return XbfRename
     */
    public final XbfRename getRename(int index) {
        return ((XbfRename)rename_.get(index));
    }

    /**
     * Sets the XbfRename property <b>rename</b> by index.
     *
     * @param index
     * @param rename
     */
    public final void setRename(int index, XbfRename rename) {
        this.rename_.set(index, rename);
        if (rename != null) {
            rename.rSetParentRNode(this);
        }
    }

    /**
     * Adds the XbfRename property <b>rename</b> by index.
     *
     * @param index
     * @param rename
     */
    public final void addRename(int index, XbfRename rename) {
        this.rename_.add(index, rename);
        if (rename != null) {
            rename.rSetParentRNode(this);
        }
    }

    /**
     * Remove the XbfRename property <b>rename</b> by index.
     *
     * @param index
     */
    public final void removeRename(int index) {
        this.rename_.remove(index);
    }

    /**
     * Remove the XbfRename property <b>rename</b> by object.
     *
     * @param rename
     */
    public final void removeRename(XbfRename rename) {
        this.rename_.remove(rename);
    }

    /**
     * Clear the XbfRename property <b>rename</b>.
     *
     */
    public final void clearRename() {
        this.rename_.clear();
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
        buffer.append("<FuseDecl");
        if (lineno_ != null) {
            buffer.append(" lineno=\"");
            buffer.append(URelaxer.escapeAttrQuot(URelaxer.getString(getLineno())));
            buffer.append("\"");
        }
        if (endlineno_ != null) {
            buffer.append(" endlineno=\"");
            buffer.append(URelaxer.escapeAttrQuot(URelaxer.getString(getEndlineno())));
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
        buffer.append(">");
        size = this.rename_.size();
        for (int i = 0;i < size;i++) {
            XbfRename value = (XbfRename)this.rename_.get(i);
            value.makeTextElement(buffer);
        }
        buffer.append("</FuseDecl>");
    }

    /**
     * Makes an XML text representation.
     *
     * @param buffer
     * @exception IOException
     */
    public void makeTextElement(Writer buffer) throws IOException {
        int size;
        buffer.write("<FuseDecl");
        if (lineno_ != null) {
            buffer.write(" lineno=\"");
            buffer.write(URelaxer.escapeAttrQuot(URelaxer.getString(getLineno())));
            buffer.write("\"");
        }
        if (endlineno_ != null) {
            buffer.write(" endlineno=\"");
            buffer.write(URelaxer.escapeAttrQuot(URelaxer.getString(getEndlineno())));
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
        buffer.write(">");
        size = this.rename_.size();
        for (int i = 0;i < size;i++) {
            XbfRename value = (XbfRename)this.rename_.get(i);
            value.makeTextElement(buffer);
        }
        buffer.write("</FuseDecl>");
    }

    /**
     * Makes an XML text representation.
     *
     * @param buffer
     */
    public void makeTextElement(PrintWriter buffer) {
        int size;
        buffer.print("<FuseDecl");
        if (lineno_ != null) {
            buffer.print(" lineno=\"");
            buffer.print(URelaxer.escapeAttrQuot(URelaxer.getString(getLineno())));
            buffer.print("\"");
        }
        if (endlineno_ != null) {
            buffer.print(" endlineno=\"");
            buffer.print(URelaxer.escapeAttrQuot(URelaxer.getString(getEndlineno())));
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
        buffer.print(">");
        size = this.rename_.size();
        for (int i = 0;i < size;i++) {
            XbfRename value = (XbfRename)this.rename_.get(i);
            value.makeTextElement(buffer);
        }
        buffer.print("</FuseDecl>");
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
    public String getEndlinenoAsString() {
        return (URelaxer.getString(getEndlineno()));
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
    public void setEndlinenoByString(String string) {
        setEndlineno(string);
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
        classNodes.addAll(rename_);
        IRNode[] nodes = new IRNode[classNodes.size()];
        return ((IRNode[])classNodes.toArray(nodes));
    }

    /**
     * Tests if a Element <code>element</code> is valid
     * for the <code>XbfFuseDecl</code>.
     *
     * @param element
     * @return boolean
     */
    public static boolean isMatch(Element element) {
        if (!URelaxer.isTargetElement(element, "FuseDecl")) {
            return (false);
        }
        RStack target = new RStack(element);
        boolean $match$ = false;
        Element child;
        while (true) {
            if (!XbfRename.isMatchHungry(target)) {
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
     * is valid for the <code>XbfFuseDecl</code>.
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
     * is valid for the <code>XbfFuseDecl</code>.
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
