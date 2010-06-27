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

import java.io.*;
import java.io.IOException;
import java.io.PrintWriter;
import java.io.Writer;
import java.net.URL;
import javax.xml.parsers.*;
import org.w3c.dom.*;
import org.xml.sax.*;

/**
 * <b>XbcArrayRef</b> is generated from XcodeML_C.rng by Relaxer.
 * This class is derived from:
 * 
 * <!-- for programmer
 * <element java:extends="xcodeml.c.obj.XmcObj" java:implements="xcodeml.c.binding.IXbcVar" name="arrayRef">
 *   <ref name="BaseExpression"/>
 *   <ref name="varRefExpression"/>
 *   <data type="string"/>
 * </element>
 * -->
 * <!-- for javadoc -->
 * <pre> &lt;element java:extends="xcodeml.c.obj.XmcObj" java:implements="xcodeml.c.binding.IXbcVar" name="arrayRef"&gt;
 *   &lt;ref name="BaseExpression"/&gt;
 *   &lt;ref name="varRefExpression"/&gt;
 *   &lt;data type="string"/&gt;
 * &lt;/element&gt;
 * </pre>
 *
 * @version XcodeML_C.rng (Thu Sep 24 16:30:18 JST 2009)
 * @author  Relaxer 1.0 (http://www.relaxer.org)
 */
public class XbcArrayRef extends xcodeml.c.obj.XmcObj implements java.io.Serializable, Cloneable, xcodeml.c.binding.IXbcVar, IRVisitable, IRNode, IXbcCastExprChoice, IXbcExprOrTypeChoice, IXbcValueChoice, IXbcDesignatedValueChoice, IXbcCompoundLiteralChoice, IXbcGotoStatementChoice, IXbcBuiltinOpChoice, IXbcExpressionsChoice {
    public static final String ISGCCSYNTAX_0 = "0";
    public static final String ISGCCSYNTAX_1 = "1";
    public static final String ISGCCSYNTAX_TRUE = "true";
    public static final String ISGCCSYNTAX_FALSE = "false";
    public static final String ISMODIFIED_0 = "0";
    public static final String ISMODIFIED_1 = "1";
    public static final String ISMODIFIED_TRUE = "true";
    public static final String ISMODIFIED_FALSE = "false";
    public static final String SCOPE_GLOBAL = "global";
    public static final String SCOPE_LOCAL = "local";
    public static final String SCOPE_PARAM = "param";

    private String content_;
    private String type_;
    private String isGccSyntax_;
    private String isModified_;
    private String scope_;
    private IRNode parentRNode_;

    /**
     * Creates a <code>XbcArrayRef</code>.
     *
     */
    public XbcArrayRef() {
    }

    /**
     * Creates a <code>XbcArrayRef</code>.
     *
     * @param source
     */
    public XbcArrayRef(XbcArrayRef source) {
        setup(source);
    }

    /**
     * Creates a <code>XbcArrayRef</code> by the Stack <code>stack</code>
     * that contains Elements.
     * This constructor is supposed to be used internally
     * by the Relaxer system.
     *
     * @param stack
     */
    public XbcArrayRef(RStack stack) {
        setup(stack);
    }

    /**
     * Creates a <code>XbcArrayRef</code> by the Document <code>doc</code>.
     *
     * @param doc
     */
    public XbcArrayRef(Document doc) {
        setup(doc.getDocumentElement());
    }

    /**
     * Creates a <code>XbcArrayRef</code> by the Element <code>element</code>.
     *
     * @param element
     */
    public XbcArrayRef(Element element) {
        setup(element);
    }

    /**
     * Creates a <code>XbcArrayRef</code> by the File <code>file</code>.
     *
     * @param file
     * @exception IOException
     * @exception SAXException
     * @exception ParserConfigurationException
     */
    public XbcArrayRef(File file) throws IOException, SAXException, ParserConfigurationException {
        setup(file);
    }

    /**
     * Creates a <code>XbcArrayRef</code>
     * by the String representation of URI <code>uri</code>.
     *
     * @param uri
     * @exception IOException
     * @exception SAXException
     * @exception ParserConfigurationException
     */
    public XbcArrayRef(String uri) throws IOException, SAXException, ParserConfigurationException {
        setup(uri);
    }

    /**
     * Creates a <code>XbcArrayRef</code> by the URL <code>url</code>.
     *
     * @param url
     * @exception IOException
     * @exception SAXException
     * @exception ParserConfigurationException
     */
    public XbcArrayRef(URL url) throws IOException, SAXException, ParserConfigurationException {
        setup(url);
    }

    /**
     * Creates a <code>XbcArrayRef</code> by the InputStream <code>in</code>.
     *
     * @param in
     * @exception IOException
     * @exception SAXException
     * @exception ParserConfigurationException
     */
    public XbcArrayRef(InputStream in) throws IOException, SAXException, ParserConfigurationException {
        setup(in);
    }

    /**
     * Creates a <code>XbcArrayRef</code> by the InputSource <code>is</code>.
     *
     * @param is
     * @exception IOException
     * @exception SAXException
     * @exception ParserConfigurationException
     */
    public XbcArrayRef(InputSource is) throws IOException, SAXException, ParserConfigurationException {
        setup(is);
    }

    /**
     * Creates a <code>XbcArrayRef</code> by the Reader <code>reader</code>.
     *
     * @param reader
     * @exception IOException
     * @exception SAXException
     * @exception ParserConfigurationException
     */
    public XbcArrayRef(Reader reader) throws IOException, SAXException, ParserConfigurationException {
        setup(reader);
    }

    /**
     * Initializes the <code>XbcArrayRef</code> by the XbcArrayRef <code>source</code>.
     *
     * @param source
     */
    public void setup(XbcArrayRef source) {
        int size;
        setContent(source.getContent());
        setType(source.getType());
        setIsGccSyntax(source.getIsGccSyntax());
        setIsModified(source.getIsModified());
        setScope(source.getScope());
    }

    /**
     * Initializes the <code>XbcArrayRef</code> by the Document <code>doc</code>.
     *
     * @param doc
     */
    public void setup(Document doc) {
        setup(doc.getDocumentElement());
    }

    /**
     * Initializes the <code>XbcArrayRef</code> by the Element <code>element</code>.
     *
     * @param element
     */
    public void setup(Element element) {
        init(element);
    }

    /**
     * Initializes the <code>XbcArrayRef</code> by the Stack <code>stack</code>
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
        IXcodeML_CFactory factory = XcodeML_CFactory.getFactory();
        RStack stack = new RStack(element);
        content_ = URelaxer.getElementPropertyAsString(element);
        type_ = URelaxer.getAttributePropertyAsString(element, "type");
        isGccSyntax_ = URelaxer.getAttributePropertyAsString(element, "is_gccSyntax");
        isModified_ = URelaxer.getAttributePropertyAsString(element, "is_modified");
        scope_ = URelaxer.getAttributePropertyAsString(element, "scope");
    }

    /**
     * @return Object
     */
    public Object clone() {
        IXcodeML_CFactory factory = XcodeML_CFactory.getFactory();
        return (factory.createXbcArrayRef(this));
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
        Element element = doc.createElement("arrayRef");
        URelaxer.setElementPropertyByString(element, this.content_);
        int size;
        if (this.type_ != null) {
            URelaxer.setAttributePropertyByString(element, "type", this.type_);
        }
        if (this.isGccSyntax_ != null) {
            URelaxer.setAttributePropertyByString(element, "is_gccSyntax", this.isGccSyntax_);
        }
        if (this.isModified_ != null) {
            URelaxer.setAttributePropertyByString(element, "is_modified", this.isModified_);
        }
        if (this.scope_ != null) {
            URelaxer.setAttributePropertyByString(element, "scope", this.scope_);
        }
        parent.appendChild(element);
    }

    /**
     * Initializes the <code>XbcArrayRef</code> by the File <code>file</code>.
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
     * Initializes the <code>XbcArrayRef</code>
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
     * Initializes the <code>XbcArrayRef</code> by the URL <code>url</code>.
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
     * Initializes the <code>XbcArrayRef</code> by the InputStream <code>in</code>.
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
     * Initializes the <code>XbcArrayRef</code> by the InputSource <code>is</code>.
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
     * Initializes the <code>XbcArrayRef</code> by the Reader <code>reader</code>.
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
     * Gets the String property <b>content</b>.
     *
     * @return String
     */
    public final String getContent() {
        return (content_);
    }

    /**
     * Sets the String property <b>content</b>.
     *
     * @param content
     */
    public final void setContent(String content) {
        this.content_ = content;
    }

    /**
     * Gets the String property <b>type</b>.
     *
     * @return String
     */
    public final String getType() {
        return (type_);
    }

    /**
     * Sets the String property <b>type</b>.
     *
     * @param type
     */
    public final void setType(String type) {
        this.type_ = type;
    }

    /**
     * Gets the String property <b>isGccSyntax</b>.
     *
     * @return String
     */
    public final String getIsGccSyntax() {
        return (isGccSyntax_);
    }

    /**
     * Sets the String property <b>isGccSyntax</b>.
     *
     * @param isGccSyntax
     */
    public final void setIsGccSyntax(String isGccSyntax) {
        this.isGccSyntax_ = isGccSyntax;
    }

    /**
     * Gets the String property <b>isModified</b>.
     *
     * @return String
     */
    public final String getIsModified() {
        return (isModified_);
    }

    /**
     * Sets the String property <b>isModified</b>.
     *
     * @param isModified
     */
    public final void setIsModified(String isModified) {
        this.isModified_ = isModified;
    }

    /**
     * Gets the String property <b>scope</b>.
     *
     * @return String
     */
    public final String getScope() {
        return (scope_);
    }

    /**
     * Sets the String property <b>scope</b>.
     *
     * @param scope
     */
    public final void setScope(String scope) {
        this.scope_ = scope;
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
        buffer.append("<arrayRef");
        if (type_ != null) {
            buffer.append(" type=\"");
            buffer.append(URelaxer.escapeAttrQuot(URelaxer.getString(getType())));
            buffer.append("\"");
        }
        if (isGccSyntax_ != null) {
            buffer.append(" is_gccSyntax=\"");
            buffer.append(URelaxer.escapeAttrQuot(URelaxer.getString(getIsGccSyntax())));
            buffer.append("\"");
        }
        if (isModified_ != null) {
            buffer.append(" is_modified=\"");
            buffer.append(URelaxer.escapeAttrQuot(URelaxer.getString(getIsModified())));
            buffer.append("\"");
        }
        if (scope_ != null) {
            buffer.append(" scope=\"");
            buffer.append(URelaxer.escapeAttrQuot(URelaxer.getString(getScope())));
            buffer.append("\"");
        }
        buffer.append(">");
        buffer.append(URelaxer.escapeCharData(URelaxer.getString(getContent())));
        buffer.append("</arrayRef>");
    }

    /**
     * Makes an XML text representation.
     *
     * @param buffer
     * @exception IOException
     */
    public void makeTextElement(Writer buffer) throws IOException {
        int size;
        buffer.write("<arrayRef");
        if (type_ != null) {
            buffer.write(" type=\"");
            buffer.write(URelaxer.escapeAttrQuot(URelaxer.getString(getType())));
            buffer.write("\"");
        }
        if (isGccSyntax_ != null) {
            buffer.write(" is_gccSyntax=\"");
            buffer.write(URelaxer.escapeAttrQuot(URelaxer.getString(getIsGccSyntax())));
            buffer.write("\"");
        }
        if (isModified_ != null) {
            buffer.write(" is_modified=\"");
            buffer.write(URelaxer.escapeAttrQuot(URelaxer.getString(getIsModified())));
            buffer.write("\"");
        }
        if (scope_ != null) {
            buffer.write(" scope=\"");
            buffer.write(URelaxer.escapeAttrQuot(URelaxer.getString(getScope())));
            buffer.write("\"");
        }
        buffer.write(">");
        buffer.write(URelaxer.escapeCharData(URelaxer.getString(getContent())));
        buffer.write("</arrayRef>");
    }

    /**
     * Makes an XML text representation.
     *
     * @param buffer
     */
    public void makeTextElement(PrintWriter buffer) {
        int size;
        buffer.print("<arrayRef");
        if (type_ != null) {
            buffer.print(" type=\"");
            buffer.print(URelaxer.escapeAttrQuot(URelaxer.getString(getType())));
            buffer.print("\"");
        }
        if (isGccSyntax_ != null) {
            buffer.print(" is_gccSyntax=\"");
            buffer.print(URelaxer.escapeAttrQuot(URelaxer.getString(getIsGccSyntax())));
            buffer.print("\"");
        }
        if (isModified_ != null) {
            buffer.print(" is_modified=\"");
            buffer.print(URelaxer.escapeAttrQuot(URelaxer.getString(getIsModified())));
            buffer.print("\"");
        }
        if (scope_ != null) {
            buffer.print(" scope=\"");
            buffer.print(URelaxer.escapeAttrQuot(URelaxer.getString(getScope())));
            buffer.print("\"");
        }
        buffer.print(">");
        buffer.print(URelaxer.escapeCharData(URelaxer.getString(getContent())));
        buffer.print("</arrayRef>");
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
    public String getContentAsString() {
        return (URelaxer.getString(getContent()));
    }

    /**
     * Gets the property value as String.
     *
     * @return String
     */
    public String getTypeAsString() {
        return (URelaxer.getString(getType()));
    }

    /**
     * Gets the property value as String.
     *
     * @return String
     */
    public String getIsGccSyntaxAsString() {
        return (URelaxer.getString(getIsGccSyntax()));
    }

    /**
     * Gets the property value as String.
     *
     * @return String
     */
    public String getIsModifiedAsString() {
        return (URelaxer.getString(getIsModified()));
    }

    /**
     * Gets the property value as String.
     *
     * @return String
     */
    public String getScopeAsString() {
        return (URelaxer.getString(getScope()));
    }

    /**
     * Sets the property value by String.
     *
     * @param string
     */
    public void setContentByString(String string) {
        setContent(string);
    }

    /**
     * Sets the property value by String.
     *
     * @param string
     */
    public void setTypeByString(String string) {
        setType(string);
    }

    /**
     * Sets the property value by String.
     *
     * @param string
     */
    public void setIsGccSyntaxByString(String string) {
        setIsGccSyntax(string);
    }

    /**
     * Sets the property value by String.
     *
     * @param string
     */
    public void setIsModifiedByString(String string) {
        setIsModified(string);
    }

    /**
     * Sets the property value by String.
     *
     * @param string
     */
    public void setScopeByString(String string) {
        setScope(string);
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
        IRNode[] nodes = new IRNode[classNodes.size()];
        return ((IRNode[])classNodes.toArray(nodes));
    }

    /**
     * Tests if a Element <code>element</code> is valid
     * for the <code>XbcArrayRef</code>.
     *
     * @param element
     * @return boolean
     */
    public static boolean isMatch(Element element) {
        if (!URelaxer.isTargetElement(element, "arrayRef")) {
            return (false);
        }
        RStack target = new RStack(element);
        boolean $match$ = false;
        Element child;
        if (!target.isEmptyElement()) {
            return (false);
        }
        return (true);
    }

    /**
     * Tests if elements contained in a Stack <code>stack</code>
     * is valid for the <code>XbcArrayRef</code>.
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
     * is valid for the <code>XbcArrayRef</code>.
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
