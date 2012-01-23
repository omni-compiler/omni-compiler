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
 * <b>XbfFunctionCall</b> is generated from XcodeML_F.rng by Relaxer.
 * This class is derived from:
 * 
 * <!-- for programmer
 * <element java:extends="xcodeml.f.XmfObj" java:implements="xcodeml.binding.IXbTypedExpr" name="functionCall">
 *   <ref name="defAttrTypeName"/>
 *   <ref name="defAttrFunction"/>
 *   <ref name="name"/>
 *   <optional>
 *     <ref name="arguments"/>
 *   </optional>
 * </element>
 * -->
 * <!-- for javadoc -->
 * <pre> &lt;element java:extends="xcodeml.f.XmfObj" java:implements="xcodeml.binding.IXbTypedExpr" name="functionCall"&gt;
 *   &lt;ref name="defAttrTypeName"/&gt;
 *   &lt;ref name="defAttrFunction"/&gt;
 *   &lt;ref name="name"/&gt;
 *   &lt;optional&gt;
 *     &lt;ref name="arguments"/&gt;
 *   &lt;/optional&gt;
 * &lt;/element&gt;
 * </pre>
 *
 * @version XcodeML_F.rng (Mon Jan 23 20:53:32 JST 2012)
 * @author  Relaxer 1.0 (http://www.relaxer.org)
 */
public class XbfFunctionCall extends xcodeml.f.XmfObj implements java.io.Serializable, Cloneable, IXbfArgumentsChoice, xcodeml.binding.IXbTypedExpr, IRVisitable, IRNode, IXbfDefModelExprChoice {
    private String type_;
    private Boolean isIntrinsic_;
    private XbfName name_;
    private XbfArguments arguments_;
    private IRNode parentRNode_;

    /**
     * Creates a <code>XbfFunctionCall</code>.
     *
     */
    public XbfFunctionCall() {
    }

    /**
     * Creates a <code>XbfFunctionCall</code>.
     *
     * @param source
     */
    public XbfFunctionCall(XbfFunctionCall source) {
        setup(source);
    }

    /**
     * Creates a <code>XbfFunctionCall</code> by the Stack <code>stack</code>
     * that contains Elements.
     * This constructor is supposed to be used internally
     * by the Relaxer system.
     *
     * @param stack
     */
    public XbfFunctionCall(RStack stack) {
        setup(stack);
    }

    /**
     * Creates a <code>XbfFunctionCall</code> by the Document <code>doc</code>.
     *
     * @param doc
     */
    public XbfFunctionCall(Document doc) {
        setup(doc.getDocumentElement());
    }

    /**
     * Creates a <code>XbfFunctionCall</code> by the Element <code>element</code>.
     *
     * @param element
     */
    public XbfFunctionCall(Element element) {
        setup(element);
    }

    /**
     * Creates a <code>XbfFunctionCall</code> by the File <code>file</code>.
     *
     * @param file
     * @exception IOException
     * @exception SAXException
     * @exception ParserConfigurationException
     */
    public XbfFunctionCall(File file) throws IOException, SAXException, ParserConfigurationException {
        setup(file);
    }

    /**
     * Creates a <code>XbfFunctionCall</code>
     * by the String representation of URI <code>uri</code>.
     *
     * @param uri
     * @exception IOException
     * @exception SAXException
     * @exception ParserConfigurationException
     */
    public XbfFunctionCall(String uri) throws IOException, SAXException, ParserConfigurationException {
        setup(uri);
    }

    /**
     * Creates a <code>XbfFunctionCall</code> by the URL <code>url</code>.
     *
     * @param url
     * @exception IOException
     * @exception SAXException
     * @exception ParserConfigurationException
     */
    public XbfFunctionCall(URL url) throws IOException, SAXException, ParserConfigurationException {
        setup(url);
    }

    /**
     * Creates a <code>XbfFunctionCall</code> by the InputStream <code>in</code>.
     *
     * @param in
     * @exception IOException
     * @exception SAXException
     * @exception ParserConfigurationException
     */
    public XbfFunctionCall(InputStream in) throws IOException, SAXException, ParserConfigurationException {
        setup(in);
    }

    /**
     * Creates a <code>XbfFunctionCall</code> by the InputSource <code>is</code>.
     *
     * @param is
     * @exception IOException
     * @exception SAXException
     * @exception ParserConfigurationException
     */
    public XbfFunctionCall(InputSource is) throws IOException, SAXException, ParserConfigurationException {
        setup(is);
    }

    /**
     * Creates a <code>XbfFunctionCall</code> by the Reader <code>reader</code>.
     *
     * @param reader
     * @exception IOException
     * @exception SAXException
     * @exception ParserConfigurationException
     */
    public XbfFunctionCall(Reader reader) throws IOException, SAXException, ParserConfigurationException {
        setup(reader);
    }

    /**
     * Initializes the <code>XbfFunctionCall</code> by the XbfFunctionCall <code>source</code>.
     *
     * @param source
     */
    public void setup(XbfFunctionCall source) {
        int size;
        setType(source.getType());
        setIsIntrinsic(source.getIsIntrinsic());
        if (source.name_ != null) {
            setName((XbfName)source.getName().clone());
        }
        if (source.arguments_ != null) {
            setArguments((XbfArguments)source.getArguments().clone());
        }
    }

    /**
     * Initializes the <code>XbfFunctionCall</code> by the Document <code>doc</code>.
     *
     * @param doc
     */
    public void setup(Document doc) {
        setup(doc.getDocumentElement());
    }

    /**
     * Initializes the <code>XbfFunctionCall</code> by the Element <code>element</code>.
     *
     * @param element
     */
    public void setup(Element element) {
        init(element);
    }

    /**
     * Initializes the <code>XbfFunctionCall</code> by the Stack <code>stack</code>
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
        type_ = URelaxer.getAttributePropertyAsString(element, "type");
        isIntrinsic_ = URelaxer.getAttributePropertyAsBooleanObject(element, "is_intrinsic");
        setName(factory.createXbfName(stack));
        if (XbfArguments.isMatch(stack)) {
            setArguments(factory.createXbfArguments(stack));
        }
    }

    /**
     * @return Object
     */
    public Object clone() {
        IXcodeML_FFactory factory = XcodeML_FFactory.getFactory();
        return (factory.createXbfFunctionCall(this));
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
        Element element = doc.createElement("functionCall");
        int size;
        if (this.type_ != null) {
            URelaxer.setAttributePropertyByString(element, "type", this.type_);
        }
        if (this.isIntrinsic_ != null) {
            URelaxer.setAttributePropertyByBoolean(element, "is_intrinsic", this.isIntrinsic_);
        }
        this.name_.makeElement(element);
        if (this.arguments_ != null) {
            this.arguments_.makeElement(element);
        }
        parent.appendChild(element);
    }

    /**
     * Initializes the <code>XbfFunctionCall</code> by the File <code>file</code>.
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
     * Initializes the <code>XbfFunctionCall</code>
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
     * Initializes the <code>XbfFunctionCall</code> by the URL <code>url</code>.
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
     * Initializes the <code>XbfFunctionCall</code> by the InputStream <code>in</code>.
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
     * Initializes the <code>XbfFunctionCall</code> by the InputSource <code>is</code>.
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
     * Initializes the <code>XbfFunctionCall</code> by the Reader <code>reader</code>.
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
     * Gets the boolean property <b>isIntrinsic</b>.
     *
     * @return boolean
     */
    public boolean getIsIntrinsic() {
        if (isIntrinsic_ == null) {
            return(false);
        }
        return (isIntrinsic_.booleanValue());
    }

    /**
     * Gets the boolean property <b>isIntrinsic</b>.
     *
     * @param isIntrinsic
     * @return boolean
     */
    public boolean getIsIntrinsic(boolean isIntrinsic) {
        if (isIntrinsic_ == null) {
            return(isIntrinsic);
        }
        return (this.isIntrinsic_.booleanValue());
    }

    /**
     * Gets the boolean property <b>isIntrinsic</b>.
     *
     * @return Boolean
     */
    public Boolean getIsIntrinsicAsBoolean() {
        return (isIntrinsic_);
    }

    /**
     * Check the boolean property <b>isIntrinsic</b>.
     *
     * @return boolean
     */
    public boolean checkIsIntrinsic() {
        return (isIntrinsic_ != null);
    }

    /**
     * Sets the boolean property <b>isIntrinsic</b>.
     *
     * @param isIntrinsic
     */
    public void setIsIntrinsic(boolean isIntrinsic) {
        this.isIntrinsic_ = new Boolean(isIntrinsic);
    }

    /**
     * Sets the boolean property <b>isIntrinsic</b>.
     *
     * @param isIntrinsic
     */
    public void setIsIntrinsic(Boolean isIntrinsic) {
        this.isIntrinsic_ = isIntrinsic;
    }

    /**
     * Gets the XbfName property <b>name</b>.
     *
     * @return XbfName
     */
    public final XbfName getName() {
        return (name_);
    }

    /**
     * Sets the XbfName property <b>name</b>.
     *
     * @param name
     */
    public final void setName(XbfName name) {
        this.name_ = name;
        if (name != null) {
            name.rSetParentRNode(this);
        }
    }

    /**
     * Gets the XbfArguments property <b>arguments</b>.
     *
     * @return XbfArguments
     */
    public final XbfArguments getArguments() {
        return (arguments_);
    }

    /**
     * Sets the XbfArguments property <b>arguments</b>.
     *
     * @param arguments
     */
    public final void setArguments(XbfArguments arguments) {
        this.arguments_ = arguments;
        if (arguments != null) {
            arguments.rSetParentRNode(this);
        }
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
        buffer.append("<functionCall");
        if (type_ != null) {
            buffer.append(" type=\"");
            buffer.append(URelaxer.escapeAttrQuot(URelaxer.getString(getType())));
            buffer.append("\"");
        }
        if (isIntrinsic_ != null) {
            buffer.append(" is_intrinsic=\"");
            buffer.append(URelaxer.getString(getIsIntrinsic()));
            buffer.append("\"");
        }
        buffer.append(">");
        name_.makeTextElement(buffer);
        if (arguments_ != null) {
            arguments_.makeTextElement(buffer);
        }
        buffer.append("</functionCall>");
    }

    /**
     * Makes an XML text representation.
     *
     * @param buffer
     * @exception IOException
     */
    public void makeTextElement(Writer buffer) throws IOException {
        int size;
        buffer.write("<functionCall");
        if (type_ != null) {
            buffer.write(" type=\"");
            buffer.write(URelaxer.escapeAttrQuot(URelaxer.getString(getType())));
            buffer.write("\"");
        }
        if (isIntrinsic_ != null) {
            buffer.write(" is_intrinsic=\"");
            buffer.write(URelaxer.getString(getIsIntrinsic()));
            buffer.write("\"");
        }
        buffer.write(">");
        name_.makeTextElement(buffer);
        if (arguments_ != null) {
            arguments_.makeTextElement(buffer);
        }
        buffer.write("</functionCall>");
    }

    /**
     * Makes an XML text representation.
     *
     * @param buffer
     */
    public void makeTextElement(PrintWriter buffer) {
        int size;
        buffer.print("<functionCall");
        if (type_ != null) {
            buffer.print(" type=\"");
            buffer.print(URelaxer.escapeAttrQuot(URelaxer.getString(getType())));
            buffer.print("\"");
        }
        if (isIntrinsic_ != null) {
            buffer.print(" is_intrinsic=\"");
            buffer.print(URelaxer.getString(getIsIntrinsic()));
            buffer.print("\"");
        }
        buffer.print(">");
        name_.makeTextElement(buffer);
        if (arguments_ != null) {
            arguments_.makeTextElement(buffer);
        }
        buffer.print("</functionCall>");
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
    public String getTypeAsString() {
        return (URelaxer.getString(getType()));
    }

    /**
     * Gets the property value as String.
     *
     * @return String
     */
    public String getIsIntrinsicAsString() {
        return (URelaxer.getString(getIsIntrinsic()));
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
    public void setIsIntrinsicByString(String string) {
        setIsIntrinsic(new Boolean(string).booleanValue());
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
        if (name_ != null) {
            classNodes.add(name_);
        }
        if (arguments_ != null) {
            classNodes.add(arguments_);
        }
        IRNode[] nodes = new IRNode[classNodes.size()];
        return ((IRNode[])classNodes.toArray(nodes));
    }

    /**
     * Tests if a Element <code>element</code> is valid
     * for the <code>XbfFunctionCall</code>.
     *
     * @param element
     * @return boolean
     */
    public static boolean isMatch(Element element) {
        if (!URelaxer.isTargetElement(element, "functionCall")) {
            return (false);
        }
        RStack target = new RStack(element);
        boolean $match$ = false;
        Element child;
        if (!XbfName.isMatchHungry(target)) {
            return (false);
        }
        $match$ = true;
        if (XbfArguments.isMatchHungry(target)) {
        }
        if (!target.isEmptyElement()) {
            return (false);
        }
        return (true);
    }

    /**
     * Tests if elements contained in a Stack <code>stack</code>
     * is valid for the <code>XbfFunctionCall</code>.
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
     * is valid for the <code>XbfFunctionCall</code>.
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
