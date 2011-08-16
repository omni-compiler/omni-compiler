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
 * <b>XbcCoArrayType</b> is generated from XcodeML_C.rng by Relaxer.
 * This class is derived from:
 * 
 * <!-- for programmer
 * <element java:extends="xcodeml.c.obj.XmcObj" java:implements="xcodeml.c.binding.IXbcArrayType" name="coArrayType">
 *   <ref name="type"/>
 *   <attribute name="element_type">
 *     <data type="string"/>
 *   </attribute>
 *   <optional>
 *     <attribute name="array_size">
 *       <data type="string"/>
 *     </attribute>
 *   </optional>
 *   <optional>
 *     <ref name="arraySize"/>
 *   </optional>
 * </element>
 * -->
 * <!-- for javadoc -->
 * <pre> &lt;element java:extends="xcodeml.c.obj.XmcObj" java:implements="xcodeml.c.binding.IXbcArrayType" name="coArrayType"&gt;
 *   &lt;ref name="type"/&gt;
 *   &lt;attribute name="element_type"&gt;
 *     &lt;data type="string"/&gt;
 *   &lt;/attribute&gt;
 *   &lt;optional&gt;
 *     &lt;attribute name="array_size"&gt;
 *       &lt;data type="string"/&gt;
 *     &lt;/attribute&gt;
 *   &lt;/optional&gt;
 *   &lt;optional&gt;
 *     &lt;ref name="arraySize"/&gt;
 *   &lt;/optional&gt;
 * &lt;/element&gt;
 * </pre>
 *
 * @version XcodeML_C.rng (Mon Aug 15 15:55:18 JST 2011)
 * @author  Relaxer 1.0 (http://www.relaxer.org)
 */
public class XbcCoArrayType extends xcodeml.c.obj.XmcObj implements java.io.Serializable, Cloneable, xcodeml.c.binding.IXbcArrayType, IRVisitable, IRNode, IXbcTypesChoice {
    private String elementType_;
    private String arraySize1_;
    private String type_;
    private XbcArraySize arraySize2_;
    private IRNode parentRNode_;

    /**
     * Creates a <code>XbcCoArrayType</code>.
     *
     */
    public XbcCoArrayType() {
        elementType_ = "";
        type_ = "";
    }

    /**
     * Creates a <code>XbcCoArrayType</code>.
     *
     * @param source
     */
    public XbcCoArrayType(XbcCoArrayType source) {
        setup(source);
    }

    /**
     * Creates a <code>XbcCoArrayType</code> by the Stack <code>stack</code>
     * that contains Elements.
     * This constructor is supposed to be used internally
     * by the Relaxer system.
     *
     * @param stack
     */
    public XbcCoArrayType(RStack stack) {
        setup(stack);
    }

    /**
     * Creates a <code>XbcCoArrayType</code> by the Document <code>doc</code>.
     *
     * @param doc
     */
    public XbcCoArrayType(Document doc) {
        setup(doc.getDocumentElement());
    }

    /**
     * Creates a <code>XbcCoArrayType</code> by the Element <code>element</code>.
     *
     * @param element
     */
    public XbcCoArrayType(Element element) {
        setup(element);
    }

    /**
     * Creates a <code>XbcCoArrayType</code> by the File <code>file</code>.
     *
     * @param file
     * @exception IOException
     * @exception SAXException
     * @exception ParserConfigurationException
     */
    public XbcCoArrayType(File file) throws IOException, SAXException, ParserConfigurationException {
        setup(file);
    }

    /**
     * Creates a <code>XbcCoArrayType</code>
     * by the String representation of URI <code>uri</code>.
     *
     * @param uri
     * @exception IOException
     * @exception SAXException
     * @exception ParserConfigurationException
     */
    public XbcCoArrayType(String uri) throws IOException, SAXException, ParserConfigurationException {
        setup(uri);
    }

    /**
     * Creates a <code>XbcCoArrayType</code> by the URL <code>url</code>.
     *
     * @param url
     * @exception IOException
     * @exception SAXException
     * @exception ParserConfigurationException
     */
    public XbcCoArrayType(URL url) throws IOException, SAXException, ParserConfigurationException {
        setup(url);
    }

    /**
     * Creates a <code>XbcCoArrayType</code> by the InputStream <code>in</code>.
     *
     * @param in
     * @exception IOException
     * @exception SAXException
     * @exception ParserConfigurationException
     */
    public XbcCoArrayType(InputStream in) throws IOException, SAXException, ParserConfigurationException {
        setup(in);
    }

    /**
     * Creates a <code>XbcCoArrayType</code> by the InputSource <code>is</code>.
     *
     * @param is
     * @exception IOException
     * @exception SAXException
     * @exception ParserConfigurationException
     */
    public XbcCoArrayType(InputSource is) throws IOException, SAXException, ParserConfigurationException {
        setup(is);
    }

    /**
     * Creates a <code>XbcCoArrayType</code> by the Reader <code>reader</code>.
     *
     * @param reader
     * @exception IOException
     * @exception SAXException
     * @exception ParserConfigurationException
     */
    public XbcCoArrayType(Reader reader) throws IOException, SAXException, ParserConfigurationException {
        setup(reader);
    }

    /**
     * Initializes the <code>XbcCoArrayType</code> by the XbcCoArrayType <code>source</code>.
     *
     * @param source
     */
    public void setup(XbcCoArrayType source) {
        int size;
        setElementType(source.getElementType());
        setArraySize1(source.getArraySize1());
        setType(source.getType());
        if (source.arraySize2_ != null) {
            setArraySize2((XbcArraySize)source.getArraySize2().clone());
        }
    }

    /**
     * Initializes the <code>XbcCoArrayType</code> by the Document <code>doc</code>.
     *
     * @param doc
     */
    public void setup(Document doc) {
        setup(doc.getDocumentElement());
    }

    /**
     * Initializes the <code>XbcCoArrayType</code> by the Element <code>element</code>.
     *
     * @param element
     */
    public void setup(Element element) {
        init(element);
    }

    /**
     * Initializes the <code>XbcCoArrayType</code> by the Stack <code>stack</code>
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
        elementType_ = URelaxer.getAttributePropertyAsString(element, "element_type");
        arraySize1_ = URelaxer.getAttributePropertyAsString(element, "array_size");
        type_ = URelaxer.getAttributePropertyAsString(element, "type");
        if (XbcArraySize.isMatch(stack)) {
            setArraySize2(factory.createXbcArraySize(stack));
        }
    }

    /**
     * @return Object
     */
    public Object clone() {
        IXcodeML_CFactory factory = XcodeML_CFactory.getFactory();
        return (factory.createXbcCoArrayType(this));
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
        Element element = doc.createElement("coArrayType");
        int size;
        if (this.elementType_ != null) {
            URelaxer.setAttributePropertyByString(element, "element_type", this.elementType_);
        }
        if (this.arraySize1_ != null) {
            URelaxer.setAttributePropertyByString(element, "array_size", this.arraySize1_);
        }
        if (this.type_ != null) {
            URelaxer.setAttributePropertyByString(element, "type", this.type_);
        }
        if (this.arraySize2_ != null) {
            this.arraySize2_.makeElement(element);
        }
        parent.appendChild(element);
    }

    /**
     * Initializes the <code>XbcCoArrayType</code> by the File <code>file</code>.
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
     * Initializes the <code>XbcCoArrayType</code>
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
     * Initializes the <code>XbcCoArrayType</code> by the URL <code>url</code>.
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
     * Initializes the <code>XbcCoArrayType</code> by the InputStream <code>in</code>.
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
     * Initializes the <code>XbcCoArrayType</code> by the InputSource <code>is</code>.
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
     * Initializes the <code>XbcCoArrayType</code> by the Reader <code>reader</code>.
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
     * Gets the String property <b>elementType</b>.
     *
     * @return String
     */
    public final String getElementType() {
        return (elementType_);
    }

    /**
     * Sets the String property <b>elementType</b>.
     *
     * @param elementType
     */
    public final void setElementType(String elementType) {
        this.elementType_ = elementType;
    }

    /**
     * Gets the String property <b>arraySize1</b>.
     *
     * @return String
     */
    public final String getArraySize1() {
        return (arraySize1_);
    }

    /**
     * Sets the String property <b>arraySize1</b>.
     *
     * @param arraySize1
     */
    public final void setArraySize1(String arraySize1) {
        this.arraySize1_ = arraySize1;
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
     * Gets the XbcArraySize property <b>arraySize2</b>.
     *
     * @return XbcArraySize
     */
    public final XbcArraySize getArraySize2() {
        return (arraySize2_);
    }

    /**
     * Sets the XbcArraySize property <b>arraySize2</b>.
     *
     * @param arraySize2
     */
    public final void setArraySize2(XbcArraySize arraySize2) {
        this.arraySize2_ = arraySize2;
        if (arraySize2 != null) {
            arraySize2.rSetParentRNode(this);
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
        buffer.append("<coArrayType");
        if (elementType_ != null) {
            buffer.append(" element_type=\"");
            buffer.append(URelaxer.escapeAttrQuot(URelaxer.getString(getElementType())));
            buffer.append("\"");
        }
        if (arraySize1_ != null) {
            buffer.append(" array_size=\"");
            buffer.append(URelaxer.escapeAttrQuot(URelaxer.getString(getArraySize1())));
            buffer.append("\"");
        }
        if (type_ != null) {
            buffer.append(" type=\"");
            buffer.append(URelaxer.escapeAttrQuot(URelaxer.getString(getType())));
            buffer.append("\"");
        }
        buffer.append(">");
        if (arraySize2_ != null) {
            arraySize2_.makeTextElement(buffer);
        }
        buffer.append("</coArrayType>");
    }

    /**
     * Makes an XML text representation.
     *
     * @param buffer
     * @exception IOException
     */
    public void makeTextElement(Writer buffer) throws IOException {
        int size;
        buffer.write("<coArrayType");
        if (elementType_ != null) {
            buffer.write(" element_type=\"");
            buffer.write(URelaxer.escapeAttrQuot(URelaxer.getString(getElementType())));
            buffer.write("\"");
        }
        if (arraySize1_ != null) {
            buffer.write(" array_size=\"");
            buffer.write(URelaxer.escapeAttrQuot(URelaxer.getString(getArraySize1())));
            buffer.write("\"");
        }
        if (type_ != null) {
            buffer.write(" type=\"");
            buffer.write(URelaxer.escapeAttrQuot(URelaxer.getString(getType())));
            buffer.write("\"");
        }
        buffer.write(">");
        if (arraySize2_ != null) {
            arraySize2_.makeTextElement(buffer);
        }
        buffer.write("</coArrayType>");
    }

    /**
     * Makes an XML text representation.
     *
     * @param buffer
     */
    public void makeTextElement(PrintWriter buffer) {
        int size;
        buffer.print("<coArrayType");
        if (elementType_ != null) {
            buffer.print(" element_type=\"");
            buffer.print(URelaxer.escapeAttrQuot(URelaxer.getString(getElementType())));
            buffer.print("\"");
        }
        if (arraySize1_ != null) {
            buffer.print(" array_size=\"");
            buffer.print(URelaxer.escapeAttrQuot(URelaxer.getString(getArraySize1())));
            buffer.print("\"");
        }
        if (type_ != null) {
            buffer.print(" type=\"");
            buffer.print(URelaxer.escapeAttrQuot(URelaxer.getString(getType())));
            buffer.print("\"");
        }
        buffer.print(">");
        if (arraySize2_ != null) {
            arraySize2_.makeTextElement(buffer);
        }
        buffer.print("</coArrayType>");
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
    public String getElementTypeAsString() {
        return (URelaxer.getString(getElementType()));
    }

    /**
     * Gets the property value as String.
     *
     * @return String
     */
    public String getArraySize1AsString() {
        return (URelaxer.getString(getArraySize1()));
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
     * Sets the property value by String.
     *
     * @param string
     */
    public void setElementTypeByString(String string) {
        setElementType(string);
    }

    /**
     * Sets the property value by String.
     *
     * @param string
     */
    public void setArraySize1ByString(String string) {
        setArraySize1(string);
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
        if (arraySize2_ != null) {
            classNodes.add(arraySize2_);
        }
        IRNode[] nodes = new IRNode[classNodes.size()];
        return ((IRNode[])classNodes.toArray(nodes));
    }

    /**
     * Tests if a Element <code>element</code> is valid
     * for the <code>XbcCoArrayType</code>.
     *
     * @param element
     * @return boolean
     */
    public static boolean isMatch(Element element) {
        if (!URelaxer.isTargetElement(element, "coArrayType")) {
            return (false);
        }
        RStack target = new RStack(element);
        boolean $match$ = false;
        Element child;
        if (!URelaxer.hasAttributeHungry(target, "element_type")) {
            return (false);
        }
        $match$ = true;
        if (!URelaxer.hasAttributeHungry(target, "type")) {
            return (false);
        }
        $match$ = true;
        if (XbcArraySize.isMatchHungry(target)) {
        }
        if (!target.isEmptyElement()) {
            return (false);
        }
        return (true);
    }

    /**
     * Tests if elements contained in a Stack <code>stack</code>
     * is valid for the <code>XbcCoArrayType</code>.
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
     * is valid for the <code>XbcCoArrayType</code>.
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
