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
 * <b>XbfFcoArrayRef</b> is generated from XcodeML_F.rng by Relaxer.
 * This class is derived from:
 * 
 * <!-- for programmer
 * <element java:extends="xcodeml.f.XmfObj" java:implements="xcodeml.binding.IXbTypedExpr" name="FcoArrayRef">
 *   <ref name="defAttrTypeName"/>
 *   <ref name="varRef"/>
 *   <zeroOrMore>
 *     <ref name="arrayIndex"/>
 *   </zeroOrMore>
 * </element>
 * -->
 * <!-- for javadoc -->
 * <pre> &lt;element java:extends="xcodeml.f.XmfObj" java:implements="xcodeml.binding.IXbTypedExpr" name="FcoArrayRef"&gt;
 *   &lt;ref name="defAttrTypeName"/&gt;
 *   &lt;ref name="varRef"/&gt;
 *   &lt;zeroOrMore&gt;
 *     &lt;ref name="arrayIndex"/&gt;
 *   &lt;/zeroOrMore&gt;
 * &lt;/element&gt;
 * </pre>
 *
 * @version XcodeML_F.rng (Mon Jan 23 20:53:32 JST 2012)
 * @author  Relaxer 1.0 (http://www.relaxer.org)
 */
public class XbfFcoArrayRef extends xcodeml.f.XmfObj implements java.io.Serializable, Cloneable, xcodeml.binding.IXbTypedExpr, IRVisitable, IRNode, IXbfVarRefChoice, IXbfDefModelExprChoice, IXbfArgumentsChoice, IXbfDefModelLValueChoice {
    private String type_;
    private XbfVarRef varRef_;
    // List<XbfArrayIndex>
    private java.util.List arrayIndex_ = new java.util.ArrayList();
    private IRNode parentRNode_;

    /**
     * Creates a <code>XbfFcoArrayRef</code>.
     *
     */
    public XbfFcoArrayRef() {
    }

    /**
     * Creates a <code>XbfFcoArrayRef</code>.
     *
     * @param source
     */
    public XbfFcoArrayRef(XbfFcoArrayRef source) {
        setup(source);
    }

    /**
     * Creates a <code>XbfFcoArrayRef</code> by the Stack <code>stack</code>
     * that contains Elements.
     * This constructor is supposed to be used internally
     * by the Relaxer system.
     *
     * @param stack
     */
    public XbfFcoArrayRef(RStack stack) {
        setup(stack);
    }

    /**
     * Creates a <code>XbfFcoArrayRef</code> by the Document <code>doc</code>.
     *
     * @param doc
     */
    public XbfFcoArrayRef(Document doc) {
        setup(doc.getDocumentElement());
    }

    /**
     * Creates a <code>XbfFcoArrayRef</code> by the Element <code>element</code>.
     *
     * @param element
     */
    public XbfFcoArrayRef(Element element) {
        setup(element);
    }

    /**
     * Creates a <code>XbfFcoArrayRef</code> by the File <code>file</code>.
     *
     * @param file
     * @exception IOException
     * @exception SAXException
     * @exception ParserConfigurationException
     */
    public XbfFcoArrayRef(File file) throws IOException, SAXException, ParserConfigurationException {
        setup(file);
    }

    /**
     * Creates a <code>XbfFcoArrayRef</code>
     * by the String representation of URI <code>uri</code>.
     *
     * @param uri
     * @exception IOException
     * @exception SAXException
     * @exception ParserConfigurationException
     */
    public XbfFcoArrayRef(String uri) throws IOException, SAXException, ParserConfigurationException {
        setup(uri);
    }

    /**
     * Creates a <code>XbfFcoArrayRef</code> by the URL <code>url</code>.
     *
     * @param url
     * @exception IOException
     * @exception SAXException
     * @exception ParserConfigurationException
     */
    public XbfFcoArrayRef(URL url) throws IOException, SAXException, ParserConfigurationException {
        setup(url);
    }

    /**
     * Creates a <code>XbfFcoArrayRef</code> by the InputStream <code>in</code>.
     *
     * @param in
     * @exception IOException
     * @exception SAXException
     * @exception ParserConfigurationException
     */
    public XbfFcoArrayRef(InputStream in) throws IOException, SAXException, ParserConfigurationException {
        setup(in);
    }

    /**
     * Creates a <code>XbfFcoArrayRef</code> by the InputSource <code>is</code>.
     *
     * @param is
     * @exception IOException
     * @exception SAXException
     * @exception ParserConfigurationException
     */
    public XbfFcoArrayRef(InputSource is) throws IOException, SAXException, ParserConfigurationException {
        setup(is);
    }

    /**
     * Creates a <code>XbfFcoArrayRef</code> by the Reader <code>reader</code>.
     *
     * @param reader
     * @exception IOException
     * @exception SAXException
     * @exception ParserConfigurationException
     */
    public XbfFcoArrayRef(Reader reader) throws IOException, SAXException, ParserConfigurationException {
        setup(reader);
    }

    /**
     * Initializes the <code>XbfFcoArrayRef</code> by the XbfFcoArrayRef <code>source</code>.
     *
     * @param source
     */
    public void setup(XbfFcoArrayRef source) {
        int size;
        setType(source.getType());
        if (source.varRef_ != null) {
            setVarRef((XbfVarRef)source.getVarRef().clone());
        }
        this.arrayIndex_.clear();
        size = source.arrayIndex_.size();
        for (int i = 0;i < size;i++) {
            addArrayIndex((XbfArrayIndex)source.getArrayIndex(i).clone());
        }
    }

    /**
     * Initializes the <code>XbfFcoArrayRef</code> by the Document <code>doc</code>.
     *
     * @param doc
     */
    public void setup(Document doc) {
        setup(doc.getDocumentElement());
    }

    /**
     * Initializes the <code>XbfFcoArrayRef</code> by the Element <code>element</code>.
     *
     * @param element
     */
    public void setup(Element element) {
        init(element);
    }

    /**
     * Initializes the <code>XbfFcoArrayRef</code> by the Stack <code>stack</code>
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
        setVarRef(factory.createXbfVarRef(stack));
        arrayIndex_.clear();
        while (true) {
            if (XbfArrayIndex.isMatch(stack)) {
                addArrayIndex(factory.createXbfArrayIndex(stack));
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
        return (factory.createXbfFcoArrayRef(this));
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
        Element element = doc.createElement("FcoArrayRef");
        int size;
        if (this.type_ != null) {
            URelaxer.setAttributePropertyByString(element, "type", this.type_);
        }
        this.varRef_.makeElement(element);
        size = this.arrayIndex_.size();
        for (int i = 0;i < size;i++) {
            XbfArrayIndex value = (XbfArrayIndex)this.arrayIndex_.get(i);
            value.makeElement(element);
        }
        parent.appendChild(element);
    }

    /**
     * Initializes the <code>XbfFcoArrayRef</code> by the File <code>file</code>.
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
     * Initializes the <code>XbfFcoArrayRef</code>
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
     * Initializes the <code>XbfFcoArrayRef</code> by the URL <code>url</code>.
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
     * Initializes the <code>XbfFcoArrayRef</code> by the InputStream <code>in</code>.
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
     * Initializes the <code>XbfFcoArrayRef</code> by the InputSource <code>is</code>.
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
     * Initializes the <code>XbfFcoArrayRef</code> by the Reader <code>reader</code>.
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
     * Gets the XbfVarRef property <b>varRef</b>.
     *
     * @return XbfVarRef
     */
    public final XbfVarRef getVarRef() {
        return (varRef_);
    }

    /**
     * Sets the XbfVarRef property <b>varRef</b>.
     *
     * @param varRef
     */
    public final void setVarRef(XbfVarRef varRef) {
        this.varRef_ = varRef;
        if (varRef != null) {
            varRef.rSetParentRNode(this);
        }
    }

    /**
     * Gets the XbfArrayIndex property <b>arrayIndex</b>.
     *
     * @return XbfArrayIndex[]
     */
    public final XbfArrayIndex[] getArrayIndex() {
        XbfArrayIndex[] array = new XbfArrayIndex[arrayIndex_.size()];
        return ((XbfArrayIndex[])arrayIndex_.toArray(array));
    }

    /**
     * Sets the XbfArrayIndex property <b>arrayIndex</b>.
     *
     * @param arrayIndex
     */
    public final void setArrayIndex(XbfArrayIndex[] arrayIndex) {
        this.arrayIndex_.clear();
        for (int i = 0;i < arrayIndex.length;i++) {
            addArrayIndex(arrayIndex[i]);
        }
        for (int i = 0;i < arrayIndex.length;i++) {
            arrayIndex[i].rSetParentRNode(this);
        }
    }

    /**
     * Sets the XbfArrayIndex property <b>arrayIndex</b>.
     *
     * @param arrayIndex
     */
    public final void setArrayIndex(XbfArrayIndex arrayIndex) {
        this.arrayIndex_.clear();
        addArrayIndex(arrayIndex);
        if (arrayIndex != null) {
            arrayIndex.rSetParentRNode(this);
        }
    }

    /**
     * Adds the XbfArrayIndex property <b>arrayIndex</b>.
     *
     * @param arrayIndex
     */
    public final void addArrayIndex(XbfArrayIndex arrayIndex) {
        this.arrayIndex_.add(arrayIndex);
        if (arrayIndex != null) {
            arrayIndex.rSetParentRNode(this);
        }
    }

    /**
     * Adds the XbfArrayIndex property <b>arrayIndex</b>.
     *
     * @param arrayIndex
     */
    public final void addArrayIndex(XbfArrayIndex[] arrayIndex) {
        for (int i = 0;i < arrayIndex.length;i++) {
            addArrayIndex(arrayIndex[i]);
        }
        for (int i = 0;i < arrayIndex.length;i++) {
            arrayIndex[i].rSetParentRNode(this);
        }
    }

    /**
     * Gets number of the XbfArrayIndex property <b>arrayIndex</b>.
     *
     * @return int
     */
    public final int sizeArrayIndex() {
        return (arrayIndex_.size());
    }

    /**
     * Gets the XbfArrayIndex property <b>arrayIndex</b> by index.
     *
     * @param index
     * @return XbfArrayIndex
     */
    public final XbfArrayIndex getArrayIndex(int index) {
        return ((XbfArrayIndex)arrayIndex_.get(index));
    }

    /**
     * Sets the XbfArrayIndex property <b>arrayIndex</b> by index.
     *
     * @param index
     * @param arrayIndex
     */
    public final void setArrayIndex(int index, XbfArrayIndex arrayIndex) {
        this.arrayIndex_.set(index, arrayIndex);
        if (arrayIndex != null) {
            arrayIndex.rSetParentRNode(this);
        }
    }

    /**
     * Adds the XbfArrayIndex property <b>arrayIndex</b> by index.
     *
     * @param index
     * @param arrayIndex
     */
    public final void addArrayIndex(int index, XbfArrayIndex arrayIndex) {
        this.arrayIndex_.add(index, arrayIndex);
        if (arrayIndex != null) {
            arrayIndex.rSetParentRNode(this);
        }
    }

    /**
     * Remove the XbfArrayIndex property <b>arrayIndex</b> by index.
     *
     * @param index
     */
    public final void removeArrayIndex(int index) {
        this.arrayIndex_.remove(index);
    }

    /**
     * Remove the XbfArrayIndex property <b>arrayIndex</b> by object.
     *
     * @param arrayIndex
     */
    public final void removeArrayIndex(XbfArrayIndex arrayIndex) {
        this.arrayIndex_.remove(arrayIndex);
    }

    /**
     * Clear the XbfArrayIndex property <b>arrayIndex</b>.
     *
     */
    public final void clearArrayIndex() {
        this.arrayIndex_.clear();
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
        buffer.append("<FcoArrayRef");
        if (type_ != null) {
            buffer.append(" type=\"");
            buffer.append(URelaxer.escapeAttrQuot(URelaxer.getString(getType())));
            buffer.append("\"");
        }
        buffer.append(">");
        varRef_.makeTextElement(buffer);
        size = this.arrayIndex_.size();
        for (int i = 0;i < size;i++) {
            XbfArrayIndex value = (XbfArrayIndex)this.arrayIndex_.get(i);
            value.makeTextElement(buffer);
        }
        buffer.append("</FcoArrayRef>");
    }

    /**
     * Makes an XML text representation.
     *
     * @param buffer
     * @exception IOException
     */
    public void makeTextElement(Writer buffer) throws IOException {
        int size;
        buffer.write("<FcoArrayRef");
        if (type_ != null) {
            buffer.write(" type=\"");
            buffer.write(URelaxer.escapeAttrQuot(URelaxer.getString(getType())));
            buffer.write("\"");
        }
        buffer.write(">");
        varRef_.makeTextElement(buffer);
        size = this.arrayIndex_.size();
        for (int i = 0;i < size;i++) {
            XbfArrayIndex value = (XbfArrayIndex)this.arrayIndex_.get(i);
            value.makeTextElement(buffer);
        }
        buffer.write("</FcoArrayRef>");
    }

    /**
     * Makes an XML text representation.
     *
     * @param buffer
     */
    public void makeTextElement(PrintWriter buffer) {
        int size;
        buffer.print("<FcoArrayRef");
        if (type_ != null) {
            buffer.print(" type=\"");
            buffer.print(URelaxer.escapeAttrQuot(URelaxer.getString(getType())));
            buffer.print("\"");
        }
        buffer.print(">");
        varRef_.makeTextElement(buffer);
        size = this.arrayIndex_.size();
        for (int i = 0;i < size;i++) {
            XbfArrayIndex value = (XbfArrayIndex)this.arrayIndex_.get(i);
            value.makeTextElement(buffer);
        }
        buffer.print("</FcoArrayRef>");
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
        if (varRef_ != null) {
            classNodes.add(varRef_);
        }
        classNodes.addAll(arrayIndex_);
        IRNode[] nodes = new IRNode[classNodes.size()];
        return ((IRNode[])classNodes.toArray(nodes));
    }

    /**
     * Tests if a Element <code>element</code> is valid
     * for the <code>XbfFcoArrayRef</code>.
     *
     * @param element
     * @return boolean
     */
    public static boolean isMatch(Element element) {
        if (!URelaxer.isTargetElement(element, "FcoArrayRef")) {
            return (false);
        }
        RStack target = new RStack(element);
        boolean $match$ = false;
        Element child;
        if (!XbfVarRef.isMatchHungry(target)) {
            return (false);
        }
        $match$ = true;
        while (true) {
            if (!XbfArrayIndex.isMatchHungry(target)) {
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
     * is valid for the <code>XbfFcoArrayRef</code>.
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
     * is valid for the <code>XbfFcoArrayRef</code>.
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
