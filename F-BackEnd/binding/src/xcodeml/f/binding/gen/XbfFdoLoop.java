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
 * <b>XbfFdoLoop</b> is generated from XcodeML_F.rng by Relaxer.
 * This class is derived from:
 * 
 * <!-- for programmer
 * <element java:extends="xcodeml.f.XmfObj" name="FdoLoop">
 *   <ref name="Var"/>
 *   <ref name="indexRange"/>
 *   <oneOrMore>
 *     <ref name="value"/>
 *   </oneOrMore>
 * </element>
 * -->
 * <!-- for javadoc -->
 * <pre> &lt;element java:extends="xcodeml.f.XmfObj" name="FdoLoop"&gt;
 *   &lt;ref name="Var"/&gt;
 *   &lt;ref name="indexRange"/&gt;
 *   &lt;oneOrMore&gt;
 *     &lt;ref name="value"/&gt;
 *   &lt;/oneOrMore&gt;
 * &lt;/element&gt;
 * </pre>
 *
 * @version XcodeML_F.rng (Fri Dec 04 19:18:15 JST 2009)
 * @author  Relaxer 1.0 (http://www.relaxer.org)
 */
public class XbfFdoLoop extends xcodeml.f.XmfObj implements java.io.Serializable, Cloneable, IRVisitable, IRNode, IXbfArgumentsChoice, IXbfDefModelExprChoice, IXbfVarListChoice {
    private XbfVar var_;
    private XbfIndexRange indexRange_;
    // List<XbfValue>
    private java.util.List value_ = new java.util.ArrayList();
    private IRNode parentRNode_;

    /**
     * Creates a <code>XbfFdoLoop</code>.
     *
     */
    public XbfFdoLoop() {
    }

    /**
     * Creates a <code>XbfFdoLoop</code>.
     *
     * @param source
     */
    public XbfFdoLoop(XbfFdoLoop source) {
        setup(source);
    }

    /**
     * Creates a <code>XbfFdoLoop</code> by the Stack <code>stack</code>
     * that contains Elements.
     * This constructor is supposed to be used internally
     * by the Relaxer system.
     *
     * @param stack
     */
    public XbfFdoLoop(RStack stack) {
        setup(stack);
    }

    /**
     * Creates a <code>XbfFdoLoop</code> by the Document <code>doc</code>.
     *
     * @param doc
     */
    public XbfFdoLoop(Document doc) {
        setup(doc.getDocumentElement());
    }

    /**
     * Creates a <code>XbfFdoLoop</code> by the Element <code>element</code>.
     *
     * @param element
     */
    public XbfFdoLoop(Element element) {
        setup(element);
    }

    /**
     * Creates a <code>XbfFdoLoop</code> by the File <code>file</code>.
     *
     * @param file
     * @exception IOException
     * @exception SAXException
     * @exception ParserConfigurationException
     */
    public XbfFdoLoop(File file) throws IOException, SAXException, ParserConfigurationException {
        setup(file);
    }

    /**
     * Creates a <code>XbfFdoLoop</code>
     * by the String representation of URI <code>uri</code>.
     *
     * @param uri
     * @exception IOException
     * @exception SAXException
     * @exception ParserConfigurationException
     */
    public XbfFdoLoop(String uri) throws IOException, SAXException, ParserConfigurationException {
        setup(uri);
    }

    /**
     * Creates a <code>XbfFdoLoop</code> by the URL <code>url</code>.
     *
     * @param url
     * @exception IOException
     * @exception SAXException
     * @exception ParserConfigurationException
     */
    public XbfFdoLoop(URL url) throws IOException, SAXException, ParserConfigurationException {
        setup(url);
    }

    /**
     * Creates a <code>XbfFdoLoop</code> by the InputStream <code>in</code>.
     *
     * @param in
     * @exception IOException
     * @exception SAXException
     * @exception ParserConfigurationException
     */
    public XbfFdoLoop(InputStream in) throws IOException, SAXException, ParserConfigurationException {
        setup(in);
    }

    /**
     * Creates a <code>XbfFdoLoop</code> by the InputSource <code>is</code>.
     *
     * @param is
     * @exception IOException
     * @exception SAXException
     * @exception ParserConfigurationException
     */
    public XbfFdoLoop(InputSource is) throws IOException, SAXException, ParserConfigurationException {
        setup(is);
    }

    /**
     * Creates a <code>XbfFdoLoop</code> by the Reader <code>reader</code>.
     *
     * @param reader
     * @exception IOException
     * @exception SAXException
     * @exception ParserConfigurationException
     */
    public XbfFdoLoop(Reader reader) throws IOException, SAXException, ParserConfigurationException {
        setup(reader);
    }

    /**
     * Initializes the <code>XbfFdoLoop</code> by the XbfFdoLoop <code>source</code>.
     *
     * @param source
     */
    public void setup(XbfFdoLoop source) {
        int size;
        if (source.var_ != null) {
            setVar((XbfVar)source.getVar().clone());
        }
        if (source.indexRange_ != null) {
            setIndexRange((XbfIndexRange)source.getIndexRange().clone());
        }
        this.value_.clear();
        size = source.value_.size();
        for (int i = 0;i < size;i++) {
            addValue((XbfValue)source.getValue(i).clone());
        }
    }

    /**
     * Initializes the <code>XbfFdoLoop</code> by the Document <code>doc</code>.
     *
     * @param doc
     */
    public void setup(Document doc) {
        setup(doc.getDocumentElement());
    }

    /**
     * Initializes the <code>XbfFdoLoop</code> by the Element <code>element</code>.
     *
     * @param element
     */
    public void setup(Element element) {
        init(element);
    }

    /**
     * Initializes the <code>XbfFdoLoop</code> by the Stack <code>stack</code>
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
        setVar(factory.createXbfVar(stack));
        setIndexRange(factory.createXbfIndexRange(stack));
        value_.clear();
        while (true) {
            if (XbfValue.isMatch(stack)) {
                addValue(factory.createXbfValue(stack));
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
        return (factory.createXbfFdoLoop(this));
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
        Element element = doc.createElement("FdoLoop");
        int size;
        this.var_.makeElement(element);
        this.indexRange_.makeElement(element);
        size = this.value_.size();
        for (int i = 0;i < size;i++) {
            XbfValue value = (XbfValue)this.value_.get(i);
            value.makeElement(element);
        }
        parent.appendChild(element);
    }

    /**
     * Initializes the <code>XbfFdoLoop</code> by the File <code>file</code>.
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
     * Initializes the <code>XbfFdoLoop</code>
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
     * Initializes the <code>XbfFdoLoop</code> by the URL <code>url</code>.
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
     * Initializes the <code>XbfFdoLoop</code> by the InputStream <code>in</code>.
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
     * Initializes the <code>XbfFdoLoop</code> by the InputSource <code>is</code>.
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
     * Initializes the <code>XbfFdoLoop</code> by the Reader <code>reader</code>.
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
     * Gets the XbfVar property <b>Var</b>.
     *
     * @return XbfVar
     */
    public final XbfVar getVar() {
        return (var_);
    }

    /**
     * Sets the XbfVar property <b>Var</b>.
     *
     * @param var
     */
    public final void setVar(XbfVar var) {
        this.var_ = var;
        if (var != null) {
            var.rSetParentRNode(this);
        }
    }

    /**
     * Gets the XbfIndexRange property <b>indexRange</b>.
     *
     * @return XbfIndexRange
     */
    public final XbfIndexRange getIndexRange() {
        return (indexRange_);
    }

    /**
     * Sets the XbfIndexRange property <b>indexRange</b>.
     *
     * @param indexRange
     */
    public final void setIndexRange(XbfIndexRange indexRange) {
        this.indexRange_ = indexRange;
        if (indexRange != null) {
            indexRange.rSetParentRNode(this);
        }
    }

    /**
     * Gets the XbfValue property <b>value</b>.
     *
     * @return XbfValue[]
     */
    public final XbfValue[] getValue() {
        XbfValue[] array = new XbfValue[value_.size()];
        return ((XbfValue[])value_.toArray(array));
    }

    /**
     * Sets the XbfValue property <b>value</b>.
     *
     * @param value
     */
    public final void setValue(XbfValue[] value) {
        this.value_.clear();
        for (int i = 0;i < value.length;i++) {
            addValue(value[i]);
        }
        for (int i = 0;i < value.length;i++) {
            value[i].rSetParentRNode(this);
        }
    }

    /**
     * Sets the XbfValue property <b>value</b>.
     *
     * @param value
     */
    public final void setValue(XbfValue value) {
        this.value_.clear();
        addValue(value);
        if (value != null) {
            value.rSetParentRNode(this);
        }
    }

    /**
     * Adds the XbfValue property <b>value</b>.
     *
     * @param value
     */
    public final void addValue(XbfValue value) {
        this.value_.add(value);
        if (value != null) {
            value.rSetParentRNode(this);
        }
    }

    /**
     * Adds the XbfValue property <b>value</b>.
     *
     * @param value
     */
    public final void addValue(XbfValue[] value) {
        for (int i = 0;i < value.length;i++) {
            addValue(value[i]);
        }
        for (int i = 0;i < value.length;i++) {
            value[i].rSetParentRNode(this);
        }
    }

    /**
     * Gets number of the XbfValue property <b>value</b>.
     *
     * @return int
     */
    public final int sizeValue() {
        return (value_.size());
    }

    /**
     * Gets the XbfValue property <b>value</b> by index.
     *
     * @param index
     * @return XbfValue
     */
    public final XbfValue getValue(int index) {
        return ((XbfValue)value_.get(index));
    }

    /**
     * Sets the XbfValue property <b>value</b> by index.
     *
     * @param index
     * @param value
     */
    public final void setValue(int index, XbfValue value) {
        this.value_.set(index, value);
        if (value != null) {
            value.rSetParentRNode(this);
        }
    }

    /**
     * Adds the XbfValue property <b>value</b> by index.
     *
     * @param index
     * @param value
     */
    public final void addValue(int index, XbfValue value) {
        this.value_.add(index, value);
        if (value != null) {
            value.rSetParentRNode(this);
        }
    }

    /**
     * Remove the XbfValue property <b>value</b> by index.
     *
     * @param index
     */
    public final void removeValue(int index) {
        this.value_.remove(index);
    }

    /**
     * Remove the XbfValue property <b>value</b> by object.
     *
     * @param value
     */
    public final void removeValue(XbfValue value) {
        this.value_.remove(value);
    }

    /**
     * Clear the XbfValue property <b>value</b>.
     *
     */
    public final void clearValue() {
        this.value_.clear();
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
        buffer.append("<FdoLoop");
        buffer.append(">");
        var_.makeTextElement(buffer);
        indexRange_.makeTextElement(buffer);
        size = this.value_.size();
        for (int i = 0;i < size;i++) {
            XbfValue value = (XbfValue)this.value_.get(i);
            value.makeTextElement(buffer);
        }
        buffer.append("</FdoLoop>");
    }

    /**
     * Makes an XML text representation.
     *
     * @param buffer
     * @exception IOException
     */
    public void makeTextElement(Writer buffer) throws IOException {
        int size;
        buffer.write("<FdoLoop");
        buffer.write(">");
        var_.makeTextElement(buffer);
        indexRange_.makeTextElement(buffer);
        size = this.value_.size();
        for (int i = 0;i < size;i++) {
            XbfValue value = (XbfValue)this.value_.get(i);
            value.makeTextElement(buffer);
        }
        buffer.write("</FdoLoop>");
    }

    /**
     * Makes an XML text representation.
     *
     * @param buffer
     */
    public void makeTextElement(PrintWriter buffer) {
        int size;
        buffer.print("<FdoLoop");
        buffer.print(">");
        var_.makeTextElement(buffer);
        indexRange_.makeTextElement(buffer);
        size = this.value_.size();
        for (int i = 0;i < size;i++) {
            XbfValue value = (XbfValue)this.value_.get(i);
            value.makeTextElement(buffer);
        }
        buffer.print("</FdoLoop>");
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
        if (var_ != null) {
            classNodes.add(var_);
        }
        if (indexRange_ != null) {
            classNodes.add(indexRange_);
        }
        classNodes.addAll(value_);
        IRNode[] nodes = new IRNode[classNodes.size()];
        return ((IRNode[])classNodes.toArray(nodes));
    }

    /**
     * Tests if a Element <code>element</code> is valid
     * for the <code>XbfFdoLoop</code>.
     *
     * @param element
     * @return boolean
     */
    public static boolean isMatch(Element element) {
        if (!URelaxer.isTargetElement(element, "FdoLoop")) {
            return (false);
        }
        RStack target = new RStack(element);
        boolean $match$ = false;
        Element child;
        if (!XbfVar.isMatchHungry(target)) {
            return (false);
        }
        $match$ = true;
        if (!XbfIndexRange.isMatchHungry(target)) {
            return (false);
        }
        $match$ = true;
        if (!XbfValue.isMatchHungry(target)) {
            return (false);
        }
        $match$ = true;
        while (true) {
            if (!XbfValue.isMatchHungry(target)) {
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
     * is valid for the <code>XbfFdoLoop</code>.
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
     * is valid for the <code>XbfFdoLoop</code>.
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
