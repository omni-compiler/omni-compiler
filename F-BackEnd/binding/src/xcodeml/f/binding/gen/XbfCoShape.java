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
 * <b>XbfCoShape</b> is generated from XcodeML_F.rng by Relaxer.
 * This class is derived from:
 * 
 * <!-- for programmer
 * <element java:extends="xcodeml.f.XmfObj" name="coShape">
 *   <zeroOrMore>
 *     <ref name="indexRange"/>
 *   </zeroOrMore>
 * </element>
 * -->
 * <!-- for javadoc -->
 * <pre> &lt;element java:extends="xcodeml.f.XmfObj" name="coShape"&gt;
 *   &lt;zeroOrMore&gt;
 *     &lt;ref name="indexRange"/&gt;
 *   &lt;/zeroOrMore&gt;
 * &lt;/element&gt;
 * </pre>
 *
 * @version XcodeML_F.rng (Mon Nov 29 15:25:55 JST 2010)
 * @author  Relaxer 1.0 (http://www.relaxer.org)
 */
public class XbfCoShape extends xcodeml.f.XmfObj implements java.io.Serializable, Cloneable, IRVisitable, IRNode {
    // List<XbfIndexRange>
    private java.util.List indexRange_ = new java.util.ArrayList();
    private IRNode parentRNode_;

    /**
     * Creates a <code>XbfCoShape</code>.
     *
     */
    public XbfCoShape() {
    }

    /**
     * Creates a <code>XbfCoShape</code>.
     *
     * @param source
     */
    public XbfCoShape(XbfCoShape source) {
        setup(source);
    }

    /**
     * Creates a <code>XbfCoShape</code> by the Stack <code>stack</code>
     * that contains Elements.
     * This constructor is supposed to be used internally
     * by the Relaxer system.
     *
     * @param stack
     */
    public XbfCoShape(RStack stack) {
        setup(stack);
    }

    /**
     * Creates a <code>XbfCoShape</code> by the Document <code>doc</code>.
     *
     * @param doc
     */
    public XbfCoShape(Document doc) {
        setup(doc.getDocumentElement());
    }

    /**
     * Creates a <code>XbfCoShape</code> by the Element <code>element</code>.
     *
     * @param element
     */
    public XbfCoShape(Element element) {
        setup(element);
    }

    /**
     * Creates a <code>XbfCoShape</code> by the File <code>file</code>.
     *
     * @param file
     * @exception IOException
     * @exception SAXException
     * @exception ParserConfigurationException
     */
    public XbfCoShape(File file) throws IOException, SAXException, ParserConfigurationException {
        setup(file);
    }

    /**
     * Creates a <code>XbfCoShape</code>
     * by the String representation of URI <code>uri</code>.
     *
     * @param uri
     * @exception IOException
     * @exception SAXException
     * @exception ParserConfigurationException
     */
    public XbfCoShape(String uri) throws IOException, SAXException, ParserConfigurationException {
        setup(uri);
    }

    /**
     * Creates a <code>XbfCoShape</code> by the URL <code>url</code>.
     *
     * @param url
     * @exception IOException
     * @exception SAXException
     * @exception ParserConfigurationException
     */
    public XbfCoShape(URL url) throws IOException, SAXException, ParserConfigurationException {
        setup(url);
    }

    /**
     * Creates a <code>XbfCoShape</code> by the InputStream <code>in</code>.
     *
     * @param in
     * @exception IOException
     * @exception SAXException
     * @exception ParserConfigurationException
     */
    public XbfCoShape(InputStream in) throws IOException, SAXException, ParserConfigurationException {
        setup(in);
    }

    /**
     * Creates a <code>XbfCoShape</code> by the InputSource <code>is</code>.
     *
     * @param is
     * @exception IOException
     * @exception SAXException
     * @exception ParserConfigurationException
     */
    public XbfCoShape(InputSource is) throws IOException, SAXException, ParserConfigurationException {
        setup(is);
    }

    /**
     * Creates a <code>XbfCoShape</code> by the Reader <code>reader</code>.
     *
     * @param reader
     * @exception IOException
     * @exception SAXException
     * @exception ParserConfigurationException
     */
    public XbfCoShape(Reader reader) throws IOException, SAXException, ParserConfigurationException {
        setup(reader);
    }

    /**
     * Initializes the <code>XbfCoShape</code> by the XbfCoShape <code>source</code>.
     *
     * @param source
     */
    public void setup(XbfCoShape source) {
        int size;
        this.indexRange_.clear();
        size = source.indexRange_.size();
        for (int i = 0;i < size;i++) {
            addIndexRange((XbfIndexRange)source.getIndexRange(i).clone());
        }
    }

    /**
     * Initializes the <code>XbfCoShape</code> by the Document <code>doc</code>.
     *
     * @param doc
     */
    public void setup(Document doc) {
        setup(doc.getDocumentElement());
    }

    /**
     * Initializes the <code>XbfCoShape</code> by the Element <code>element</code>.
     *
     * @param element
     */
    public void setup(Element element) {
        init(element);
    }

    /**
     * Initializes the <code>XbfCoShape</code> by the Stack <code>stack</code>
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
        indexRange_.clear();
        while (true) {
            if (XbfIndexRange.isMatch(stack)) {
                addIndexRange(factory.createXbfIndexRange(stack));
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
        return (factory.createXbfCoShape(this));
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
        Element element = doc.createElement("coShape");
        int size;
        size = this.indexRange_.size();
        for (int i = 0;i < size;i++) {
            XbfIndexRange value = (XbfIndexRange)this.indexRange_.get(i);
            value.makeElement(element);
        }
        parent.appendChild(element);
    }

    /**
     * Initializes the <code>XbfCoShape</code> by the File <code>file</code>.
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
     * Initializes the <code>XbfCoShape</code>
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
     * Initializes the <code>XbfCoShape</code> by the URL <code>url</code>.
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
     * Initializes the <code>XbfCoShape</code> by the InputStream <code>in</code>.
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
     * Initializes the <code>XbfCoShape</code> by the InputSource <code>is</code>.
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
     * Initializes the <code>XbfCoShape</code> by the Reader <code>reader</code>.
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
     * Gets the XbfIndexRange property <b>indexRange</b>.
     *
     * @return XbfIndexRange[]
     */
    public final XbfIndexRange[] getIndexRange() {
        XbfIndexRange[] array = new XbfIndexRange[indexRange_.size()];
        return ((XbfIndexRange[])indexRange_.toArray(array));
    }

    /**
     * Sets the XbfIndexRange property <b>indexRange</b>.
     *
     * @param indexRange
     */
    public final void setIndexRange(XbfIndexRange[] indexRange) {
        this.indexRange_.clear();
        for (int i = 0;i < indexRange.length;i++) {
            addIndexRange(indexRange[i]);
        }
        for (int i = 0;i < indexRange.length;i++) {
            indexRange[i].rSetParentRNode(this);
        }
    }

    /**
     * Sets the XbfIndexRange property <b>indexRange</b>.
     *
     * @param indexRange
     */
    public final void setIndexRange(XbfIndexRange indexRange) {
        this.indexRange_.clear();
        addIndexRange(indexRange);
        if (indexRange != null) {
            indexRange.rSetParentRNode(this);
        }
    }

    /**
     * Adds the XbfIndexRange property <b>indexRange</b>.
     *
     * @param indexRange
     */
    public final void addIndexRange(XbfIndexRange indexRange) {
        this.indexRange_.add(indexRange);
        if (indexRange != null) {
            indexRange.rSetParentRNode(this);
        }
    }

    /**
     * Adds the XbfIndexRange property <b>indexRange</b>.
     *
     * @param indexRange
     */
    public final void addIndexRange(XbfIndexRange[] indexRange) {
        for (int i = 0;i < indexRange.length;i++) {
            addIndexRange(indexRange[i]);
        }
        for (int i = 0;i < indexRange.length;i++) {
            indexRange[i].rSetParentRNode(this);
        }
    }

    /**
     * Gets number of the XbfIndexRange property <b>indexRange</b>.
     *
     * @return int
     */
    public final int sizeIndexRange() {
        return (indexRange_.size());
    }

    /**
     * Gets the XbfIndexRange property <b>indexRange</b> by index.
     *
     * @param index
     * @return XbfIndexRange
     */
    public final XbfIndexRange getIndexRange(int index) {
        return ((XbfIndexRange)indexRange_.get(index));
    }

    /**
     * Sets the XbfIndexRange property <b>indexRange</b> by index.
     *
     * @param index
     * @param indexRange
     */
    public final void setIndexRange(int index, XbfIndexRange indexRange) {
        this.indexRange_.set(index, indexRange);
        if (indexRange != null) {
            indexRange.rSetParentRNode(this);
        }
    }

    /**
     * Adds the XbfIndexRange property <b>indexRange</b> by index.
     *
     * @param index
     * @param indexRange
     */
    public final void addIndexRange(int index, XbfIndexRange indexRange) {
        this.indexRange_.add(index, indexRange);
        if (indexRange != null) {
            indexRange.rSetParentRNode(this);
        }
    }

    /**
     * Remove the XbfIndexRange property <b>indexRange</b> by index.
     *
     * @param index
     */
    public final void removeIndexRange(int index) {
        this.indexRange_.remove(index);
    }

    /**
     * Remove the XbfIndexRange property <b>indexRange</b> by object.
     *
     * @param indexRange
     */
    public final void removeIndexRange(XbfIndexRange indexRange) {
        this.indexRange_.remove(indexRange);
    }

    /**
     * Clear the XbfIndexRange property <b>indexRange</b>.
     *
     */
    public final void clearIndexRange() {
        this.indexRange_.clear();
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
        buffer.append("<coShape");
        buffer.append(">");
        size = this.indexRange_.size();
        for (int i = 0;i < size;i++) {
            XbfIndexRange value = (XbfIndexRange)this.indexRange_.get(i);
            value.makeTextElement(buffer);
        }
        buffer.append("</coShape>");
    }

    /**
     * Makes an XML text representation.
     *
     * @param buffer
     * @exception IOException
     */
    public void makeTextElement(Writer buffer) throws IOException {
        int size;
        buffer.write("<coShape");
        buffer.write(">");
        size = this.indexRange_.size();
        for (int i = 0;i < size;i++) {
            XbfIndexRange value = (XbfIndexRange)this.indexRange_.get(i);
            value.makeTextElement(buffer);
        }
        buffer.write("</coShape>");
    }

    /**
     * Makes an XML text representation.
     *
     * @param buffer
     */
    public void makeTextElement(PrintWriter buffer) {
        int size;
        buffer.print("<coShape");
        buffer.print(">");
        size = this.indexRange_.size();
        for (int i = 0;i < size;i++) {
            XbfIndexRange value = (XbfIndexRange)this.indexRange_.get(i);
            value.makeTextElement(buffer);
        }
        buffer.print("</coShape>");
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
        classNodes.addAll(indexRange_);
        IRNode[] nodes = new IRNode[classNodes.size()];
        return ((IRNode[])classNodes.toArray(nodes));
    }

    /**
     * Tests if a Element <code>element</code> is valid
     * for the <code>XbfCoShape</code>.
     *
     * @param element
     * @return boolean
     */
    public static boolean isMatch(Element element) {
        if (!URelaxer.isTargetElement(element, "coShape")) {
            return (false);
        }
        RStack target = new RStack(element);
        boolean $match$ = false;
        Element child;
        while (true) {
            if (!XbfIndexRange.isMatchHungry(target)) {
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
     * is valid for the <code>XbfCoShape</code>.
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
     * is valid for the <code>XbfCoShape</code>.
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
