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
 * <b>XbfAlloc</b> is generated from XcodeML_F.rng by Relaxer.
 * This class is derived from:
 * 
 * <!-- for programmer
 * <element java:extends="xcodeml.f.XmfObj" name="alloc">
 *   <choice>
 *     <ref name="Var"/>
 *     <ref name="FmemberRef"/>
 *   </choice>
 *   <ref name="defModelArraySubscriptSequence0"/>
 *   <ref name="coShape"/>
 * </element>
 * -->
 * <!-- for javadoc -->
 * <pre> &lt;element java:extends="xcodeml.f.XmfObj" name="alloc"&gt;
 *   &lt;choice&gt;
 *     &lt;ref name="Var"/&gt;
 *     &lt;ref name="FmemberRef"/&gt;
 *   &lt;/choice&gt;
 *   &lt;ref name="defModelArraySubscriptSequence0"/&gt;
 *   &lt;ref name="coShape"/&gt;
 * &lt;/element&gt;
 * </pre>
 *
 * @version XcodeML_F.rng (Mon Nov 29 15:25:56 JST 2010)
 * @author  Relaxer 1.0 (http://www.relaxer.org)
 */
public class XbfAlloc extends xcodeml.f.XmfObj implements java.io.Serializable, Cloneable, IRVisitable, IRNode {
    private IXbfAllocChoice1 content_;
    // List<IXbfDefModelArraySubscriptChoice>
    private java.util.List defModelArraySubscript_ = new java.util.ArrayList();
    private XbfCoShape coShape_;
    private IRNode parentRNode_;

    /**
     * Creates a <code>XbfAlloc</code>.
     *
     */
    public XbfAlloc() {
    }

    /**
     * Creates a <code>XbfAlloc</code>.
     *
     * @param source
     */
    public XbfAlloc(XbfAlloc source) {
        setup(source);
    }

    /**
     * Creates a <code>XbfAlloc</code> by the Stack <code>stack</code>
     * that contains Elements.
     * This constructor is supposed to be used internally
     * by the Relaxer system.
     *
     * @param stack
     */
    public XbfAlloc(RStack stack) {
        setup(stack);
    }

    /**
     * Creates a <code>XbfAlloc</code> by the Document <code>doc</code>.
     *
     * @param doc
     */
    public XbfAlloc(Document doc) {
        setup(doc.getDocumentElement());
    }

    /**
     * Creates a <code>XbfAlloc</code> by the Element <code>element</code>.
     *
     * @param element
     */
    public XbfAlloc(Element element) {
        setup(element);
    }

    /**
     * Creates a <code>XbfAlloc</code> by the File <code>file</code>.
     *
     * @param file
     * @exception IOException
     * @exception SAXException
     * @exception ParserConfigurationException
     */
    public XbfAlloc(File file) throws IOException, SAXException, ParserConfigurationException {
        setup(file);
    }

    /**
     * Creates a <code>XbfAlloc</code>
     * by the String representation of URI <code>uri</code>.
     *
     * @param uri
     * @exception IOException
     * @exception SAXException
     * @exception ParserConfigurationException
     */
    public XbfAlloc(String uri) throws IOException, SAXException, ParserConfigurationException {
        setup(uri);
    }

    /**
     * Creates a <code>XbfAlloc</code> by the URL <code>url</code>.
     *
     * @param url
     * @exception IOException
     * @exception SAXException
     * @exception ParserConfigurationException
     */
    public XbfAlloc(URL url) throws IOException, SAXException, ParserConfigurationException {
        setup(url);
    }

    /**
     * Creates a <code>XbfAlloc</code> by the InputStream <code>in</code>.
     *
     * @param in
     * @exception IOException
     * @exception SAXException
     * @exception ParserConfigurationException
     */
    public XbfAlloc(InputStream in) throws IOException, SAXException, ParserConfigurationException {
        setup(in);
    }

    /**
     * Creates a <code>XbfAlloc</code> by the InputSource <code>is</code>.
     *
     * @param is
     * @exception IOException
     * @exception SAXException
     * @exception ParserConfigurationException
     */
    public XbfAlloc(InputSource is) throws IOException, SAXException, ParserConfigurationException {
        setup(is);
    }

    /**
     * Creates a <code>XbfAlloc</code> by the Reader <code>reader</code>.
     *
     * @param reader
     * @exception IOException
     * @exception SAXException
     * @exception ParserConfigurationException
     */
    public XbfAlloc(Reader reader) throws IOException, SAXException, ParserConfigurationException {
        setup(reader);
    }

    /**
     * Initializes the <code>XbfAlloc</code> by the XbfAlloc <code>source</code>.
     *
     * @param source
     */
    public void setup(XbfAlloc source) {
        int size;
        if (source.content_ != null) {
            setContent((IXbfAllocChoice1)source.getContent().clone());
        }
        this.defModelArraySubscript_.clear();
        size = source.defModelArraySubscript_.size();
        for (int i = 0;i < size;i++) {
            addDefModelArraySubscript((IXbfDefModelArraySubscriptChoice)source.getDefModelArraySubscript(i).clone());
        }
        if (source.coShape_ != null) {
            setCoShape((XbfCoShape)source.getCoShape().clone());
        }
    }

    /**
     * Initializes the <code>XbfAlloc</code> by the Document <code>doc</code>.
     *
     * @param doc
     */
    public void setup(Document doc) {
        setup(doc.getDocumentElement());
    }

    /**
     * Initializes the <code>XbfAlloc</code> by the Element <code>element</code>.
     *
     * @param element
     */
    public void setup(Element element) {
        init(element);
    }

    /**
     * Initializes the <code>XbfAlloc</code> by the Stack <code>stack</code>
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
        if (XbfFmemberRef.isMatch(stack)) {
            setContent(factory.createXbfFmemberRef(stack));
        } else if (XbfVar.isMatch(stack)) {
            setContent(factory.createXbfVar(stack));
        } else {
            throw (new IllegalArgumentException());
        }
        defModelArraySubscript_.clear();
        while (true) {
            if (XbfIndexRange.isMatch(stack)) {
                addDefModelArraySubscript(factory.createXbfIndexRange(stack));
            } else if (XbfArrayIndex.isMatch(stack)) {
                addDefModelArraySubscript(factory.createXbfArrayIndex(stack));
            } else {
                break;
            }
        }
        setCoShape(factory.createXbfCoShape(stack));
    }

    /**
     * @return Object
     */
    public Object clone() {
        IXcodeML_FFactory factory = XcodeML_FFactory.getFactory();
        return (factory.createXbfAlloc(this));
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
        Element element = doc.createElement("alloc");
        int size;
        this.content_.makeElement(element);
        size = this.defModelArraySubscript_.size();
        for (int i = 0;i < size;i++) {
            IXbfDefModelArraySubscriptChoice value = (IXbfDefModelArraySubscriptChoice)this.defModelArraySubscript_.get(i);
            value.makeElement(element);
        }
        this.coShape_.makeElement(element);
        parent.appendChild(element);
    }

    /**
     * Initializes the <code>XbfAlloc</code> by the File <code>file</code>.
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
     * Initializes the <code>XbfAlloc</code>
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
     * Initializes the <code>XbfAlloc</code> by the URL <code>url</code>.
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
     * Initializes the <code>XbfAlloc</code> by the InputStream <code>in</code>.
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
     * Initializes the <code>XbfAlloc</code> by the InputSource <code>is</code>.
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
     * Initializes the <code>XbfAlloc</code> by the Reader <code>reader</code>.
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
     * Gets the IXbfAllocChoice1 property <b>content</b>.
     *
     * @return IXbfAllocChoice1
     */
    public final IXbfAllocChoice1 getContent() {
        return (content_);
    }

    /**
     * Sets the IXbfAllocChoice1 property <b>content</b>.
     *
     * @param content
     */
    public final void setContent(IXbfAllocChoice1 content) {
        this.content_ = content;
        if (content != null) {
            content.rSetParentRNode(this);
        }
    }

    /**
     * Gets the IXbfDefModelArraySubscriptChoice property <b>defModelArraySubscript</b>.
     *
     * @return IXbfDefModelArraySubscriptChoice[]
     */
    public final IXbfDefModelArraySubscriptChoice[] getDefModelArraySubscript() {
        IXbfDefModelArraySubscriptChoice[] array = new IXbfDefModelArraySubscriptChoice[defModelArraySubscript_.size()];
        return ((IXbfDefModelArraySubscriptChoice[])defModelArraySubscript_.toArray(array));
    }

    /**
     * Sets the IXbfDefModelArraySubscriptChoice property <b>defModelArraySubscript</b>.
     *
     * @param defModelArraySubscript
     */
    public final void setDefModelArraySubscript(IXbfDefModelArraySubscriptChoice[] defModelArraySubscript) {
        this.defModelArraySubscript_.clear();
        for (int i = 0;i < defModelArraySubscript.length;i++) {
            addDefModelArraySubscript(defModelArraySubscript[i]);
        }
        for (int i = 0;i < defModelArraySubscript.length;i++) {
            defModelArraySubscript[i].rSetParentRNode(this);
        }
    }

    /**
     * Sets the IXbfDefModelArraySubscriptChoice property <b>defModelArraySubscript</b>.
     *
     * @param defModelArraySubscript
     */
    public final void setDefModelArraySubscript(IXbfDefModelArraySubscriptChoice defModelArraySubscript) {
        this.defModelArraySubscript_.clear();
        addDefModelArraySubscript(defModelArraySubscript);
        if (defModelArraySubscript != null) {
            defModelArraySubscript.rSetParentRNode(this);
        }
    }

    /**
     * Adds the IXbfDefModelArraySubscriptChoice property <b>defModelArraySubscript</b>.
     *
     * @param defModelArraySubscript
     */
    public final void addDefModelArraySubscript(IXbfDefModelArraySubscriptChoice defModelArraySubscript) {
        this.defModelArraySubscript_.add(defModelArraySubscript);
        if (defModelArraySubscript != null) {
            defModelArraySubscript.rSetParentRNode(this);
        }
    }

    /**
     * Adds the IXbfDefModelArraySubscriptChoice property <b>defModelArraySubscript</b>.
     *
     * @param defModelArraySubscript
     */
    public final void addDefModelArraySubscript(IXbfDefModelArraySubscriptChoice[] defModelArraySubscript) {
        for (int i = 0;i < defModelArraySubscript.length;i++) {
            addDefModelArraySubscript(defModelArraySubscript[i]);
        }
        for (int i = 0;i < defModelArraySubscript.length;i++) {
            defModelArraySubscript[i].rSetParentRNode(this);
        }
    }

    /**
     * Gets number of the IXbfDefModelArraySubscriptChoice property <b>defModelArraySubscript</b>.
     *
     * @return int
     */
    public final int sizeDefModelArraySubscript() {
        return (defModelArraySubscript_.size());
    }

    /**
     * Gets the IXbfDefModelArraySubscriptChoice property <b>defModelArraySubscript</b> by index.
     *
     * @param index
     * @return IXbfDefModelArraySubscriptChoice
     */
    public final IXbfDefModelArraySubscriptChoice getDefModelArraySubscript(int index) {
        return ((IXbfDefModelArraySubscriptChoice)defModelArraySubscript_.get(index));
    }

    /**
     * Sets the IXbfDefModelArraySubscriptChoice property <b>defModelArraySubscript</b> by index.
     *
     * @param index
     * @param defModelArraySubscript
     */
    public final void setDefModelArraySubscript(int index, IXbfDefModelArraySubscriptChoice defModelArraySubscript) {
        this.defModelArraySubscript_.set(index, defModelArraySubscript);
        if (defModelArraySubscript != null) {
            defModelArraySubscript.rSetParentRNode(this);
        }
    }

    /**
     * Adds the IXbfDefModelArraySubscriptChoice property <b>defModelArraySubscript</b> by index.
     *
     * @param index
     * @param defModelArraySubscript
     */
    public final void addDefModelArraySubscript(int index, IXbfDefModelArraySubscriptChoice defModelArraySubscript) {
        this.defModelArraySubscript_.add(index, defModelArraySubscript);
        if (defModelArraySubscript != null) {
            defModelArraySubscript.rSetParentRNode(this);
        }
    }

    /**
     * Remove the IXbfDefModelArraySubscriptChoice property <b>defModelArraySubscript</b> by index.
     *
     * @param index
     */
    public final void removeDefModelArraySubscript(int index) {
        this.defModelArraySubscript_.remove(index);
    }

    /**
     * Remove the IXbfDefModelArraySubscriptChoice property <b>defModelArraySubscript</b> by object.
     *
     * @param defModelArraySubscript
     */
    public final void removeDefModelArraySubscript(IXbfDefModelArraySubscriptChoice defModelArraySubscript) {
        this.defModelArraySubscript_.remove(defModelArraySubscript);
    }

    /**
     * Clear the IXbfDefModelArraySubscriptChoice property <b>defModelArraySubscript</b>.
     *
     */
    public final void clearDefModelArraySubscript() {
        this.defModelArraySubscript_.clear();
    }

    /**
     * Gets the XbfCoShape property <b>coShape</b>.
     *
     * @return XbfCoShape
     */
    public final XbfCoShape getCoShape() {
        return (coShape_);
    }

    /**
     * Sets the XbfCoShape property <b>coShape</b>.
     *
     * @param coShape
     */
    public final void setCoShape(XbfCoShape coShape) {
        this.coShape_ = coShape;
        if (coShape != null) {
            coShape.rSetParentRNode(this);
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
        buffer.append("<alloc");
        content_.makeTextAttribute(buffer);
        size = this.defModelArraySubscript_.size();
        for (int i = 0;i < size;i++) {
            IXbfDefModelArraySubscriptChoice value = (IXbfDefModelArraySubscriptChoice)this.defModelArraySubscript_.get(i);
            value.makeTextAttribute(buffer);
        }
        buffer.append(">");
        content_.makeTextElement(buffer);
        size = this.defModelArraySubscript_.size();
        for (int i = 0;i < size;i++) {
            IXbfDefModelArraySubscriptChoice value = (IXbfDefModelArraySubscriptChoice)this.defModelArraySubscript_.get(i);
            value.makeTextElement(buffer);
        }
        coShape_.makeTextElement(buffer);
        buffer.append("</alloc>");
    }

    /**
     * Makes an XML text representation.
     *
     * @param buffer
     * @exception IOException
     */
    public void makeTextElement(Writer buffer) throws IOException {
        int size;
        buffer.write("<alloc");
        content_.makeTextAttribute(buffer);
        size = this.defModelArraySubscript_.size();
        for (int i = 0;i < size;i++) {
            IXbfDefModelArraySubscriptChoice value = (IXbfDefModelArraySubscriptChoice)this.defModelArraySubscript_.get(i);
            value.makeTextAttribute(buffer);
        }
        buffer.write(">");
        content_.makeTextElement(buffer);
        size = this.defModelArraySubscript_.size();
        for (int i = 0;i < size;i++) {
            IXbfDefModelArraySubscriptChoice value = (IXbfDefModelArraySubscriptChoice)this.defModelArraySubscript_.get(i);
            value.makeTextElement(buffer);
        }
        coShape_.makeTextElement(buffer);
        buffer.write("</alloc>");
    }

    /**
     * Makes an XML text representation.
     *
     * @param buffer
     */
    public void makeTextElement(PrintWriter buffer) {
        int size;
        buffer.print("<alloc");
        content_.makeTextAttribute(buffer);
        size = this.defModelArraySubscript_.size();
        for (int i = 0;i < size;i++) {
            IXbfDefModelArraySubscriptChoice value = (IXbfDefModelArraySubscriptChoice)this.defModelArraySubscript_.get(i);
            value.makeTextAttribute(buffer);
        }
        buffer.print(">");
        content_.makeTextElement(buffer);
        size = this.defModelArraySubscript_.size();
        for (int i = 0;i < size;i++) {
            IXbfDefModelArraySubscriptChoice value = (IXbfDefModelArraySubscriptChoice)this.defModelArraySubscript_.get(i);
            value.makeTextElement(buffer);
        }
        coShape_.makeTextElement(buffer);
        buffer.print("</alloc>");
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
        if (content_ != null) {
            classNodes.add(content_);
        }
        classNodes.addAll(defModelArraySubscript_);
        if (coShape_ != null) {
            classNodes.add(coShape_);
        }
        IRNode[] nodes = new IRNode[classNodes.size()];
        return ((IRNode[])classNodes.toArray(nodes));
    }

    /**
     * Tests if a Element <code>element</code> is valid
     * for the <code>XbfAlloc</code>.
     *
     * @param element
     * @return boolean
     */
    public static boolean isMatch(Element element) {
        if (!URelaxer.isTargetElement(element, "alloc")) {
            return (false);
        }
        RStack target = new RStack(element);
        boolean $match$ = false;
        Element child;
        if (XbfFmemberRef.isMatchHungry(target)) {
            $match$ = true;
        } else if (XbfVar.isMatchHungry(target)) {
            $match$ = true;
        } else {
            return (false);
        }
        while (true) {
            if (XbfIndexRange.isMatchHungry(target)) {
                $match$ = true;
            } else if (XbfArrayIndex.isMatchHungry(target)) {
                $match$ = true;
            } else {
                break;
            }
        }
        if (!XbfCoShape.isMatchHungry(target)) {
            return (false);
        }
        $match$ = true;
        if (!target.isEmptyElement()) {
            return (false);
        }
        return (true);
    }

    /**
     * Tests if elements contained in a Stack <code>stack</code>
     * is valid for the <code>XbfAlloc</code>.
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
     * is valid for the <code>XbfAlloc</code>.
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
