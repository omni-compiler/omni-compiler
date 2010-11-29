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
 * <b>XbfTypeTable</b> is generated from XcodeML_F.rng by Relaxer.
 * This class is derived from:
 * 
 * <!-- for programmer
 * <element java:extends="xcodeml.f.XmfObj" name="typeTable">
 *   <zeroOrMore>
 *     <choice>
 *       <ref name="FbasicType"/>
 *       <ref name="FfunctionType"/>
 *       <ref name="FstructType"/>
 *     </choice>
 *   </zeroOrMore>
 * </element>
 * -->
 * <!-- for javadoc -->
 * <pre> &lt;element java:extends="xcodeml.f.XmfObj" name="typeTable"&gt;
 *   &lt;zeroOrMore&gt;
 *     &lt;choice&gt;
 *       &lt;ref name="FbasicType"/&gt;
 *       &lt;ref name="FfunctionType"/&gt;
 *       &lt;ref name="FstructType"/&gt;
 *     &lt;/choice&gt;
 *   &lt;/zeroOrMore&gt;
 * &lt;/element&gt;
 * </pre>
 *
 * @version XcodeML_F.rng (Mon Nov 29 15:25:55 JST 2010)
 * @author  Relaxer 1.0 (http://www.relaxer.org)
 */
public class XbfTypeTable extends xcodeml.f.XmfObj implements java.io.Serializable, Cloneable, IRVisitable, IRNode {
    // List<IXbfTypeTableChoice>
    private java.util.List content_ = new java.util.ArrayList();
    private IRNode parentRNode_;

    /**
     * Creates a <code>XbfTypeTable</code>.
     *
     */
    public XbfTypeTable() {
    }

    /**
     * Creates a <code>XbfTypeTable</code>.
     *
     * @param source
     */
    public XbfTypeTable(XbfTypeTable source) {
        setup(source);
    }

    /**
     * Creates a <code>XbfTypeTable</code> by the Stack <code>stack</code>
     * that contains Elements.
     * This constructor is supposed to be used internally
     * by the Relaxer system.
     *
     * @param stack
     */
    public XbfTypeTable(RStack stack) {
        setup(stack);
    }

    /**
     * Creates a <code>XbfTypeTable</code> by the Document <code>doc</code>.
     *
     * @param doc
     */
    public XbfTypeTable(Document doc) {
        setup(doc.getDocumentElement());
    }

    /**
     * Creates a <code>XbfTypeTable</code> by the Element <code>element</code>.
     *
     * @param element
     */
    public XbfTypeTable(Element element) {
        setup(element);
    }

    /**
     * Creates a <code>XbfTypeTable</code> by the File <code>file</code>.
     *
     * @param file
     * @exception IOException
     * @exception SAXException
     * @exception ParserConfigurationException
     */
    public XbfTypeTable(File file) throws IOException, SAXException, ParserConfigurationException {
        setup(file);
    }

    /**
     * Creates a <code>XbfTypeTable</code>
     * by the String representation of URI <code>uri</code>.
     *
     * @param uri
     * @exception IOException
     * @exception SAXException
     * @exception ParserConfigurationException
     */
    public XbfTypeTable(String uri) throws IOException, SAXException, ParserConfigurationException {
        setup(uri);
    }

    /**
     * Creates a <code>XbfTypeTable</code> by the URL <code>url</code>.
     *
     * @param url
     * @exception IOException
     * @exception SAXException
     * @exception ParserConfigurationException
     */
    public XbfTypeTable(URL url) throws IOException, SAXException, ParserConfigurationException {
        setup(url);
    }

    /**
     * Creates a <code>XbfTypeTable</code> by the InputStream <code>in</code>.
     *
     * @param in
     * @exception IOException
     * @exception SAXException
     * @exception ParserConfigurationException
     */
    public XbfTypeTable(InputStream in) throws IOException, SAXException, ParserConfigurationException {
        setup(in);
    }

    /**
     * Creates a <code>XbfTypeTable</code> by the InputSource <code>is</code>.
     *
     * @param is
     * @exception IOException
     * @exception SAXException
     * @exception ParserConfigurationException
     */
    public XbfTypeTable(InputSource is) throws IOException, SAXException, ParserConfigurationException {
        setup(is);
    }

    /**
     * Creates a <code>XbfTypeTable</code> by the Reader <code>reader</code>.
     *
     * @param reader
     * @exception IOException
     * @exception SAXException
     * @exception ParserConfigurationException
     */
    public XbfTypeTable(Reader reader) throws IOException, SAXException, ParserConfigurationException {
        setup(reader);
    }

    /**
     * Initializes the <code>XbfTypeTable</code> by the XbfTypeTable <code>source</code>.
     *
     * @param source
     */
    public void setup(XbfTypeTable source) {
        int size;
        this.content_.clear();
        size = source.content_.size();
        for (int i = 0;i < size;i++) {
            addContent((IXbfTypeTableChoice)source.getContent(i).clone());
        }
    }

    /**
     * Initializes the <code>XbfTypeTable</code> by the Document <code>doc</code>.
     *
     * @param doc
     */
    public void setup(Document doc) {
        setup(doc.getDocumentElement());
    }

    /**
     * Initializes the <code>XbfTypeTable</code> by the Element <code>element</code>.
     *
     * @param element
     */
    public void setup(Element element) {
        init(element);
    }

    /**
     * Initializes the <code>XbfTypeTable</code> by the Stack <code>stack</code>
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
        content_.clear();
        while (true) {
            if (XbfFfunctionType.isMatch(stack)) {
                addContent(factory.createXbfFfunctionType(stack));
            } else if (XbfFbasicType.isMatch(stack)) {
                addContent(factory.createXbfFbasicType(stack));
            } else if (XbfFstructType.isMatch(stack)) {
                addContent(factory.createXbfFstructType(stack));
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
        return (factory.createXbfTypeTable(this));
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
        Element element = doc.createElement("typeTable");
        int size;
        size = this.content_.size();
        for (int i = 0;i < size;i++) {
            IXbfTypeTableChoice value = (IXbfTypeTableChoice)this.content_.get(i);
            value.makeElement(element);
        }
        parent.appendChild(element);
    }

    /**
     * Initializes the <code>XbfTypeTable</code> by the File <code>file</code>.
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
     * Initializes the <code>XbfTypeTable</code>
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
     * Initializes the <code>XbfTypeTable</code> by the URL <code>url</code>.
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
     * Initializes the <code>XbfTypeTable</code> by the InputStream <code>in</code>.
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
     * Initializes the <code>XbfTypeTable</code> by the InputSource <code>is</code>.
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
     * Initializes the <code>XbfTypeTable</code> by the Reader <code>reader</code>.
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
     * Gets the IXbfTypeTableChoice property <b>content</b>.
     *
     * @return IXbfTypeTableChoice[]
     */
    public final IXbfTypeTableChoice[] getContent() {
        IXbfTypeTableChoice[] array = new IXbfTypeTableChoice[content_.size()];
        return ((IXbfTypeTableChoice[])content_.toArray(array));
    }

    /**
     * Sets the IXbfTypeTableChoice property <b>content</b>.
     *
     * @param content
     */
    public final void setContent(IXbfTypeTableChoice[] content) {
        this.content_.clear();
        for (int i = 0;i < content.length;i++) {
            addContent(content[i]);
        }
        for (int i = 0;i < content.length;i++) {
            content[i].rSetParentRNode(this);
        }
    }

    /**
     * Sets the IXbfTypeTableChoice property <b>content</b>.
     *
     * @param content
     */
    public final void setContent(IXbfTypeTableChoice content) {
        this.content_.clear();
        addContent(content);
        if (content != null) {
            content.rSetParentRNode(this);
        }
    }

    /**
     * Adds the IXbfTypeTableChoice property <b>content</b>.
     *
     * @param content
     */
    public final void addContent(IXbfTypeTableChoice content) {
        this.content_.add(content);
        if (content != null) {
            content.rSetParentRNode(this);
        }
    }

    /**
     * Adds the IXbfTypeTableChoice property <b>content</b>.
     *
     * @param content
     */
    public final void addContent(IXbfTypeTableChoice[] content) {
        for (int i = 0;i < content.length;i++) {
            addContent(content[i]);
        }
        for (int i = 0;i < content.length;i++) {
            content[i].rSetParentRNode(this);
        }
    }

    /**
     * Gets number of the IXbfTypeTableChoice property <b>content</b>.
     *
     * @return int
     */
    public final int sizeContent() {
        return (content_.size());
    }

    /**
     * Gets the IXbfTypeTableChoice property <b>content</b> by index.
     *
     * @param index
     * @return IXbfTypeTableChoice
     */
    public final IXbfTypeTableChoice getContent(int index) {
        return ((IXbfTypeTableChoice)content_.get(index));
    }

    /**
     * Sets the IXbfTypeTableChoice property <b>content</b> by index.
     *
     * @param index
     * @param content
     */
    public final void setContent(int index, IXbfTypeTableChoice content) {
        this.content_.set(index, content);
        if (content != null) {
            content.rSetParentRNode(this);
        }
    }

    /**
     * Adds the IXbfTypeTableChoice property <b>content</b> by index.
     *
     * @param index
     * @param content
     */
    public final void addContent(int index, IXbfTypeTableChoice content) {
        this.content_.add(index, content);
        if (content != null) {
            content.rSetParentRNode(this);
        }
    }

    /**
     * Remove the IXbfTypeTableChoice property <b>content</b> by index.
     *
     * @param index
     */
    public final void removeContent(int index) {
        this.content_.remove(index);
    }

    /**
     * Remove the IXbfTypeTableChoice property <b>content</b> by object.
     *
     * @param content
     */
    public final void removeContent(IXbfTypeTableChoice content) {
        this.content_.remove(content);
    }

    /**
     * Clear the IXbfTypeTableChoice property <b>content</b>.
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
        buffer.append("<typeTable");
        size = this.content_.size();
        for (int i = 0;i < size;i++) {
            IXbfTypeTableChoice value = (IXbfTypeTableChoice)this.content_.get(i);
            value.makeTextAttribute(buffer);
        }
        buffer.append(">");
        size = this.content_.size();
        for (int i = 0;i < size;i++) {
            IXbfTypeTableChoice value = (IXbfTypeTableChoice)this.content_.get(i);
            value.makeTextElement(buffer);
        }
        buffer.append("</typeTable>");
    }

    /**
     * Makes an XML text representation.
     *
     * @param buffer
     * @exception IOException
     */
    public void makeTextElement(Writer buffer) throws IOException {
        int size;
        buffer.write("<typeTable");
        size = this.content_.size();
        for (int i = 0;i < size;i++) {
            IXbfTypeTableChoice value = (IXbfTypeTableChoice)this.content_.get(i);
            value.makeTextAttribute(buffer);
        }
        buffer.write(">");
        size = this.content_.size();
        for (int i = 0;i < size;i++) {
            IXbfTypeTableChoice value = (IXbfTypeTableChoice)this.content_.get(i);
            value.makeTextElement(buffer);
        }
        buffer.write("</typeTable>");
    }

    /**
     * Makes an XML text representation.
     *
     * @param buffer
     */
    public void makeTextElement(PrintWriter buffer) {
        int size;
        buffer.print("<typeTable");
        size = this.content_.size();
        for (int i = 0;i < size;i++) {
            IXbfTypeTableChoice value = (IXbfTypeTableChoice)this.content_.get(i);
            value.makeTextAttribute(buffer);
        }
        buffer.print(">");
        size = this.content_.size();
        for (int i = 0;i < size;i++) {
            IXbfTypeTableChoice value = (IXbfTypeTableChoice)this.content_.get(i);
            value.makeTextElement(buffer);
        }
        buffer.print("</typeTable>");
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
        classNodes.addAll(content_);
        IRNode[] nodes = new IRNode[classNodes.size()];
        return ((IRNode[])classNodes.toArray(nodes));
    }

    /**
     * Tests if a Element <code>element</code> is valid
     * for the <code>XbfTypeTable</code>.
     *
     * @param element
     * @return boolean
     */
    public static boolean isMatch(Element element) {
        if (!URelaxer.isTargetElement(element, "typeTable")) {
            return (false);
        }
        RStack target = new RStack(element);
        boolean $match$ = false;
        Element child;
        while (true) {
            if (XbfFfunctionType.isMatchHungry(target)) {
                $match$ = true;
            } else if (XbfFbasicType.isMatchHungry(target)) {
                $match$ = true;
            } else if (XbfFstructType.isMatchHungry(target)) {
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
     * is valid for the <code>XbfTypeTable</code>.
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
     * is valid for the <code>XbfTypeTable</code>.
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
