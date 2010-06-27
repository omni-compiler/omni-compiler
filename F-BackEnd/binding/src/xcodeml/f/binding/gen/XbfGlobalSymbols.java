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
 * <b>XbfGlobalSymbols</b> is generated from XcodeML_F.rng by Relaxer.
 * This class is derived from:
 * 
 * <!-- for programmer
 * <element java:extends="xcodeml.f.XmfObj" java:implements="xcodeml.binding.IXbSymbols" name="globalSymbols">
 *   <zeroOrMore>
 *     <ref name="id"/>
 *   </zeroOrMore>
 * </element>
 * -->
 * <!-- for javadoc -->
 * <pre> &lt;element java:extends="xcodeml.f.XmfObj" java:implements="xcodeml.binding.IXbSymbols" name="globalSymbols"&gt;
 *   &lt;zeroOrMore&gt;
 *     &lt;ref name="id"/&gt;
 *   &lt;/zeroOrMore&gt;
 * &lt;/element&gt;
 * </pre>
 *
 * @version XcodeML_F.rng (Fri Dec 04 19:18:15 JST 2009)
 * @author  Relaxer 1.0 (http://www.relaxer.org)
 */
public class XbfGlobalSymbols extends xcodeml.f.XmfObj implements java.io.Serializable, Cloneable, xcodeml.binding.IXbSymbols, IRVisitable, IRNode {
    // List<XbfId>
    private java.util.List id_ = new java.util.ArrayList();
    private IRNode parentRNode_;

    /**
     * Creates a <code>XbfGlobalSymbols</code>.
     *
     */
    public XbfGlobalSymbols() {
    }

    /**
     * Creates a <code>XbfGlobalSymbols</code>.
     *
     * @param source
     */
    public XbfGlobalSymbols(XbfGlobalSymbols source) {
        setup(source);
    }

    /**
     * Creates a <code>XbfGlobalSymbols</code> by the Stack <code>stack</code>
     * that contains Elements.
     * This constructor is supposed to be used internally
     * by the Relaxer system.
     *
     * @param stack
     */
    public XbfGlobalSymbols(RStack stack) {
        setup(stack);
    }

    /**
     * Creates a <code>XbfGlobalSymbols</code> by the Document <code>doc</code>.
     *
     * @param doc
     */
    public XbfGlobalSymbols(Document doc) {
        setup(doc.getDocumentElement());
    }

    /**
     * Creates a <code>XbfGlobalSymbols</code> by the Element <code>element</code>.
     *
     * @param element
     */
    public XbfGlobalSymbols(Element element) {
        setup(element);
    }

    /**
     * Creates a <code>XbfGlobalSymbols</code> by the File <code>file</code>.
     *
     * @param file
     * @exception IOException
     * @exception SAXException
     * @exception ParserConfigurationException
     */
    public XbfGlobalSymbols(File file) throws IOException, SAXException, ParserConfigurationException {
        setup(file);
    }

    /**
     * Creates a <code>XbfGlobalSymbols</code>
     * by the String representation of URI <code>uri</code>.
     *
     * @param uri
     * @exception IOException
     * @exception SAXException
     * @exception ParserConfigurationException
     */
    public XbfGlobalSymbols(String uri) throws IOException, SAXException, ParserConfigurationException {
        setup(uri);
    }

    /**
     * Creates a <code>XbfGlobalSymbols</code> by the URL <code>url</code>.
     *
     * @param url
     * @exception IOException
     * @exception SAXException
     * @exception ParserConfigurationException
     */
    public XbfGlobalSymbols(URL url) throws IOException, SAXException, ParserConfigurationException {
        setup(url);
    }

    /**
     * Creates a <code>XbfGlobalSymbols</code> by the InputStream <code>in</code>.
     *
     * @param in
     * @exception IOException
     * @exception SAXException
     * @exception ParserConfigurationException
     */
    public XbfGlobalSymbols(InputStream in) throws IOException, SAXException, ParserConfigurationException {
        setup(in);
    }

    /**
     * Creates a <code>XbfGlobalSymbols</code> by the InputSource <code>is</code>.
     *
     * @param is
     * @exception IOException
     * @exception SAXException
     * @exception ParserConfigurationException
     */
    public XbfGlobalSymbols(InputSource is) throws IOException, SAXException, ParserConfigurationException {
        setup(is);
    }

    /**
     * Creates a <code>XbfGlobalSymbols</code> by the Reader <code>reader</code>.
     *
     * @param reader
     * @exception IOException
     * @exception SAXException
     * @exception ParserConfigurationException
     */
    public XbfGlobalSymbols(Reader reader) throws IOException, SAXException, ParserConfigurationException {
        setup(reader);
    }

    /**
     * Initializes the <code>XbfGlobalSymbols</code> by the XbfGlobalSymbols <code>source</code>.
     *
     * @param source
     */
    public void setup(XbfGlobalSymbols source) {
        int size;
        this.id_.clear();
        size = source.id_.size();
        for (int i = 0;i < size;i++) {
            addId((XbfId)source.getId(i).clone());
        }
    }

    /**
     * Initializes the <code>XbfGlobalSymbols</code> by the Document <code>doc</code>.
     *
     * @param doc
     */
    public void setup(Document doc) {
        setup(doc.getDocumentElement());
    }

    /**
     * Initializes the <code>XbfGlobalSymbols</code> by the Element <code>element</code>.
     *
     * @param element
     */
    public void setup(Element element) {
        init(element);
    }

    /**
     * Initializes the <code>XbfGlobalSymbols</code> by the Stack <code>stack</code>
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
        id_.clear();
        while (true) {
            if (XbfId.isMatch(stack)) {
                addId(factory.createXbfId(stack));
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
        return (factory.createXbfGlobalSymbols(this));
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
        Element element = doc.createElement("globalSymbols");
        int size;
        size = this.id_.size();
        for (int i = 0;i < size;i++) {
            XbfId value = (XbfId)this.id_.get(i);
            value.makeElement(element);
        }
        parent.appendChild(element);
    }

    /**
     * Initializes the <code>XbfGlobalSymbols</code> by the File <code>file</code>.
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
     * Initializes the <code>XbfGlobalSymbols</code>
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
     * Initializes the <code>XbfGlobalSymbols</code> by the URL <code>url</code>.
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
     * Initializes the <code>XbfGlobalSymbols</code> by the InputStream <code>in</code>.
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
     * Initializes the <code>XbfGlobalSymbols</code> by the InputSource <code>is</code>.
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
     * Initializes the <code>XbfGlobalSymbols</code> by the Reader <code>reader</code>.
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
     * Gets the XbfId property <b>id</b>.
     *
     * @return XbfId[]
     */
    public final XbfId[] getId() {
        XbfId[] array = new XbfId[id_.size()];
        return ((XbfId[])id_.toArray(array));
    }

    /**
     * Sets the XbfId property <b>id</b>.
     *
     * @param id
     */
    public final void setId(XbfId[] id) {
        this.id_.clear();
        for (int i = 0;i < id.length;i++) {
            addId(id[i]);
        }
        for (int i = 0;i < id.length;i++) {
            id[i].rSetParentRNode(this);
        }
    }

    /**
     * Sets the XbfId property <b>id</b>.
     *
     * @param id
     */
    public final void setId(XbfId id) {
        this.id_.clear();
        addId(id);
        if (id != null) {
            id.rSetParentRNode(this);
        }
    }

    /**
     * Adds the XbfId property <b>id</b>.
     *
     * @param id
     */
    public final void addId(XbfId id) {
        this.id_.add(id);
        if (id != null) {
            id.rSetParentRNode(this);
        }
    }

    /**
     * Adds the XbfId property <b>id</b>.
     *
     * @param id
     */
    public final void addId(XbfId[] id) {
        for (int i = 0;i < id.length;i++) {
            addId(id[i]);
        }
        for (int i = 0;i < id.length;i++) {
            id[i].rSetParentRNode(this);
        }
    }

    /**
     * Gets number of the XbfId property <b>id</b>.
     *
     * @return int
     */
    public final int sizeId() {
        return (id_.size());
    }

    /**
     * Gets the XbfId property <b>id</b> by index.
     *
     * @param index
     * @return XbfId
     */
    public final XbfId getId(int index) {
        return ((XbfId)id_.get(index));
    }

    /**
     * Sets the XbfId property <b>id</b> by index.
     *
     * @param index
     * @param id
     */
    public final void setId(int index, XbfId id) {
        this.id_.set(index, id);
        if (id != null) {
            id.rSetParentRNode(this);
        }
    }

    /**
     * Adds the XbfId property <b>id</b> by index.
     *
     * @param index
     * @param id
     */
    public final void addId(int index, XbfId id) {
        this.id_.add(index, id);
        if (id != null) {
            id.rSetParentRNode(this);
        }
    }

    /**
     * Remove the XbfId property <b>id</b> by index.
     *
     * @param index
     */
    public final void removeId(int index) {
        this.id_.remove(index);
    }

    /**
     * Remove the XbfId property <b>id</b> by object.
     *
     * @param id
     */
    public final void removeId(XbfId id) {
        this.id_.remove(id);
    }

    /**
     * Clear the XbfId property <b>id</b>.
     *
     */
    public final void clearId() {
        this.id_.clear();
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
        buffer.append("<globalSymbols");
        buffer.append(">");
        size = this.id_.size();
        for (int i = 0;i < size;i++) {
            XbfId value = (XbfId)this.id_.get(i);
            value.makeTextElement(buffer);
        }
        buffer.append("</globalSymbols>");
    }

    /**
     * Makes an XML text representation.
     *
     * @param buffer
     * @exception IOException
     */
    public void makeTextElement(Writer buffer) throws IOException {
        int size;
        buffer.write("<globalSymbols");
        buffer.write(">");
        size = this.id_.size();
        for (int i = 0;i < size;i++) {
            XbfId value = (XbfId)this.id_.get(i);
            value.makeTextElement(buffer);
        }
        buffer.write("</globalSymbols>");
    }

    /**
     * Makes an XML text representation.
     *
     * @param buffer
     */
    public void makeTextElement(PrintWriter buffer) {
        int size;
        buffer.print("<globalSymbols");
        buffer.print(">");
        size = this.id_.size();
        for (int i = 0;i < size;i++) {
            XbfId value = (XbfId)this.id_.get(i);
            value.makeTextElement(buffer);
        }
        buffer.print("</globalSymbols>");
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
        classNodes.addAll(id_);
        IRNode[] nodes = new IRNode[classNodes.size()];
        return ((IRNode[])classNodes.toArray(nodes));
    }

    /**
     * Tests if a Element <code>element</code> is valid
     * for the <code>XbfGlobalSymbols</code>.
     *
     * @param element
     * @return boolean
     */
    public static boolean isMatch(Element element) {
        if (!URelaxer.isTargetElement(element, "globalSymbols")) {
            return (false);
        }
        RStack target = new RStack(element);
        boolean $match$ = false;
        Element child;
        while (true) {
            if (!XbfId.isMatchHungry(target)) {
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
     * is valid for the <code>XbfGlobalSymbols</code>.
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
     * is valid for the <code>XbfGlobalSymbols</code>.
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
