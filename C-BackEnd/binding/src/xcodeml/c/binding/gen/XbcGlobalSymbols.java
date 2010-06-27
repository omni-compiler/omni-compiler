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
 * <b>XbcGlobalSymbols</b> is generated from XcodeML_C.rng by Relaxer.
 * This class is derived from:
 * 
 * <!-- for programmer
 * <element java:extends="xcodeml.c.obj.XmcObj" java:implements="xcodeml.binding.IXbSymbols" name="globalSymbols">
 *   <zeroOrMore>
 *     <ref name="id"/>
 *   </zeroOrMore>
 * </element>
 * -->
 * <!-- for javadoc -->
 * <pre> &lt;element java:extends="xcodeml.c.obj.XmcObj" java:implements="xcodeml.binding.IXbSymbols" name="globalSymbols"&gt;
 *   &lt;zeroOrMore&gt;
 *     &lt;ref name="id"/&gt;
 *   &lt;/zeroOrMore&gt;
 * &lt;/element&gt;
 * </pre>
 *
 * @version XcodeML_C.rng (Thu Sep 24 16:30:19 JST 2009)
 * @author  Relaxer 1.0 (http://www.relaxer.org)
 */
public class XbcGlobalSymbols extends xcodeml.c.obj.XmcObj implements java.io.Serializable, Cloneable, xcodeml.binding.IXbSymbols, IRVisitable, IRNode {
    // List<XbcId>
    private java.util.List id_ = new java.util.ArrayList();
    private IRNode parentRNode_;

    /**
     * Creates a <code>XbcGlobalSymbols</code>.
     *
     */
    public XbcGlobalSymbols() {
    }

    /**
     * Creates a <code>XbcGlobalSymbols</code>.
     *
     * @param source
     */
    public XbcGlobalSymbols(XbcGlobalSymbols source) {
        setup(source);
    }

    /**
     * Creates a <code>XbcGlobalSymbols</code> by the Stack <code>stack</code>
     * that contains Elements.
     * This constructor is supposed to be used internally
     * by the Relaxer system.
     *
     * @param stack
     */
    public XbcGlobalSymbols(RStack stack) {
        setup(stack);
    }

    /**
     * Creates a <code>XbcGlobalSymbols</code> by the Document <code>doc</code>.
     *
     * @param doc
     */
    public XbcGlobalSymbols(Document doc) {
        setup(doc.getDocumentElement());
    }

    /**
     * Creates a <code>XbcGlobalSymbols</code> by the Element <code>element</code>.
     *
     * @param element
     */
    public XbcGlobalSymbols(Element element) {
        setup(element);
    }

    /**
     * Creates a <code>XbcGlobalSymbols</code> by the File <code>file</code>.
     *
     * @param file
     * @exception IOException
     * @exception SAXException
     * @exception ParserConfigurationException
     */
    public XbcGlobalSymbols(File file) throws IOException, SAXException, ParserConfigurationException {
        setup(file);
    }

    /**
     * Creates a <code>XbcGlobalSymbols</code>
     * by the String representation of URI <code>uri</code>.
     *
     * @param uri
     * @exception IOException
     * @exception SAXException
     * @exception ParserConfigurationException
     */
    public XbcGlobalSymbols(String uri) throws IOException, SAXException, ParserConfigurationException {
        setup(uri);
    }

    /**
     * Creates a <code>XbcGlobalSymbols</code> by the URL <code>url</code>.
     *
     * @param url
     * @exception IOException
     * @exception SAXException
     * @exception ParserConfigurationException
     */
    public XbcGlobalSymbols(URL url) throws IOException, SAXException, ParserConfigurationException {
        setup(url);
    }

    /**
     * Creates a <code>XbcGlobalSymbols</code> by the InputStream <code>in</code>.
     *
     * @param in
     * @exception IOException
     * @exception SAXException
     * @exception ParserConfigurationException
     */
    public XbcGlobalSymbols(InputStream in) throws IOException, SAXException, ParserConfigurationException {
        setup(in);
    }

    /**
     * Creates a <code>XbcGlobalSymbols</code> by the InputSource <code>is</code>.
     *
     * @param is
     * @exception IOException
     * @exception SAXException
     * @exception ParserConfigurationException
     */
    public XbcGlobalSymbols(InputSource is) throws IOException, SAXException, ParserConfigurationException {
        setup(is);
    }

    /**
     * Creates a <code>XbcGlobalSymbols</code> by the Reader <code>reader</code>.
     *
     * @param reader
     * @exception IOException
     * @exception SAXException
     * @exception ParserConfigurationException
     */
    public XbcGlobalSymbols(Reader reader) throws IOException, SAXException, ParserConfigurationException {
        setup(reader);
    }

    /**
     * Initializes the <code>XbcGlobalSymbols</code> by the XbcGlobalSymbols <code>source</code>.
     *
     * @param source
     */
    public void setup(XbcGlobalSymbols source) {
        int size;
        this.id_.clear();
        size = source.id_.size();
        for (int i = 0;i < size;i++) {
            addId((XbcId)source.getId(i).clone());
        }
    }

    /**
     * Initializes the <code>XbcGlobalSymbols</code> by the Document <code>doc</code>.
     *
     * @param doc
     */
    public void setup(Document doc) {
        setup(doc.getDocumentElement());
    }

    /**
     * Initializes the <code>XbcGlobalSymbols</code> by the Element <code>element</code>.
     *
     * @param element
     */
    public void setup(Element element) {
        init(element);
    }

    /**
     * Initializes the <code>XbcGlobalSymbols</code> by the Stack <code>stack</code>
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
        id_.clear();
        while (true) {
            if (XbcId.isMatch(stack)) {
                addId(factory.createXbcId(stack));
            } else {
                break;
            }
        }
    }

    /**
     * @return Object
     */
    public Object clone() {
        IXcodeML_CFactory factory = XcodeML_CFactory.getFactory();
        return (factory.createXbcGlobalSymbols(this));
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
            XbcId value = (XbcId)this.id_.get(i);
            value.makeElement(element);
        }
        parent.appendChild(element);
    }

    /**
     * Initializes the <code>XbcGlobalSymbols</code> by the File <code>file</code>.
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
     * Initializes the <code>XbcGlobalSymbols</code>
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
     * Initializes the <code>XbcGlobalSymbols</code> by the URL <code>url</code>.
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
     * Initializes the <code>XbcGlobalSymbols</code> by the InputStream <code>in</code>.
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
     * Initializes the <code>XbcGlobalSymbols</code> by the InputSource <code>is</code>.
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
     * Initializes the <code>XbcGlobalSymbols</code> by the Reader <code>reader</code>.
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
     * Gets the XbcId property <b>id</b>.
     *
     * @return XbcId[]
     */
    public final XbcId[] getId() {
        XbcId[] array = new XbcId[id_.size()];
        return ((XbcId[])id_.toArray(array));
    }

    /**
     * Sets the XbcId property <b>id</b>.
     *
     * @param id
     */
    public final void setId(XbcId[] id) {
        this.id_.clear();
        for (int i = 0;i < id.length;i++) {
            addId(id[i]);
        }
        for (int i = 0;i < id.length;i++) {
            id[i].rSetParentRNode(this);
        }
    }

    /**
     * Sets the XbcId property <b>id</b>.
     *
     * @param id
     */
    public final void setId(XbcId id) {
        this.id_.clear();
        addId(id);
        if (id != null) {
            id.rSetParentRNode(this);
        }
    }

    /**
     * Adds the XbcId property <b>id</b>.
     *
     * @param id
     */
    public final void addId(XbcId id) {
        this.id_.add(id);
        if (id != null) {
            id.rSetParentRNode(this);
        }
    }

    /**
     * Adds the XbcId property <b>id</b>.
     *
     * @param id
     */
    public final void addId(XbcId[] id) {
        for (int i = 0;i < id.length;i++) {
            addId(id[i]);
        }
        for (int i = 0;i < id.length;i++) {
            id[i].rSetParentRNode(this);
        }
    }

    /**
     * Gets number of the XbcId property <b>id</b>.
     *
     * @return int
     */
    public final int sizeId() {
        return (id_.size());
    }

    /**
     * Gets the XbcId property <b>id</b> by index.
     *
     * @param index
     * @return XbcId
     */
    public final XbcId getId(int index) {
        return ((XbcId)id_.get(index));
    }

    /**
     * Sets the XbcId property <b>id</b> by index.
     *
     * @param index
     * @param id
     */
    public final void setId(int index, XbcId id) {
        this.id_.set(index, id);
        if (id != null) {
            id.rSetParentRNode(this);
        }
    }

    /**
     * Adds the XbcId property <b>id</b> by index.
     *
     * @param index
     * @param id
     */
    public final void addId(int index, XbcId id) {
        this.id_.add(index, id);
        if (id != null) {
            id.rSetParentRNode(this);
        }
    }

    /**
     * Remove the XbcId property <b>id</b> by index.
     *
     * @param index
     */
    public final void removeId(int index) {
        this.id_.remove(index);
    }

    /**
     * Remove the XbcId property <b>id</b> by object.
     *
     * @param id
     */
    public final void removeId(XbcId id) {
        this.id_.remove(id);
    }

    /**
     * Clear the XbcId property <b>id</b>.
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
            XbcId value = (XbcId)this.id_.get(i);
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
            XbcId value = (XbcId)this.id_.get(i);
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
            XbcId value = (XbcId)this.id_.get(i);
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
     * for the <code>XbcGlobalSymbols</code>.
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
            if (!XbcId.isMatchHungry(target)) {
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
     * is valid for the <code>XbcGlobalSymbols</code>.
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
     * is valid for the <code>XbcGlobalSymbols</code>.
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
