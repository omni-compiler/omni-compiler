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
 * <b>XbcGccAsmClobbers</b> is generated from XcodeML_C.rng by Relaxer.
 * This class is derived from:
 * 
 * <!-- for programmer
 * <element java:extends="xcodeml.c.obj.XmcObj" name="gccAsmClobbers">
 *   <zeroOrMore>
 *     <ref name="stringConstant"/>
 *   </zeroOrMore>
 * </element>
 * -->
 * <!-- for javadoc -->
 * <pre> &lt;element java:extends="xcodeml.c.obj.XmcObj" name="gccAsmClobbers"&gt;
 *   &lt;zeroOrMore&gt;
 *     &lt;ref name="stringConstant"/&gt;
 *   &lt;/zeroOrMore&gt;
 * &lt;/element&gt;
 * </pre>
 *
 * @version XcodeML_C.rng (Thu Feb 02 16:55:19 JST 2012)
 * @author  Relaxer 1.0 (http://www.relaxer.org)
 */
public class XbcGccAsmClobbers extends xcodeml.c.obj.XmcObj implements java.io.Serializable, Cloneable, IRVisitable, IRNode {
    // List<XbcStringConstant>
    private java.util.List stringConstant_ = new java.util.ArrayList();
    private IRNode parentRNode_;

    /**
     * Creates a <code>XbcGccAsmClobbers</code>.
     *
     */
    public XbcGccAsmClobbers() {
    }

    /**
     * Creates a <code>XbcGccAsmClobbers</code>.
     *
     * @param source
     */
    public XbcGccAsmClobbers(XbcGccAsmClobbers source) {
        setup(source);
    }

    /**
     * Creates a <code>XbcGccAsmClobbers</code> by the Stack <code>stack</code>
     * that contains Elements.
     * This constructor is supposed to be used internally
     * by the Relaxer system.
     *
     * @param stack
     */
    public XbcGccAsmClobbers(RStack stack) {
        setup(stack);
    }

    /**
     * Creates a <code>XbcGccAsmClobbers</code> by the Document <code>doc</code>.
     *
     * @param doc
     */
    public XbcGccAsmClobbers(Document doc) {
        setup(doc.getDocumentElement());
    }

    /**
     * Creates a <code>XbcGccAsmClobbers</code> by the Element <code>element</code>.
     *
     * @param element
     */
    public XbcGccAsmClobbers(Element element) {
        setup(element);
    }

    /**
     * Creates a <code>XbcGccAsmClobbers</code> by the File <code>file</code>.
     *
     * @param file
     * @exception IOException
     * @exception SAXException
     * @exception ParserConfigurationException
     */
    public XbcGccAsmClobbers(File file) throws IOException, SAXException, ParserConfigurationException {
        setup(file);
    }

    /**
     * Creates a <code>XbcGccAsmClobbers</code>
     * by the String representation of URI <code>uri</code>.
     *
     * @param uri
     * @exception IOException
     * @exception SAXException
     * @exception ParserConfigurationException
     */
    public XbcGccAsmClobbers(String uri) throws IOException, SAXException, ParserConfigurationException {
        setup(uri);
    }

    /**
     * Creates a <code>XbcGccAsmClobbers</code> by the URL <code>url</code>.
     *
     * @param url
     * @exception IOException
     * @exception SAXException
     * @exception ParserConfigurationException
     */
    public XbcGccAsmClobbers(URL url) throws IOException, SAXException, ParserConfigurationException {
        setup(url);
    }

    /**
     * Creates a <code>XbcGccAsmClobbers</code> by the InputStream <code>in</code>.
     *
     * @param in
     * @exception IOException
     * @exception SAXException
     * @exception ParserConfigurationException
     */
    public XbcGccAsmClobbers(InputStream in) throws IOException, SAXException, ParserConfigurationException {
        setup(in);
    }

    /**
     * Creates a <code>XbcGccAsmClobbers</code> by the InputSource <code>is</code>.
     *
     * @param is
     * @exception IOException
     * @exception SAXException
     * @exception ParserConfigurationException
     */
    public XbcGccAsmClobbers(InputSource is) throws IOException, SAXException, ParserConfigurationException {
        setup(is);
    }

    /**
     * Creates a <code>XbcGccAsmClobbers</code> by the Reader <code>reader</code>.
     *
     * @param reader
     * @exception IOException
     * @exception SAXException
     * @exception ParserConfigurationException
     */
    public XbcGccAsmClobbers(Reader reader) throws IOException, SAXException, ParserConfigurationException {
        setup(reader);
    }

    /**
     * Initializes the <code>XbcGccAsmClobbers</code> by the XbcGccAsmClobbers <code>source</code>.
     *
     * @param source
     */
    public void setup(XbcGccAsmClobbers source) {
        int size;
        this.stringConstant_.clear();
        size = source.stringConstant_.size();
        for (int i = 0;i < size;i++) {
            addStringConstant((XbcStringConstant)source.getStringConstant(i).clone());
        }
    }

    /**
     * Initializes the <code>XbcGccAsmClobbers</code> by the Document <code>doc</code>.
     *
     * @param doc
     */
    public void setup(Document doc) {
        setup(doc.getDocumentElement());
    }

    /**
     * Initializes the <code>XbcGccAsmClobbers</code> by the Element <code>element</code>.
     *
     * @param element
     */
    public void setup(Element element) {
        init(element);
    }

    /**
     * Initializes the <code>XbcGccAsmClobbers</code> by the Stack <code>stack</code>
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
        stringConstant_.clear();
        while (true) {
            if (XbcStringConstant.isMatch(stack)) {
                addStringConstant(factory.createXbcStringConstant(stack));
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
        return (factory.createXbcGccAsmClobbers(this));
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
        Element element = doc.createElement("gccAsmClobbers");
        int size;
        size = this.stringConstant_.size();
        for (int i = 0;i < size;i++) {
            XbcStringConstant value = (XbcStringConstant)this.stringConstant_.get(i);
            value.makeElement(element);
        }
        parent.appendChild(element);
    }

    /**
     * Initializes the <code>XbcGccAsmClobbers</code> by the File <code>file</code>.
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
     * Initializes the <code>XbcGccAsmClobbers</code>
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
     * Initializes the <code>XbcGccAsmClobbers</code> by the URL <code>url</code>.
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
     * Initializes the <code>XbcGccAsmClobbers</code> by the InputStream <code>in</code>.
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
     * Initializes the <code>XbcGccAsmClobbers</code> by the InputSource <code>is</code>.
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
     * Initializes the <code>XbcGccAsmClobbers</code> by the Reader <code>reader</code>.
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
     * Gets the XbcStringConstant property <b>stringConstant</b>.
     *
     * @return XbcStringConstant[]
     */
    public final XbcStringConstant[] getStringConstant() {
        XbcStringConstant[] array = new XbcStringConstant[stringConstant_.size()];
        return ((XbcStringConstant[])stringConstant_.toArray(array));
    }

    /**
     * Sets the XbcStringConstant property <b>stringConstant</b>.
     *
     * @param stringConstant
     */
    public final void setStringConstant(XbcStringConstant[] stringConstant) {
        this.stringConstant_.clear();
        for (int i = 0;i < stringConstant.length;i++) {
            addStringConstant(stringConstant[i]);
        }
        for (int i = 0;i < stringConstant.length;i++) {
            stringConstant[i].rSetParentRNode(this);
        }
    }

    /**
     * Sets the XbcStringConstant property <b>stringConstant</b>.
     *
     * @param stringConstant
     */
    public final void setStringConstant(XbcStringConstant stringConstant) {
        this.stringConstant_.clear();
        addStringConstant(stringConstant);
        if (stringConstant != null) {
            stringConstant.rSetParentRNode(this);
        }
    }

    /**
     * Adds the XbcStringConstant property <b>stringConstant</b>.
     *
     * @param stringConstant
     */
    public final void addStringConstant(XbcStringConstant stringConstant) {
        this.stringConstant_.add(stringConstant);
        if (stringConstant != null) {
            stringConstant.rSetParentRNode(this);
        }
    }

    /**
     * Adds the XbcStringConstant property <b>stringConstant</b>.
     *
     * @param stringConstant
     */
    public final void addStringConstant(XbcStringConstant[] stringConstant) {
        for (int i = 0;i < stringConstant.length;i++) {
            addStringConstant(stringConstant[i]);
        }
        for (int i = 0;i < stringConstant.length;i++) {
            stringConstant[i].rSetParentRNode(this);
        }
    }

    /**
     * Gets number of the XbcStringConstant property <b>stringConstant</b>.
     *
     * @return int
     */
    public final int sizeStringConstant() {
        return (stringConstant_.size());
    }

    /**
     * Gets the XbcStringConstant property <b>stringConstant</b> by index.
     *
     * @param index
     * @return XbcStringConstant
     */
    public final XbcStringConstant getStringConstant(int index) {
        return ((XbcStringConstant)stringConstant_.get(index));
    }

    /**
     * Sets the XbcStringConstant property <b>stringConstant</b> by index.
     *
     * @param index
     * @param stringConstant
     */
    public final void setStringConstant(int index, XbcStringConstant stringConstant) {
        this.stringConstant_.set(index, stringConstant);
        if (stringConstant != null) {
            stringConstant.rSetParentRNode(this);
        }
    }

    /**
     * Adds the XbcStringConstant property <b>stringConstant</b> by index.
     *
     * @param index
     * @param stringConstant
     */
    public final void addStringConstant(int index, XbcStringConstant stringConstant) {
        this.stringConstant_.add(index, stringConstant);
        if (stringConstant != null) {
            stringConstant.rSetParentRNode(this);
        }
    }

    /**
     * Remove the XbcStringConstant property <b>stringConstant</b> by index.
     *
     * @param index
     */
    public final void removeStringConstant(int index) {
        this.stringConstant_.remove(index);
    }

    /**
     * Remove the XbcStringConstant property <b>stringConstant</b> by object.
     *
     * @param stringConstant
     */
    public final void removeStringConstant(XbcStringConstant stringConstant) {
        this.stringConstant_.remove(stringConstant);
    }

    /**
     * Clear the XbcStringConstant property <b>stringConstant</b>.
     *
     */
    public final void clearStringConstant() {
        this.stringConstant_.clear();
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
        buffer.append("<gccAsmClobbers");
        buffer.append(">");
        size = this.stringConstant_.size();
        for (int i = 0;i < size;i++) {
            XbcStringConstant value = (XbcStringConstant)this.stringConstant_.get(i);
            value.makeTextElement(buffer);
        }
        buffer.append("</gccAsmClobbers>");
    }

    /**
     * Makes an XML text representation.
     *
     * @param buffer
     * @exception IOException
     */
    public void makeTextElement(Writer buffer) throws IOException {
        int size;
        buffer.write("<gccAsmClobbers");
        buffer.write(">");
        size = this.stringConstant_.size();
        for (int i = 0;i < size;i++) {
            XbcStringConstant value = (XbcStringConstant)this.stringConstant_.get(i);
            value.makeTextElement(buffer);
        }
        buffer.write("</gccAsmClobbers>");
    }

    /**
     * Makes an XML text representation.
     *
     * @param buffer
     */
    public void makeTextElement(PrintWriter buffer) {
        int size;
        buffer.print("<gccAsmClobbers");
        buffer.print(">");
        size = this.stringConstant_.size();
        for (int i = 0;i < size;i++) {
            XbcStringConstant value = (XbcStringConstant)this.stringConstant_.get(i);
            value.makeTextElement(buffer);
        }
        buffer.print("</gccAsmClobbers>");
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
        classNodes.addAll(stringConstant_);
        IRNode[] nodes = new IRNode[classNodes.size()];
        return ((IRNode[])classNodes.toArray(nodes));
    }

    /**
     * Tests if a Element <code>element</code> is valid
     * for the <code>XbcGccAsmClobbers</code>.
     *
     * @param element
     * @return boolean
     */
    public static boolean isMatch(Element element) {
        if (!URelaxer.isTargetElement(element, "gccAsmClobbers")) {
            return (false);
        }
        RStack target = new RStack(element);
        boolean $match$ = false;
        Element child;
        while (true) {
            if (!XbcStringConstant.isMatchHungry(target)) {
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
     * is valid for the <code>XbcGccAsmClobbers</code>.
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
     * is valid for the <code>XbcGccAsmClobbers</code>.
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
