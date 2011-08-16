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
 * <b>XbcParams</b> is generated from XcodeML_C.rng by Relaxer.
 * This class is derived from:
 * 
 * <!-- for programmer
 * <element java:extends="xcodeml.c.obj.XmcObj" name="params">
 *   <zeroOrMore>
 *     <ref name="name"/>
 *   </zeroOrMore>
 *   <optional>
 *     <ref name="ellipsis"/>
 *   </optional>
 * </element>
 * -->
 * <!-- for javadoc -->
 * <pre> &lt;element java:extends="xcodeml.c.obj.XmcObj" name="params"&gt;
 *   &lt;zeroOrMore&gt;
 *     &lt;ref name="name"/&gt;
 *   &lt;/zeroOrMore&gt;
 *   &lt;optional&gt;
 *     &lt;ref name="ellipsis"/&gt;
 *   &lt;/optional&gt;
 * &lt;/element&gt;
 * </pre>
 *
 * @version XcodeML_C.rng (Mon Aug 15 15:55:18 JST 2011)
 * @author  Relaxer 1.0 (http://www.relaxer.org)
 */
public class XbcParams extends xcodeml.c.obj.XmcObj implements java.io.Serializable, Cloneable, IRVisitable, IRNode {
    // List<XbcName>
    private java.util.List name_ = new java.util.ArrayList();
    private String ellipsis_;
    private IRNode parentRNode_;

    /**
     * Creates a <code>XbcParams</code>.
     *
     */
    public XbcParams() {
    }

    /**
     * Creates a <code>XbcParams</code>.
     *
     * @param source
     */
    public XbcParams(XbcParams source) {
        setup(source);
    }

    /**
     * Creates a <code>XbcParams</code> by the Stack <code>stack</code>
     * that contains Elements.
     * This constructor is supposed to be used internally
     * by the Relaxer system.
     *
     * @param stack
     */
    public XbcParams(RStack stack) {
        setup(stack);
    }

    /**
     * Creates a <code>XbcParams</code> by the Document <code>doc</code>.
     *
     * @param doc
     */
    public XbcParams(Document doc) {
        setup(doc.getDocumentElement());
    }

    /**
     * Creates a <code>XbcParams</code> by the Element <code>element</code>.
     *
     * @param element
     */
    public XbcParams(Element element) {
        setup(element);
    }

    /**
     * Creates a <code>XbcParams</code> by the File <code>file</code>.
     *
     * @param file
     * @exception IOException
     * @exception SAXException
     * @exception ParserConfigurationException
     */
    public XbcParams(File file) throws IOException, SAXException, ParserConfigurationException {
        setup(file);
    }

    /**
     * Creates a <code>XbcParams</code>
     * by the String representation of URI <code>uri</code>.
     *
     * @param uri
     * @exception IOException
     * @exception SAXException
     * @exception ParserConfigurationException
     */
    public XbcParams(String uri) throws IOException, SAXException, ParserConfigurationException {
        setup(uri);
    }

    /**
     * Creates a <code>XbcParams</code> by the URL <code>url</code>.
     *
     * @param url
     * @exception IOException
     * @exception SAXException
     * @exception ParserConfigurationException
     */
    public XbcParams(URL url) throws IOException, SAXException, ParserConfigurationException {
        setup(url);
    }

    /**
     * Creates a <code>XbcParams</code> by the InputStream <code>in</code>.
     *
     * @param in
     * @exception IOException
     * @exception SAXException
     * @exception ParserConfigurationException
     */
    public XbcParams(InputStream in) throws IOException, SAXException, ParserConfigurationException {
        setup(in);
    }

    /**
     * Creates a <code>XbcParams</code> by the InputSource <code>is</code>.
     *
     * @param is
     * @exception IOException
     * @exception SAXException
     * @exception ParserConfigurationException
     */
    public XbcParams(InputSource is) throws IOException, SAXException, ParserConfigurationException {
        setup(is);
    }

    /**
     * Creates a <code>XbcParams</code> by the Reader <code>reader</code>.
     *
     * @param reader
     * @exception IOException
     * @exception SAXException
     * @exception ParserConfigurationException
     */
    public XbcParams(Reader reader) throws IOException, SAXException, ParserConfigurationException {
        setup(reader);
    }

    /**
     * Initializes the <code>XbcParams</code> by the XbcParams <code>source</code>.
     *
     * @param source
     */
    public void setup(XbcParams source) {
        int size;
        this.name_.clear();
        size = source.name_.size();
        for (int i = 0;i < size;i++) {
            addName((XbcName)source.getName(i).clone());
        }
        setEllipsis(source.getEllipsis());
    }

    /**
     * Initializes the <code>XbcParams</code> by the Document <code>doc</code>.
     *
     * @param doc
     */
    public void setup(Document doc) {
        setup(doc.getDocumentElement());
    }

    /**
     * Initializes the <code>XbcParams</code> by the Element <code>element</code>.
     *
     * @param element
     */
    public void setup(Element element) {
        init(element);
    }

    /**
     * Initializes the <code>XbcParams</code> by the Stack <code>stack</code>
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
        name_.clear();
        while (true) {
            if (XbcName.isMatch(stack)) {
                addName(factory.createXbcName(stack));
            } else {
                break;
            }
        }
        ellipsis_ = URelaxer.getElementPropertyAsStringByStack(stack, "ellipsis");
    }

    /**
     * @return Object
     */
    public Object clone() {
        IXcodeML_CFactory factory = XcodeML_CFactory.getFactory();
        return (factory.createXbcParams(this));
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
        Element element = doc.createElement("params");
        int size;
        size = this.name_.size();
        for (int i = 0;i < size;i++) {
            XbcName value = (XbcName)this.name_.get(i);
            value.makeElement(element);
        }
        if (this.ellipsis_ != null) {
            URelaxer.setElementPropertyByString(element, "ellipsis", this.ellipsis_);
        }
        parent.appendChild(element);
    }

    /**
     * Initializes the <code>XbcParams</code> by the File <code>file</code>.
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
     * Initializes the <code>XbcParams</code>
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
     * Initializes the <code>XbcParams</code> by the URL <code>url</code>.
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
     * Initializes the <code>XbcParams</code> by the InputStream <code>in</code>.
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
     * Initializes the <code>XbcParams</code> by the InputSource <code>is</code>.
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
     * Initializes the <code>XbcParams</code> by the Reader <code>reader</code>.
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
     * Gets the XbcName property <b>name</b>.
     *
     * @return XbcName[]
     */
    public final XbcName[] getName() {
        XbcName[] array = new XbcName[name_.size()];
        return ((XbcName[])name_.toArray(array));
    }

    /**
     * Sets the XbcName property <b>name</b>.
     *
     * @param name
     */
    public final void setName(XbcName[] name) {
        this.name_.clear();
        for (int i = 0;i < name.length;i++) {
            addName(name[i]);
        }
        for (int i = 0;i < name.length;i++) {
            name[i].rSetParentRNode(this);
        }
    }

    /**
     * Sets the XbcName property <b>name</b>.
     *
     * @param name
     */
    public final void setName(XbcName name) {
        this.name_.clear();
        addName(name);
        if (name != null) {
            name.rSetParentRNode(this);
        }
    }

    /**
     * Adds the XbcName property <b>name</b>.
     *
     * @param name
     */
    public final void addName(XbcName name) {
        this.name_.add(name);
        if (name != null) {
            name.rSetParentRNode(this);
        }
    }

    /**
     * Adds the XbcName property <b>name</b>.
     *
     * @param name
     */
    public final void addName(XbcName[] name) {
        for (int i = 0;i < name.length;i++) {
            addName(name[i]);
        }
        for (int i = 0;i < name.length;i++) {
            name[i].rSetParentRNode(this);
        }
    }

    /**
     * Gets number of the XbcName property <b>name</b>.
     *
     * @return int
     */
    public final int sizeName() {
        return (name_.size());
    }

    /**
     * Gets the XbcName property <b>name</b> by index.
     *
     * @param index
     * @return XbcName
     */
    public final XbcName getName(int index) {
        return ((XbcName)name_.get(index));
    }

    /**
     * Sets the XbcName property <b>name</b> by index.
     *
     * @param index
     * @param name
     */
    public final void setName(int index, XbcName name) {
        this.name_.set(index, name);
        if (name != null) {
            name.rSetParentRNode(this);
        }
    }

    /**
     * Adds the XbcName property <b>name</b> by index.
     *
     * @param index
     * @param name
     */
    public final void addName(int index, XbcName name) {
        this.name_.add(index, name);
        if (name != null) {
            name.rSetParentRNode(this);
        }
    }

    /**
     * Remove the XbcName property <b>name</b> by index.
     *
     * @param index
     */
    public final void removeName(int index) {
        this.name_.remove(index);
    }

    /**
     * Remove the XbcName property <b>name</b> by object.
     *
     * @param name
     */
    public final void removeName(XbcName name) {
        this.name_.remove(name);
    }

    /**
     * Clear the XbcName property <b>name</b>.
     *
     */
    public final void clearName() {
        this.name_.clear();
    }

    /**
     * Gets the String property <b>ellipsis</b>.
     *
     * @return String
     */
    public final String getEllipsis() {
        return (ellipsis_);
    }

    /**
     * Sets the String property <b>ellipsis</b>.
     *
     * @param ellipsis
     */
    public final void setEllipsis(String ellipsis) {
        this.ellipsis_ = ellipsis;
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
        buffer.append("<params");
        buffer.append(">");
        size = this.name_.size();
        for (int i = 0;i < size;i++) {
            XbcName value = (XbcName)this.name_.get(i);
            value.makeTextElement(buffer);
        }
        if (ellipsis_ != null) {
            buffer.append("<ellipsis>");
            buffer.append(URelaxer.escapeCharData(URelaxer.getString(getEllipsis())));
            buffer.append("</ellipsis>");
        }
        buffer.append("</params>");
    }

    /**
     * Makes an XML text representation.
     *
     * @param buffer
     * @exception IOException
     */
    public void makeTextElement(Writer buffer) throws IOException {
        int size;
        buffer.write("<params");
        buffer.write(">");
        size = this.name_.size();
        for (int i = 0;i < size;i++) {
            XbcName value = (XbcName)this.name_.get(i);
            value.makeTextElement(buffer);
        }
        if (ellipsis_ != null) {
            buffer.write("<ellipsis>");
            buffer.write(URelaxer.escapeCharData(URelaxer.getString(getEllipsis())));
            buffer.write("</ellipsis>");
        }
        buffer.write("</params>");
    }

    /**
     * Makes an XML text representation.
     *
     * @param buffer
     */
    public void makeTextElement(PrintWriter buffer) {
        int size;
        buffer.print("<params");
        buffer.print(">");
        size = this.name_.size();
        for (int i = 0;i < size;i++) {
            XbcName value = (XbcName)this.name_.get(i);
            value.makeTextElement(buffer);
        }
        if (ellipsis_ != null) {
            buffer.print("<ellipsis>");
            buffer.print(URelaxer.escapeCharData(URelaxer.getString(getEllipsis())));
            buffer.print("</ellipsis>");
        }
        buffer.print("</params>");
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
    public String getEllipsisAsString() {
        return (URelaxer.getString(getEllipsis()));
    }

    /**
     * Sets the property value by String.
     *
     * @param string
     */
    public void setEllipsisByString(String string) {
        setEllipsis(string);
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
        classNodes.addAll(name_);
        IRNode[] nodes = new IRNode[classNodes.size()];
        return ((IRNode[])classNodes.toArray(nodes));
    }

    /**
     * Tests if a Element <code>element</code> is valid
     * for the <code>XbcParams</code>.
     *
     * @param element
     * @return boolean
     */
    public static boolean isMatch(Element element) {
        if (!URelaxer.isTargetElement(element, "params")) {
            return (false);
        }
        RStack target = new RStack(element);
        boolean $match$ = false;
        Element child;
        while (true) {
            if (!XbcName.isMatchHungry(target)) {
                break;
            }
            $match$ = true;
        }
        child = target.peekElement();
        if (child != null) {
            if (URelaxer.isTargetElement(child, "ellipsis")) {
                target.popElement();
            }
        }
        $match$ = true;
        if (!target.isEmptyElement()) {
            return (false);
        }
        return (true);
    }

    /**
     * Tests if elements contained in a Stack <code>stack</code>
     * is valid for the <code>XbcParams</code>.
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
     * is valid for the <code>XbcParams</code>.
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
