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
 * <b>XbcGlobalDeclarations</b> is generated from XcodeML_C.rng by Relaxer.
 * This class is derived from:
 * 
 * <!-- for programmer
 * <element java:extends="xcodeml.c.obj.XmcObj" name="globalDeclarations">
 *   <zeroOrMore>
 *     <choice>
 *       <ref name="functionDefinition"/>
 *       <ref name="varDecl"/>
 *       <ref name="functionDecl"/>
 *       <ref name="gccAsmDefinition"/>
 *       <ref name="pragma"/>
 *       <ref name="text"/>
 *     </choice>
 *   </zeroOrMore>
 * </element>
 * -->
 * <!-- for javadoc -->
 * <pre> &lt;element java:extends="xcodeml.c.obj.XmcObj" name="globalDeclarations"&gt;
 *   &lt;zeroOrMore&gt;
 *     &lt;choice&gt;
 *       &lt;ref name="functionDefinition"/&gt;
 *       &lt;ref name="varDecl"/&gt;
 *       &lt;ref name="functionDecl"/&gt;
 *       &lt;ref name="gccAsmDefinition"/&gt;
 *       &lt;ref name="pragma"/&gt;
 *       &lt;ref name="text"/&gt;
 *     &lt;/choice&gt;
 *   &lt;/zeroOrMore&gt;
 * &lt;/element&gt;
 * </pre>
 *
 * @version XcodeML_C.rng (Fri Oct 07 17:51:13 JST 2011)
 * @author  Relaxer 1.0 (http://www.relaxer.org)
 */
public class XbcGlobalDeclarations extends xcodeml.c.obj.XmcObj implements java.io.Serializable, Cloneable, IRVisitable, IRNode {
    // List<IXbcGlobalDeclarationsChoice>
    private java.util.List content_ = new java.util.ArrayList();
    private IRNode parentRNode_;

    /**
     * Creates a <code>XbcGlobalDeclarations</code>.
     *
     */
    public XbcGlobalDeclarations() {
    }

    /**
     * Creates a <code>XbcGlobalDeclarations</code>.
     *
     * @param source
     */
    public XbcGlobalDeclarations(XbcGlobalDeclarations source) {
        setup(source);
    }

    /**
     * Creates a <code>XbcGlobalDeclarations</code> by the Stack <code>stack</code>
     * that contains Elements.
     * This constructor is supposed to be used internally
     * by the Relaxer system.
     *
     * @param stack
     */
    public XbcGlobalDeclarations(RStack stack) {
        setup(stack);
    }

    /**
     * Creates a <code>XbcGlobalDeclarations</code> by the Document <code>doc</code>.
     *
     * @param doc
     */
    public XbcGlobalDeclarations(Document doc) {
        setup(doc.getDocumentElement());
    }

    /**
     * Creates a <code>XbcGlobalDeclarations</code> by the Element <code>element</code>.
     *
     * @param element
     */
    public XbcGlobalDeclarations(Element element) {
        setup(element);
    }

    /**
     * Creates a <code>XbcGlobalDeclarations</code> by the File <code>file</code>.
     *
     * @param file
     * @exception IOException
     * @exception SAXException
     * @exception ParserConfigurationException
     */
    public XbcGlobalDeclarations(File file) throws IOException, SAXException, ParserConfigurationException {
        setup(file);
    }

    /**
     * Creates a <code>XbcGlobalDeclarations</code>
     * by the String representation of URI <code>uri</code>.
     *
     * @param uri
     * @exception IOException
     * @exception SAXException
     * @exception ParserConfigurationException
     */
    public XbcGlobalDeclarations(String uri) throws IOException, SAXException, ParserConfigurationException {
        setup(uri);
    }

    /**
     * Creates a <code>XbcGlobalDeclarations</code> by the URL <code>url</code>.
     *
     * @param url
     * @exception IOException
     * @exception SAXException
     * @exception ParserConfigurationException
     */
    public XbcGlobalDeclarations(URL url) throws IOException, SAXException, ParserConfigurationException {
        setup(url);
    }

    /**
     * Creates a <code>XbcGlobalDeclarations</code> by the InputStream <code>in</code>.
     *
     * @param in
     * @exception IOException
     * @exception SAXException
     * @exception ParserConfigurationException
     */
    public XbcGlobalDeclarations(InputStream in) throws IOException, SAXException, ParserConfigurationException {
        setup(in);
    }

    /**
     * Creates a <code>XbcGlobalDeclarations</code> by the InputSource <code>is</code>.
     *
     * @param is
     * @exception IOException
     * @exception SAXException
     * @exception ParserConfigurationException
     */
    public XbcGlobalDeclarations(InputSource is) throws IOException, SAXException, ParserConfigurationException {
        setup(is);
    }

    /**
     * Creates a <code>XbcGlobalDeclarations</code> by the Reader <code>reader</code>.
     *
     * @param reader
     * @exception IOException
     * @exception SAXException
     * @exception ParserConfigurationException
     */
    public XbcGlobalDeclarations(Reader reader) throws IOException, SAXException, ParserConfigurationException {
        setup(reader);
    }

    /**
     * Initializes the <code>XbcGlobalDeclarations</code> by the XbcGlobalDeclarations <code>source</code>.
     *
     * @param source
     */
    public void setup(XbcGlobalDeclarations source) {
        int size;
        this.content_.clear();
        size = source.content_.size();
        for (int i = 0;i < size;i++) {
            addContent((IXbcGlobalDeclarationsChoice)source.getContent(i).clone());
        }
    }

    /**
     * Initializes the <code>XbcGlobalDeclarations</code> by the Document <code>doc</code>.
     *
     * @param doc
     */
    public void setup(Document doc) {
        setup(doc.getDocumentElement());
    }

    /**
     * Initializes the <code>XbcGlobalDeclarations</code> by the Element <code>element</code>.
     *
     * @param element
     */
    public void setup(Element element) {
        init(element);
    }

    /**
     * Initializes the <code>XbcGlobalDeclarations</code> by the Stack <code>stack</code>
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
        content_.clear();
        while (true) {
            if (XbcFunctionDefinition.isMatch(stack)) {
                addContent(factory.createXbcFunctionDefinition(stack));
            } else if (XbcVarDecl.isMatch(stack)) {
                addContent(factory.createXbcVarDecl(stack));
            } else if (XbcFunctionDecl.isMatch(stack)) {
                addContent(factory.createXbcFunctionDecl(stack));
            } else if (XbcGccAsmDefinition.isMatch(stack)) {
                addContent(factory.createXbcGccAsmDefinition(stack));
            } else if (XbcPragma.isMatch(stack)) {
                addContent(factory.createXbcPragma(stack));
            } else if (XbcText.isMatch(stack)) {
                addContent(factory.createXbcText(stack));
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
        return (factory.createXbcGlobalDeclarations(this));
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
        Element element = doc.createElement("globalDeclarations");
        int size;
        size = this.content_.size();
        for (int i = 0;i < size;i++) {
            IXbcGlobalDeclarationsChoice value = (IXbcGlobalDeclarationsChoice)this.content_.get(i);
            value.makeElement(element);
        }
        parent.appendChild(element);
    }

    /**
     * Initializes the <code>XbcGlobalDeclarations</code> by the File <code>file</code>.
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
     * Initializes the <code>XbcGlobalDeclarations</code>
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
     * Initializes the <code>XbcGlobalDeclarations</code> by the URL <code>url</code>.
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
     * Initializes the <code>XbcGlobalDeclarations</code> by the InputStream <code>in</code>.
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
     * Initializes the <code>XbcGlobalDeclarations</code> by the InputSource <code>is</code>.
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
     * Initializes the <code>XbcGlobalDeclarations</code> by the Reader <code>reader</code>.
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
     * Gets the IXbcGlobalDeclarationsChoice property <b>content</b>.
     *
     * @return IXbcGlobalDeclarationsChoice[]
     */
    public final IXbcGlobalDeclarationsChoice[] getContent() {
        IXbcGlobalDeclarationsChoice[] array = new IXbcGlobalDeclarationsChoice[content_.size()];
        return ((IXbcGlobalDeclarationsChoice[])content_.toArray(array));
    }

    /**
     * Sets the IXbcGlobalDeclarationsChoice property <b>content</b>.
     *
     * @param content
     */
    public final void setContent(IXbcGlobalDeclarationsChoice[] content) {
        this.content_.clear();
        for (int i = 0;i < content.length;i++) {
            addContent(content[i]);
        }
        for (int i = 0;i < content.length;i++) {
            content[i].rSetParentRNode(this);
        }
    }

    /**
     * Sets the IXbcGlobalDeclarationsChoice property <b>content</b>.
     *
     * @param content
     */
    public final void setContent(IXbcGlobalDeclarationsChoice content) {
        this.content_.clear();
        addContent(content);
        if (content != null) {
            content.rSetParentRNode(this);
        }
    }

    /**
     * Adds the IXbcGlobalDeclarationsChoice property <b>content</b>.
     *
     * @param content
     */
    public final void addContent(IXbcGlobalDeclarationsChoice content) {
        this.content_.add(content);
        if (content != null) {
            content.rSetParentRNode(this);
        }
    }

    /**
     * Adds the IXbcGlobalDeclarationsChoice property <b>content</b>.
     *
     * @param content
     */
    public final void addContent(IXbcGlobalDeclarationsChoice[] content) {
        for (int i = 0;i < content.length;i++) {
            addContent(content[i]);
        }
        for (int i = 0;i < content.length;i++) {
            content[i].rSetParentRNode(this);
        }
    }

    /**
     * Gets number of the IXbcGlobalDeclarationsChoice property <b>content</b>.
     *
     * @return int
     */
    public final int sizeContent() {
        return (content_.size());
    }

    /**
     * Gets the IXbcGlobalDeclarationsChoice property <b>content</b> by index.
     *
     * @param index
     * @return IXbcGlobalDeclarationsChoice
     */
    public final IXbcGlobalDeclarationsChoice getContent(int index) {
        return ((IXbcGlobalDeclarationsChoice)content_.get(index));
    }

    /**
     * Sets the IXbcGlobalDeclarationsChoice property <b>content</b> by index.
     *
     * @param index
     * @param content
     */
    public final void setContent(int index, IXbcGlobalDeclarationsChoice content) {
        this.content_.set(index, content);
        if (content != null) {
            content.rSetParentRNode(this);
        }
    }

    /**
     * Adds the IXbcGlobalDeclarationsChoice property <b>content</b> by index.
     *
     * @param index
     * @param content
     */
    public final void addContent(int index, IXbcGlobalDeclarationsChoice content) {
        this.content_.add(index, content);
        if (content != null) {
            content.rSetParentRNode(this);
        }
    }

    /**
     * Remove the IXbcGlobalDeclarationsChoice property <b>content</b> by index.
     *
     * @param index
     */
    public final void removeContent(int index) {
        this.content_.remove(index);
    }

    /**
     * Remove the IXbcGlobalDeclarationsChoice property <b>content</b> by object.
     *
     * @param content
     */
    public final void removeContent(IXbcGlobalDeclarationsChoice content) {
        this.content_.remove(content);
    }

    /**
     * Clear the IXbcGlobalDeclarationsChoice property <b>content</b>.
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
        buffer.append("<globalDeclarations");
        size = this.content_.size();
        for (int i = 0;i < size;i++) {
            IXbcGlobalDeclarationsChoice value = (IXbcGlobalDeclarationsChoice)this.content_.get(i);
            value.makeTextAttribute(buffer);
        }
        buffer.append(">");
        size = this.content_.size();
        for (int i = 0;i < size;i++) {
            IXbcGlobalDeclarationsChoice value = (IXbcGlobalDeclarationsChoice)this.content_.get(i);
            value.makeTextElement(buffer);
        }
        buffer.append("</globalDeclarations>");
    }

    /**
     * Makes an XML text representation.
     *
     * @param buffer
     * @exception IOException
     */
    public void makeTextElement(Writer buffer) throws IOException {
        int size;
        buffer.write("<globalDeclarations");
        size = this.content_.size();
        for (int i = 0;i < size;i++) {
            IXbcGlobalDeclarationsChoice value = (IXbcGlobalDeclarationsChoice)this.content_.get(i);
            value.makeTextAttribute(buffer);
        }
        buffer.write(">");
        size = this.content_.size();
        for (int i = 0;i < size;i++) {
            IXbcGlobalDeclarationsChoice value = (IXbcGlobalDeclarationsChoice)this.content_.get(i);
            value.makeTextElement(buffer);
        }
        buffer.write("</globalDeclarations>");
    }

    /**
     * Makes an XML text representation.
     *
     * @param buffer
     */
    public void makeTextElement(PrintWriter buffer) {
        int size;
        buffer.print("<globalDeclarations");
        size = this.content_.size();
        for (int i = 0;i < size;i++) {
            IXbcGlobalDeclarationsChoice value = (IXbcGlobalDeclarationsChoice)this.content_.get(i);
            value.makeTextAttribute(buffer);
        }
        buffer.print(">");
        size = this.content_.size();
        for (int i = 0;i < size;i++) {
            IXbcGlobalDeclarationsChoice value = (IXbcGlobalDeclarationsChoice)this.content_.get(i);
            value.makeTextElement(buffer);
        }
        buffer.print("</globalDeclarations>");
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
     * for the <code>XbcGlobalDeclarations</code>.
     *
     * @param element
     * @return boolean
     */
    public static boolean isMatch(Element element) {
        if (!URelaxer.isTargetElement(element, "globalDeclarations")) {
            return (false);
        }
        RStack target = new RStack(element);
        boolean $match$ = false;
        Element child;
        while (true) {
            if (XbcFunctionDefinition.isMatchHungry(target)) {
                $match$ = true;
            } else if (XbcVarDecl.isMatchHungry(target)) {
                $match$ = true;
            } else if (XbcFunctionDecl.isMatchHungry(target)) {
                $match$ = true;
            } else if (XbcGccAsmDefinition.isMatchHungry(target)) {
                $match$ = true;
            } else if (XbcPragma.isMatchHungry(target)) {
                $match$ = true;
            } else if (XbcText.isMatchHungry(target)) {
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
     * is valid for the <code>XbcGlobalDeclarations</code>.
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
     * is valid for the <code>XbcGlobalDeclarations</code>.
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
