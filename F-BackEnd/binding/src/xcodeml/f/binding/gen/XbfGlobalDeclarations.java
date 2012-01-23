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
 * <b>XbfGlobalDeclarations</b> is generated from XcodeML_F.rng by Relaxer.
 * This class is derived from:
 * 
 * <!-- for programmer
 * <element java:extends="xcodeml.f.XmfObj" name="globalDeclarations">
 *   <zeroOrMore>
 *     <choice>
 *       <ref name="FfunctionDefinition"/>
 *       <ref name="FmoduleDefinition"/>
 *       <ref name="FblockDataDefinition"/>
 *       <ref name="text"/>
 *     </choice>
 *   </zeroOrMore>
 * </element>
 * -->
 * <!-- for javadoc -->
 * <pre> &lt;element java:extends="xcodeml.f.XmfObj" name="globalDeclarations"&gt;
 *   &lt;zeroOrMore&gt;
 *     &lt;choice&gt;
 *       &lt;ref name="FfunctionDefinition"/&gt;
 *       &lt;ref name="FmoduleDefinition"/&gt;
 *       &lt;ref name="FblockDataDefinition"/&gt;
 *       &lt;ref name="text"/&gt;
 *     &lt;/choice&gt;
 *   &lt;/zeroOrMore&gt;
 * &lt;/element&gt;
 * </pre>
 *
 * @version XcodeML_F.rng (Mon Jan 23 20:53:32 JST 2012)
 * @author  Relaxer 1.0 (http://www.relaxer.org)
 */
public class XbfGlobalDeclarations extends xcodeml.f.XmfObj implements java.io.Serializable, Cloneable, IRVisitable, IRNode {
    // List<IXbfGlobalDeclarationsChoice>
    private java.util.List content_ = new java.util.ArrayList();
    private IRNode parentRNode_;

    /**
     * Creates a <code>XbfGlobalDeclarations</code>.
     *
     */
    public XbfGlobalDeclarations() {
    }

    /**
     * Creates a <code>XbfGlobalDeclarations</code>.
     *
     * @param source
     */
    public XbfGlobalDeclarations(XbfGlobalDeclarations source) {
        setup(source);
    }

    /**
     * Creates a <code>XbfGlobalDeclarations</code> by the Stack <code>stack</code>
     * that contains Elements.
     * This constructor is supposed to be used internally
     * by the Relaxer system.
     *
     * @param stack
     */
    public XbfGlobalDeclarations(RStack stack) {
        setup(stack);
    }

    /**
     * Creates a <code>XbfGlobalDeclarations</code> by the Document <code>doc</code>.
     *
     * @param doc
     */
    public XbfGlobalDeclarations(Document doc) {
        setup(doc.getDocumentElement());
    }

    /**
     * Creates a <code>XbfGlobalDeclarations</code> by the Element <code>element</code>.
     *
     * @param element
     */
    public XbfGlobalDeclarations(Element element) {
        setup(element);
    }

    /**
     * Creates a <code>XbfGlobalDeclarations</code> by the File <code>file</code>.
     *
     * @param file
     * @exception IOException
     * @exception SAXException
     * @exception ParserConfigurationException
     */
    public XbfGlobalDeclarations(File file) throws IOException, SAXException, ParserConfigurationException {
        setup(file);
    }

    /**
     * Creates a <code>XbfGlobalDeclarations</code>
     * by the String representation of URI <code>uri</code>.
     *
     * @param uri
     * @exception IOException
     * @exception SAXException
     * @exception ParserConfigurationException
     */
    public XbfGlobalDeclarations(String uri) throws IOException, SAXException, ParserConfigurationException {
        setup(uri);
    }

    /**
     * Creates a <code>XbfGlobalDeclarations</code> by the URL <code>url</code>.
     *
     * @param url
     * @exception IOException
     * @exception SAXException
     * @exception ParserConfigurationException
     */
    public XbfGlobalDeclarations(URL url) throws IOException, SAXException, ParserConfigurationException {
        setup(url);
    }

    /**
     * Creates a <code>XbfGlobalDeclarations</code> by the InputStream <code>in</code>.
     *
     * @param in
     * @exception IOException
     * @exception SAXException
     * @exception ParserConfigurationException
     */
    public XbfGlobalDeclarations(InputStream in) throws IOException, SAXException, ParserConfigurationException {
        setup(in);
    }

    /**
     * Creates a <code>XbfGlobalDeclarations</code> by the InputSource <code>is</code>.
     *
     * @param is
     * @exception IOException
     * @exception SAXException
     * @exception ParserConfigurationException
     */
    public XbfGlobalDeclarations(InputSource is) throws IOException, SAXException, ParserConfigurationException {
        setup(is);
    }

    /**
     * Creates a <code>XbfGlobalDeclarations</code> by the Reader <code>reader</code>.
     *
     * @param reader
     * @exception IOException
     * @exception SAXException
     * @exception ParserConfigurationException
     */
    public XbfGlobalDeclarations(Reader reader) throws IOException, SAXException, ParserConfigurationException {
        setup(reader);
    }

    /**
     * Initializes the <code>XbfGlobalDeclarations</code> by the XbfGlobalDeclarations <code>source</code>.
     *
     * @param source
     */
    public void setup(XbfGlobalDeclarations source) {
        int size;
        this.content_.clear();
        size = source.content_.size();
        for (int i = 0;i < size;i++) {
            addContent((IXbfGlobalDeclarationsChoice)source.getContent(i).clone());
        }
    }

    /**
     * Initializes the <code>XbfGlobalDeclarations</code> by the Document <code>doc</code>.
     *
     * @param doc
     */
    public void setup(Document doc) {
        setup(doc.getDocumentElement());
    }

    /**
     * Initializes the <code>XbfGlobalDeclarations</code> by the Element <code>element</code>.
     *
     * @param element
     */
    public void setup(Element element) {
        init(element);
    }

    /**
     * Initializes the <code>XbfGlobalDeclarations</code> by the Stack <code>stack</code>
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
            if (XbfFfunctionDefinition.isMatch(stack)) {
                addContent(factory.createXbfFfunctionDefinition(stack));
            } else if (XbfFmoduleDefinition.isMatch(stack)) {
                addContent(factory.createXbfFmoduleDefinition(stack));
            } else if (XbfFblockDataDefinition.isMatch(stack)) {
                addContent(factory.createXbfFblockDataDefinition(stack));
            } else if (XbfText.isMatch(stack)) {
                addContent(factory.createXbfText(stack));
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
        return (factory.createXbfGlobalDeclarations(this));
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
            IXbfGlobalDeclarationsChoice value = (IXbfGlobalDeclarationsChoice)this.content_.get(i);
            value.makeElement(element);
        }
        parent.appendChild(element);
    }

    /**
     * Initializes the <code>XbfGlobalDeclarations</code> by the File <code>file</code>.
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
     * Initializes the <code>XbfGlobalDeclarations</code>
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
     * Initializes the <code>XbfGlobalDeclarations</code> by the URL <code>url</code>.
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
     * Initializes the <code>XbfGlobalDeclarations</code> by the InputStream <code>in</code>.
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
     * Initializes the <code>XbfGlobalDeclarations</code> by the InputSource <code>is</code>.
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
     * Initializes the <code>XbfGlobalDeclarations</code> by the Reader <code>reader</code>.
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
     * Gets the IXbfGlobalDeclarationsChoice property <b>content</b>.
     *
     * @return IXbfGlobalDeclarationsChoice[]
     */
    public final IXbfGlobalDeclarationsChoice[] getContent() {
        IXbfGlobalDeclarationsChoice[] array = new IXbfGlobalDeclarationsChoice[content_.size()];
        return ((IXbfGlobalDeclarationsChoice[])content_.toArray(array));
    }

    /**
     * Sets the IXbfGlobalDeclarationsChoice property <b>content</b>.
     *
     * @param content
     */
    public final void setContent(IXbfGlobalDeclarationsChoice[] content) {
        this.content_.clear();
        for (int i = 0;i < content.length;i++) {
            addContent(content[i]);
        }
        for (int i = 0;i < content.length;i++) {
            content[i].rSetParentRNode(this);
        }
    }

    /**
     * Sets the IXbfGlobalDeclarationsChoice property <b>content</b>.
     *
     * @param content
     */
    public final void setContent(IXbfGlobalDeclarationsChoice content) {
        this.content_.clear();
        addContent(content);
        if (content != null) {
            content.rSetParentRNode(this);
        }
    }

    /**
     * Adds the IXbfGlobalDeclarationsChoice property <b>content</b>.
     *
     * @param content
     */
    public final void addContent(IXbfGlobalDeclarationsChoice content) {
        this.content_.add(content);
        if (content != null) {
            content.rSetParentRNode(this);
        }
    }

    /**
     * Adds the IXbfGlobalDeclarationsChoice property <b>content</b>.
     *
     * @param content
     */
    public final void addContent(IXbfGlobalDeclarationsChoice[] content) {
        for (int i = 0;i < content.length;i++) {
            addContent(content[i]);
        }
        for (int i = 0;i < content.length;i++) {
            content[i].rSetParentRNode(this);
        }
    }

    /**
     * Gets number of the IXbfGlobalDeclarationsChoice property <b>content</b>.
     *
     * @return int
     */
    public final int sizeContent() {
        return (content_.size());
    }

    /**
     * Gets the IXbfGlobalDeclarationsChoice property <b>content</b> by index.
     *
     * @param index
     * @return IXbfGlobalDeclarationsChoice
     */
    public final IXbfGlobalDeclarationsChoice getContent(int index) {
        return ((IXbfGlobalDeclarationsChoice)content_.get(index));
    }

    /**
     * Sets the IXbfGlobalDeclarationsChoice property <b>content</b> by index.
     *
     * @param index
     * @param content
     */
    public final void setContent(int index, IXbfGlobalDeclarationsChoice content) {
        this.content_.set(index, content);
        if (content != null) {
            content.rSetParentRNode(this);
        }
    }

    /**
     * Adds the IXbfGlobalDeclarationsChoice property <b>content</b> by index.
     *
     * @param index
     * @param content
     */
    public final void addContent(int index, IXbfGlobalDeclarationsChoice content) {
        this.content_.add(index, content);
        if (content != null) {
            content.rSetParentRNode(this);
        }
    }

    /**
     * Remove the IXbfGlobalDeclarationsChoice property <b>content</b> by index.
     *
     * @param index
     */
    public final void removeContent(int index) {
        this.content_.remove(index);
    }

    /**
     * Remove the IXbfGlobalDeclarationsChoice property <b>content</b> by object.
     *
     * @param content
     */
    public final void removeContent(IXbfGlobalDeclarationsChoice content) {
        this.content_.remove(content);
    }

    /**
     * Clear the IXbfGlobalDeclarationsChoice property <b>content</b>.
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
            IXbfGlobalDeclarationsChoice value = (IXbfGlobalDeclarationsChoice)this.content_.get(i);
            value.makeTextAttribute(buffer);
        }
        buffer.append(">");
        size = this.content_.size();
        for (int i = 0;i < size;i++) {
            IXbfGlobalDeclarationsChoice value = (IXbfGlobalDeclarationsChoice)this.content_.get(i);
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
            IXbfGlobalDeclarationsChoice value = (IXbfGlobalDeclarationsChoice)this.content_.get(i);
            value.makeTextAttribute(buffer);
        }
        buffer.write(">");
        size = this.content_.size();
        for (int i = 0;i < size;i++) {
            IXbfGlobalDeclarationsChoice value = (IXbfGlobalDeclarationsChoice)this.content_.get(i);
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
            IXbfGlobalDeclarationsChoice value = (IXbfGlobalDeclarationsChoice)this.content_.get(i);
            value.makeTextAttribute(buffer);
        }
        buffer.print(">");
        size = this.content_.size();
        for (int i = 0;i < size;i++) {
            IXbfGlobalDeclarationsChoice value = (IXbfGlobalDeclarationsChoice)this.content_.get(i);
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
     * for the <code>XbfGlobalDeclarations</code>.
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
            if (XbfFfunctionDefinition.isMatchHungry(target)) {
                $match$ = true;
            } else if (XbfFmoduleDefinition.isMatchHungry(target)) {
                $match$ = true;
            } else if (XbfFblockDataDefinition.isMatchHungry(target)) {
                $match$ = true;
            } else if (XbfText.isMatchHungry(target)) {
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
     * is valid for the <code>XbfGlobalDeclarations</code>.
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
     * is valid for the <code>XbfGlobalDeclarations</code>.
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
