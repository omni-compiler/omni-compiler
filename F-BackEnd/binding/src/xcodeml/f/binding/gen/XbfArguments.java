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
 * <b>XbfArguments</b> is generated from XcodeML_F.rng by Relaxer.
 * This class is derived from:
 * 
 * <!-- for programmer
 * <element java:extends="xcodeml.f.XmfObj" name="arguments">
 *   <zeroOrMore>
 *     <choice>
 *       <ref name="defModelExpr"/>
 *       <ref name="Ffunction"/>
 *       <ref name="namedValue"/>
 *     </choice>
 *   </zeroOrMore>
 * </element>
 * -->
 * <!-- for javadoc -->
 * <pre> &lt;element java:extends="xcodeml.f.XmfObj" name="arguments"&gt;
 *   &lt;zeroOrMore&gt;
 *     &lt;choice&gt;
 *       &lt;ref name="defModelExpr"/&gt;
 *       &lt;ref name="Ffunction"/&gt;
 *       &lt;ref name="namedValue"/&gt;
 *     &lt;/choice&gt;
 *   &lt;/zeroOrMore&gt;
 * &lt;/element&gt;
 * </pre>
 *
 * @version XcodeML_F.rng (Mon Jan 23 20:53:32 JST 2012)
 * @author  Relaxer 1.0 (http://www.relaxer.org)
 */
public class XbfArguments extends xcodeml.f.XmfObj implements java.io.Serializable, Cloneable, IRVisitable, IRNode {
    // List<IXbfArgumentsChoice>
    private java.util.List content_ = new java.util.ArrayList();
    private IRNode parentRNode_;

    /**
     * Creates a <code>XbfArguments</code>.
     *
     */
    public XbfArguments() {
    }

    /**
     * Creates a <code>XbfArguments</code>.
     *
     * @param source
     */
    public XbfArguments(XbfArguments source) {
        setup(source);
    }

    /**
     * Creates a <code>XbfArguments</code> by the Stack <code>stack</code>
     * that contains Elements.
     * This constructor is supposed to be used internally
     * by the Relaxer system.
     *
     * @param stack
     */
    public XbfArguments(RStack stack) {
        setup(stack);
    }

    /**
     * Creates a <code>XbfArguments</code> by the Document <code>doc</code>.
     *
     * @param doc
     */
    public XbfArguments(Document doc) {
        setup(doc.getDocumentElement());
    }

    /**
     * Creates a <code>XbfArguments</code> by the Element <code>element</code>.
     *
     * @param element
     */
    public XbfArguments(Element element) {
        setup(element);
    }

    /**
     * Creates a <code>XbfArguments</code> by the File <code>file</code>.
     *
     * @param file
     * @exception IOException
     * @exception SAXException
     * @exception ParserConfigurationException
     */
    public XbfArguments(File file) throws IOException, SAXException, ParserConfigurationException {
        setup(file);
    }

    /**
     * Creates a <code>XbfArguments</code>
     * by the String representation of URI <code>uri</code>.
     *
     * @param uri
     * @exception IOException
     * @exception SAXException
     * @exception ParserConfigurationException
     */
    public XbfArguments(String uri) throws IOException, SAXException, ParserConfigurationException {
        setup(uri);
    }

    /**
     * Creates a <code>XbfArguments</code> by the URL <code>url</code>.
     *
     * @param url
     * @exception IOException
     * @exception SAXException
     * @exception ParserConfigurationException
     */
    public XbfArguments(URL url) throws IOException, SAXException, ParserConfigurationException {
        setup(url);
    }

    /**
     * Creates a <code>XbfArguments</code> by the InputStream <code>in</code>.
     *
     * @param in
     * @exception IOException
     * @exception SAXException
     * @exception ParserConfigurationException
     */
    public XbfArguments(InputStream in) throws IOException, SAXException, ParserConfigurationException {
        setup(in);
    }

    /**
     * Creates a <code>XbfArguments</code> by the InputSource <code>is</code>.
     *
     * @param is
     * @exception IOException
     * @exception SAXException
     * @exception ParserConfigurationException
     */
    public XbfArguments(InputSource is) throws IOException, SAXException, ParserConfigurationException {
        setup(is);
    }

    /**
     * Creates a <code>XbfArguments</code> by the Reader <code>reader</code>.
     *
     * @param reader
     * @exception IOException
     * @exception SAXException
     * @exception ParserConfigurationException
     */
    public XbfArguments(Reader reader) throws IOException, SAXException, ParserConfigurationException {
        setup(reader);
    }

    /**
     * Initializes the <code>XbfArguments</code> by the XbfArguments <code>source</code>.
     *
     * @param source
     */
    public void setup(XbfArguments source) {
        int size;
        this.content_.clear();
        size = source.content_.size();
        for (int i = 0;i < size;i++) {
            addContent((IXbfArgumentsChoice)source.getContent(i).clone());
        }
    }

    /**
     * Initializes the <code>XbfArguments</code> by the Document <code>doc</code>.
     *
     * @param doc
     */
    public void setup(Document doc) {
        setup(doc.getDocumentElement());
    }

    /**
     * Initializes the <code>XbfArguments</code> by the Element <code>element</code>.
     *
     * @param element
     */
    public void setup(Element element) {
        init(element);
    }

    /**
     * Initializes the <code>XbfArguments</code> by the Stack <code>stack</code>
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
            if (XbfFarrayRef.isMatch(stack)) {
                addContent(factory.createXbfFarrayRef(stack));
            } else if (XbfFcharacterRef.isMatch(stack)) {
                addContent(factory.createXbfFcharacterRef(stack));
            } else if (XbfFmemberRef.isMatch(stack)) {
                addContent(factory.createXbfFmemberRef(stack));
            } else if (XbfFcoArrayRef.isMatch(stack)) {
                addContent(factory.createXbfFcoArrayRef(stack));
            } else if (XbfFdoLoop.isMatch(stack)) {
                addContent(factory.createXbfFdoLoop(stack));
            } else if (XbfNamedValue.isMatch(stack)) {
                addContent(factory.createXbfNamedValue(stack));
            } else if (XbfFintConstant.isMatch(stack)) {
                addContent(factory.createXbfFintConstant(stack));
            } else if (XbfFrealConstant.isMatch(stack)) {
                addContent(factory.createXbfFrealConstant(stack));
            } else if (XbfFcharacterConstant.isMatch(stack)) {
                addContent(factory.createXbfFcharacterConstant(stack));
            } else if (XbfFlogicalConstant.isMatch(stack)) {
                addContent(factory.createXbfFlogicalConstant(stack));
            } else if (XbfFcomplexConstant.isMatch(stack)) {
                addContent(factory.createXbfFcomplexConstant(stack));
            } else if (XbfVar.isMatch(stack)) {
                addContent(factory.createXbfVar(stack));
            } else if (XbfVarRef.isMatch(stack)) {
                addContent(factory.createXbfVarRef(stack));
            } else if (XbfUserUnaryExpr.isMatch(stack)) {
                addContent(factory.createXbfUserUnaryExpr(stack));
            } else if (XbfFarrayConstructor.isMatch(stack)) {
                addContent(factory.createXbfFarrayConstructor(stack));
            } else if (XbfFstructConstructor.isMatch(stack)) {
                addContent(factory.createXbfFstructConstructor(stack));
            } else if (XbfFunctionCall.isMatch(stack)) {
                addContent(factory.createXbfFunctionCall(stack));
            } else if (XbfUserBinaryExpr.isMatch(stack)) {
                addContent(factory.createXbfUserBinaryExpr(stack));
            } else if (XbfLogNotExpr.isMatch(stack)) {
                addContent(factory.createXbfLogNotExpr(stack));
            } else if (XbfFfunction.isMatch(stack)) {
                addContent(factory.createXbfFfunction(stack));
            } else if (XbfLogNEQExpr.isMatch(stack)) {
                addContent(factory.createXbfLogNEQExpr(stack));
            } else if (XbfLogGTExpr.isMatch(stack)) {
                addContent(factory.createXbfLogGTExpr(stack));
            } else if (XbfLogNEQVExpr.isMatch(stack)) {
                addContent(factory.createXbfLogNEQVExpr(stack));
            } else if (XbfPlusExpr.isMatch(stack)) {
                addContent(factory.createXbfPlusExpr(stack));
            } else if (XbfMinusExpr.isMatch(stack)) {
                addContent(factory.createXbfMinusExpr(stack));
            } else if (XbfMulExpr.isMatch(stack)) {
                addContent(factory.createXbfMulExpr(stack));
            } else if (XbfDivExpr.isMatch(stack)) {
                addContent(factory.createXbfDivExpr(stack));
            } else if (XbfFpowerExpr.isMatch(stack)) {
                addContent(factory.createXbfFpowerExpr(stack));
            } else if (XbfFconcatExpr.isMatch(stack)) {
                addContent(factory.createXbfFconcatExpr(stack));
            } else if (XbfLogEQExpr.isMatch(stack)) {
                addContent(factory.createXbfLogEQExpr(stack));
            } else if (XbfLogGEExpr.isMatch(stack)) {
                addContent(factory.createXbfLogGEExpr(stack));
            } else if (XbfLogLEExpr.isMatch(stack)) {
                addContent(factory.createXbfLogLEExpr(stack));
            } else if (XbfLogLTExpr.isMatch(stack)) {
                addContent(factory.createXbfLogLTExpr(stack));
            } else if (XbfLogAndExpr.isMatch(stack)) {
                addContent(factory.createXbfLogAndExpr(stack));
            } else if (XbfLogOrExpr.isMatch(stack)) {
                addContent(factory.createXbfLogOrExpr(stack));
            } else if (XbfLogEQVExpr.isMatch(stack)) {
                addContent(factory.createXbfLogEQVExpr(stack));
            } else if (XbfUnaryMinusExpr.isMatch(stack)) {
                addContent(factory.createXbfUnaryMinusExpr(stack));
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
        return (factory.createXbfArguments(this));
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
        Element element = doc.createElement("arguments");
        int size;
        size = this.content_.size();
        for (int i = 0;i < size;i++) {
            IXbfArgumentsChoice value = (IXbfArgumentsChoice)this.content_.get(i);
            value.makeElement(element);
        }
        parent.appendChild(element);
    }

    /**
     * Initializes the <code>XbfArguments</code> by the File <code>file</code>.
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
     * Initializes the <code>XbfArguments</code>
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
     * Initializes the <code>XbfArguments</code> by the URL <code>url</code>.
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
     * Initializes the <code>XbfArguments</code> by the InputStream <code>in</code>.
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
     * Initializes the <code>XbfArguments</code> by the InputSource <code>is</code>.
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
     * Initializes the <code>XbfArguments</code> by the Reader <code>reader</code>.
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
     * Gets the IXbfArgumentsChoice property <b>content</b>.
     *
     * @return IXbfArgumentsChoice[]
     */
    public final IXbfArgumentsChoice[] getContent() {
        IXbfArgumentsChoice[] array = new IXbfArgumentsChoice[content_.size()];
        return ((IXbfArgumentsChoice[])content_.toArray(array));
    }

    /**
     * Sets the IXbfArgumentsChoice property <b>content</b>.
     *
     * @param content
     */
    public final void setContent(IXbfArgumentsChoice[] content) {
        this.content_.clear();
        for (int i = 0;i < content.length;i++) {
            addContent(content[i]);
        }
        for (int i = 0;i < content.length;i++) {
            content[i].rSetParentRNode(this);
        }
    }

    /**
     * Sets the IXbfArgumentsChoice property <b>content</b>.
     *
     * @param content
     */
    public final void setContent(IXbfArgumentsChoice content) {
        this.content_.clear();
        addContent(content);
        if (content != null) {
            content.rSetParentRNode(this);
        }
    }

    /**
     * Adds the IXbfArgumentsChoice property <b>content</b>.
     *
     * @param content
     */
    public final void addContent(IXbfArgumentsChoice content) {
        this.content_.add(content);
        if (content != null) {
            content.rSetParentRNode(this);
        }
    }

    /**
     * Adds the IXbfArgumentsChoice property <b>content</b>.
     *
     * @param content
     */
    public final void addContent(IXbfArgumentsChoice[] content) {
        for (int i = 0;i < content.length;i++) {
            addContent(content[i]);
        }
        for (int i = 0;i < content.length;i++) {
            content[i].rSetParentRNode(this);
        }
    }

    /**
     * Gets number of the IXbfArgumentsChoice property <b>content</b>.
     *
     * @return int
     */
    public final int sizeContent() {
        return (content_.size());
    }

    /**
     * Gets the IXbfArgumentsChoice property <b>content</b> by index.
     *
     * @param index
     * @return IXbfArgumentsChoice
     */
    public final IXbfArgumentsChoice getContent(int index) {
        return ((IXbfArgumentsChoice)content_.get(index));
    }

    /**
     * Sets the IXbfArgumentsChoice property <b>content</b> by index.
     *
     * @param index
     * @param content
     */
    public final void setContent(int index, IXbfArgumentsChoice content) {
        this.content_.set(index, content);
        if (content != null) {
            content.rSetParentRNode(this);
        }
    }

    /**
     * Adds the IXbfArgumentsChoice property <b>content</b> by index.
     *
     * @param index
     * @param content
     */
    public final void addContent(int index, IXbfArgumentsChoice content) {
        this.content_.add(index, content);
        if (content != null) {
            content.rSetParentRNode(this);
        }
    }

    /**
     * Remove the IXbfArgumentsChoice property <b>content</b> by index.
     *
     * @param index
     */
    public final void removeContent(int index) {
        this.content_.remove(index);
    }

    /**
     * Remove the IXbfArgumentsChoice property <b>content</b> by object.
     *
     * @param content
     */
    public final void removeContent(IXbfArgumentsChoice content) {
        this.content_.remove(content);
    }

    /**
     * Clear the IXbfArgumentsChoice property <b>content</b>.
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
        buffer.append("<arguments");
        size = this.content_.size();
        for (int i = 0;i < size;i++) {
            IXbfArgumentsChoice value = (IXbfArgumentsChoice)this.content_.get(i);
            value.makeTextAttribute(buffer);
        }
        buffer.append(">");
        size = this.content_.size();
        for (int i = 0;i < size;i++) {
            IXbfArgumentsChoice value = (IXbfArgumentsChoice)this.content_.get(i);
            value.makeTextElement(buffer);
        }
        buffer.append("</arguments>");
    }

    /**
     * Makes an XML text representation.
     *
     * @param buffer
     * @exception IOException
     */
    public void makeTextElement(Writer buffer) throws IOException {
        int size;
        buffer.write("<arguments");
        size = this.content_.size();
        for (int i = 0;i < size;i++) {
            IXbfArgumentsChoice value = (IXbfArgumentsChoice)this.content_.get(i);
            value.makeTextAttribute(buffer);
        }
        buffer.write(">");
        size = this.content_.size();
        for (int i = 0;i < size;i++) {
            IXbfArgumentsChoice value = (IXbfArgumentsChoice)this.content_.get(i);
            value.makeTextElement(buffer);
        }
        buffer.write("</arguments>");
    }

    /**
     * Makes an XML text representation.
     *
     * @param buffer
     */
    public void makeTextElement(PrintWriter buffer) {
        int size;
        buffer.print("<arguments");
        size = this.content_.size();
        for (int i = 0;i < size;i++) {
            IXbfArgumentsChoice value = (IXbfArgumentsChoice)this.content_.get(i);
            value.makeTextAttribute(buffer);
        }
        buffer.print(">");
        size = this.content_.size();
        for (int i = 0;i < size;i++) {
            IXbfArgumentsChoice value = (IXbfArgumentsChoice)this.content_.get(i);
            value.makeTextElement(buffer);
        }
        buffer.print("</arguments>");
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
     * for the <code>XbfArguments</code>.
     *
     * @param element
     * @return boolean
     */
    public static boolean isMatch(Element element) {
        if (!URelaxer.isTargetElement(element, "arguments")) {
            return (false);
        }
        RStack target = new RStack(element);
        boolean $match$ = false;
        Element child;
        while (true) {
            if (XbfFarrayRef.isMatchHungry(target)) {
                $match$ = true;
            } else if (XbfFcharacterRef.isMatchHungry(target)) {
                $match$ = true;
            } else if (XbfFmemberRef.isMatchHungry(target)) {
                $match$ = true;
            } else if (XbfFcoArrayRef.isMatchHungry(target)) {
                $match$ = true;
            } else if (XbfFdoLoop.isMatchHungry(target)) {
                $match$ = true;
            } else if (XbfNamedValue.isMatchHungry(target)) {
                $match$ = true;
            } else if (XbfFintConstant.isMatchHungry(target)) {
                $match$ = true;
            } else if (XbfFrealConstant.isMatchHungry(target)) {
                $match$ = true;
            } else if (XbfFcharacterConstant.isMatchHungry(target)) {
                $match$ = true;
            } else if (XbfFlogicalConstant.isMatchHungry(target)) {
                $match$ = true;
            } else if (XbfFcomplexConstant.isMatchHungry(target)) {
                $match$ = true;
            } else if (XbfVar.isMatchHungry(target)) {
                $match$ = true;
            } else if (XbfVarRef.isMatchHungry(target)) {
                $match$ = true;
            } else if (XbfUserUnaryExpr.isMatchHungry(target)) {
                $match$ = true;
            } else if (XbfFarrayConstructor.isMatchHungry(target)) {
                $match$ = true;
            } else if (XbfFstructConstructor.isMatchHungry(target)) {
                $match$ = true;
            } else if (XbfFunctionCall.isMatchHungry(target)) {
                $match$ = true;
            } else if (XbfUserBinaryExpr.isMatchHungry(target)) {
                $match$ = true;
            } else if (XbfLogNotExpr.isMatchHungry(target)) {
                $match$ = true;
            } else if (XbfFfunction.isMatchHungry(target)) {
                $match$ = true;
            } else if (XbfLogNEQExpr.isMatchHungry(target)) {
                $match$ = true;
            } else if (XbfLogGTExpr.isMatchHungry(target)) {
                $match$ = true;
            } else if (XbfLogNEQVExpr.isMatchHungry(target)) {
                $match$ = true;
            } else if (XbfPlusExpr.isMatchHungry(target)) {
                $match$ = true;
            } else if (XbfMinusExpr.isMatchHungry(target)) {
                $match$ = true;
            } else if (XbfMulExpr.isMatchHungry(target)) {
                $match$ = true;
            } else if (XbfDivExpr.isMatchHungry(target)) {
                $match$ = true;
            } else if (XbfFpowerExpr.isMatchHungry(target)) {
                $match$ = true;
            } else if (XbfFconcatExpr.isMatchHungry(target)) {
                $match$ = true;
            } else if (XbfLogEQExpr.isMatchHungry(target)) {
                $match$ = true;
            } else if (XbfLogGEExpr.isMatchHungry(target)) {
                $match$ = true;
            } else if (XbfLogLEExpr.isMatchHungry(target)) {
                $match$ = true;
            } else if (XbfLogLTExpr.isMatchHungry(target)) {
                $match$ = true;
            } else if (XbfLogAndExpr.isMatchHungry(target)) {
                $match$ = true;
            } else if (XbfLogOrExpr.isMatchHungry(target)) {
                $match$ = true;
            } else if (XbfLogEQVExpr.isMatchHungry(target)) {
                $match$ = true;
            } else if (XbfUnaryMinusExpr.isMatchHungry(target)) {
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
     * is valid for the <code>XbfArguments</code>.
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
     * is valid for the <code>XbfArguments</code>.
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
