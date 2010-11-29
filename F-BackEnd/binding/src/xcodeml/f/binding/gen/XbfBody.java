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
 * <b>XbfBody</b> is generated from XcodeML_F.rng by Relaxer.
 * This class is derived from:
 * 
 * <!-- for programmer
 * <element java:extends="xcodeml.f.XmfObj" name="body">
 *   <zeroOrMore>
 *     <ref name="defModelStatement"/>
 *   </zeroOrMore>
 * </element>
 * -->
 * <!-- for javadoc -->
 * <pre> &lt;element java:extends="xcodeml.f.XmfObj" name="body"&gt;
 *   &lt;zeroOrMore&gt;
 *     &lt;ref name="defModelStatement"/&gt;
 *   &lt;/zeroOrMore&gt;
 * &lt;/element&gt;
 * </pre>
 *
 * @version XcodeML_F.rng (Mon Nov 29 15:25:56 JST 2010)
 * @author  Relaxer 1.0 (http://www.relaxer.org)
 */
public class XbfBody extends xcodeml.f.XmfObj implements java.io.Serializable, Cloneable, IRVisitable, IRNode {
    // List<IXbfDefModelStatementChoice>
    private java.util.List defModelStatement_ = new java.util.ArrayList();
    private IRNode parentRNode_;

    /**
     * Creates a <code>XbfBody</code>.
     *
     */
    public XbfBody() {
    }

    /**
     * Creates a <code>XbfBody</code>.
     *
     * @param source
     */
    public XbfBody(XbfBody source) {
        setup(source);
    }

    /**
     * Creates a <code>XbfBody</code> by the Stack <code>stack</code>
     * that contains Elements.
     * This constructor is supposed to be used internally
     * by the Relaxer system.
     *
     * @param stack
     */
    public XbfBody(RStack stack) {
        setup(stack);
    }

    /**
     * Creates a <code>XbfBody</code> by the Document <code>doc</code>.
     *
     * @param doc
     */
    public XbfBody(Document doc) {
        setup(doc.getDocumentElement());
    }

    /**
     * Creates a <code>XbfBody</code> by the Element <code>element</code>.
     *
     * @param element
     */
    public XbfBody(Element element) {
        setup(element);
    }

    /**
     * Creates a <code>XbfBody</code> by the File <code>file</code>.
     *
     * @param file
     * @exception IOException
     * @exception SAXException
     * @exception ParserConfigurationException
     */
    public XbfBody(File file) throws IOException, SAXException, ParserConfigurationException {
        setup(file);
    }

    /**
     * Creates a <code>XbfBody</code>
     * by the String representation of URI <code>uri</code>.
     *
     * @param uri
     * @exception IOException
     * @exception SAXException
     * @exception ParserConfigurationException
     */
    public XbfBody(String uri) throws IOException, SAXException, ParserConfigurationException {
        setup(uri);
    }

    /**
     * Creates a <code>XbfBody</code> by the URL <code>url</code>.
     *
     * @param url
     * @exception IOException
     * @exception SAXException
     * @exception ParserConfigurationException
     */
    public XbfBody(URL url) throws IOException, SAXException, ParserConfigurationException {
        setup(url);
    }

    /**
     * Creates a <code>XbfBody</code> by the InputStream <code>in</code>.
     *
     * @param in
     * @exception IOException
     * @exception SAXException
     * @exception ParserConfigurationException
     */
    public XbfBody(InputStream in) throws IOException, SAXException, ParserConfigurationException {
        setup(in);
    }

    /**
     * Creates a <code>XbfBody</code> by the InputSource <code>is</code>.
     *
     * @param is
     * @exception IOException
     * @exception SAXException
     * @exception ParserConfigurationException
     */
    public XbfBody(InputSource is) throws IOException, SAXException, ParserConfigurationException {
        setup(is);
    }

    /**
     * Creates a <code>XbfBody</code> by the Reader <code>reader</code>.
     *
     * @param reader
     * @exception IOException
     * @exception SAXException
     * @exception ParserConfigurationException
     */
    public XbfBody(Reader reader) throws IOException, SAXException, ParserConfigurationException {
        setup(reader);
    }

    /**
     * Initializes the <code>XbfBody</code> by the XbfBody <code>source</code>.
     *
     * @param source
     */
    public void setup(XbfBody source) {
        int size;
        this.defModelStatement_.clear();
        size = source.defModelStatement_.size();
        for (int i = 0;i < size;i++) {
            addDefModelStatement((IXbfDefModelStatementChoice)source.getDefModelStatement(i).clone());
        }
    }

    /**
     * Initializes the <code>XbfBody</code> by the Document <code>doc</code>.
     *
     * @param doc
     */
    public void setup(Document doc) {
        setup(doc.getDocumentElement());
    }

    /**
     * Initializes the <code>XbfBody</code> by the Element <code>element</code>.
     *
     * @param element
     */
    public void setup(Element element) {
        init(element);
    }

    /**
     * Initializes the <code>XbfBody</code> by the Stack <code>stack</code>
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
        defModelStatement_.clear();
        while (true) {
            if (XbfFdoStatement.isMatch(stack)) {
                addDefModelStatement(factory.createXbfFdoStatement(stack));
            } else if (XbfFselectCaseStatement.isMatch(stack)) {
                addDefModelStatement(factory.createXbfFselectCaseStatement(stack));
            } else if (XbfFcaseLabel.isMatch(stack)) {
                addDefModelStatement(factory.createXbfFcaseLabel(stack));
            } else if (XbfGotoStatement.isMatch(stack)) {
                addDefModelStatement(factory.createXbfGotoStatement(stack));
            } else if (XbfFstopStatement.isMatch(stack)) {
                addDefModelStatement(factory.createXbfFstopStatement(stack));
            } else if (XbfFpauseStatement.isMatch(stack)) {
                addDefModelStatement(factory.createXbfFpauseStatement(stack));
            } else if (XbfExprStatement.isMatch(stack)) {
                addDefModelStatement(factory.createXbfExprStatement(stack));
            } else if (XbfFifStatement.isMatch(stack)) {
                addDefModelStatement(factory.createXbfFifStatement(stack));
            } else if (XbfFdoWhileStatement.isMatch(stack)) {
                addDefModelStatement(factory.createXbfFdoWhileStatement(stack));
            } else if (XbfFcycleStatement.isMatch(stack)) {
                addDefModelStatement(factory.createXbfFcycleStatement(stack));
            } else if (XbfFexitStatement.isMatch(stack)) {
                addDefModelStatement(factory.createXbfFexitStatement(stack));
            } else if (XbfStatementLabel.isMatch(stack)) {
                addDefModelStatement(factory.createXbfStatementLabel(stack));
            } else if (XbfFreadStatement.isMatch(stack)) {
                addDefModelStatement(factory.createXbfFreadStatement(stack));
            } else if (XbfFwriteStatement.isMatch(stack)) {
                addDefModelStatement(factory.createXbfFwriteStatement(stack));
            } else if (XbfFprintStatement.isMatch(stack)) {
                addDefModelStatement(factory.createXbfFprintStatement(stack));
            } else if (XbfFrewindStatement.isMatch(stack)) {
                addDefModelStatement(factory.createXbfFrewindStatement(stack));
            } else if (XbfFendFileStatement.isMatch(stack)) {
                addDefModelStatement(factory.createXbfFendFileStatement(stack));
            } else if (XbfFbackspaceStatement.isMatch(stack)) {
                addDefModelStatement(factory.createXbfFbackspaceStatement(stack));
            } else if (XbfFopenStatement.isMatch(stack)) {
                addDefModelStatement(factory.createXbfFopenStatement(stack));
            } else if (XbfFcloseStatement.isMatch(stack)) {
                addDefModelStatement(factory.createXbfFcloseStatement(stack));
            } else if (XbfFinquireStatement.isMatch(stack)) {
                addDefModelStatement(factory.createXbfFinquireStatement(stack));
            } else if (XbfFformatDecl.isMatch(stack)) {
                addDefModelStatement(factory.createXbfFformatDecl(stack));
            } else if (XbfFentryDecl.isMatch(stack)) {
                addDefModelStatement(factory.createXbfFentryDecl(stack));
            } else if (XbfFallocateStatement.isMatch(stack)) {
                addDefModelStatement(factory.createXbfFallocateStatement(stack));
            } else if (XbfFdeallocateStatement.isMatch(stack)) {
                addDefModelStatement(factory.createXbfFdeallocateStatement(stack));
            } else if (XbfFcontainsStatement.isMatch(stack)) {
                addDefModelStatement(factory.createXbfFcontainsStatement(stack));
            } else if (XbfContinueStatement.isMatch(stack)) {
                addDefModelStatement(factory.createXbfContinueStatement(stack));
            } else if (XbfFreturnStatement.isMatch(stack)) {
                addDefModelStatement(factory.createXbfFreturnStatement(stack));
            } else if (XbfFwhereStatement.isMatch(stack)) {
                addDefModelStatement(factory.createXbfFwhereStatement(stack));
            } else if (XbfFdataDecl.isMatch(stack)) {
                addDefModelStatement(factory.createXbfFdataDecl(stack));
            } else if (XbfFnullifyStatement.isMatch(stack)) {
                addDefModelStatement(factory.createXbfFnullifyStatement(stack));
            } else if (XbfFpragmaStatement.isMatch(stack)) {
                addDefModelStatement(factory.createXbfFpragmaStatement(stack));
            } else if (XbfText.isMatch(stack)) {
                addDefModelStatement(factory.createXbfText(stack));
            } else if (XbfFassignStatement.isMatch(stack)) {
                addDefModelStatement(factory.createXbfFassignStatement(stack));
            } else if (XbfFpointerAssignStatement.isMatch(stack)) {
                addDefModelStatement(factory.createXbfFpointerAssignStatement(stack));
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
        return (factory.createXbfBody(this));
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
        Element element = doc.createElement("body");
        int size;
        size = this.defModelStatement_.size();
        for (int i = 0;i < size;i++) {
            IXbfDefModelStatementChoice value = (IXbfDefModelStatementChoice)this.defModelStatement_.get(i);
            value.makeElement(element);
        }
        parent.appendChild(element);
    }

    /**
     * Initializes the <code>XbfBody</code> by the File <code>file</code>.
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
     * Initializes the <code>XbfBody</code>
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
     * Initializes the <code>XbfBody</code> by the URL <code>url</code>.
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
     * Initializes the <code>XbfBody</code> by the InputStream <code>in</code>.
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
     * Initializes the <code>XbfBody</code> by the InputSource <code>is</code>.
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
     * Initializes the <code>XbfBody</code> by the Reader <code>reader</code>.
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
     * Gets the IXbfDefModelStatementChoice property <b>defModelStatement</b>.
     *
     * @return IXbfDefModelStatementChoice[]
     */
    public final IXbfDefModelStatementChoice[] getDefModelStatement() {
        IXbfDefModelStatementChoice[] array = new IXbfDefModelStatementChoice[defModelStatement_.size()];
        return ((IXbfDefModelStatementChoice[])defModelStatement_.toArray(array));
    }

    /**
     * Sets the IXbfDefModelStatementChoice property <b>defModelStatement</b>.
     *
     * @param defModelStatement
     */
    public final void setDefModelStatement(IXbfDefModelStatementChoice[] defModelStatement) {
        this.defModelStatement_.clear();
        for (int i = 0;i < defModelStatement.length;i++) {
            addDefModelStatement(defModelStatement[i]);
        }
        for (int i = 0;i < defModelStatement.length;i++) {
            defModelStatement[i].rSetParentRNode(this);
        }
    }

    /**
     * Sets the IXbfDefModelStatementChoice property <b>defModelStatement</b>.
     *
     * @param defModelStatement
     */
    public final void setDefModelStatement(IXbfDefModelStatementChoice defModelStatement) {
        this.defModelStatement_.clear();
        addDefModelStatement(defModelStatement);
        if (defModelStatement != null) {
            defModelStatement.rSetParentRNode(this);
        }
    }

    /**
     * Adds the IXbfDefModelStatementChoice property <b>defModelStatement</b>.
     *
     * @param defModelStatement
     */
    public final void addDefModelStatement(IXbfDefModelStatementChoice defModelStatement) {
        this.defModelStatement_.add(defModelStatement);
        if (defModelStatement != null) {
            defModelStatement.rSetParentRNode(this);
        }
    }

    /**
     * Adds the IXbfDefModelStatementChoice property <b>defModelStatement</b>.
     *
     * @param defModelStatement
     */
    public final void addDefModelStatement(IXbfDefModelStatementChoice[] defModelStatement) {
        for (int i = 0;i < defModelStatement.length;i++) {
            addDefModelStatement(defModelStatement[i]);
        }
        for (int i = 0;i < defModelStatement.length;i++) {
            defModelStatement[i].rSetParentRNode(this);
        }
    }

    /**
     * Gets number of the IXbfDefModelStatementChoice property <b>defModelStatement</b>.
     *
     * @return int
     */
    public final int sizeDefModelStatement() {
        return (defModelStatement_.size());
    }

    /**
     * Gets the IXbfDefModelStatementChoice property <b>defModelStatement</b> by index.
     *
     * @param index
     * @return IXbfDefModelStatementChoice
     */
    public final IXbfDefModelStatementChoice getDefModelStatement(int index) {
        return ((IXbfDefModelStatementChoice)defModelStatement_.get(index));
    }

    /**
     * Sets the IXbfDefModelStatementChoice property <b>defModelStatement</b> by index.
     *
     * @param index
     * @param defModelStatement
     */
    public final void setDefModelStatement(int index, IXbfDefModelStatementChoice defModelStatement) {
        this.defModelStatement_.set(index, defModelStatement);
        if (defModelStatement != null) {
            defModelStatement.rSetParentRNode(this);
        }
    }

    /**
     * Adds the IXbfDefModelStatementChoice property <b>defModelStatement</b> by index.
     *
     * @param index
     * @param defModelStatement
     */
    public final void addDefModelStatement(int index, IXbfDefModelStatementChoice defModelStatement) {
        this.defModelStatement_.add(index, defModelStatement);
        if (defModelStatement != null) {
            defModelStatement.rSetParentRNode(this);
        }
    }

    /**
     * Remove the IXbfDefModelStatementChoice property <b>defModelStatement</b> by index.
     *
     * @param index
     */
    public final void removeDefModelStatement(int index) {
        this.defModelStatement_.remove(index);
    }

    /**
     * Remove the IXbfDefModelStatementChoice property <b>defModelStatement</b> by object.
     *
     * @param defModelStatement
     */
    public final void removeDefModelStatement(IXbfDefModelStatementChoice defModelStatement) {
        this.defModelStatement_.remove(defModelStatement);
    }

    /**
     * Clear the IXbfDefModelStatementChoice property <b>defModelStatement</b>.
     *
     */
    public final void clearDefModelStatement() {
        this.defModelStatement_.clear();
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
        buffer.append("<body");
        size = this.defModelStatement_.size();
        for (int i = 0;i < size;i++) {
            IXbfDefModelStatementChoice value = (IXbfDefModelStatementChoice)this.defModelStatement_.get(i);
            value.makeTextAttribute(buffer);
        }
        buffer.append(">");
        size = this.defModelStatement_.size();
        for (int i = 0;i < size;i++) {
            IXbfDefModelStatementChoice value = (IXbfDefModelStatementChoice)this.defModelStatement_.get(i);
            value.makeTextElement(buffer);
        }
        buffer.append("</body>");
    }

    /**
     * Makes an XML text representation.
     *
     * @param buffer
     * @exception IOException
     */
    public void makeTextElement(Writer buffer) throws IOException {
        int size;
        buffer.write("<body");
        size = this.defModelStatement_.size();
        for (int i = 0;i < size;i++) {
            IXbfDefModelStatementChoice value = (IXbfDefModelStatementChoice)this.defModelStatement_.get(i);
            value.makeTextAttribute(buffer);
        }
        buffer.write(">");
        size = this.defModelStatement_.size();
        for (int i = 0;i < size;i++) {
            IXbfDefModelStatementChoice value = (IXbfDefModelStatementChoice)this.defModelStatement_.get(i);
            value.makeTextElement(buffer);
        }
        buffer.write("</body>");
    }

    /**
     * Makes an XML text representation.
     *
     * @param buffer
     */
    public void makeTextElement(PrintWriter buffer) {
        int size;
        buffer.print("<body");
        size = this.defModelStatement_.size();
        for (int i = 0;i < size;i++) {
            IXbfDefModelStatementChoice value = (IXbfDefModelStatementChoice)this.defModelStatement_.get(i);
            value.makeTextAttribute(buffer);
        }
        buffer.print(">");
        size = this.defModelStatement_.size();
        for (int i = 0;i < size;i++) {
            IXbfDefModelStatementChoice value = (IXbfDefModelStatementChoice)this.defModelStatement_.get(i);
            value.makeTextElement(buffer);
        }
        buffer.print("</body>");
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
        classNodes.addAll(defModelStatement_);
        IRNode[] nodes = new IRNode[classNodes.size()];
        return ((IRNode[])classNodes.toArray(nodes));
    }

    /**
     * Tests if a Element <code>element</code> is valid
     * for the <code>XbfBody</code>.
     *
     * @param element
     * @return boolean
     */
    public static boolean isMatch(Element element) {
        if (!URelaxer.isTargetElement(element, "body")) {
            return (false);
        }
        RStack target = new RStack(element);
        boolean $match$ = false;
        Element child;
        while (true) {
            if (XbfFdoStatement.isMatchHungry(target)) {
                $match$ = true;
            } else if (XbfFselectCaseStatement.isMatchHungry(target)) {
                $match$ = true;
            } else if (XbfFcaseLabel.isMatchHungry(target)) {
                $match$ = true;
            } else if (XbfGotoStatement.isMatchHungry(target)) {
                $match$ = true;
            } else if (XbfFstopStatement.isMatchHungry(target)) {
                $match$ = true;
            } else if (XbfFpauseStatement.isMatchHungry(target)) {
                $match$ = true;
            } else if (XbfExprStatement.isMatchHungry(target)) {
                $match$ = true;
            } else if (XbfFifStatement.isMatchHungry(target)) {
                $match$ = true;
            } else if (XbfFdoWhileStatement.isMatchHungry(target)) {
                $match$ = true;
            } else if (XbfFcycleStatement.isMatchHungry(target)) {
                $match$ = true;
            } else if (XbfFexitStatement.isMatchHungry(target)) {
                $match$ = true;
            } else if (XbfStatementLabel.isMatchHungry(target)) {
                $match$ = true;
            } else if (XbfFreadStatement.isMatchHungry(target)) {
                $match$ = true;
            } else if (XbfFwriteStatement.isMatchHungry(target)) {
                $match$ = true;
            } else if (XbfFprintStatement.isMatchHungry(target)) {
                $match$ = true;
            } else if (XbfFrewindStatement.isMatchHungry(target)) {
                $match$ = true;
            } else if (XbfFendFileStatement.isMatchHungry(target)) {
                $match$ = true;
            } else if (XbfFbackspaceStatement.isMatchHungry(target)) {
                $match$ = true;
            } else if (XbfFopenStatement.isMatchHungry(target)) {
                $match$ = true;
            } else if (XbfFcloseStatement.isMatchHungry(target)) {
                $match$ = true;
            } else if (XbfFinquireStatement.isMatchHungry(target)) {
                $match$ = true;
            } else if (XbfFformatDecl.isMatchHungry(target)) {
                $match$ = true;
            } else if (XbfFentryDecl.isMatchHungry(target)) {
                $match$ = true;
            } else if (XbfFallocateStatement.isMatchHungry(target)) {
                $match$ = true;
            } else if (XbfFdeallocateStatement.isMatchHungry(target)) {
                $match$ = true;
            } else if (XbfFcontainsStatement.isMatchHungry(target)) {
                $match$ = true;
            } else if (XbfContinueStatement.isMatchHungry(target)) {
                $match$ = true;
            } else if (XbfFreturnStatement.isMatchHungry(target)) {
                $match$ = true;
            } else if (XbfFwhereStatement.isMatchHungry(target)) {
                $match$ = true;
            } else if (XbfFdataDecl.isMatchHungry(target)) {
                $match$ = true;
            } else if (XbfFnullifyStatement.isMatchHungry(target)) {
                $match$ = true;
            } else if (XbfFpragmaStatement.isMatchHungry(target)) {
                $match$ = true;
            } else if (XbfText.isMatchHungry(target)) {
                $match$ = true;
            } else if (XbfFassignStatement.isMatchHungry(target)) {
                $match$ = true;
            } else if (XbfFpointerAssignStatement.isMatchHungry(target)) {
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
     * is valid for the <code>XbfBody</code>.
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
     * is valid for the <code>XbfBody</code>.
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
