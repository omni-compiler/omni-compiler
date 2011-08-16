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
 * <b>XbcBody</b> is generated from XcodeML_C.rng by Relaxer.
 * This class is derived from:
 * 
 * <!-- for programmer
 * <element java:extends="xcodeml.c.obj.XmcObj" name="body">
 *   <zeroOrMore>
 *     <ref name="statements"/>
 *   </zeroOrMore>
 * </element>
 * -->
 * <!-- for javadoc -->
 * <pre> &lt;element java:extends="xcodeml.c.obj.XmcObj" name="body"&gt;
 *   &lt;zeroOrMore&gt;
 *     &lt;ref name="statements"/&gt;
 *   &lt;/zeroOrMore&gt;
 * &lt;/element&gt;
 * </pre>
 *
 * @version XcodeML_C.rng (Mon Aug 15 15:55:18 JST 2011)
 * @author  Relaxer 1.0 (http://www.relaxer.org)
 */
public class XbcBody extends xcodeml.c.obj.XmcObj implements java.io.Serializable, Cloneable, IRVisitable, IRNode {
    // List<IXbcStatementsChoice>
    private java.util.List statements_ = new java.util.ArrayList();
    private IRNode parentRNode_;

    /**
     * Creates a <code>XbcBody</code>.
     *
     */
    public XbcBody() {
    }

    /**
     * Creates a <code>XbcBody</code>.
     *
     * @param source
     */
    public XbcBody(XbcBody source) {
        setup(source);
    }

    /**
     * Creates a <code>XbcBody</code> by the Stack <code>stack</code>
     * that contains Elements.
     * This constructor is supposed to be used internally
     * by the Relaxer system.
     *
     * @param stack
     */
    public XbcBody(RStack stack) {
        setup(stack);
    }

    /**
     * Creates a <code>XbcBody</code> by the Document <code>doc</code>.
     *
     * @param doc
     */
    public XbcBody(Document doc) {
        setup(doc.getDocumentElement());
    }

    /**
     * Creates a <code>XbcBody</code> by the Element <code>element</code>.
     *
     * @param element
     */
    public XbcBody(Element element) {
        setup(element);
    }

    /**
     * Creates a <code>XbcBody</code> by the File <code>file</code>.
     *
     * @param file
     * @exception IOException
     * @exception SAXException
     * @exception ParserConfigurationException
     */
    public XbcBody(File file) throws IOException, SAXException, ParserConfigurationException {
        setup(file);
    }

    /**
     * Creates a <code>XbcBody</code>
     * by the String representation of URI <code>uri</code>.
     *
     * @param uri
     * @exception IOException
     * @exception SAXException
     * @exception ParserConfigurationException
     */
    public XbcBody(String uri) throws IOException, SAXException, ParserConfigurationException {
        setup(uri);
    }

    /**
     * Creates a <code>XbcBody</code> by the URL <code>url</code>.
     *
     * @param url
     * @exception IOException
     * @exception SAXException
     * @exception ParserConfigurationException
     */
    public XbcBody(URL url) throws IOException, SAXException, ParserConfigurationException {
        setup(url);
    }

    /**
     * Creates a <code>XbcBody</code> by the InputStream <code>in</code>.
     *
     * @param in
     * @exception IOException
     * @exception SAXException
     * @exception ParserConfigurationException
     */
    public XbcBody(InputStream in) throws IOException, SAXException, ParserConfigurationException {
        setup(in);
    }

    /**
     * Creates a <code>XbcBody</code> by the InputSource <code>is</code>.
     *
     * @param is
     * @exception IOException
     * @exception SAXException
     * @exception ParserConfigurationException
     */
    public XbcBody(InputSource is) throws IOException, SAXException, ParserConfigurationException {
        setup(is);
    }

    /**
     * Creates a <code>XbcBody</code> by the Reader <code>reader</code>.
     *
     * @param reader
     * @exception IOException
     * @exception SAXException
     * @exception ParserConfigurationException
     */
    public XbcBody(Reader reader) throws IOException, SAXException, ParserConfigurationException {
        setup(reader);
    }

    /**
     * Initializes the <code>XbcBody</code> by the XbcBody <code>source</code>.
     *
     * @param source
     */
    public void setup(XbcBody source) {
        int size;
        this.statements_.clear();
        size = source.statements_.size();
        for (int i = 0;i < size;i++) {
            addStatements((IXbcStatementsChoice)source.getStatements(i).clone());
        }
    }

    /**
     * Initializes the <code>XbcBody</code> by the Document <code>doc</code>.
     *
     * @param doc
     */
    public void setup(Document doc) {
        setup(doc.getDocumentElement());
    }

    /**
     * Initializes the <code>XbcBody</code> by the Element <code>element</code>.
     *
     * @param element
     */
    public void setup(Element element) {
        init(element);
    }

    /**
     * Initializes the <code>XbcBody</code> by the Stack <code>stack</code>
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
        statements_.clear();
        while (true) {
            if (XbcGccAsmStatement.isMatch(stack)) {
                addStatements(factory.createXbcGccAsmStatement(stack));
            } else if (XbcForStatement.isMatch(stack)) {
                addStatements(factory.createXbcForStatement(stack));
            } else if (XbcCompoundStatement.isMatch(stack)) {
                addStatements(factory.createXbcCompoundStatement(stack));
            } else if (XbcIfStatement.isMatch(stack)) {
                addStatements(factory.createXbcIfStatement(stack));
            } else if (XbcWhileStatement.isMatch(stack)) {
                addStatements(factory.createXbcWhileStatement(stack));
            } else if (XbcDoStatement.isMatch(stack)) {
                addStatements(factory.createXbcDoStatement(stack));
            } else if (XbcSwitchStatement.isMatch(stack)) {
                addStatements(factory.createXbcSwitchStatement(stack));
            } else if (XbcGccRangedCaseLabel.isMatch(stack)) {
                addStatements(factory.createXbcGccRangedCaseLabel(stack));
            } else if (XbcStatementLabel.isMatch(stack)) {
                addStatements(factory.createXbcStatementLabel(stack));
            } else if (XbcCaseLabel.isMatch(stack)) {
                addStatements(factory.createXbcCaseLabel(stack));
            } else if (XbcBreakStatement.isMatch(stack)) {
                addStatements(factory.createXbcBreakStatement(stack));
            } else if (XbcContinueStatement.isMatch(stack)) {
                addStatements(factory.createXbcContinueStatement(stack));
            } else if (XbcReturnStatement.isMatch(stack)) {
                addStatements(factory.createXbcReturnStatement(stack));
            } else if (XbcGotoStatement.isMatch(stack)) {
                addStatements(factory.createXbcGotoStatement(stack));
            } else if (XbcDefaultLabel.isMatch(stack)) {
                addStatements(factory.createXbcDefaultLabel(stack));
            } else if (XbcPragma.isMatch(stack)) {
                addStatements(factory.createXbcPragma(stack));
            } else if (XbcText.isMatch(stack)) {
                addStatements(factory.createXbcText(stack));
            } else if (XbcExprStatement.isMatch(stack)) {
                addStatements(factory.createXbcExprStatement(stack));
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
        return (factory.createXbcBody(this));
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
        size = this.statements_.size();
        for (int i = 0;i < size;i++) {
            IXbcStatementsChoice value = (IXbcStatementsChoice)this.statements_.get(i);
            value.makeElement(element);
        }
        parent.appendChild(element);
    }

    /**
     * Initializes the <code>XbcBody</code> by the File <code>file</code>.
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
     * Initializes the <code>XbcBody</code>
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
     * Initializes the <code>XbcBody</code> by the URL <code>url</code>.
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
     * Initializes the <code>XbcBody</code> by the InputStream <code>in</code>.
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
     * Initializes the <code>XbcBody</code> by the InputSource <code>is</code>.
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
     * Initializes the <code>XbcBody</code> by the Reader <code>reader</code>.
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
     * Gets the IXbcStatementsChoice property <b>statements</b>.
     *
     * @return IXbcStatementsChoice[]
     */
    public final IXbcStatementsChoice[] getStatements() {
        IXbcStatementsChoice[] array = new IXbcStatementsChoice[statements_.size()];
        return ((IXbcStatementsChoice[])statements_.toArray(array));
    }

    /**
     * Sets the IXbcStatementsChoice property <b>statements</b>.
     *
     * @param statements
     */
    public final void setStatements(IXbcStatementsChoice[] statements) {
        this.statements_.clear();
        for (int i = 0;i < statements.length;i++) {
            addStatements(statements[i]);
        }
        for (int i = 0;i < statements.length;i++) {
            statements[i].rSetParentRNode(this);
        }
    }

    /**
     * Sets the IXbcStatementsChoice property <b>statements</b>.
     *
     * @param statements
     */
    public final void setStatements(IXbcStatementsChoice statements) {
        this.statements_.clear();
        addStatements(statements);
        if (statements != null) {
            statements.rSetParentRNode(this);
        }
    }

    /**
     * Adds the IXbcStatementsChoice property <b>statements</b>.
     *
     * @param statements
     */
    public final void addStatements(IXbcStatementsChoice statements) {
        this.statements_.add(statements);
        if (statements != null) {
            statements.rSetParentRNode(this);
        }
    }

    /**
     * Adds the IXbcStatementsChoice property <b>statements</b>.
     *
     * @param statements
     */
    public final void addStatements(IXbcStatementsChoice[] statements) {
        for (int i = 0;i < statements.length;i++) {
            addStatements(statements[i]);
        }
        for (int i = 0;i < statements.length;i++) {
            statements[i].rSetParentRNode(this);
        }
    }

    /**
     * Gets number of the IXbcStatementsChoice property <b>statements</b>.
     *
     * @return int
     */
    public final int sizeStatements() {
        return (statements_.size());
    }

    /**
     * Gets the IXbcStatementsChoice property <b>statements</b> by index.
     *
     * @param index
     * @return IXbcStatementsChoice
     */
    public final IXbcStatementsChoice getStatements(int index) {
        return ((IXbcStatementsChoice)statements_.get(index));
    }

    /**
     * Sets the IXbcStatementsChoice property <b>statements</b> by index.
     *
     * @param index
     * @param statements
     */
    public final void setStatements(int index, IXbcStatementsChoice statements) {
        this.statements_.set(index, statements);
        if (statements != null) {
            statements.rSetParentRNode(this);
        }
    }

    /**
     * Adds the IXbcStatementsChoice property <b>statements</b> by index.
     *
     * @param index
     * @param statements
     */
    public final void addStatements(int index, IXbcStatementsChoice statements) {
        this.statements_.add(index, statements);
        if (statements != null) {
            statements.rSetParentRNode(this);
        }
    }

    /**
     * Remove the IXbcStatementsChoice property <b>statements</b> by index.
     *
     * @param index
     */
    public final void removeStatements(int index) {
        this.statements_.remove(index);
    }

    /**
     * Remove the IXbcStatementsChoice property <b>statements</b> by object.
     *
     * @param statements
     */
    public final void removeStatements(IXbcStatementsChoice statements) {
        this.statements_.remove(statements);
    }

    /**
     * Clear the IXbcStatementsChoice property <b>statements</b>.
     *
     */
    public final void clearStatements() {
        this.statements_.clear();
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
        size = this.statements_.size();
        for (int i = 0;i < size;i++) {
            IXbcStatementsChoice value = (IXbcStatementsChoice)this.statements_.get(i);
            value.makeTextAttribute(buffer);
        }
        buffer.append(">");
        size = this.statements_.size();
        for (int i = 0;i < size;i++) {
            IXbcStatementsChoice value = (IXbcStatementsChoice)this.statements_.get(i);
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
        size = this.statements_.size();
        for (int i = 0;i < size;i++) {
            IXbcStatementsChoice value = (IXbcStatementsChoice)this.statements_.get(i);
            value.makeTextAttribute(buffer);
        }
        buffer.write(">");
        size = this.statements_.size();
        for (int i = 0;i < size;i++) {
            IXbcStatementsChoice value = (IXbcStatementsChoice)this.statements_.get(i);
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
        size = this.statements_.size();
        for (int i = 0;i < size;i++) {
            IXbcStatementsChoice value = (IXbcStatementsChoice)this.statements_.get(i);
            value.makeTextAttribute(buffer);
        }
        buffer.print(">");
        size = this.statements_.size();
        for (int i = 0;i < size;i++) {
            IXbcStatementsChoice value = (IXbcStatementsChoice)this.statements_.get(i);
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
        classNodes.addAll(statements_);
        IRNode[] nodes = new IRNode[classNodes.size()];
        return ((IRNode[])classNodes.toArray(nodes));
    }

    /**
     * Tests if a Element <code>element</code> is valid
     * for the <code>XbcBody</code>.
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
            if (XbcGccAsmStatement.isMatchHungry(target)) {
                $match$ = true;
            } else if (XbcForStatement.isMatchHungry(target)) {
                $match$ = true;
            } else if (XbcCompoundStatement.isMatchHungry(target)) {
                $match$ = true;
            } else if (XbcIfStatement.isMatchHungry(target)) {
                $match$ = true;
            } else if (XbcWhileStatement.isMatchHungry(target)) {
                $match$ = true;
            } else if (XbcDoStatement.isMatchHungry(target)) {
                $match$ = true;
            } else if (XbcSwitchStatement.isMatchHungry(target)) {
                $match$ = true;
            } else if (XbcGccRangedCaseLabel.isMatchHungry(target)) {
                $match$ = true;
            } else if (XbcStatementLabel.isMatchHungry(target)) {
                $match$ = true;
            } else if (XbcCaseLabel.isMatchHungry(target)) {
                $match$ = true;
            } else if (XbcBreakStatement.isMatchHungry(target)) {
                $match$ = true;
            } else if (XbcContinueStatement.isMatchHungry(target)) {
                $match$ = true;
            } else if (XbcReturnStatement.isMatchHungry(target)) {
                $match$ = true;
            } else if (XbcGotoStatement.isMatchHungry(target)) {
                $match$ = true;
            } else if (XbcDefaultLabel.isMatchHungry(target)) {
                $match$ = true;
            } else if (XbcPragma.isMatchHungry(target)) {
                $match$ = true;
            } else if (XbcText.isMatchHungry(target)) {
                $match$ = true;
            } else if (XbcExprStatement.isMatchHungry(target)) {
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
     * is valid for the <code>XbcBody</code>.
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
     * is valid for the <code>XbcBody</code>.
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
