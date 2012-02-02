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
 * <b>XbcThen</b> is generated from XcodeML_C.rng by Relaxer.
 * This class is derived from:
 * 
 * <!-- for programmer
 * <element java:extends="xcodeml.c.obj.XmcObj" name="then">
 *   <optional>
 *     <ref name="statements"/>
 *   </optional>
 * </element>
 * -->
 * <!-- for javadoc -->
 * <pre> &lt;element java:extends="xcodeml.c.obj.XmcObj" name="then"&gt;
 *   &lt;optional&gt;
 *     &lt;ref name="statements"/&gt;
 *   &lt;/optional&gt;
 * &lt;/element&gt;
 * </pre>
 *
 * @version XcodeML_C.rng (Thu Feb 02 16:55:19 JST 2012)
 * @author  Relaxer 1.0 (http://www.relaxer.org)
 */
public class XbcThen extends xcodeml.c.obj.XmcObj implements java.io.Serializable, Cloneable, IRVisitable, IRNode {
    private IXbcStatementsChoice statements_;
    private IRNode parentRNode_;

    /**
     * Creates a <code>XbcThen</code>.
     *
     */
    public XbcThen() {
    }

    /**
     * Creates a <code>XbcThen</code>.
     *
     * @param source
     */
    public XbcThen(XbcThen source) {
        setup(source);
    }

    /**
     * Creates a <code>XbcThen</code> by the Stack <code>stack</code>
     * that contains Elements.
     * This constructor is supposed to be used internally
     * by the Relaxer system.
     *
     * @param stack
     */
    public XbcThen(RStack stack) {
        setup(stack);
    }

    /**
     * Creates a <code>XbcThen</code> by the Document <code>doc</code>.
     *
     * @param doc
     */
    public XbcThen(Document doc) {
        setup(doc.getDocumentElement());
    }

    /**
     * Creates a <code>XbcThen</code> by the Element <code>element</code>.
     *
     * @param element
     */
    public XbcThen(Element element) {
        setup(element);
    }

    /**
     * Creates a <code>XbcThen</code> by the File <code>file</code>.
     *
     * @param file
     * @exception IOException
     * @exception SAXException
     * @exception ParserConfigurationException
     */
    public XbcThen(File file) throws IOException, SAXException, ParserConfigurationException {
        setup(file);
    }

    /**
     * Creates a <code>XbcThen</code>
     * by the String representation of URI <code>uri</code>.
     *
     * @param uri
     * @exception IOException
     * @exception SAXException
     * @exception ParserConfigurationException
     */
    public XbcThen(String uri) throws IOException, SAXException, ParserConfigurationException {
        setup(uri);
    }

    /**
     * Creates a <code>XbcThen</code> by the URL <code>url</code>.
     *
     * @param url
     * @exception IOException
     * @exception SAXException
     * @exception ParserConfigurationException
     */
    public XbcThen(URL url) throws IOException, SAXException, ParserConfigurationException {
        setup(url);
    }

    /**
     * Creates a <code>XbcThen</code> by the InputStream <code>in</code>.
     *
     * @param in
     * @exception IOException
     * @exception SAXException
     * @exception ParserConfigurationException
     */
    public XbcThen(InputStream in) throws IOException, SAXException, ParserConfigurationException {
        setup(in);
    }

    /**
     * Creates a <code>XbcThen</code> by the InputSource <code>is</code>.
     *
     * @param is
     * @exception IOException
     * @exception SAXException
     * @exception ParserConfigurationException
     */
    public XbcThen(InputSource is) throws IOException, SAXException, ParserConfigurationException {
        setup(is);
    }

    /**
     * Creates a <code>XbcThen</code> by the Reader <code>reader</code>.
     *
     * @param reader
     * @exception IOException
     * @exception SAXException
     * @exception ParserConfigurationException
     */
    public XbcThen(Reader reader) throws IOException, SAXException, ParserConfigurationException {
        setup(reader);
    }

    /**
     * Initializes the <code>XbcThen</code> by the XbcThen <code>source</code>.
     *
     * @param source
     */
    public void setup(XbcThen source) {
        int size;
        if (source.statements_ != null) {
            setStatements((IXbcStatementsChoice)source.getStatements().clone());
        }
    }

    /**
     * Initializes the <code>XbcThen</code> by the Document <code>doc</code>.
     *
     * @param doc
     */
    public void setup(Document doc) {
        setup(doc.getDocumentElement());
    }

    /**
     * Initializes the <code>XbcThen</code> by the Element <code>element</code>.
     *
     * @param element
     */
    public void setup(Element element) {
        init(element);
    }

    /**
     * Initializes the <code>XbcThen</code> by the Stack <code>stack</code>
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
        if (XbcGccAsmStatement.isMatch(stack)) {
            setStatements(factory.createXbcGccAsmStatement(stack));
        } else if (XbcForStatement.isMatch(stack)) {
            setStatements(factory.createXbcForStatement(stack));
        } else if (XbcCompoundStatement.isMatch(stack)) {
            setStatements(factory.createXbcCompoundStatement(stack));
        } else if (XbcIfStatement.isMatch(stack)) {
            setStatements(factory.createXbcIfStatement(stack));
        } else if (XbcWhileStatement.isMatch(stack)) {
            setStatements(factory.createXbcWhileStatement(stack));
        } else if (XbcDoStatement.isMatch(stack)) {
            setStatements(factory.createXbcDoStatement(stack));
        } else if (XbcSwitchStatement.isMatch(stack)) {
            setStatements(factory.createXbcSwitchStatement(stack));
        } else if (XbcGccRangedCaseLabel.isMatch(stack)) {
            setStatements(factory.createXbcGccRangedCaseLabel(stack));
        } else if (XbcStatementLabel.isMatch(stack)) {
            setStatements(factory.createXbcStatementLabel(stack));
        } else if (XbcCaseLabel.isMatch(stack)) {
            setStatements(factory.createXbcCaseLabel(stack));
        } else if (XbcBreakStatement.isMatch(stack)) {
            setStatements(factory.createXbcBreakStatement(stack));
        } else if (XbcContinueStatement.isMatch(stack)) {
            setStatements(factory.createXbcContinueStatement(stack));
        } else if (XbcReturnStatement.isMatch(stack)) {
            setStatements(factory.createXbcReturnStatement(stack));
        } else if (XbcGotoStatement.isMatch(stack)) {
            setStatements(factory.createXbcGotoStatement(stack));
        } else if (XbcDefaultLabel.isMatch(stack)) {
            setStatements(factory.createXbcDefaultLabel(stack));
        } else if (XbcPragma.isMatch(stack)) {
            setStatements(factory.createXbcPragma(stack));
        } else if (XbcText.isMatch(stack)) {
            setStatements(factory.createXbcText(stack));
        } else if (XbcExprStatement.isMatch(stack)) {
            setStatements(factory.createXbcExprStatement(stack));
        } else {
        }
    }

    /**
     * @return Object
     */
    public Object clone() {
        IXcodeML_CFactory factory = XcodeML_CFactory.getFactory();
        return (factory.createXbcThen(this));
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
        Element element = doc.createElement("then");
        int size;
        if (this.statements_ != null) {
            this.statements_.makeElement(element);
        }
        parent.appendChild(element);
    }

    /**
     * Initializes the <code>XbcThen</code> by the File <code>file</code>.
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
     * Initializes the <code>XbcThen</code>
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
     * Initializes the <code>XbcThen</code> by the URL <code>url</code>.
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
     * Initializes the <code>XbcThen</code> by the InputStream <code>in</code>.
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
     * Initializes the <code>XbcThen</code> by the InputSource <code>is</code>.
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
     * Initializes the <code>XbcThen</code> by the Reader <code>reader</code>.
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
     * @return IXbcStatementsChoice
     */
    public final IXbcStatementsChoice getStatements() {
        return (statements_);
    }

    /**
     * Sets the IXbcStatementsChoice property <b>statements</b>.
     *
     * @param statements
     */
    public final void setStatements(IXbcStatementsChoice statements) {
        this.statements_ = statements;
        if (statements != null) {
            statements.rSetParentRNode(this);
        }
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
        buffer.append("<then");
        if (statements_ != null) {
            statements_.makeTextAttribute(buffer);
        }
        buffer.append(">");
        if (statements_ != null) {
            statements_.makeTextElement(buffer);
        }
        buffer.append("</then>");
    }

    /**
     * Makes an XML text representation.
     *
     * @param buffer
     * @exception IOException
     */
    public void makeTextElement(Writer buffer) throws IOException {
        int size;
        buffer.write("<then");
        if (statements_ != null) {
            statements_.makeTextAttribute(buffer);
        }
        buffer.write(">");
        if (statements_ != null) {
            statements_.makeTextElement(buffer);
        }
        buffer.write("</then>");
    }

    /**
     * Makes an XML text representation.
     *
     * @param buffer
     */
    public void makeTextElement(PrintWriter buffer) {
        int size;
        buffer.print("<then");
        if (statements_ != null) {
            statements_.makeTextAttribute(buffer);
        }
        buffer.print(">");
        if (statements_ != null) {
            statements_.makeTextElement(buffer);
        }
        buffer.print("</then>");
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
        if (statements_ != null) {
            classNodes.add(statements_);
        }
        IRNode[] nodes = new IRNode[classNodes.size()];
        return ((IRNode[])classNodes.toArray(nodes));
    }

    /**
     * Tests if a Element <code>element</code> is valid
     * for the <code>XbcThen</code>.
     *
     * @param element
     * @return boolean
     */
    public static boolean isMatch(Element element) {
        if (!URelaxer.isTargetElement(element, "then")) {
            return (false);
        }
        RStack target = new RStack(element);
        boolean $match$ = false;
        Element child;
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
        }
        if (!target.isEmptyElement()) {
            return (false);
        }
        return (true);
    }

    /**
     * Tests if elements contained in a Stack <code>stack</code>
     * is valid for the <code>XbcThen</code>.
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
     * is valid for the <code>XbcThen</code>.
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
