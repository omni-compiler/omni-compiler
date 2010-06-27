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
 * <b>XbfFpowerExpr</b> is generated from XcodeML_F.rng by Relaxer.
 * This class is derived from:
 * 
 * <!-- for programmer
 * <element java:extends="xcodeml.f.XmfObj" java:implements="xcodeml.f.binding.IXbfBinaryExpr" name="FpowerExpr">
 *   <ref name="defModelBinaryOperation"/>
 * </element>
 * -->
 * <!-- for javadoc -->
 * <pre> &lt;element java:extends="xcodeml.f.XmfObj" java:implements="xcodeml.f.binding.IXbfBinaryExpr" name="FpowerExpr"&gt;
 *   &lt;ref name="defModelBinaryOperation"/&gt;
 * &lt;/element&gt;
 * </pre>
 *
 * @version XcodeML_F.rng (Fri Dec 04 19:18:15 JST 2009)
 * @author  Relaxer 1.0 (http://www.relaxer.org)
 */
public class XbfFpowerExpr extends xcodeml.f.XmfObj implements java.io.Serializable, Cloneable, xcodeml.f.binding.IXbfBinaryExpr, IRVisitable, IRNode, IXbfArgumentsChoice, IXbfDefModelExprChoice {
    private String type_;
    private IXbfDefModelExprChoice defModelExpr1_;
    private IXbfDefModelExprChoice defModelExpr2_;
    private IRNode parentRNode_;

    /**
     * Creates a <code>XbfFpowerExpr</code>.
     *
     */
    public XbfFpowerExpr() {
    }

    /**
     * Creates a <code>XbfFpowerExpr</code>.
     *
     * @param source
     */
    public XbfFpowerExpr(XbfFpowerExpr source) {
        setup(source);
    }

    /**
     * Creates a <code>XbfFpowerExpr</code> by the Stack <code>stack</code>
     * that contains Elements.
     * This constructor is supposed to be used internally
     * by the Relaxer system.
     *
     * @param stack
     */
    public XbfFpowerExpr(RStack stack) {
        setup(stack);
    }

    /**
     * Creates a <code>XbfFpowerExpr</code> by the Document <code>doc</code>.
     *
     * @param doc
     */
    public XbfFpowerExpr(Document doc) {
        setup(doc.getDocumentElement());
    }

    /**
     * Creates a <code>XbfFpowerExpr</code> by the Element <code>element</code>.
     *
     * @param element
     */
    public XbfFpowerExpr(Element element) {
        setup(element);
    }

    /**
     * Creates a <code>XbfFpowerExpr</code> by the File <code>file</code>.
     *
     * @param file
     * @exception IOException
     * @exception SAXException
     * @exception ParserConfigurationException
     */
    public XbfFpowerExpr(File file) throws IOException, SAXException, ParserConfigurationException {
        setup(file);
    }

    /**
     * Creates a <code>XbfFpowerExpr</code>
     * by the String representation of URI <code>uri</code>.
     *
     * @param uri
     * @exception IOException
     * @exception SAXException
     * @exception ParserConfigurationException
     */
    public XbfFpowerExpr(String uri) throws IOException, SAXException, ParserConfigurationException {
        setup(uri);
    }

    /**
     * Creates a <code>XbfFpowerExpr</code> by the URL <code>url</code>.
     *
     * @param url
     * @exception IOException
     * @exception SAXException
     * @exception ParserConfigurationException
     */
    public XbfFpowerExpr(URL url) throws IOException, SAXException, ParserConfigurationException {
        setup(url);
    }

    /**
     * Creates a <code>XbfFpowerExpr</code> by the InputStream <code>in</code>.
     *
     * @param in
     * @exception IOException
     * @exception SAXException
     * @exception ParserConfigurationException
     */
    public XbfFpowerExpr(InputStream in) throws IOException, SAXException, ParserConfigurationException {
        setup(in);
    }

    /**
     * Creates a <code>XbfFpowerExpr</code> by the InputSource <code>is</code>.
     *
     * @param is
     * @exception IOException
     * @exception SAXException
     * @exception ParserConfigurationException
     */
    public XbfFpowerExpr(InputSource is) throws IOException, SAXException, ParserConfigurationException {
        setup(is);
    }

    /**
     * Creates a <code>XbfFpowerExpr</code> by the Reader <code>reader</code>.
     *
     * @param reader
     * @exception IOException
     * @exception SAXException
     * @exception ParserConfigurationException
     */
    public XbfFpowerExpr(Reader reader) throws IOException, SAXException, ParserConfigurationException {
        setup(reader);
    }

    /**
     * Initializes the <code>XbfFpowerExpr</code> by the XbfFpowerExpr <code>source</code>.
     *
     * @param source
     */
    public void setup(XbfFpowerExpr source) {
        int size;
        setType(source.getType());
        if (source.defModelExpr1_ != null) {
            setDefModelExpr1((IXbfDefModelExprChoice)source.getDefModelExpr1().clone());
        }
        if (source.defModelExpr2_ != null) {
            setDefModelExpr2((IXbfDefModelExprChoice)source.getDefModelExpr2().clone());
        }
    }

    /**
     * Initializes the <code>XbfFpowerExpr</code> by the Document <code>doc</code>.
     *
     * @param doc
     */
    public void setup(Document doc) {
        setup(doc.getDocumentElement());
    }

    /**
     * Initializes the <code>XbfFpowerExpr</code> by the Element <code>element</code>.
     *
     * @param element
     */
    public void setup(Element element) {
        init(element);
    }

    /**
     * Initializes the <code>XbfFpowerExpr</code> by the Stack <code>stack</code>
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
        type_ = URelaxer.getAttributePropertyAsString(element, "type");
        if (XbfFunctionCall.isMatch(stack)) {
            setDefModelExpr1(factory.createXbfFunctionCall(stack));
        } else if (XbfUserBinaryExpr.isMatch(stack)) {
            setDefModelExpr1(factory.createXbfUserBinaryExpr(stack));
        } else if (XbfFarrayRef.isMatch(stack)) {
            setDefModelExpr1(factory.createXbfFarrayRef(stack));
        } else if (XbfFmemberRef.isMatch(stack)) {
            setDefModelExpr1(factory.createXbfFmemberRef(stack));
        } else if (XbfFdoLoop.isMatch(stack)) {
            setDefModelExpr1(factory.createXbfFdoLoop(stack));
        } else if (XbfUserUnaryExpr.isMatch(stack)) {
            setDefModelExpr1(factory.createXbfUserUnaryExpr(stack));
        } else if (XbfPlusExpr.isMatch(stack)) {
            setDefModelExpr1(factory.createXbfPlusExpr(stack));
        } else if (XbfMinusExpr.isMatch(stack)) {
            setDefModelExpr1(factory.createXbfMinusExpr(stack));
        } else if (XbfLogEQExpr.isMatch(stack)) {
            setDefModelExpr1(factory.createXbfLogEQExpr(stack));
        } else if (XbfDivExpr.isMatch(stack)) {
            setDefModelExpr1(factory.createXbfDivExpr(stack));
        } else if (XbfFcharacterConstant.isMatch(stack)) {
            setDefModelExpr1(factory.createXbfFcharacterConstant(stack));
        } else if (XbfVar.isMatch(stack)) {
            setDefModelExpr1(factory.createXbfVar(stack));
        } else if (XbfVarRef.isMatch(stack)) {
            setDefModelExpr1(factory.createXbfVarRef(stack));
        } else if (XbfFintConstant.isMatch(stack)) {
            setDefModelExpr1(factory.createXbfFintConstant(stack));
        } else if (XbfFrealConstant.isMatch(stack)) {
            setDefModelExpr1(factory.createXbfFrealConstant(stack));
        } else if (XbfFlogicalConstant.isMatch(stack)) {
            setDefModelExpr1(factory.createXbfFlogicalConstant(stack));
        } else if (XbfFarrayConstructor.isMatch(stack)) {
            setDefModelExpr1(factory.createXbfFarrayConstructor(stack));
        } else if (XbfMulExpr.isMatch(stack)) {
            setDefModelExpr1(factory.createXbfMulExpr(stack));
        } else if (XbfUnaryMinusExpr.isMatch(stack)) {
            setDefModelExpr1(factory.createXbfUnaryMinusExpr(stack));
        } else if (XbfLogNotExpr.isMatch(stack)) {
            setDefModelExpr1(factory.createXbfLogNotExpr(stack));
        } else if (XbfLogLEExpr.isMatch(stack)) {
            setDefModelExpr1(factory.createXbfLogLEExpr(stack));
        } else if (XbfFpowerExpr.isMatch(stack)) {
            setDefModelExpr1(factory.createXbfFpowerExpr(stack));
        } else if (XbfLogNEQExpr.isMatch(stack)) {
            setDefModelExpr1(factory.createXbfLogNEQExpr(stack));
        } else if (XbfLogGEExpr.isMatch(stack)) {
            setDefModelExpr1(factory.createXbfLogGEExpr(stack));
        } else if (XbfLogGTExpr.isMatch(stack)) {
            setDefModelExpr1(factory.createXbfLogGTExpr(stack));
        } else if (XbfFcharacterRef.isMatch(stack)) {
            setDefModelExpr1(factory.createXbfFcharacterRef(stack));
        } else if (XbfFcomplexConstant.isMatch(stack)) {
            setDefModelExpr1(factory.createXbfFcomplexConstant(stack));
        } else if (XbfFstructConstructor.isMatch(stack)) {
            setDefModelExpr1(factory.createXbfFstructConstructor(stack));
        } else if (XbfFconcatExpr.isMatch(stack)) {
            setDefModelExpr1(factory.createXbfFconcatExpr(stack));
        } else if (XbfLogLTExpr.isMatch(stack)) {
            setDefModelExpr1(factory.createXbfLogLTExpr(stack));
        } else if (XbfLogAndExpr.isMatch(stack)) {
            setDefModelExpr1(factory.createXbfLogAndExpr(stack));
        } else if (XbfLogOrExpr.isMatch(stack)) {
            setDefModelExpr1(factory.createXbfLogOrExpr(stack));
        } else if (XbfLogEQVExpr.isMatch(stack)) {
            setDefModelExpr1(factory.createXbfLogEQVExpr(stack));
        } else if (XbfLogNEQVExpr.isMatch(stack)) {
            setDefModelExpr1(factory.createXbfLogNEQVExpr(stack));
        } else {
            throw (new IllegalArgumentException());
        }
        if (XbfFunctionCall.isMatch(stack)) {
            setDefModelExpr2(factory.createXbfFunctionCall(stack));
        } else if (XbfUserBinaryExpr.isMatch(stack)) {
            setDefModelExpr2(factory.createXbfUserBinaryExpr(stack));
        } else if (XbfFarrayRef.isMatch(stack)) {
            setDefModelExpr2(factory.createXbfFarrayRef(stack));
        } else if (XbfFmemberRef.isMatch(stack)) {
            setDefModelExpr2(factory.createXbfFmemberRef(stack));
        } else if (XbfUserUnaryExpr.isMatch(stack)) {
            setDefModelExpr2(factory.createXbfUserUnaryExpr(stack));
        } else if (XbfFdoLoop.isMatch(stack)) {
            setDefModelExpr2(factory.createXbfFdoLoop(stack));
        } else if (XbfPlusExpr.isMatch(stack)) {
            setDefModelExpr2(factory.createXbfPlusExpr(stack));
        } else if (XbfMinusExpr.isMatch(stack)) {
            setDefModelExpr2(factory.createXbfMinusExpr(stack));
        } else if (XbfLogEQExpr.isMatch(stack)) {
            setDefModelExpr2(factory.createXbfLogEQExpr(stack));
        } else if (XbfFcomplexConstant.isMatch(stack)) {
            setDefModelExpr2(factory.createXbfFcomplexConstant(stack));
        } else if (XbfMulExpr.isMatch(stack)) {
            setDefModelExpr2(factory.createXbfMulExpr(stack));
        } else if (XbfDivExpr.isMatch(stack)) {
            setDefModelExpr2(factory.createXbfDivExpr(stack));
        } else if (XbfLogGEExpr.isMatch(stack)) {
            setDefModelExpr2(factory.createXbfLogGEExpr(stack));
        } else if (XbfLogGTExpr.isMatch(stack)) {
            setDefModelExpr2(factory.createXbfLogGTExpr(stack));
        } else if (XbfLogLTExpr.isMatch(stack)) {
            setDefModelExpr2(factory.createXbfLogLTExpr(stack));
        } else if (XbfFcharacterConstant.isMatch(stack)) {
            setDefModelExpr2(factory.createXbfFcharacterConstant(stack));
        } else if (XbfVar.isMatch(stack)) {
            setDefModelExpr2(factory.createXbfVar(stack));
        } else if (XbfVarRef.isMatch(stack)) {
            setDefModelExpr2(factory.createXbfVarRef(stack));
        } else if (XbfFintConstant.isMatch(stack)) {
            setDefModelExpr2(factory.createXbfFintConstant(stack));
        } else if (XbfFlogicalConstant.isMatch(stack)) {
            setDefModelExpr2(factory.createXbfFlogicalConstant(stack));
        } else if (XbfUnaryMinusExpr.isMatch(stack)) {
            setDefModelExpr2(factory.createXbfUnaryMinusExpr(stack));
        } else if (XbfLogNotExpr.isMatch(stack)) {
            setDefModelExpr2(factory.createXbfLogNotExpr(stack));
        } else if (XbfFrealConstant.isMatch(stack)) {
            setDefModelExpr2(factory.createXbfFrealConstant(stack));
        } else if (XbfFarrayConstructor.isMatch(stack)) {
            setDefModelExpr2(factory.createXbfFarrayConstructor(stack));
        } else if (XbfFstructConstructor.isMatch(stack)) {
            setDefModelExpr2(factory.createXbfFstructConstructor(stack));
        } else if (XbfFpowerExpr.isMatch(stack)) {
            setDefModelExpr2(factory.createXbfFpowerExpr(stack));
        } else if (XbfLogLEExpr.isMatch(stack)) {
            setDefModelExpr2(factory.createXbfLogLEExpr(stack));
        } else if (XbfLogNEQExpr.isMatch(stack)) {
            setDefModelExpr2(factory.createXbfLogNEQExpr(stack));
        } else if (XbfLogNEQVExpr.isMatch(stack)) {
            setDefModelExpr2(factory.createXbfLogNEQVExpr(stack));
        } else if (XbfFcharacterRef.isMatch(stack)) {
            setDefModelExpr2(factory.createXbfFcharacterRef(stack));
        } else if (XbfFconcatExpr.isMatch(stack)) {
            setDefModelExpr2(factory.createXbfFconcatExpr(stack));
        } else if (XbfLogAndExpr.isMatch(stack)) {
            setDefModelExpr2(factory.createXbfLogAndExpr(stack));
        } else if (XbfLogOrExpr.isMatch(stack)) {
            setDefModelExpr2(factory.createXbfLogOrExpr(stack));
        } else if (XbfLogEQVExpr.isMatch(stack)) {
            setDefModelExpr2(factory.createXbfLogEQVExpr(stack));
        } else {
            throw (new IllegalArgumentException());
        }
    }

    /**
     * @return Object
     */
    public Object clone() {
        IXcodeML_FFactory factory = XcodeML_FFactory.getFactory();
        return (factory.createXbfFpowerExpr(this));
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
        Element element = doc.createElement("FpowerExpr");
        int size;
        if (this.type_ != null) {
            URelaxer.setAttributePropertyByString(element, "type", this.type_);
        }
        this.defModelExpr1_.makeElement(element);
        this.defModelExpr2_.makeElement(element);
        parent.appendChild(element);
    }

    /**
     * Initializes the <code>XbfFpowerExpr</code> by the File <code>file</code>.
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
     * Initializes the <code>XbfFpowerExpr</code>
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
     * Initializes the <code>XbfFpowerExpr</code> by the URL <code>url</code>.
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
     * Initializes the <code>XbfFpowerExpr</code> by the InputStream <code>in</code>.
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
     * Initializes the <code>XbfFpowerExpr</code> by the InputSource <code>is</code>.
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
     * Initializes the <code>XbfFpowerExpr</code> by the Reader <code>reader</code>.
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
     * Gets the String property <b>type</b>.
     *
     * @return String
     */
    public final String getType() {
        return (type_);
    }

    /**
     * Sets the String property <b>type</b>.
     *
     * @param type
     */
    public final void setType(String type) {
        this.type_ = type;
    }

    /**
     * Gets the IXbfDefModelExprChoice property <b>defModelExpr1</b>.
     *
     * @return IXbfDefModelExprChoice
     */
    public final IXbfDefModelExprChoice getDefModelExpr1() {
        return (defModelExpr1_);
    }

    /**
     * Sets the IXbfDefModelExprChoice property <b>defModelExpr1</b>.
     *
     * @param defModelExpr1
     */
    public final void setDefModelExpr1(IXbfDefModelExprChoice defModelExpr1) {
        this.defModelExpr1_ = defModelExpr1;
        if (defModelExpr1 != null) {
            defModelExpr1.rSetParentRNode(this);
        }
    }

    /**
     * Gets the IXbfDefModelExprChoice property <b>defModelExpr2</b>.
     *
     * @return IXbfDefModelExprChoice
     */
    public final IXbfDefModelExprChoice getDefModelExpr2() {
        return (defModelExpr2_);
    }

    /**
     * Sets the IXbfDefModelExprChoice property <b>defModelExpr2</b>.
     *
     * @param defModelExpr2
     */
    public final void setDefModelExpr2(IXbfDefModelExprChoice defModelExpr2) {
        this.defModelExpr2_ = defModelExpr2;
        if (defModelExpr2 != null) {
            defModelExpr2.rSetParentRNode(this);
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
        buffer.append("<FpowerExpr");
        if (type_ != null) {
            buffer.append(" type=\"");
            buffer.append(URelaxer.escapeAttrQuot(URelaxer.getString(getType())));
            buffer.append("\"");
        }
        defModelExpr1_.makeTextAttribute(buffer);
        defModelExpr2_.makeTextAttribute(buffer);
        buffer.append(">");
        defModelExpr1_.makeTextElement(buffer);
        defModelExpr2_.makeTextElement(buffer);
        buffer.append("</FpowerExpr>");
    }

    /**
     * Makes an XML text representation.
     *
     * @param buffer
     * @exception IOException
     */
    public void makeTextElement(Writer buffer) throws IOException {
        int size;
        buffer.write("<FpowerExpr");
        if (type_ != null) {
            buffer.write(" type=\"");
            buffer.write(URelaxer.escapeAttrQuot(URelaxer.getString(getType())));
            buffer.write("\"");
        }
        defModelExpr1_.makeTextAttribute(buffer);
        defModelExpr2_.makeTextAttribute(buffer);
        buffer.write(">");
        defModelExpr1_.makeTextElement(buffer);
        defModelExpr2_.makeTextElement(buffer);
        buffer.write("</FpowerExpr>");
    }

    /**
     * Makes an XML text representation.
     *
     * @param buffer
     */
    public void makeTextElement(PrintWriter buffer) {
        int size;
        buffer.print("<FpowerExpr");
        if (type_ != null) {
            buffer.print(" type=\"");
            buffer.print(URelaxer.escapeAttrQuot(URelaxer.getString(getType())));
            buffer.print("\"");
        }
        defModelExpr1_.makeTextAttribute(buffer);
        defModelExpr2_.makeTextAttribute(buffer);
        buffer.print(">");
        defModelExpr1_.makeTextElement(buffer);
        defModelExpr2_.makeTextElement(buffer);
        buffer.print("</FpowerExpr>");
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
    public String getTypeAsString() {
        return (URelaxer.getString(getType()));
    }

    /**
     * Sets the property value by String.
     *
     * @param string
     */
    public void setTypeByString(String string) {
        setType(string);
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
        if (defModelExpr1_ != null) {
            classNodes.add(defModelExpr1_);
        }
        if (defModelExpr2_ != null) {
            classNodes.add(defModelExpr2_);
        }
        IRNode[] nodes = new IRNode[classNodes.size()];
        return ((IRNode[])classNodes.toArray(nodes));
    }

    /**
     * Tests if a Element <code>element</code> is valid
     * for the <code>XbfFpowerExpr</code>.
     *
     * @param element
     * @return boolean
     */
    public static boolean isMatch(Element element) {
        if (!URelaxer.isTargetElement(element, "FpowerExpr")) {
            return (false);
        }
        RStack target = new RStack(element);
        boolean $match$ = false;
        Element child;
        if (XbfFunctionCall.isMatchHungry(target)) {
            $match$ = true;
        } else if (XbfUserBinaryExpr.isMatchHungry(target)) {
            $match$ = true;
        } else if (XbfFarrayRef.isMatchHungry(target)) {
            $match$ = true;
        } else if (XbfFmemberRef.isMatchHungry(target)) {
            $match$ = true;
        } else if (XbfFdoLoop.isMatchHungry(target)) {
            $match$ = true;
        } else if (XbfUserUnaryExpr.isMatchHungry(target)) {
            $match$ = true;
        } else if (XbfPlusExpr.isMatchHungry(target)) {
            $match$ = true;
        } else if (XbfMinusExpr.isMatchHungry(target)) {
            $match$ = true;
        } else if (XbfLogEQExpr.isMatchHungry(target)) {
            $match$ = true;
        } else if (XbfDivExpr.isMatchHungry(target)) {
            $match$ = true;
        } else if (XbfFcharacterConstant.isMatchHungry(target)) {
            $match$ = true;
        } else if (XbfVar.isMatchHungry(target)) {
            $match$ = true;
        } else if (XbfVarRef.isMatchHungry(target)) {
            $match$ = true;
        } else if (XbfFintConstant.isMatchHungry(target)) {
            $match$ = true;
        } else if (XbfFrealConstant.isMatchHungry(target)) {
            $match$ = true;
        } else if (XbfFlogicalConstant.isMatchHungry(target)) {
            $match$ = true;
        } else if (XbfFarrayConstructor.isMatchHungry(target)) {
            $match$ = true;
        } else if (XbfMulExpr.isMatchHungry(target)) {
            $match$ = true;
        } else if (XbfUnaryMinusExpr.isMatchHungry(target)) {
            $match$ = true;
        } else if (XbfLogNotExpr.isMatchHungry(target)) {
            $match$ = true;
        } else if (XbfLogLEExpr.isMatchHungry(target)) {
            $match$ = true;
        } else if (XbfFpowerExpr.isMatchHungry(target)) {
            $match$ = true;
        } else if (XbfLogNEQExpr.isMatchHungry(target)) {
            $match$ = true;
        } else if (XbfLogGEExpr.isMatchHungry(target)) {
            $match$ = true;
        } else if (XbfLogGTExpr.isMatchHungry(target)) {
            $match$ = true;
        } else if (XbfFcharacterRef.isMatchHungry(target)) {
            $match$ = true;
        } else if (XbfFcomplexConstant.isMatchHungry(target)) {
            $match$ = true;
        } else if (XbfFstructConstructor.isMatchHungry(target)) {
            $match$ = true;
        } else if (XbfFconcatExpr.isMatchHungry(target)) {
            $match$ = true;
        } else if (XbfLogLTExpr.isMatchHungry(target)) {
            $match$ = true;
        } else if (XbfLogAndExpr.isMatchHungry(target)) {
            $match$ = true;
        } else if (XbfLogOrExpr.isMatchHungry(target)) {
            $match$ = true;
        } else if (XbfLogEQVExpr.isMatchHungry(target)) {
            $match$ = true;
        } else if (XbfLogNEQVExpr.isMatchHungry(target)) {
            $match$ = true;
        } else {
            return (false);
        }
        if (XbfFunctionCall.isMatchHungry(target)) {
            $match$ = true;
        } else if (XbfUserBinaryExpr.isMatchHungry(target)) {
            $match$ = true;
        } else if (XbfFarrayRef.isMatchHungry(target)) {
            $match$ = true;
        } else if (XbfFmemberRef.isMatchHungry(target)) {
            $match$ = true;
        } else if (XbfUserUnaryExpr.isMatchHungry(target)) {
            $match$ = true;
        } else if (XbfFdoLoop.isMatchHungry(target)) {
            $match$ = true;
        } else if (XbfPlusExpr.isMatchHungry(target)) {
            $match$ = true;
        } else if (XbfMinusExpr.isMatchHungry(target)) {
            $match$ = true;
        } else if (XbfLogEQExpr.isMatchHungry(target)) {
            $match$ = true;
        } else if (XbfFcomplexConstant.isMatchHungry(target)) {
            $match$ = true;
        } else if (XbfMulExpr.isMatchHungry(target)) {
            $match$ = true;
        } else if (XbfDivExpr.isMatchHungry(target)) {
            $match$ = true;
        } else if (XbfLogGEExpr.isMatchHungry(target)) {
            $match$ = true;
        } else if (XbfLogGTExpr.isMatchHungry(target)) {
            $match$ = true;
        } else if (XbfLogLTExpr.isMatchHungry(target)) {
            $match$ = true;
        } else if (XbfFcharacterConstant.isMatchHungry(target)) {
            $match$ = true;
        } else if (XbfVar.isMatchHungry(target)) {
            $match$ = true;
        } else if (XbfVarRef.isMatchHungry(target)) {
            $match$ = true;
        } else if (XbfFintConstant.isMatchHungry(target)) {
            $match$ = true;
        } else if (XbfFlogicalConstant.isMatchHungry(target)) {
            $match$ = true;
        } else if (XbfUnaryMinusExpr.isMatchHungry(target)) {
            $match$ = true;
        } else if (XbfLogNotExpr.isMatchHungry(target)) {
            $match$ = true;
        } else if (XbfFrealConstant.isMatchHungry(target)) {
            $match$ = true;
        } else if (XbfFarrayConstructor.isMatchHungry(target)) {
            $match$ = true;
        } else if (XbfFstructConstructor.isMatchHungry(target)) {
            $match$ = true;
        } else if (XbfFpowerExpr.isMatchHungry(target)) {
            $match$ = true;
        } else if (XbfLogLEExpr.isMatchHungry(target)) {
            $match$ = true;
        } else if (XbfLogNEQExpr.isMatchHungry(target)) {
            $match$ = true;
        } else if (XbfLogNEQVExpr.isMatchHungry(target)) {
            $match$ = true;
        } else if (XbfFcharacterRef.isMatchHungry(target)) {
            $match$ = true;
        } else if (XbfFconcatExpr.isMatchHungry(target)) {
            $match$ = true;
        } else if (XbfLogAndExpr.isMatchHungry(target)) {
            $match$ = true;
        } else if (XbfLogOrExpr.isMatchHungry(target)) {
            $match$ = true;
        } else if (XbfLogEQVExpr.isMatchHungry(target)) {
            $match$ = true;
        } else {
            return (false);
        }
        if (!target.isEmptyElement()) {
            return (false);
        }
        return (true);
    }

    /**
     * Tests if elements contained in a Stack <code>stack</code>
     * is valid for the <code>XbfFpowerExpr</code>.
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
     * is valid for the <code>XbfFpowerExpr</code>.
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
