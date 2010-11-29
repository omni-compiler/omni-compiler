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
 * <b>XbfFstructConstructor</b> is generated from XcodeML_F.rng by Relaxer.
 * This class is derived from:
 * 
 * <!-- for programmer
 * <element java:extends="xcodeml.f.XmfObj" java:implements="xcodeml.binding.IXbTypedExpr" name="FstructConstructor">
 *   <ref name="defAttrTypeName"/>
 *   <ref name="defModelExprList"/>
 * </element>
 * -->
 * <!-- for javadoc -->
 * <pre> &lt;element java:extends="xcodeml.f.XmfObj" java:implements="xcodeml.binding.IXbTypedExpr" name="FstructConstructor"&gt;
 *   &lt;ref name="defAttrTypeName"/&gt;
 *   &lt;ref name="defModelExprList"/&gt;
 * &lt;/element&gt;
 * </pre>
 *
 * @version XcodeML_F.rng (Mon Nov 29 15:25:55 JST 2010)
 * @author  Relaxer 1.0 (http://www.relaxer.org)
 */
public class XbfFstructConstructor extends xcodeml.f.XmfObj implements java.io.Serializable, Cloneable, xcodeml.binding.IXbTypedExpr, IRVisitable, IRNode, IXbfArgumentsChoice, IXbfDefModelExprChoice {
    private String type_;
    // List<IXbfDefModelExprChoice>
    private java.util.List defModelExpr_ = new java.util.ArrayList();
    private IRNode parentRNode_;

    /**
     * Creates a <code>XbfFstructConstructor</code>.
     *
     */
    public XbfFstructConstructor() {
    }

    /**
     * Creates a <code>XbfFstructConstructor</code>.
     *
     * @param source
     */
    public XbfFstructConstructor(XbfFstructConstructor source) {
        setup(source);
    }

    /**
     * Creates a <code>XbfFstructConstructor</code> by the Stack <code>stack</code>
     * that contains Elements.
     * This constructor is supposed to be used internally
     * by the Relaxer system.
     *
     * @param stack
     */
    public XbfFstructConstructor(RStack stack) {
        setup(stack);
    }

    /**
     * Creates a <code>XbfFstructConstructor</code> by the Document <code>doc</code>.
     *
     * @param doc
     */
    public XbfFstructConstructor(Document doc) {
        setup(doc.getDocumentElement());
    }

    /**
     * Creates a <code>XbfFstructConstructor</code> by the Element <code>element</code>.
     *
     * @param element
     */
    public XbfFstructConstructor(Element element) {
        setup(element);
    }

    /**
     * Creates a <code>XbfFstructConstructor</code> by the File <code>file</code>.
     *
     * @param file
     * @exception IOException
     * @exception SAXException
     * @exception ParserConfigurationException
     */
    public XbfFstructConstructor(File file) throws IOException, SAXException, ParserConfigurationException {
        setup(file);
    }

    /**
     * Creates a <code>XbfFstructConstructor</code>
     * by the String representation of URI <code>uri</code>.
     *
     * @param uri
     * @exception IOException
     * @exception SAXException
     * @exception ParserConfigurationException
     */
    public XbfFstructConstructor(String uri) throws IOException, SAXException, ParserConfigurationException {
        setup(uri);
    }

    /**
     * Creates a <code>XbfFstructConstructor</code> by the URL <code>url</code>.
     *
     * @param url
     * @exception IOException
     * @exception SAXException
     * @exception ParserConfigurationException
     */
    public XbfFstructConstructor(URL url) throws IOException, SAXException, ParserConfigurationException {
        setup(url);
    }

    /**
     * Creates a <code>XbfFstructConstructor</code> by the InputStream <code>in</code>.
     *
     * @param in
     * @exception IOException
     * @exception SAXException
     * @exception ParserConfigurationException
     */
    public XbfFstructConstructor(InputStream in) throws IOException, SAXException, ParserConfigurationException {
        setup(in);
    }

    /**
     * Creates a <code>XbfFstructConstructor</code> by the InputSource <code>is</code>.
     *
     * @param is
     * @exception IOException
     * @exception SAXException
     * @exception ParserConfigurationException
     */
    public XbfFstructConstructor(InputSource is) throws IOException, SAXException, ParserConfigurationException {
        setup(is);
    }

    /**
     * Creates a <code>XbfFstructConstructor</code> by the Reader <code>reader</code>.
     *
     * @param reader
     * @exception IOException
     * @exception SAXException
     * @exception ParserConfigurationException
     */
    public XbfFstructConstructor(Reader reader) throws IOException, SAXException, ParserConfigurationException {
        setup(reader);
    }

    /**
     * Initializes the <code>XbfFstructConstructor</code> by the XbfFstructConstructor <code>source</code>.
     *
     * @param source
     */
    public void setup(XbfFstructConstructor source) {
        int size;
        setType(source.getType());
        this.defModelExpr_.clear();
        size = source.defModelExpr_.size();
        for (int i = 0;i < size;i++) {
            addDefModelExpr((IXbfDefModelExprChoice)source.getDefModelExpr(i).clone());
        }
    }

    /**
     * Initializes the <code>XbfFstructConstructor</code> by the Document <code>doc</code>.
     *
     * @param doc
     */
    public void setup(Document doc) {
        setup(doc.getDocumentElement());
    }

    /**
     * Initializes the <code>XbfFstructConstructor</code> by the Element <code>element</code>.
     *
     * @param element
     */
    public void setup(Element element) {
        init(element);
    }

    /**
     * Initializes the <code>XbfFstructConstructor</code> by the Stack <code>stack</code>
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
        defModelExpr_.clear();
        while (true) {
            if (XbfFunctionCall.isMatch(stack)) {
                addDefModelExpr(factory.createXbfFunctionCall(stack));
            } else if (XbfFcharacterRef.isMatch(stack)) {
                addDefModelExpr(factory.createXbfFcharacterRef(stack));
            } else if (XbfFmemberRef.isMatch(stack)) {
                addDefModelExpr(factory.createXbfFmemberRef(stack));
            } else if (XbfFcoArrayRef.isMatch(stack)) {
                addDefModelExpr(factory.createXbfFcoArrayRef(stack));
            } else if (XbfFdoLoop.isMatch(stack)) {
                addDefModelExpr(factory.createXbfFdoLoop(stack));
            } else if (XbfFarrayRef.isMatch(stack)) {
                addDefModelExpr(factory.createXbfFarrayRef(stack));
            } else if (XbfFintConstant.isMatch(stack)) {
                addDefModelExpr(factory.createXbfFintConstant(stack));
            } else if (XbfFrealConstant.isMatch(stack)) {
                addDefModelExpr(factory.createXbfFrealConstant(stack));
            } else if (XbfFcharacterConstant.isMatch(stack)) {
                addDefModelExpr(factory.createXbfFcharacterConstant(stack));
            } else if (XbfFlogicalConstant.isMatch(stack)) {
                addDefModelExpr(factory.createXbfFlogicalConstant(stack));
            } else if (XbfVar.isMatch(stack)) {
                addDefModelExpr(factory.createXbfVar(stack));
            } else if (XbfVarRef.isMatch(stack)) {
                addDefModelExpr(factory.createXbfVarRef(stack));
            } else if (XbfFarrayConstructor.isMatch(stack)) {
                addDefModelExpr(factory.createXbfFarrayConstructor(stack));
            } else if (XbfFcomplexConstant.isMatch(stack)) {
                addDefModelExpr(factory.createXbfFcomplexConstant(stack));
            } else if (XbfFstructConstructor.isMatch(stack)) {
                addDefModelExpr(factory.createXbfFstructConstructor(stack));
            } else if (XbfUserBinaryExpr.isMatch(stack)) {
                addDefModelExpr(factory.createXbfUserBinaryExpr(stack));
            } else if (XbfUserUnaryExpr.isMatch(stack)) {
                addDefModelExpr(factory.createXbfUserUnaryExpr(stack));
            } else if (XbfPlusExpr.isMatch(stack)) {
                addDefModelExpr(factory.createXbfPlusExpr(stack));
            } else if (XbfMulExpr.isMatch(stack)) {
                addDefModelExpr(factory.createXbfMulExpr(stack));
            } else if (XbfDivExpr.isMatch(stack)) {
                addDefModelExpr(factory.createXbfDivExpr(stack));
            } else if (XbfLogNEQExpr.isMatch(stack)) {
                addDefModelExpr(factory.createXbfLogNEQExpr(stack));
            } else if (XbfLogGEExpr.isMatch(stack)) {
                addDefModelExpr(factory.createXbfLogGEExpr(stack));
            } else if (XbfLogOrExpr.isMatch(stack)) {
                addDefModelExpr(factory.createXbfLogOrExpr(stack));
            } else if (XbfLogEQVExpr.isMatch(stack)) {
                addDefModelExpr(factory.createXbfLogEQVExpr(stack));
            } else if (XbfMinusExpr.isMatch(stack)) {
                addDefModelExpr(factory.createXbfMinusExpr(stack));
            } else if (XbfFpowerExpr.isMatch(stack)) {
                addDefModelExpr(factory.createXbfFpowerExpr(stack));
            } else if (XbfFconcatExpr.isMatch(stack)) {
                addDefModelExpr(factory.createXbfFconcatExpr(stack));
            } else if (XbfLogEQExpr.isMatch(stack)) {
                addDefModelExpr(factory.createXbfLogEQExpr(stack));
            } else if (XbfLogGTExpr.isMatch(stack)) {
                addDefModelExpr(factory.createXbfLogGTExpr(stack));
            } else if (XbfLogLEExpr.isMatch(stack)) {
                addDefModelExpr(factory.createXbfLogLEExpr(stack));
            } else if (XbfLogLTExpr.isMatch(stack)) {
                addDefModelExpr(factory.createXbfLogLTExpr(stack));
            } else if (XbfLogAndExpr.isMatch(stack)) {
                addDefModelExpr(factory.createXbfLogAndExpr(stack));
            } else if (XbfLogNEQVExpr.isMatch(stack)) {
                addDefModelExpr(factory.createXbfLogNEQVExpr(stack));
            } else if (XbfUnaryMinusExpr.isMatch(stack)) {
                addDefModelExpr(factory.createXbfUnaryMinusExpr(stack));
            } else if (XbfLogNotExpr.isMatch(stack)) {
                addDefModelExpr(factory.createXbfLogNotExpr(stack));
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
        return (factory.createXbfFstructConstructor(this));
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
        Element element = doc.createElement("FstructConstructor");
        int size;
        if (this.type_ != null) {
            URelaxer.setAttributePropertyByString(element, "type", this.type_);
        }
        size = this.defModelExpr_.size();
        for (int i = 0;i < size;i++) {
            IXbfDefModelExprChoice value = (IXbfDefModelExprChoice)this.defModelExpr_.get(i);
            value.makeElement(element);
        }
        parent.appendChild(element);
    }

    /**
     * Initializes the <code>XbfFstructConstructor</code> by the File <code>file</code>.
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
     * Initializes the <code>XbfFstructConstructor</code>
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
     * Initializes the <code>XbfFstructConstructor</code> by the URL <code>url</code>.
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
     * Initializes the <code>XbfFstructConstructor</code> by the InputStream <code>in</code>.
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
     * Initializes the <code>XbfFstructConstructor</code> by the InputSource <code>is</code>.
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
     * Initializes the <code>XbfFstructConstructor</code> by the Reader <code>reader</code>.
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
     * Gets the IXbfDefModelExprChoice property <b>defModelExpr</b>.
     *
     * @return IXbfDefModelExprChoice[]
     */
    public final IXbfDefModelExprChoice[] getDefModelExpr() {
        IXbfDefModelExprChoice[] array = new IXbfDefModelExprChoice[defModelExpr_.size()];
        return ((IXbfDefModelExprChoice[])defModelExpr_.toArray(array));
    }

    /**
     * Sets the IXbfDefModelExprChoice property <b>defModelExpr</b>.
     *
     * @param defModelExpr
     */
    public final void setDefModelExpr(IXbfDefModelExprChoice[] defModelExpr) {
        this.defModelExpr_.clear();
        for (int i = 0;i < defModelExpr.length;i++) {
            addDefModelExpr(defModelExpr[i]);
        }
        for (int i = 0;i < defModelExpr.length;i++) {
            defModelExpr[i].rSetParentRNode(this);
        }
    }

    /**
     * Sets the IXbfDefModelExprChoice property <b>defModelExpr</b>.
     *
     * @param defModelExpr
     */
    public final void setDefModelExpr(IXbfDefModelExprChoice defModelExpr) {
        this.defModelExpr_.clear();
        addDefModelExpr(defModelExpr);
        if (defModelExpr != null) {
            defModelExpr.rSetParentRNode(this);
        }
    }

    /**
     * Adds the IXbfDefModelExprChoice property <b>defModelExpr</b>.
     *
     * @param defModelExpr
     */
    public final void addDefModelExpr(IXbfDefModelExprChoice defModelExpr) {
        this.defModelExpr_.add(defModelExpr);
        if (defModelExpr != null) {
            defModelExpr.rSetParentRNode(this);
        }
    }

    /**
     * Adds the IXbfDefModelExprChoice property <b>defModelExpr</b>.
     *
     * @param defModelExpr
     */
    public final void addDefModelExpr(IXbfDefModelExprChoice[] defModelExpr) {
        for (int i = 0;i < defModelExpr.length;i++) {
            addDefModelExpr(defModelExpr[i]);
        }
        for (int i = 0;i < defModelExpr.length;i++) {
            defModelExpr[i].rSetParentRNode(this);
        }
    }

    /**
     * Gets number of the IXbfDefModelExprChoice property <b>defModelExpr</b>.
     *
     * @return int
     */
    public final int sizeDefModelExpr() {
        return (defModelExpr_.size());
    }

    /**
     * Gets the IXbfDefModelExprChoice property <b>defModelExpr</b> by index.
     *
     * @param index
     * @return IXbfDefModelExprChoice
     */
    public final IXbfDefModelExprChoice getDefModelExpr(int index) {
        return ((IXbfDefModelExprChoice)defModelExpr_.get(index));
    }

    /**
     * Sets the IXbfDefModelExprChoice property <b>defModelExpr</b> by index.
     *
     * @param index
     * @param defModelExpr
     */
    public final void setDefModelExpr(int index, IXbfDefModelExprChoice defModelExpr) {
        this.defModelExpr_.set(index, defModelExpr);
        if (defModelExpr != null) {
            defModelExpr.rSetParentRNode(this);
        }
    }

    /**
     * Adds the IXbfDefModelExprChoice property <b>defModelExpr</b> by index.
     *
     * @param index
     * @param defModelExpr
     */
    public final void addDefModelExpr(int index, IXbfDefModelExprChoice defModelExpr) {
        this.defModelExpr_.add(index, defModelExpr);
        if (defModelExpr != null) {
            defModelExpr.rSetParentRNode(this);
        }
    }

    /**
     * Remove the IXbfDefModelExprChoice property <b>defModelExpr</b> by index.
     *
     * @param index
     */
    public final void removeDefModelExpr(int index) {
        this.defModelExpr_.remove(index);
    }

    /**
     * Remove the IXbfDefModelExprChoice property <b>defModelExpr</b> by object.
     *
     * @param defModelExpr
     */
    public final void removeDefModelExpr(IXbfDefModelExprChoice defModelExpr) {
        this.defModelExpr_.remove(defModelExpr);
    }

    /**
     * Clear the IXbfDefModelExprChoice property <b>defModelExpr</b>.
     *
     */
    public final void clearDefModelExpr() {
        this.defModelExpr_.clear();
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
        buffer.append("<FstructConstructor");
        if (type_ != null) {
            buffer.append(" type=\"");
            buffer.append(URelaxer.escapeAttrQuot(URelaxer.getString(getType())));
            buffer.append("\"");
        }
        size = this.defModelExpr_.size();
        for (int i = 0;i < size;i++) {
            IXbfDefModelExprChoice value = (IXbfDefModelExprChoice)this.defModelExpr_.get(i);
            value.makeTextAttribute(buffer);
        }
        buffer.append(">");
        size = this.defModelExpr_.size();
        for (int i = 0;i < size;i++) {
            IXbfDefModelExprChoice value = (IXbfDefModelExprChoice)this.defModelExpr_.get(i);
            value.makeTextElement(buffer);
        }
        buffer.append("</FstructConstructor>");
    }

    /**
     * Makes an XML text representation.
     *
     * @param buffer
     * @exception IOException
     */
    public void makeTextElement(Writer buffer) throws IOException {
        int size;
        buffer.write("<FstructConstructor");
        if (type_ != null) {
            buffer.write(" type=\"");
            buffer.write(URelaxer.escapeAttrQuot(URelaxer.getString(getType())));
            buffer.write("\"");
        }
        size = this.defModelExpr_.size();
        for (int i = 0;i < size;i++) {
            IXbfDefModelExprChoice value = (IXbfDefModelExprChoice)this.defModelExpr_.get(i);
            value.makeTextAttribute(buffer);
        }
        buffer.write(">");
        size = this.defModelExpr_.size();
        for (int i = 0;i < size;i++) {
            IXbfDefModelExprChoice value = (IXbfDefModelExprChoice)this.defModelExpr_.get(i);
            value.makeTextElement(buffer);
        }
        buffer.write("</FstructConstructor>");
    }

    /**
     * Makes an XML text representation.
     *
     * @param buffer
     */
    public void makeTextElement(PrintWriter buffer) {
        int size;
        buffer.print("<FstructConstructor");
        if (type_ != null) {
            buffer.print(" type=\"");
            buffer.print(URelaxer.escapeAttrQuot(URelaxer.getString(getType())));
            buffer.print("\"");
        }
        size = this.defModelExpr_.size();
        for (int i = 0;i < size;i++) {
            IXbfDefModelExprChoice value = (IXbfDefModelExprChoice)this.defModelExpr_.get(i);
            value.makeTextAttribute(buffer);
        }
        buffer.print(">");
        size = this.defModelExpr_.size();
        for (int i = 0;i < size;i++) {
            IXbfDefModelExprChoice value = (IXbfDefModelExprChoice)this.defModelExpr_.get(i);
            value.makeTextElement(buffer);
        }
        buffer.print("</FstructConstructor>");
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
        classNodes.addAll(defModelExpr_);
        IRNode[] nodes = new IRNode[classNodes.size()];
        return ((IRNode[])classNodes.toArray(nodes));
    }

    /**
     * Tests if a Element <code>element</code> is valid
     * for the <code>XbfFstructConstructor</code>.
     *
     * @param element
     * @return boolean
     */
    public static boolean isMatch(Element element) {
        if (!URelaxer.isTargetElement(element, "FstructConstructor")) {
            return (false);
        }
        RStack target = new RStack(element);
        boolean $match$ = false;
        Element child;
        while (true) {
            if (XbfFunctionCall.isMatchHungry(target)) {
                $match$ = true;
            } else if (XbfFcharacterRef.isMatchHungry(target)) {
                $match$ = true;
            } else if (XbfFmemberRef.isMatchHungry(target)) {
                $match$ = true;
            } else if (XbfFcoArrayRef.isMatchHungry(target)) {
                $match$ = true;
            } else if (XbfFdoLoop.isMatchHungry(target)) {
                $match$ = true;
            } else if (XbfFarrayRef.isMatchHungry(target)) {
                $match$ = true;
            } else if (XbfFintConstant.isMatchHungry(target)) {
                $match$ = true;
            } else if (XbfFrealConstant.isMatchHungry(target)) {
                $match$ = true;
            } else if (XbfFcharacterConstant.isMatchHungry(target)) {
                $match$ = true;
            } else if (XbfFlogicalConstant.isMatchHungry(target)) {
                $match$ = true;
            } else if (XbfVar.isMatchHungry(target)) {
                $match$ = true;
            } else if (XbfVarRef.isMatchHungry(target)) {
                $match$ = true;
            } else if (XbfFarrayConstructor.isMatchHungry(target)) {
                $match$ = true;
            } else if (XbfFcomplexConstant.isMatchHungry(target)) {
                $match$ = true;
            } else if (XbfFstructConstructor.isMatchHungry(target)) {
                $match$ = true;
            } else if (XbfUserBinaryExpr.isMatchHungry(target)) {
                $match$ = true;
            } else if (XbfUserUnaryExpr.isMatchHungry(target)) {
                $match$ = true;
            } else if (XbfPlusExpr.isMatchHungry(target)) {
                $match$ = true;
            } else if (XbfMulExpr.isMatchHungry(target)) {
                $match$ = true;
            } else if (XbfDivExpr.isMatchHungry(target)) {
                $match$ = true;
            } else if (XbfLogNEQExpr.isMatchHungry(target)) {
                $match$ = true;
            } else if (XbfLogGEExpr.isMatchHungry(target)) {
                $match$ = true;
            } else if (XbfLogOrExpr.isMatchHungry(target)) {
                $match$ = true;
            } else if (XbfLogEQVExpr.isMatchHungry(target)) {
                $match$ = true;
            } else if (XbfMinusExpr.isMatchHungry(target)) {
                $match$ = true;
            } else if (XbfFpowerExpr.isMatchHungry(target)) {
                $match$ = true;
            } else if (XbfFconcatExpr.isMatchHungry(target)) {
                $match$ = true;
            } else if (XbfLogEQExpr.isMatchHungry(target)) {
                $match$ = true;
            } else if (XbfLogGTExpr.isMatchHungry(target)) {
                $match$ = true;
            } else if (XbfLogLEExpr.isMatchHungry(target)) {
                $match$ = true;
            } else if (XbfLogLTExpr.isMatchHungry(target)) {
                $match$ = true;
            } else if (XbfLogAndExpr.isMatchHungry(target)) {
                $match$ = true;
            } else if (XbfLogNEQVExpr.isMatchHungry(target)) {
                $match$ = true;
            } else if (XbfUnaryMinusExpr.isMatchHungry(target)) {
                $match$ = true;
            } else if (XbfLogNotExpr.isMatchHungry(target)) {
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
     * is valid for the <code>XbfFstructConstructor</code>.
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
     * is valid for the <code>XbfFstructConstructor</code>.
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
