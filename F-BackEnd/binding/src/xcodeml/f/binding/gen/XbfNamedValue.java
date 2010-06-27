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
 * <b>XbfNamedValue</b> is generated from XcodeML_F.rng by Relaxer.
 * This class is derived from:
 * 
 * <!-- for programmer
 * <element java:extends="xcodeml.f.XmfObj" name="namedValue">
 *   <ref name="defAttrGenericName"/>
 *   <ref name="defAttrValue"/>
 *   <optional>
 *     <ref name="defModelExpr"/>
 *   </optional>
 * </element>
 * -->
 * <!-- for javadoc -->
 * <pre> &lt;element java:extends="xcodeml.f.XmfObj" name="namedValue"&gt;
 *   &lt;ref name="defAttrGenericName"/&gt;
 *   &lt;ref name="defAttrValue"/&gt;
 *   &lt;optional&gt;
 *     &lt;ref name="defModelExpr"/&gt;
 *   &lt;/optional&gt;
 * &lt;/element&gt;
 * </pre>
 *
 * @version XcodeML_F.rng (Fri Dec 04 19:18:15 JST 2009)
 * @author  Relaxer 1.0 (http://www.relaxer.org)
 */
public class XbfNamedValue extends xcodeml.f.XmfObj implements java.io.Serializable, Cloneable, IRVisitable, IRNode, IXbfArgumentsChoice {
    private String name_;
    private String value_;
    private IXbfDefModelExprChoice defModelExpr_;
    private IRNode parentRNode_;

    /**
     * Creates a <code>XbfNamedValue</code>.
     *
     */
    public XbfNamedValue() {
    }

    /**
     * Creates a <code>XbfNamedValue</code>.
     *
     * @param source
     */
    public XbfNamedValue(XbfNamedValue source) {
        setup(source);
    }

    /**
     * Creates a <code>XbfNamedValue</code> by the Stack <code>stack</code>
     * that contains Elements.
     * This constructor is supposed to be used internally
     * by the Relaxer system.
     *
     * @param stack
     */
    public XbfNamedValue(RStack stack) {
        setup(stack);
    }

    /**
     * Creates a <code>XbfNamedValue</code> by the Document <code>doc</code>.
     *
     * @param doc
     */
    public XbfNamedValue(Document doc) {
        setup(doc.getDocumentElement());
    }

    /**
     * Creates a <code>XbfNamedValue</code> by the Element <code>element</code>.
     *
     * @param element
     */
    public XbfNamedValue(Element element) {
        setup(element);
    }

    /**
     * Creates a <code>XbfNamedValue</code> by the File <code>file</code>.
     *
     * @param file
     * @exception IOException
     * @exception SAXException
     * @exception ParserConfigurationException
     */
    public XbfNamedValue(File file) throws IOException, SAXException, ParserConfigurationException {
        setup(file);
    }

    /**
     * Creates a <code>XbfNamedValue</code>
     * by the String representation of URI <code>uri</code>.
     *
     * @param uri
     * @exception IOException
     * @exception SAXException
     * @exception ParserConfigurationException
     */
    public XbfNamedValue(String uri) throws IOException, SAXException, ParserConfigurationException {
        setup(uri);
    }

    /**
     * Creates a <code>XbfNamedValue</code> by the URL <code>url</code>.
     *
     * @param url
     * @exception IOException
     * @exception SAXException
     * @exception ParserConfigurationException
     */
    public XbfNamedValue(URL url) throws IOException, SAXException, ParserConfigurationException {
        setup(url);
    }

    /**
     * Creates a <code>XbfNamedValue</code> by the InputStream <code>in</code>.
     *
     * @param in
     * @exception IOException
     * @exception SAXException
     * @exception ParserConfigurationException
     */
    public XbfNamedValue(InputStream in) throws IOException, SAXException, ParserConfigurationException {
        setup(in);
    }

    /**
     * Creates a <code>XbfNamedValue</code> by the InputSource <code>is</code>.
     *
     * @param is
     * @exception IOException
     * @exception SAXException
     * @exception ParserConfigurationException
     */
    public XbfNamedValue(InputSource is) throws IOException, SAXException, ParserConfigurationException {
        setup(is);
    }

    /**
     * Creates a <code>XbfNamedValue</code> by the Reader <code>reader</code>.
     *
     * @param reader
     * @exception IOException
     * @exception SAXException
     * @exception ParserConfigurationException
     */
    public XbfNamedValue(Reader reader) throws IOException, SAXException, ParserConfigurationException {
        setup(reader);
    }

    /**
     * Initializes the <code>XbfNamedValue</code> by the XbfNamedValue <code>source</code>.
     *
     * @param source
     */
    public void setup(XbfNamedValue source) {
        int size;
        setName(source.getName());
        setValue(source.getValue());
        if (source.defModelExpr_ != null) {
            setDefModelExpr((IXbfDefModelExprChoice)source.getDefModelExpr().clone());
        }
    }

    /**
     * Initializes the <code>XbfNamedValue</code> by the Document <code>doc</code>.
     *
     * @param doc
     */
    public void setup(Document doc) {
        setup(doc.getDocumentElement());
    }

    /**
     * Initializes the <code>XbfNamedValue</code> by the Element <code>element</code>.
     *
     * @param element
     */
    public void setup(Element element) {
        init(element);
    }

    /**
     * Initializes the <code>XbfNamedValue</code> by the Stack <code>stack</code>
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
        name_ = URelaxer.getAttributePropertyAsString(element, "name");
        value_ = URelaxer.getAttributePropertyAsString(element, "value");
        if (XbfUserUnaryExpr.isMatch(stack)) {
            setDefModelExpr(factory.createXbfUserUnaryExpr(stack));
        } else if (XbfFunctionCall.isMatch(stack)) {
            setDefModelExpr(factory.createXbfFunctionCall(stack));
        } else if (XbfFarrayRef.isMatch(stack)) {
            setDefModelExpr(factory.createXbfFarrayRef(stack));
        } else if (XbfFcharacterRef.isMatch(stack)) {
            setDefModelExpr(factory.createXbfFcharacterRef(stack));
        } else if (XbfFmemberRef.isMatch(stack)) {
            setDefModelExpr(factory.createXbfFmemberRef(stack));
        } else if (XbfFcharacterConstant.isMatch(stack)) {
            setDefModelExpr(factory.createXbfFcharacterConstant(stack));
        } else if (XbfVar.isMatch(stack)) {
            setDefModelExpr(factory.createXbfVar(stack));
        } else if (XbfVarRef.isMatch(stack)) {
            setDefModelExpr(factory.createXbfVarRef(stack));
        } else if (XbfPlusExpr.isMatch(stack)) {
            setDefModelExpr(factory.createXbfPlusExpr(stack));
        } else if (XbfMinusExpr.isMatch(stack)) {
            setDefModelExpr(factory.createXbfMinusExpr(stack));
        } else if (XbfLogEQExpr.isMatch(stack)) {
            setDefModelExpr(factory.createXbfLogEQExpr(stack));
        } else if (XbfLogLEExpr.isMatch(stack)) {
            setDefModelExpr(factory.createXbfLogLEExpr(stack));
        } else if (XbfFintConstant.isMatch(stack)) {
            setDefModelExpr(factory.createXbfFintConstant(stack));
        } else if (XbfFrealConstant.isMatch(stack)) {
            setDefModelExpr(factory.createXbfFrealConstant(stack));
        } else if (XbfFcomplexConstant.isMatch(stack)) {
            setDefModelExpr(factory.createXbfFcomplexConstant(stack));
        } else if (XbfFlogicalConstant.isMatch(stack)) {
            setDefModelExpr(factory.createXbfFlogicalConstant(stack));
        } else if (XbfFarrayConstructor.isMatch(stack)) {
            setDefModelExpr(factory.createXbfFarrayConstructor(stack));
        } else if (XbfFstructConstructor.isMatch(stack)) {
            setDefModelExpr(factory.createXbfFstructConstructor(stack));
        } else if (XbfMulExpr.isMatch(stack)) {
            setDefModelExpr(factory.createXbfMulExpr(stack));
        } else if (XbfDivExpr.isMatch(stack)) {
            setDefModelExpr(factory.createXbfDivExpr(stack));
        } else if (XbfFpowerExpr.isMatch(stack)) {
            setDefModelExpr(factory.createXbfFpowerExpr(stack));
        } else if (XbfFconcatExpr.isMatch(stack)) {
            setDefModelExpr(factory.createXbfFconcatExpr(stack));
        } else if (XbfLogNEQExpr.isMatch(stack)) {
            setDefModelExpr(factory.createXbfLogNEQExpr(stack));
        } else if (XbfLogGEExpr.isMatch(stack)) {
            setDefModelExpr(factory.createXbfLogGEExpr(stack));
        } else if (XbfLogGTExpr.isMatch(stack)) {
            setDefModelExpr(factory.createXbfLogGTExpr(stack));
        } else if (XbfLogLTExpr.isMatch(stack)) {
            setDefModelExpr(factory.createXbfLogLTExpr(stack));
        } else if (XbfLogAndExpr.isMatch(stack)) {
            setDefModelExpr(factory.createXbfLogAndExpr(stack));
        } else if (XbfLogOrExpr.isMatch(stack)) {
            setDefModelExpr(factory.createXbfLogOrExpr(stack));
        } else if (XbfLogEQVExpr.isMatch(stack)) {
            setDefModelExpr(factory.createXbfLogEQVExpr(stack));
        } else if (XbfLogNEQVExpr.isMatch(stack)) {
            setDefModelExpr(factory.createXbfLogNEQVExpr(stack));
        } else if (XbfUnaryMinusExpr.isMatch(stack)) {
            setDefModelExpr(factory.createXbfUnaryMinusExpr(stack));
        } else if (XbfLogNotExpr.isMatch(stack)) {
            setDefModelExpr(factory.createXbfLogNotExpr(stack));
        } else if (XbfUserBinaryExpr.isMatch(stack)) {
            setDefModelExpr(factory.createXbfUserBinaryExpr(stack));
        } else if (XbfFdoLoop.isMatch(stack)) {
            setDefModelExpr(factory.createXbfFdoLoop(stack));
        } else {
        }
    }

    /**
     * @return Object
     */
    public Object clone() {
        IXcodeML_FFactory factory = XcodeML_FFactory.getFactory();
        return (factory.createXbfNamedValue(this));
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
        Element element = doc.createElement("namedValue");
        int size;
        if (this.name_ != null) {
            URelaxer.setAttributePropertyByString(element, "name", this.name_);
        }
        if (this.value_ != null) {
            URelaxer.setAttributePropertyByString(element, "value", this.value_);
        }
        if (this.defModelExpr_ != null) {
            this.defModelExpr_.makeElement(element);
        }
        parent.appendChild(element);
    }

    /**
     * Initializes the <code>XbfNamedValue</code> by the File <code>file</code>.
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
     * Initializes the <code>XbfNamedValue</code>
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
     * Initializes the <code>XbfNamedValue</code> by the URL <code>url</code>.
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
     * Initializes the <code>XbfNamedValue</code> by the InputStream <code>in</code>.
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
     * Initializes the <code>XbfNamedValue</code> by the InputSource <code>is</code>.
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
     * Initializes the <code>XbfNamedValue</code> by the Reader <code>reader</code>.
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
     * Gets the String property <b>name</b>.
     *
     * @return String
     */
    public final String getName() {
        return (name_);
    }

    /**
     * Sets the String property <b>name</b>.
     *
     * @param name
     */
    public final void setName(String name) {
        this.name_ = name;
    }

    /**
     * Gets the String property <b>value</b>.
     *
     * @return String
     */
    public final String getValue() {
        return (value_);
    }

    /**
     * Sets the String property <b>value</b>.
     *
     * @param value
     */
    public final void setValue(String value) {
        this.value_ = value;
    }

    /**
     * Gets the IXbfDefModelExprChoice property <b>defModelExpr</b>.
     *
     * @return IXbfDefModelExprChoice
     */
    public final IXbfDefModelExprChoice getDefModelExpr() {
        return (defModelExpr_);
    }

    /**
     * Sets the IXbfDefModelExprChoice property <b>defModelExpr</b>.
     *
     * @param defModelExpr
     */
    public final void setDefModelExpr(IXbfDefModelExprChoice defModelExpr) {
        this.defModelExpr_ = defModelExpr;
        if (defModelExpr != null) {
            defModelExpr.rSetParentRNode(this);
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
        buffer.append("<namedValue");
        if (name_ != null) {
            buffer.append(" name=\"");
            buffer.append(URelaxer.escapeAttrQuot(URelaxer.getString(getName())));
            buffer.append("\"");
        }
        if (value_ != null) {
            buffer.append(" value=\"");
            buffer.append(URelaxer.escapeAttrQuot(URelaxer.getString(getValue())));
            buffer.append("\"");
        }
        if (defModelExpr_ != null) {
            defModelExpr_.makeTextAttribute(buffer);
        }
        buffer.append(">");
        if (defModelExpr_ != null) {
            defModelExpr_.makeTextElement(buffer);
        }
        buffer.append("</namedValue>");
    }

    /**
     * Makes an XML text representation.
     *
     * @param buffer
     * @exception IOException
     */
    public void makeTextElement(Writer buffer) throws IOException {
        int size;
        buffer.write("<namedValue");
        if (name_ != null) {
            buffer.write(" name=\"");
            buffer.write(URelaxer.escapeAttrQuot(URelaxer.getString(getName())));
            buffer.write("\"");
        }
        if (value_ != null) {
            buffer.write(" value=\"");
            buffer.write(URelaxer.escapeAttrQuot(URelaxer.getString(getValue())));
            buffer.write("\"");
        }
        if (defModelExpr_ != null) {
            defModelExpr_.makeTextAttribute(buffer);
        }
        buffer.write(">");
        if (defModelExpr_ != null) {
            defModelExpr_.makeTextElement(buffer);
        }
        buffer.write("</namedValue>");
    }

    /**
     * Makes an XML text representation.
     *
     * @param buffer
     */
    public void makeTextElement(PrintWriter buffer) {
        int size;
        buffer.print("<namedValue");
        if (name_ != null) {
            buffer.print(" name=\"");
            buffer.print(URelaxer.escapeAttrQuot(URelaxer.getString(getName())));
            buffer.print("\"");
        }
        if (value_ != null) {
            buffer.print(" value=\"");
            buffer.print(URelaxer.escapeAttrQuot(URelaxer.getString(getValue())));
            buffer.print("\"");
        }
        if (defModelExpr_ != null) {
            defModelExpr_.makeTextAttribute(buffer);
        }
        buffer.print(">");
        if (defModelExpr_ != null) {
            defModelExpr_.makeTextElement(buffer);
        }
        buffer.print("</namedValue>");
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
    public String getNameAsString() {
        return (URelaxer.getString(getName()));
    }

    /**
     * Gets the property value as String.
     *
     * @return String
     */
    public String getValueAsString() {
        return (URelaxer.getString(getValue()));
    }

    /**
     * Sets the property value by String.
     *
     * @param string
     */
    public void setNameByString(String string) {
        setName(string);
    }

    /**
     * Sets the property value by String.
     *
     * @param string
     */
    public void setValueByString(String string) {
        setValue(string);
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
        if (defModelExpr_ != null) {
            classNodes.add(defModelExpr_);
        }
        IRNode[] nodes = new IRNode[classNodes.size()];
        return ((IRNode[])classNodes.toArray(nodes));
    }

    /**
     * Tests if a Element <code>element</code> is valid
     * for the <code>XbfNamedValue</code>.
     *
     * @param element
     * @return boolean
     */
    public static boolean isMatch(Element element) {
        if (!URelaxer.isTargetElement(element, "namedValue")) {
            return (false);
        }
        RStack target = new RStack(element);
        boolean $match$ = false;
        Element child;
        if (XbfUserUnaryExpr.isMatchHungry(target)) {
            $match$ = true;
        } else if (XbfFunctionCall.isMatchHungry(target)) {
            $match$ = true;
        } else if (XbfFarrayRef.isMatchHungry(target)) {
            $match$ = true;
        } else if (XbfFcharacterRef.isMatchHungry(target)) {
            $match$ = true;
        } else if (XbfFmemberRef.isMatchHungry(target)) {
            $match$ = true;
        } else if (XbfFcharacterConstant.isMatchHungry(target)) {
            $match$ = true;
        } else if (XbfVar.isMatchHungry(target)) {
            $match$ = true;
        } else if (XbfVarRef.isMatchHungry(target)) {
            $match$ = true;
        } else if (XbfPlusExpr.isMatchHungry(target)) {
            $match$ = true;
        } else if (XbfMinusExpr.isMatchHungry(target)) {
            $match$ = true;
        } else if (XbfLogEQExpr.isMatchHungry(target)) {
            $match$ = true;
        } else if (XbfLogLEExpr.isMatchHungry(target)) {
            $match$ = true;
        } else if (XbfFintConstant.isMatchHungry(target)) {
            $match$ = true;
        } else if (XbfFrealConstant.isMatchHungry(target)) {
            $match$ = true;
        } else if (XbfFcomplexConstant.isMatchHungry(target)) {
            $match$ = true;
        } else if (XbfFlogicalConstant.isMatchHungry(target)) {
            $match$ = true;
        } else if (XbfFarrayConstructor.isMatchHungry(target)) {
            $match$ = true;
        } else if (XbfFstructConstructor.isMatchHungry(target)) {
            $match$ = true;
        } else if (XbfMulExpr.isMatchHungry(target)) {
            $match$ = true;
        } else if (XbfDivExpr.isMatchHungry(target)) {
            $match$ = true;
        } else if (XbfFpowerExpr.isMatchHungry(target)) {
            $match$ = true;
        } else if (XbfFconcatExpr.isMatchHungry(target)) {
            $match$ = true;
        } else if (XbfLogNEQExpr.isMatchHungry(target)) {
            $match$ = true;
        } else if (XbfLogGEExpr.isMatchHungry(target)) {
            $match$ = true;
        } else if (XbfLogGTExpr.isMatchHungry(target)) {
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
        } else if (XbfUnaryMinusExpr.isMatchHungry(target)) {
            $match$ = true;
        } else if (XbfLogNotExpr.isMatchHungry(target)) {
            $match$ = true;
        } else if (XbfUserBinaryExpr.isMatchHungry(target)) {
            $match$ = true;
        } else if (XbfFdoLoop.isMatchHungry(target)) {
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
     * is valid for the <code>XbfNamedValue</code>.
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
     * is valid for the <code>XbfNamedValue</code>.
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
