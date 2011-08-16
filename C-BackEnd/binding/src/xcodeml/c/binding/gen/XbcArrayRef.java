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
 * <b>XbcArrayRef</b> is generated from XcodeML_C.rng by Relaxer.
 * This class is derived from:
 * 
 * <!-- for programmer
 * <element java:extends="xcodeml.c.obj.XmcObj" java:implements="xcodeml.c.binding.IXbcTypedExpr" name="arrayRef">
 *   <ref name="BaseExpression"/>
 *   <ref name="varRefExpression"/>
 *   <ref name="arrayAddr"/>
 *   <ref name="arrayDimension"/>
 * </element>
 * -->
 * <!-- for javadoc -->
 * <pre> &lt;element java:extends="xcodeml.c.obj.XmcObj" java:implements="xcodeml.c.binding.IXbcTypedExpr" name="arrayRef"&gt;
 *   &lt;ref name="BaseExpression"/&gt;
 *   &lt;ref name="varRefExpression"/&gt;
 *   &lt;ref name="arrayAddr"/&gt;
 *   &lt;ref name="arrayDimension"/&gt;
 * &lt;/element&gt;
 * </pre>
 *
 * @version XcodeML_C.rng (Mon Aug 15 15:55:17 JST 2011)
 * @author  Relaxer 1.0 (http://www.relaxer.org)
 */
public class XbcArrayRef extends xcodeml.c.obj.XmcObj implements java.io.Serializable, Cloneable, xcodeml.c.binding.IXbcTypedExpr, IRVisitable, IRNode, IXbcBuiltinOpChoice, IXbcSubArrayDimensionChoice, IXbcValueChoice, IXbcCoArrayRefChoice1, IXbcCastExprChoice, IXbcExprOrTypeChoice, IXbcCompoundLiteralChoice, IXbcDesignatedValueChoice, IXbcGotoStatementChoice, IXbcExpressionsChoice {
    public static final String ISGCCSYNTAX_0 = "0";
    public static final String ISGCCSYNTAX_1 = "1";
    public static final String ISGCCSYNTAX_TRUE = "true";
    public static final String ISGCCSYNTAX_FALSE = "false";
    public static final String ISMODIFIED_0 = "0";
    public static final String ISMODIFIED_1 = "1";
    public static final String ISMODIFIED_TRUE = "true";
    public static final String ISMODIFIED_FALSE = "false";
    public static final String SCOPE_GLOBAL = "global";
    public static final String SCOPE_LOCAL = "local";
    public static final String SCOPE_PARAM = "param";

    private String type_;
    private String isGccSyntax_;
    private String isModified_;
    private String scope_;
    private XbcArrayAddr arrayAddr_;
    // List<IXbcExpressionsChoice>
    private java.util.List expressions_ = new java.util.ArrayList();
    private IRNode parentRNode_;

    /**
     * Creates a <code>XbcArrayRef</code>.
     *
     */
    public XbcArrayRef() {
    }

    /**
     * Creates a <code>XbcArrayRef</code>.
     *
     * @param source
     */
    public XbcArrayRef(XbcArrayRef source) {
        setup(source);
    }

    /**
     * Creates a <code>XbcArrayRef</code> by the Stack <code>stack</code>
     * that contains Elements.
     * This constructor is supposed to be used internally
     * by the Relaxer system.
     *
     * @param stack
     */
    public XbcArrayRef(RStack stack) {
        setup(stack);
    }

    /**
     * Creates a <code>XbcArrayRef</code> by the Document <code>doc</code>.
     *
     * @param doc
     */
    public XbcArrayRef(Document doc) {
        setup(doc.getDocumentElement());
    }

    /**
     * Creates a <code>XbcArrayRef</code> by the Element <code>element</code>.
     *
     * @param element
     */
    public XbcArrayRef(Element element) {
        setup(element);
    }

    /**
     * Creates a <code>XbcArrayRef</code> by the File <code>file</code>.
     *
     * @param file
     * @exception IOException
     * @exception SAXException
     * @exception ParserConfigurationException
     */
    public XbcArrayRef(File file) throws IOException, SAXException, ParserConfigurationException {
        setup(file);
    }

    /**
     * Creates a <code>XbcArrayRef</code>
     * by the String representation of URI <code>uri</code>.
     *
     * @param uri
     * @exception IOException
     * @exception SAXException
     * @exception ParserConfigurationException
     */
    public XbcArrayRef(String uri) throws IOException, SAXException, ParserConfigurationException {
        setup(uri);
    }

    /**
     * Creates a <code>XbcArrayRef</code> by the URL <code>url</code>.
     *
     * @param url
     * @exception IOException
     * @exception SAXException
     * @exception ParserConfigurationException
     */
    public XbcArrayRef(URL url) throws IOException, SAXException, ParserConfigurationException {
        setup(url);
    }

    /**
     * Creates a <code>XbcArrayRef</code> by the InputStream <code>in</code>.
     *
     * @param in
     * @exception IOException
     * @exception SAXException
     * @exception ParserConfigurationException
     */
    public XbcArrayRef(InputStream in) throws IOException, SAXException, ParserConfigurationException {
        setup(in);
    }

    /**
     * Creates a <code>XbcArrayRef</code> by the InputSource <code>is</code>.
     *
     * @param is
     * @exception IOException
     * @exception SAXException
     * @exception ParserConfigurationException
     */
    public XbcArrayRef(InputSource is) throws IOException, SAXException, ParserConfigurationException {
        setup(is);
    }

    /**
     * Creates a <code>XbcArrayRef</code> by the Reader <code>reader</code>.
     *
     * @param reader
     * @exception IOException
     * @exception SAXException
     * @exception ParserConfigurationException
     */
    public XbcArrayRef(Reader reader) throws IOException, SAXException, ParserConfigurationException {
        setup(reader);
    }

    /**
     * Initializes the <code>XbcArrayRef</code> by the XbcArrayRef <code>source</code>.
     *
     * @param source
     */
    public void setup(XbcArrayRef source) {
        int size;
        setType(source.getType());
        setIsGccSyntax(source.getIsGccSyntax());
        setIsModified(source.getIsModified());
        setScope(source.getScope());
        if (source.arrayAddr_ != null) {
            setArrayAddr((XbcArrayAddr)source.getArrayAddr().clone());
        }
        this.expressions_.clear();
        size = source.expressions_.size();
        for (int i = 0;i < size;i++) {
            addExpressions((IXbcExpressionsChoice)source.getExpressions(i).clone());
        }
    }

    /**
     * Initializes the <code>XbcArrayRef</code> by the Document <code>doc</code>.
     *
     * @param doc
     */
    public void setup(Document doc) {
        setup(doc.getDocumentElement());
    }

    /**
     * Initializes the <code>XbcArrayRef</code> by the Element <code>element</code>.
     *
     * @param element
     */
    public void setup(Element element) {
        init(element);
    }

    /**
     * Initializes the <code>XbcArrayRef</code> by the Stack <code>stack</code>
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
        type_ = URelaxer.getAttributePropertyAsString(element, "type");
        isGccSyntax_ = URelaxer.getAttributePropertyAsString(element, "is_gccSyntax");
        isModified_ = URelaxer.getAttributePropertyAsString(element, "is_modified");
        scope_ = URelaxer.getAttributePropertyAsString(element, "scope");
        setArrayAddr(factory.createXbcArrayAddr(stack));
        expressions_.clear();
        while (true) {
            if (XbcBuiltinOp.isMatch(stack)) {
                addExpressions(factory.createXbcBuiltinOp(stack));
            } else if (XbcSubArrayRef.isMatch(stack)) {
                addExpressions(factory.createXbcSubArrayRef(stack));
            } else if (XbcArrayRef.isMatch(stack)) {
                addExpressions(factory.createXbcArrayRef(stack));
            } else if (XbcFunctionCall.isMatch(stack)) {
                addExpressions(factory.createXbcFunctionCall(stack));
            } else if (XbcGccCompoundExpr.isMatch(stack)) {
                addExpressions(factory.createXbcGccCompoundExpr(stack));
            } else if (XbcCoArrayRef.isMatch(stack)) {
                addExpressions(factory.createXbcCoArrayRef(stack));
            } else if (XbcCastExpr.isMatch(stack)) {
                addExpressions(factory.createXbcCastExpr(stack));
            } else if (XbcStringConstant.isMatch(stack)) {
                addExpressions(factory.createXbcStringConstant(stack));
            } else if (XbcVar.isMatch(stack)) {
                addExpressions(factory.createXbcVar(stack));
            } else if (XbcVarAddr.isMatch(stack)) {
                addExpressions(factory.createXbcVarAddr(stack));
            } else if (XbcArrayAddr.isMatch(stack)) {
                addExpressions(factory.createXbcArrayAddr(stack));
            } else if (XbcCompoundValueExpr.isMatch(stack)) {
                addExpressions(factory.createXbcCompoundValueExpr(stack));
            } else if (XbcCompoundValueAddrExpr.isMatch(stack)) {
                addExpressions(factory.createXbcCompoundValueAddrExpr(stack));
            } else if (XbcSizeOfExpr.isMatch(stack)) {
                addExpressions(factory.createXbcSizeOfExpr(stack));
            } else if (XbcIntConstant.isMatch(stack)) {
                addExpressions(factory.createXbcIntConstant(stack));
            } else if (XbcFloatConstant.isMatch(stack)) {
                addExpressions(factory.createXbcFloatConstant(stack));
            } else if (XbcLonglongConstant.isMatch(stack)) {
                addExpressions(factory.createXbcLonglongConstant(stack));
            } else if (XbcMoeConstant.isMatch(stack)) {
                addExpressions(factory.createXbcMoeConstant(stack));
            } else if (XbcFuncAddr.isMatch(stack)) {
                addExpressions(factory.createXbcFuncAddr(stack));
            } else if (XbcGccAlignOfExpr.isMatch(stack)) {
                addExpressions(factory.createXbcGccAlignOfExpr(stack));
            } else if (XbcGccLabelAddr.isMatch(stack)) {
                addExpressions(factory.createXbcGccLabelAddr(stack));
            } else if (XbcCoArrayAssignExpr.isMatch(stack)) {
                addExpressions(factory.createXbcCoArrayAssignExpr(stack));
            } else if (XbcMemberAddr.isMatch(stack)) {
                addExpressions(factory.createXbcMemberAddr(stack));
            } else if (XbcMemberRef.isMatch(stack)) {
                addExpressions(factory.createXbcMemberRef(stack));
            } else if (XbcMemberArrayRef.isMatch(stack)) {
                addExpressions(factory.createXbcMemberArrayRef(stack));
            } else if (XbcMemberArrayAddr.isMatch(stack)) {
                addExpressions(factory.createXbcMemberArrayAddr(stack));
            } else if (XbcPointerRef.isMatch(stack)) {
                addExpressions(factory.createXbcPointerRef(stack));
            } else if (XbcAssignExpr.isMatch(stack)) {
                addExpressions(factory.createXbcAssignExpr(stack));
            } else if (XbcPlusExpr.isMatch(stack)) {
                addExpressions(factory.createXbcPlusExpr(stack));
            } else if (XbcMinusExpr.isMatch(stack)) {
                addExpressions(factory.createXbcMinusExpr(stack));
            } else if (XbcMulExpr.isMatch(stack)) {
                addExpressions(factory.createXbcMulExpr(stack));
            } else if (XbcDivExpr.isMatch(stack)) {
                addExpressions(factory.createXbcDivExpr(stack));
            } else if (XbcModExpr.isMatch(stack)) {
                addExpressions(factory.createXbcModExpr(stack));
            } else if (XbcLshiftExpr.isMatch(stack)) {
                addExpressions(factory.createXbcLshiftExpr(stack));
            } else if (XbcRshiftExpr.isMatch(stack)) {
                addExpressions(factory.createXbcRshiftExpr(stack));
            } else if (XbcBitAndExpr.isMatch(stack)) {
                addExpressions(factory.createXbcBitAndExpr(stack));
            } else if (XbcBitOrExpr.isMatch(stack)) {
                addExpressions(factory.createXbcBitOrExpr(stack));
            } else if (XbcBitXorExpr.isMatch(stack)) {
                addExpressions(factory.createXbcBitXorExpr(stack));
            } else if (XbcAsgPlusExpr.isMatch(stack)) {
                addExpressions(factory.createXbcAsgPlusExpr(stack));
            } else if (XbcAsgMinusExpr.isMatch(stack)) {
                addExpressions(factory.createXbcAsgMinusExpr(stack));
            } else if (XbcAsgMulExpr.isMatch(stack)) {
                addExpressions(factory.createXbcAsgMulExpr(stack));
            } else if (XbcAsgDivExpr.isMatch(stack)) {
                addExpressions(factory.createXbcAsgDivExpr(stack));
            } else if (XbcAsgModExpr.isMatch(stack)) {
                addExpressions(factory.createXbcAsgModExpr(stack));
            } else if (XbcAsgLshiftExpr.isMatch(stack)) {
                addExpressions(factory.createXbcAsgLshiftExpr(stack));
            } else if (XbcAsgRshiftExpr.isMatch(stack)) {
                addExpressions(factory.createXbcAsgRshiftExpr(stack));
            } else if (XbcAsgBitAndExpr.isMatch(stack)) {
                addExpressions(factory.createXbcAsgBitAndExpr(stack));
            } else if (XbcAsgBitOrExpr.isMatch(stack)) {
                addExpressions(factory.createXbcAsgBitOrExpr(stack));
            } else if (XbcAsgBitXorExpr.isMatch(stack)) {
                addExpressions(factory.createXbcAsgBitXorExpr(stack));
            } else if (XbcLogEQExpr.isMatch(stack)) {
                addExpressions(factory.createXbcLogEQExpr(stack));
            } else if (XbcLogNEQExpr.isMatch(stack)) {
                addExpressions(factory.createXbcLogNEQExpr(stack));
            } else if (XbcLogGEExpr.isMatch(stack)) {
                addExpressions(factory.createXbcLogGEExpr(stack));
            } else if (XbcLogGTExpr.isMatch(stack)) {
                addExpressions(factory.createXbcLogGTExpr(stack));
            } else if (XbcLogLEExpr.isMatch(stack)) {
                addExpressions(factory.createXbcLogLEExpr(stack));
            } else if (XbcLogLTExpr.isMatch(stack)) {
                addExpressions(factory.createXbcLogLTExpr(stack));
            } else if (XbcLogAndExpr.isMatch(stack)) {
                addExpressions(factory.createXbcLogAndExpr(stack));
            } else if (XbcLogOrExpr.isMatch(stack)) {
                addExpressions(factory.createXbcLogOrExpr(stack));
            } else if (XbcUnaryMinusExpr.isMatch(stack)) {
                addExpressions(factory.createXbcUnaryMinusExpr(stack));
            } else if (XbcBitNotExpr.isMatch(stack)) {
                addExpressions(factory.createXbcBitNotExpr(stack));
            } else if (XbcLogNotExpr.isMatch(stack)) {
                addExpressions(factory.createXbcLogNotExpr(stack));
            } else if (XbcCommaExpr.isMatch(stack)) {
                addExpressions(factory.createXbcCommaExpr(stack));
            } else if (XbcPostIncrExpr.isMatch(stack)) {
                addExpressions(factory.createXbcPostIncrExpr(stack));
            } else if (XbcPostDecrExpr.isMatch(stack)) {
                addExpressions(factory.createXbcPostDecrExpr(stack));
            } else if (XbcPreIncrExpr.isMatch(stack)) {
                addExpressions(factory.createXbcPreIncrExpr(stack));
            } else if (XbcPreDecrExpr.isMatch(stack)) {
                addExpressions(factory.createXbcPreDecrExpr(stack));
            } else if (XbcCondExpr.isMatch(stack)) {
                addExpressions(factory.createXbcCondExpr(stack));
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
        return (factory.createXbcArrayRef(this));
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
        Element element = doc.createElement("arrayRef");
        int size;
        if (this.type_ != null) {
            URelaxer.setAttributePropertyByString(element, "type", this.type_);
        }
        if (this.isGccSyntax_ != null) {
            URelaxer.setAttributePropertyByString(element, "is_gccSyntax", this.isGccSyntax_);
        }
        if (this.isModified_ != null) {
            URelaxer.setAttributePropertyByString(element, "is_modified", this.isModified_);
        }
        if (this.scope_ != null) {
            URelaxer.setAttributePropertyByString(element, "scope", this.scope_);
        }
        this.arrayAddr_.makeElement(element);
        size = this.expressions_.size();
        for (int i = 0;i < size;i++) {
            IXbcExpressionsChoice value = (IXbcExpressionsChoice)this.expressions_.get(i);
            value.makeElement(element);
        }
        parent.appendChild(element);
    }

    /**
     * Initializes the <code>XbcArrayRef</code> by the File <code>file</code>.
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
     * Initializes the <code>XbcArrayRef</code>
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
     * Initializes the <code>XbcArrayRef</code> by the URL <code>url</code>.
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
     * Initializes the <code>XbcArrayRef</code> by the InputStream <code>in</code>.
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
     * Initializes the <code>XbcArrayRef</code> by the InputSource <code>is</code>.
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
     * Initializes the <code>XbcArrayRef</code> by the Reader <code>reader</code>.
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
     * Gets the String property <b>isGccSyntax</b>.
     *
     * @return String
     */
    public final String getIsGccSyntax() {
        return (isGccSyntax_);
    }

    /**
     * Sets the String property <b>isGccSyntax</b>.
     *
     * @param isGccSyntax
     */
    public final void setIsGccSyntax(String isGccSyntax) {
        this.isGccSyntax_ = isGccSyntax;
    }

    /**
     * Gets the String property <b>isModified</b>.
     *
     * @return String
     */
    public final String getIsModified() {
        return (isModified_);
    }

    /**
     * Sets the String property <b>isModified</b>.
     *
     * @param isModified
     */
    public final void setIsModified(String isModified) {
        this.isModified_ = isModified;
    }

    /**
     * Gets the String property <b>scope</b>.
     *
     * @return String
     */
    public final String getScope() {
        return (scope_);
    }

    /**
     * Sets the String property <b>scope</b>.
     *
     * @param scope
     */
    public final void setScope(String scope) {
        this.scope_ = scope;
    }

    /**
     * Gets the XbcArrayAddr property <b>arrayAddr</b>.
     *
     * @return XbcArrayAddr
     */
    public final XbcArrayAddr getArrayAddr() {
        return (arrayAddr_);
    }

    /**
     * Sets the XbcArrayAddr property <b>arrayAddr</b>.
     *
     * @param arrayAddr
     */
    public final void setArrayAddr(XbcArrayAddr arrayAddr) {
        this.arrayAddr_ = arrayAddr;
        if (arrayAddr != null) {
            arrayAddr.rSetParentRNode(this);
        }
    }

    /**
     * Gets the IXbcExpressionsChoice property <b>expressions</b>.
     *
     * @return IXbcExpressionsChoice[]
     */
    public final IXbcExpressionsChoice[] getExpressions() {
        IXbcExpressionsChoice[] array = new IXbcExpressionsChoice[expressions_.size()];
        return ((IXbcExpressionsChoice[])expressions_.toArray(array));
    }

    /**
     * Sets the IXbcExpressionsChoice property <b>expressions</b>.
     *
     * @param expressions
     */
    public final void setExpressions(IXbcExpressionsChoice[] expressions) {
        this.expressions_.clear();
        for (int i = 0;i < expressions.length;i++) {
            addExpressions(expressions[i]);
        }
        for (int i = 0;i < expressions.length;i++) {
            expressions[i].rSetParentRNode(this);
        }
    }

    /**
     * Sets the IXbcExpressionsChoice property <b>expressions</b>.
     *
     * @param expressions
     */
    public final void setExpressions(IXbcExpressionsChoice expressions) {
        this.expressions_.clear();
        addExpressions(expressions);
        if (expressions != null) {
            expressions.rSetParentRNode(this);
        }
    }

    /**
     * Adds the IXbcExpressionsChoice property <b>expressions</b>.
     *
     * @param expressions
     */
    public final void addExpressions(IXbcExpressionsChoice expressions) {
        this.expressions_.add(expressions);
        if (expressions != null) {
            expressions.rSetParentRNode(this);
        }
    }

    /**
     * Adds the IXbcExpressionsChoice property <b>expressions</b>.
     *
     * @param expressions
     */
    public final void addExpressions(IXbcExpressionsChoice[] expressions) {
        for (int i = 0;i < expressions.length;i++) {
            addExpressions(expressions[i]);
        }
        for (int i = 0;i < expressions.length;i++) {
            expressions[i].rSetParentRNode(this);
        }
    }

    /**
     * Gets number of the IXbcExpressionsChoice property <b>expressions</b>.
     *
     * @return int
     */
    public final int sizeExpressions() {
        return (expressions_.size());
    }

    /**
     * Gets the IXbcExpressionsChoice property <b>expressions</b> by index.
     *
     * @param index
     * @return IXbcExpressionsChoice
     */
    public final IXbcExpressionsChoice getExpressions(int index) {
        return ((IXbcExpressionsChoice)expressions_.get(index));
    }

    /**
     * Sets the IXbcExpressionsChoice property <b>expressions</b> by index.
     *
     * @param index
     * @param expressions
     */
    public final void setExpressions(int index, IXbcExpressionsChoice expressions) {
        this.expressions_.set(index, expressions);
        if (expressions != null) {
            expressions.rSetParentRNode(this);
        }
    }

    /**
     * Adds the IXbcExpressionsChoice property <b>expressions</b> by index.
     *
     * @param index
     * @param expressions
     */
    public final void addExpressions(int index, IXbcExpressionsChoice expressions) {
        this.expressions_.add(index, expressions);
        if (expressions != null) {
            expressions.rSetParentRNode(this);
        }
    }

    /**
     * Remove the IXbcExpressionsChoice property <b>expressions</b> by index.
     *
     * @param index
     */
    public final void removeExpressions(int index) {
        this.expressions_.remove(index);
    }

    /**
     * Remove the IXbcExpressionsChoice property <b>expressions</b> by object.
     *
     * @param expressions
     */
    public final void removeExpressions(IXbcExpressionsChoice expressions) {
        this.expressions_.remove(expressions);
    }

    /**
     * Clear the IXbcExpressionsChoice property <b>expressions</b>.
     *
     */
    public final void clearExpressions() {
        this.expressions_.clear();
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
        buffer.append("<arrayRef");
        if (type_ != null) {
            buffer.append(" type=\"");
            buffer.append(URelaxer.escapeAttrQuot(URelaxer.getString(getType())));
            buffer.append("\"");
        }
        if (isGccSyntax_ != null) {
            buffer.append(" is_gccSyntax=\"");
            buffer.append(URelaxer.escapeAttrQuot(URelaxer.getString(getIsGccSyntax())));
            buffer.append("\"");
        }
        if (isModified_ != null) {
            buffer.append(" is_modified=\"");
            buffer.append(URelaxer.escapeAttrQuot(URelaxer.getString(getIsModified())));
            buffer.append("\"");
        }
        if (scope_ != null) {
            buffer.append(" scope=\"");
            buffer.append(URelaxer.escapeAttrQuot(URelaxer.getString(getScope())));
            buffer.append("\"");
        }
        size = this.expressions_.size();
        for (int i = 0;i < size;i++) {
            IXbcExpressionsChoice value = (IXbcExpressionsChoice)this.expressions_.get(i);
            value.makeTextAttribute(buffer);
        }
        buffer.append(">");
        arrayAddr_.makeTextElement(buffer);
        size = this.expressions_.size();
        for (int i = 0;i < size;i++) {
            IXbcExpressionsChoice value = (IXbcExpressionsChoice)this.expressions_.get(i);
            value.makeTextElement(buffer);
        }
        buffer.append("</arrayRef>");
    }

    /**
     * Makes an XML text representation.
     *
     * @param buffer
     * @exception IOException
     */
    public void makeTextElement(Writer buffer) throws IOException {
        int size;
        buffer.write("<arrayRef");
        if (type_ != null) {
            buffer.write(" type=\"");
            buffer.write(URelaxer.escapeAttrQuot(URelaxer.getString(getType())));
            buffer.write("\"");
        }
        if (isGccSyntax_ != null) {
            buffer.write(" is_gccSyntax=\"");
            buffer.write(URelaxer.escapeAttrQuot(URelaxer.getString(getIsGccSyntax())));
            buffer.write("\"");
        }
        if (isModified_ != null) {
            buffer.write(" is_modified=\"");
            buffer.write(URelaxer.escapeAttrQuot(URelaxer.getString(getIsModified())));
            buffer.write("\"");
        }
        if (scope_ != null) {
            buffer.write(" scope=\"");
            buffer.write(URelaxer.escapeAttrQuot(URelaxer.getString(getScope())));
            buffer.write("\"");
        }
        size = this.expressions_.size();
        for (int i = 0;i < size;i++) {
            IXbcExpressionsChoice value = (IXbcExpressionsChoice)this.expressions_.get(i);
            value.makeTextAttribute(buffer);
        }
        buffer.write(">");
        arrayAddr_.makeTextElement(buffer);
        size = this.expressions_.size();
        for (int i = 0;i < size;i++) {
            IXbcExpressionsChoice value = (IXbcExpressionsChoice)this.expressions_.get(i);
            value.makeTextElement(buffer);
        }
        buffer.write("</arrayRef>");
    }

    /**
     * Makes an XML text representation.
     *
     * @param buffer
     */
    public void makeTextElement(PrintWriter buffer) {
        int size;
        buffer.print("<arrayRef");
        if (type_ != null) {
            buffer.print(" type=\"");
            buffer.print(URelaxer.escapeAttrQuot(URelaxer.getString(getType())));
            buffer.print("\"");
        }
        if (isGccSyntax_ != null) {
            buffer.print(" is_gccSyntax=\"");
            buffer.print(URelaxer.escapeAttrQuot(URelaxer.getString(getIsGccSyntax())));
            buffer.print("\"");
        }
        if (isModified_ != null) {
            buffer.print(" is_modified=\"");
            buffer.print(URelaxer.escapeAttrQuot(URelaxer.getString(getIsModified())));
            buffer.print("\"");
        }
        if (scope_ != null) {
            buffer.print(" scope=\"");
            buffer.print(URelaxer.escapeAttrQuot(URelaxer.getString(getScope())));
            buffer.print("\"");
        }
        size = this.expressions_.size();
        for (int i = 0;i < size;i++) {
            IXbcExpressionsChoice value = (IXbcExpressionsChoice)this.expressions_.get(i);
            value.makeTextAttribute(buffer);
        }
        buffer.print(">");
        arrayAddr_.makeTextElement(buffer);
        size = this.expressions_.size();
        for (int i = 0;i < size;i++) {
            IXbcExpressionsChoice value = (IXbcExpressionsChoice)this.expressions_.get(i);
            value.makeTextElement(buffer);
        }
        buffer.print("</arrayRef>");
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
     * Gets the property value as String.
     *
     * @return String
     */
    public String getIsGccSyntaxAsString() {
        return (URelaxer.getString(getIsGccSyntax()));
    }

    /**
     * Gets the property value as String.
     *
     * @return String
     */
    public String getIsModifiedAsString() {
        return (URelaxer.getString(getIsModified()));
    }

    /**
     * Gets the property value as String.
     *
     * @return String
     */
    public String getScopeAsString() {
        return (URelaxer.getString(getScope()));
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
     * Sets the property value by String.
     *
     * @param string
     */
    public void setIsGccSyntaxByString(String string) {
        setIsGccSyntax(string);
    }

    /**
     * Sets the property value by String.
     *
     * @param string
     */
    public void setIsModifiedByString(String string) {
        setIsModified(string);
    }

    /**
     * Sets the property value by String.
     *
     * @param string
     */
    public void setScopeByString(String string) {
        setScope(string);
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
        if (arrayAddr_ != null) {
            classNodes.add(arrayAddr_);
        }
        classNodes.addAll(expressions_);
        IRNode[] nodes = new IRNode[classNodes.size()];
        return ((IRNode[])classNodes.toArray(nodes));
    }

    /**
     * Tests if a Element <code>element</code> is valid
     * for the <code>XbcArrayRef</code>.
     *
     * @param element
     * @return boolean
     */
    public static boolean isMatch(Element element) {
        if (!URelaxer.isTargetElement(element, "arrayRef")) {
            return (false);
        }
        RStack target = new RStack(element);
        boolean $match$ = false;
        Element child;
        if (!XbcArrayAddr.isMatchHungry(target)) {
            return (false);
        }
        $match$ = true;
        if (XbcBuiltinOp.isMatchHungry(target)) {
            $match$ = true;
        } else if (XbcSubArrayRef.isMatchHungry(target)) {
            $match$ = true;
        } else if (XbcArrayRef.isMatchHungry(target)) {
            $match$ = true;
        } else if (XbcFunctionCall.isMatchHungry(target)) {
            $match$ = true;
        } else if (XbcGccCompoundExpr.isMatchHungry(target)) {
            $match$ = true;
        } else if (XbcCoArrayRef.isMatchHungry(target)) {
            $match$ = true;
        } else if (XbcCastExpr.isMatchHungry(target)) {
            $match$ = true;
        } else if (XbcStringConstant.isMatchHungry(target)) {
            $match$ = true;
        } else if (XbcVar.isMatchHungry(target)) {
            $match$ = true;
        } else if (XbcVarAddr.isMatchHungry(target)) {
            $match$ = true;
        } else if (XbcArrayAddr.isMatchHungry(target)) {
            $match$ = true;
        } else if (XbcCompoundValueExpr.isMatchHungry(target)) {
            $match$ = true;
        } else if (XbcCompoundValueAddrExpr.isMatchHungry(target)) {
            $match$ = true;
        } else if (XbcSizeOfExpr.isMatchHungry(target)) {
            $match$ = true;
        } else if (XbcIntConstant.isMatchHungry(target)) {
            $match$ = true;
        } else if (XbcFloatConstant.isMatchHungry(target)) {
            $match$ = true;
        } else if (XbcLonglongConstant.isMatchHungry(target)) {
            $match$ = true;
        } else if (XbcMoeConstant.isMatchHungry(target)) {
            $match$ = true;
        } else if (XbcFuncAddr.isMatchHungry(target)) {
            $match$ = true;
        } else if (XbcGccAlignOfExpr.isMatchHungry(target)) {
            $match$ = true;
        } else if (XbcGccLabelAddr.isMatchHungry(target)) {
            $match$ = true;
        } else if (XbcCoArrayAssignExpr.isMatchHungry(target)) {
            $match$ = true;
        } else if (XbcMemberAddr.isMatchHungry(target)) {
            $match$ = true;
        } else if (XbcMemberRef.isMatchHungry(target)) {
            $match$ = true;
        } else if (XbcMemberArrayRef.isMatchHungry(target)) {
            $match$ = true;
        } else if (XbcMemberArrayAddr.isMatchHungry(target)) {
            $match$ = true;
        } else if (XbcPointerRef.isMatchHungry(target)) {
            $match$ = true;
        } else if (XbcAssignExpr.isMatchHungry(target)) {
            $match$ = true;
        } else if (XbcPlusExpr.isMatchHungry(target)) {
            $match$ = true;
        } else if (XbcMinusExpr.isMatchHungry(target)) {
            $match$ = true;
        } else if (XbcMulExpr.isMatchHungry(target)) {
            $match$ = true;
        } else if (XbcDivExpr.isMatchHungry(target)) {
            $match$ = true;
        } else if (XbcModExpr.isMatchHungry(target)) {
            $match$ = true;
        } else if (XbcLshiftExpr.isMatchHungry(target)) {
            $match$ = true;
        } else if (XbcRshiftExpr.isMatchHungry(target)) {
            $match$ = true;
        } else if (XbcBitAndExpr.isMatchHungry(target)) {
            $match$ = true;
        } else if (XbcBitOrExpr.isMatchHungry(target)) {
            $match$ = true;
        } else if (XbcBitXorExpr.isMatchHungry(target)) {
            $match$ = true;
        } else if (XbcAsgPlusExpr.isMatchHungry(target)) {
            $match$ = true;
        } else if (XbcAsgMinusExpr.isMatchHungry(target)) {
            $match$ = true;
        } else if (XbcAsgMulExpr.isMatchHungry(target)) {
            $match$ = true;
        } else if (XbcAsgDivExpr.isMatchHungry(target)) {
            $match$ = true;
        } else if (XbcAsgModExpr.isMatchHungry(target)) {
            $match$ = true;
        } else if (XbcAsgLshiftExpr.isMatchHungry(target)) {
            $match$ = true;
        } else if (XbcAsgRshiftExpr.isMatchHungry(target)) {
            $match$ = true;
        } else if (XbcAsgBitAndExpr.isMatchHungry(target)) {
            $match$ = true;
        } else if (XbcAsgBitOrExpr.isMatchHungry(target)) {
            $match$ = true;
        } else if (XbcAsgBitXorExpr.isMatchHungry(target)) {
            $match$ = true;
        } else if (XbcLogEQExpr.isMatchHungry(target)) {
            $match$ = true;
        } else if (XbcLogNEQExpr.isMatchHungry(target)) {
            $match$ = true;
        } else if (XbcLogGEExpr.isMatchHungry(target)) {
            $match$ = true;
        } else if (XbcLogGTExpr.isMatchHungry(target)) {
            $match$ = true;
        } else if (XbcLogLEExpr.isMatchHungry(target)) {
            $match$ = true;
        } else if (XbcLogLTExpr.isMatchHungry(target)) {
            $match$ = true;
        } else if (XbcLogAndExpr.isMatchHungry(target)) {
            $match$ = true;
        } else if (XbcLogOrExpr.isMatchHungry(target)) {
            $match$ = true;
        } else if (XbcUnaryMinusExpr.isMatchHungry(target)) {
            $match$ = true;
        } else if (XbcBitNotExpr.isMatchHungry(target)) {
            $match$ = true;
        } else if (XbcLogNotExpr.isMatchHungry(target)) {
            $match$ = true;
        } else if (XbcCommaExpr.isMatchHungry(target)) {
            $match$ = true;
        } else if (XbcPostIncrExpr.isMatchHungry(target)) {
            $match$ = true;
        } else if (XbcPostDecrExpr.isMatchHungry(target)) {
            $match$ = true;
        } else if (XbcPreIncrExpr.isMatchHungry(target)) {
            $match$ = true;
        } else if (XbcPreDecrExpr.isMatchHungry(target)) {
            $match$ = true;
        } else if (XbcCondExpr.isMatchHungry(target)) {
            $match$ = true;
        } else {
            return (false);
        }
        while (true) {
            if (XbcBuiltinOp.isMatchHungry(target)) {
                $match$ = true;
            } else if (XbcSubArrayRef.isMatchHungry(target)) {
                $match$ = true;
            } else if (XbcArrayRef.isMatchHungry(target)) {
                $match$ = true;
            } else if (XbcFunctionCall.isMatchHungry(target)) {
                $match$ = true;
            } else if (XbcGccCompoundExpr.isMatchHungry(target)) {
                $match$ = true;
            } else if (XbcCoArrayRef.isMatchHungry(target)) {
                $match$ = true;
            } else if (XbcCastExpr.isMatchHungry(target)) {
                $match$ = true;
            } else if (XbcStringConstant.isMatchHungry(target)) {
                $match$ = true;
            } else if (XbcVar.isMatchHungry(target)) {
                $match$ = true;
            } else if (XbcVarAddr.isMatchHungry(target)) {
                $match$ = true;
            } else if (XbcArrayAddr.isMatchHungry(target)) {
                $match$ = true;
            } else if (XbcCompoundValueExpr.isMatchHungry(target)) {
                $match$ = true;
            } else if (XbcCompoundValueAddrExpr.isMatchHungry(target)) {
                $match$ = true;
            } else if (XbcSizeOfExpr.isMatchHungry(target)) {
                $match$ = true;
            } else if (XbcIntConstant.isMatchHungry(target)) {
                $match$ = true;
            } else if (XbcFloatConstant.isMatchHungry(target)) {
                $match$ = true;
            } else if (XbcLonglongConstant.isMatchHungry(target)) {
                $match$ = true;
            } else if (XbcMoeConstant.isMatchHungry(target)) {
                $match$ = true;
            } else if (XbcFuncAddr.isMatchHungry(target)) {
                $match$ = true;
            } else if (XbcGccAlignOfExpr.isMatchHungry(target)) {
                $match$ = true;
            } else if (XbcGccLabelAddr.isMatchHungry(target)) {
                $match$ = true;
            } else if (XbcCoArrayAssignExpr.isMatchHungry(target)) {
                $match$ = true;
            } else if (XbcMemberAddr.isMatchHungry(target)) {
                $match$ = true;
            } else if (XbcMemberRef.isMatchHungry(target)) {
                $match$ = true;
            } else if (XbcMemberArrayRef.isMatchHungry(target)) {
                $match$ = true;
            } else if (XbcMemberArrayAddr.isMatchHungry(target)) {
                $match$ = true;
            } else if (XbcPointerRef.isMatchHungry(target)) {
                $match$ = true;
            } else if (XbcAssignExpr.isMatchHungry(target)) {
                $match$ = true;
            } else if (XbcPlusExpr.isMatchHungry(target)) {
                $match$ = true;
            } else if (XbcMinusExpr.isMatchHungry(target)) {
                $match$ = true;
            } else if (XbcMulExpr.isMatchHungry(target)) {
                $match$ = true;
            } else if (XbcDivExpr.isMatchHungry(target)) {
                $match$ = true;
            } else if (XbcModExpr.isMatchHungry(target)) {
                $match$ = true;
            } else if (XbcLshiftExpr.isMatchHungry(target)) {
                $match$ = true;
            } else if (XbcRshiftExpr.isMatchHungry(target)) {
                $match$ = true;
            } else if (XbcBitAndExpr.isMatchHungry(target)) {
                $match$ = true;
            } else if (XbcBitOrExpr.isMatchHungry(target)) {
                $match$ = true;
            } else if (XbcBitXorExpr.isMatchHungry(target)) {
                $match$ = true;
            } else if (XbcAsgPlusExpr.isMatchHungry(target)) {
                $match$ = true;
            } else if (XbcAsgMinusExpr.isMatchHungry(target)) {
                $match$ = true;
            } else if (XbcAsgMulExpr.isMatchHungry(target)) {
                $match$ = true;
            } else if (XbcAsgDivExpr.isMatchHungry(target)) {
                $match$ = true;
            } else if (XbcAsgModExpr.isMatchHungry(target)) {
                $match$ = true;
            } else if (XbcAsgLshiftExpr.isMatchHungry(target)) {
                $match$ = true;
            } else if (XbcAsgRshiftExpr.isMatchHungry(target)) {
                $match$ = true;
            } else if (XbcAsgBitAndExpr.isMatchHungry(target)) {
                $match$ = true;
            } else if (XbcAsgBitOrExpr.isMatchHungry(target)) {
                $match$ = true;
            } else if (XbcAsgBitXorExpr.isMatchHungry(target)) {
                $match$ = true;
            } else if (XbcLogEQExpr.isMatchHungry(target)) {
                $match$ = true;
            } else if (XbcLogNEQExpr.isMatchHungry(target)) {
                $match$ = true;
            } else if (XbcLogGEExpr.isMatchHungry(target)) {
                $match$ = true;
            } else if (XbcLogGTExpr.isMatchHungry(target)) {
                $match$ = true;
            } else if (XbcLogLEExpr.isMatchHungry(target)) {
                $match$ = true;
            } else if (XbcLogLTExpr.isMatchHungry(target)) {
                $match$ = true;
            } else if (XbcLogAndExpr.isMatchHungry(target)) {
                $match$ = true;
            } else if (XbcLogOrExpr.isMatchHungry(target)) {
                $match$ = true;
            } else if (XbcUnaryMinusExpr.isMatchHungry(target)) {
                $match$ = true;
            } else if (XbcBitNotExpr.isMatchHungry(target)) {
                $match$ = true;
            } else if (XbcLogNotExpr.isMatchHungry(target)) {
                $match$ = true;
            } else if (XbcCommaExpr.isMatchHungry(target)) {
                $match$ = true;
            } else if (XbcPostIncrExpr.isMatchHungry(target)) {
                $match$ = true;
            } else if (XbcPostDecrExpr.isMatchHungry(target)) {
                $match$ = true;
            } else if (XbcPreIncrExpr.isMatchHungry(target)) {
                $match$ = true;
            } else if (XbcPreDecrExpr.isMatchHungry(target)) {
                $match$ = true;
            } else if (XbcCondExpr.isMatchHungry(target)) {
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
     * is valid for the <code>XbcArrayRef</code>.
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
     * is valid for the <code>XbcArrayRef</code>.
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
