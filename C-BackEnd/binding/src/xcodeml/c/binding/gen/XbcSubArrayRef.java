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
 * <b>XbcSubArrayRef</b> is generated from XcodeML_C.rng by Relaxer.
 * This class is derived from:
 * 
 * <!-- for programmer
 * <element java:extends="xcodeml.c.obj.XmcObj" java:implements="xcodeml.c.binding.IXbcTypedExpr" name="subArrayRef">
 *   <ref name="BaseExpression"/>
 *   <ref name="varRefExpression"/>
 *   <ref name="arrayAddr"/>
 *   <ref name="subArrayDimension"/>
 * </element>
 * -->
 * <!-- for javadoc -->
 * <pre> &lt;element java:extends="xcodeml.c.obj.XmcObj" java:implements="xcodeml.c.binding.IXbcTypedExpr" name="subArrayRef"&gt;
 *   &lt;ref name="BaseExpression"/&gt;
 *   &lt;ref name="varRefExpression"/&gt;
 *   &lt;ref name="arrayAddr"/&gt;
 *   &lt;ref name="subArrayDimension"/&gt;
 * &lt;/element&gt;
 * </pre>
 *
 * @version XcodeML_C.rng (Thu Feb 02 16:55:18 JST 2012)
 * @author  Relaxer 1.0 (http://www.relaxer.org)
 */
public class XbcSubArrayRef extends xcodeml.c.obj.XmcObj implements java.io.Serializable, Cloneable, IXbcSubArrayDimensionChoice, IXbcCoArrayRefChoice1, IXbcCastExprChoice, IXbcExprOrTypeChoice, xcodeml.c.binding.IXbcTypedExpr, IRVisitable, IRNode, IXbcValueChoice, IXbcDesignatedValueChoice, IXbcCompoundLiteralChoice, IXbcGotoStatementChoice, IXbcBuiltinOpChoice, IXbcExpressionsChoice {
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
    // List<IXbcSubArrayDimensionChoice>
    private java.util.List subArrayDimension_ = new java.util.ArrayList();
    private IRNode parentRNode_;

    /**
     * Creates a <code>XbcSubArrayRef</code>.
     *
     */
    public XbcSubArrayRef() {
    }

    /**
     * Creates a <code>XbcSubArrayRef</code>.
     *
     * @param source
     */
    public XbcSubArrayRef(XbcSubArrayRef source) {
        setup(source);
    }

    /**
     * Creates a <code>XbcSubArrayRef</code> by the Stack <code>stack</code>
     * that contains Elements.
     * This constructor is supposed to be used internally
     * by the Relaxer system.
     *
     * @param stack
     */
    public XbcSubArrayRef(RStack stack) {
        setup(stack);
    }

    /**
     * Creates a <code>XbcSubArrayRef</code> by the Document <code>doc</code>.
     *
     * @param doc
     */
    public XbcSubArrayRef(Document doc) {
        setup(doc.getDocumentElement());
    }

    /**
     * Creates a <code>XbcSubArrayRef</code> by the Element <code>element</code>.
     *
     * @param element
     */
    public XbcSubArrayRef(Element element) {
        setup(element);
    }

    /**
     * Creates a <code>XbcSubArrayRef</code> by the File <code>file</code>.
     *
     * @param file
     * @exception IOException
     * @exception SAXException
     * @exception ParserConfigurationException
     */
    public XbcSubArrayRef(File file) throws IOException, SAXException, ParserConfigurationException {
        setup(file);
    }

    /**
     * Creates a <code>XbcSubArrayRef</code>
     * by the String representation of URI <code>uri</code>.
     *
     * @param uri
     * @exception IOException
     * @exception SAXException
     * @exception ParserConfigurationException
     */
    public XbcSubArrayRef(String uri) throws IOException, SAXException, ParserConfigurationException {
        setup(uri);
    }

    /**
     * Creates a <code>XbcSubArrayRef</code> by the URL <code>url</code>.
     *
     * @param url
     * @exception IOException
     * @exception SAXException
     * @exception ParserConfigurationException
     */
    public XbcSubArrayRef(URL url) throws IOException, SAXException, ParserConfigurationException {
        setup(url);
    }

    /**
     * Creates a <code>XbcSubArrayRef</code> by the InputStream <code>in</code>.
     *
     * @param in
     * @exception IOException
     * @exception SAXException
     * @exception ParserConfigurationException
     */
    public XbcSubArrayRef(InputStream in) throws IOException, SAXException, ParserConfigurationException {
        setup(in);
    }

    /**
     * Creates a <code>XbcSubArrayRef</code> by the InputSource <code>is</code>.
     *
     * @param is
     * @exception IOException
     * @exception SAXException
     * @exception ParserConfigurationException
     */
    public XbcSubArrayRef(InputSource is) throws IOException, SAXException, ParserConfigurationException {
        setup(is);
    }

    /**
     * Creates a <code>XbcSubArrayRef</code> by the Reader <code>reader</code>.
     *
     * @param reader
     * @exception IOException
     * @exception SAXException
     * @exception ParserConfigurationException
     */
    public XbcSubArrayRef(Reader reader) throws IOException, SAXException, ParserConfigurationException {
        setup(reader);
    }

    /**
     * Initializes the <code>XbcSubArrayRef</code> by the XbcSubArrayRef <code>source</code>.
     *
     * @param source
     */
    public void setup(XbcSubArrayRef source) {
        int size;
        setType(source.getType());
        setIsGccSyntax(source.getIsGccSyntax());
        setIsModified(source.getIsModified());
        setScope(source.getScope());
        if (source.arrayAddr_ != null) {
            setArrayAddr((XbcArrayAddr)source.getArrayAddr().clone());
        }
        this.subArrayDimension_.clear();
        size = source.subArrayDimension_.size();
        for (int i = 0;i < size;i++) {
            addSubArrayDimension((IXbcSubArrayDimensionChoice)source.getSubArrayDimension(i).clone());
        }
    }

    /**
     * Initializes the <code>XbcSubArrayRef</code> by the Document <code>doc</code>.
     *
     * @param doc
     */
    public void setup(Document doc) {
        setup(doc.getDocumentElement());
    }

    /**
     * Initializes the <code>XbcSubArrayRef</code> by the Element <code>element</code>.
     *
     * @param element
     */
    public void setup(Element element) {
        init(element);
    }

    /**
     * Initializes the <code>XbcSubArrayRef</code> by the Stack <code>stack</code>
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
        subArrayDimension_.clear();
        while (true) {
            if (XbcBuiltinOp.isMatch(stack)) {
                addSubArrayDimension(factory.createXbcBuiltinOp(stack));
            } else if (XbcArrayRef.isMatch(stack)) {
                addSubArrayDimension(factory.createXbcArrayRef(stack));
            } else if (XbcFunctionCall.isMatch(stack)) {
                addSubArrayDimension(factory.createXbcFunctionCall(stack));
            } else if (XbcGccCompoundExpr.isMatch(stack)) {
                addSubArrayDimension(factory.createXbcGccCompoundExpr(stack));
            } else if (XbcSubArrayRef.isMatch(stack)) {
                addSubArrayDimension(factory.createXbcSubArrayRef(stack));
            } else if (XbcCoArrayRef.isMatch(stack)) {
                addSubArrayDimension(factory.createXbcCoArrayRef(stack));
            } else if (XbcCastExpr.isMatch(stack)) {
                addSubArrayDimension(factory.createXbcCastExpr(stack));
            } else if (XbcStringConstant.isMatch(stack)) {
                addSubArrayDimension(factory.createXbcStringConstant(stack));
            } else if (XbcVar.isMatch(stack)) {
                addSubArrayDimension(factory.createXbcVar(stack));
            } else if (XbcVarAddr.isMatch(stack)) {
                addSubArrayDimension(factory.createXbcVarAddr(stack));
            } else if (XbcArrayAddr.isMatch(stack)) {
                addSubArrayDimension(factory.createXbcArrayAddr(stack));
            } else if (XbcCompoundValueExpr.isMatch(stack)) {
                addSubArrayDimension(factory.createXbcCompoundValueExpr(stack));
            } else if (XbcCompoundValueAddrExpr.isMatch(stack)) {
                addSubArrayDimension(factory.createXbcCompoundValueAddrExpr(stack));
            } else if (XbcXmpDescOf.isMatch(stack)) {
                addSubArrayDimension(factory.createXbcXmpDescOf(stack));
            } else if (XbcCoArrayAssignExpr.isMatch(stack)) {
                addSubArrayDimension(factory.createXbcCoArrayAssignExpr(stack));
            } else if (XbcIndexRange.isMatch(stack)) {
                addSubArrayDimension(factory.createXbcIndexRange(stack));
            } else if (XbcIntConstant.isMatch(stack)) {
                addSubArrayDimension(factory.createXbcIntConstant(stack));
            } else if (XbcFloatConstant.isMatch(stack)) {
                addSubArrayDimension(factory.createXbcFloatConstant(stack));
            } else if (XbcLonglongConstant.isMatch(stack)) {
                addSubArrayDimension(factory.createXbcLonglongConstant(stack));
            } else if (XbcMoeConstant.isMatch(stack)) {
                addSubArrayDimension(factory.createXbcMoeConstant(stack));
            } else if (XbcFuncAddr.isMatch(stack)) {
                addSubArrayDimension(factory.createXbcFuncAddr(stack));
            } else if (XbcSizeOfExpr.isMatch(stack)) {
                addSubArrayDimension(factory.createXbcSizeOfExpr(stack));
            } else if (XbcAddrOfExpr.isMatch(stack)) {
                addSubArrayDimension(factory.createXbcAddrOfExpr(stack));
            } else if (XbcGccAlignOfExpr.isMatch(stack)) {
                addSubArrayDimension(factory.createXbcGccAlignOfExpr(stack));
            } else if (XbcGccLabelAddr.isMatch(stack)) {
                addSubArrayDimension(factory.createXbcGccLabelAddr(stack));
            } else if (XbcMemberAddr.isMatch(stack)) {
                addSubArrayDimension(factory.createXbcMemberAddr(stack));
            } else if (XbcMemberRef.isMatch(stack)) {
                addSubArrayDimension(factory.createXbcMemberRef(stack));
            } else if (XbcMemberArrayRef.isMatch(stack)) {
                addSubArrayDimension(factory.createXbcMemberArrayRef(stack));
            } else if (XbcMemberArrayAddr.isMatch(stack)) {
                addSubArrayDimension(factory.createXbcMemberArrayAddr(stack));
            } else if (XbcPointerRef.isMatch(stack)) {
                addSubArrayDimension(factory.createXbcPointerRef(stack));
            } else if (XbcAssignExpr.isMatch(stack)) {
                addSubArrayDimension(factory.createXbcAssignExpr(stack));
            } else if (XbcPlusExpr.isMatch(stack)) {
                addSubArrayDimension(factory.createXbcPlusExpr(stack));
            } else if (XbcMinusExpr.isMatch(stack)) {
                addSubArrayDimension(factory.createXbcMinusExpr(stack));
            } else if (XbcMulExpr.isMatch(stack)) {
                addSubArrayDimension(factory.createXbcMulExpr(stack));
            } else if (XbcDivExpr.isMatch(stack)) {
                addSubArrayDimension(factory.createXbcDivExpr(stack));
            } else if (XbcModExpr.isMatch(stack)) {
                addSubArrayDimension(factory.createXbcModExpr(stack));
            } else if (XbcLshiftExpr.isMatch(stack)) {
                addSubArrayDimension(factory.createXbcLshiftExpr(stack));
            } else if (XbcRshiftExpr.isMatch(stack)) {
                addSubArrayDimension(factory.createXbcRshiftExpr(stack));
            } else if (XbcBitAndExpr.isMatch(stack)) {
                addSubArrayDimension(factory.createXbcBitAndExpr(stack));
            } else if (XbcBitOrExpr.isMatch(stack)) {
                addSubArrayDimension(factory.createXbcBitOrExpr(stack));
            } else if (XbcBitXorExpr.isMatch(stack)) {
                addSubArrayDimension(factory.createXbcBitXorExpr(stack));
            } else if (XbcAsgPlusExpr.isMatch(stack)) {
                addSubArrayDimension(factory.createXbcAsgPlusExpr(stack));
            } else if (XbcAsgMinusExpr.isMatch(stack)) {
                addSubArrayDimension(factory.createXbcAsgMinusExpr(stack));
            } else if (XbcAsgMulExpr.isMatch(stack)) {
                addSubArrayDimension(factory.createXbcAsgMulExpr(stack));
            } else if (XbcAsgDivExpr.isMatch(stack)) {
                addSubArrayDimension(factory.createXbcAsgDivExpr(stack));
            } else if (XbcAsgModExpr.isMatch(stack)) {
                addSubArrayDimension(factory.createXbcAsgModExpr(stack));
            } else if (XbcAsgLshiftExpr.isMatch(stack)) {
                addSubArrayDimension(factory.createXbcAsgLshiftExpr(stack));
            } else if (XbcAsgRshiftExpr.isMatch(stack)) {
                addSubArrayDimension(factory.createXbcAsgRshiftExpr(stack));
            } else if (XbcAsgBitAndExpr.isMatch(stack)) {
                addSubArrayDimension(factory.createXbcAsgBitAndExpr(stack));
            } else if (XbcAsgBitOrExpr.isMatch(stack)) {
                addSubArrayDimension(factory.createXbcAsgBitOrExpr(stack));
            } else if (XbcAsgBitXorExpr.isMatch(stack)) {
                addSubArrayDimension(factory.createXbcAsgBitXorExpr(stack));
            } else if (XbcLogEQExpr.isMatch(stack)) {
                addSubArrayDimension(factory.createXbcLogEQExpr(stack));
            } else if (XbcLogNEQExpr.isMatch(stack)) {
                addSubArrayDimension(factory.createXbcLogNEQExpr(stack));
            } else if (XbcLogGEExpr.isMatch(stack)) {
                addSubArrayDimension(factory.createXbcLogGEExpr(stack));
            } else if (XbcLogGTExpr.isMatch(stack)) {
                addSubArrayDimension(factory.createXbcLogGTExpr(stack));
            } else if (XbcLogLEExpr.isMatch(stack)) {
                addSubArrayDimension(factory.createXbcLogLEExpr(stack));
            } else if (XbcLogLTExpr.isMatch(stack)) {
                addSubArrayDimension(factory.createXbcLogLTExpr(stack));
            } else if (XbcLogAndExpr.isMatch(stack)) {
                addSubArrayDimension(factory.createXbcLogAndExpr(stack));
            } else if (XbcLogOrExpr.isMatch(stack)) {
                addSubArrayDimension(factory.createXbcLogOrExpr(stack));
            } else if (XbcUnaryMinusExpr.isMatch(stack)) {
                addSubArrayDimension(factory.createXbcUnaryMinusExpr(stack));
            } else if (XbcBitNotExpr.isMatch(stack)) {
                addSubArrayDimension(factory.createXbcBitNotExpr(stack));
            } else if (XbcLogNotExpr.isMatch(stack)) {
                addSubArrayDimension(factory.createXbcLogNotExpr(stack));
            } else if (XbcCommaExpr.isMatch(stack)) {
                addSubArrayDimension(factory.createXbcCommaExpr(stack));
            } else if (XbcPostIncrExpr.isMatch(stack)) {
                addSubArrayDimension(factory.createXbcPostIncrExpr(stack));
            } else if (XbcPostDecrExpr.isMatch(stack)) {
                addSubArrayDimension(factory.createXbcPostDecrExpr(stack));
            } else if (XbcPreIncrExpr.isMatch(stack)) {
                addSubArrayDimension(factory.createXbcPreIncrExpr(stack));
            } else if (XbcPreDecrExpr.isMatch(stack)) {
                addSubArrayDimension(factory.createXbcPreDecrExpr(stack));
            } else if (XbcCondExpr.isMatch(stack)) {
                addSubArrayDimension(factory.createXbcCondExpr(stack));
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
        return (factory.createXbcSubArrayRef(this));
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
        Element element = doc.createElement("subArrayRef");
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
        size = this.subArrayDimension_.size();
        for (int i = 0;i < size;i++) {
            IXbcSubArrayDimensionChoice value = (IXbcSubArrayDimensionChoice)this.subArrayDimension_.get(i);
            value.makeElement(element);
        }
        parent.appendChild(element);
    }

    /**
     * Initializes the <code>XbcSubArrayRef</code> by the File <code>file</code>.
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
     * Initializes the <code>XbcSubArrayRef</code>
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
     * Initializes the <code>XbcSubArrayRef</code> by the URL <code>url</code>.
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
     * Initializes the <code>XbcSubArrayRef</code> by the InputStream <code>in</code>.
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
     * Initializes the <code>XbcSubArrayRef</code> by the InputSource <code>is</code>.
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
     * Initializes the <code>XbcSubArrayRef</code> by the Reader <code>reader</code>.
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
     * Gets the IXbcSubArrayDimensionChoice property <b>subArrayDimension</b>.
     *
     * @return IXbcSubArrayDimensionChoice[]
     */
    public final IXbcSubArrayDimensionChoice[] getSubArrayDimension() {
        IXbcSubArrayDimensionChoice[] array = new IXbcSubArrayDimensionChoice[subArrayDimension_.size()];
        return ((IXbcSubArrayDimensionChoice[])subArrayDimension_.toArray(array));
    }

    /**
     * Sets the IXbcSubArrayDimensionChoice property <b>subArrayDimension</b>.
     *
     * @param subArrayDimension
     */
    public final void setSubArrayDimension(IXbcSubArrayDimensionChoice[] subArrayDimension) {
        this.subArrayDimension_.clear();
        for (int i = 0;i < subArrayDimension.length;i++) {
            addSubArrayDimension(subArrayDimension[i]);
        }
        for (int i = 0;i < subArrayDimension.length;i++) {
            subArrayDimension[i].rSetParentRNode(this);
        }
    }

    /**
     * Sets the IXbcSubArrayDimensionChoice property <b>subArrayDimension</b>.
     *
     * @param subArrayDimension
     */
    public final void setSubArrayDimension(IXbcSubArrayDimensionChoice subArrayDimension) {
        this.subArrayDimension_.clear();
        addSubArrayDimension(subArrayDimension);
        if (subArrayDimension != null) {
            subArrayDimension.rSetParentRNode(this);
        }
    }

    /**
     * Adds the IXbcSubArrayDimensionChoice property <b>subArrayDimension</b>.
     *
     * @param subArrayDimension
     */
    public final void addSubArrayDimension(IXbcSubArrayDimensionChoice subArrayDimension) {
        this.subArrayDimension_.add(subArrayDimension);
        if (subArrayDimension != null) {
            subArrayDimension.rSetParentRNode(this);
        }
    }

    /**
     * Adds the IXbcSubArrayDimensionChoice property <b>subArrayDimension</b>.
     *
     * @param subArrayDimension
     */
    public final void addSubArrayDimension(IXbcSubArrayDimensionChoice[] subArrayDimension) {
        for (int i = 0;i < subArrayDimension.length;i++) {
            addSubArrayDimension(subArrayDimension[i]);
        }
        for (int i = 0;i < subArrayDimension.length;i++) {
            subArrayDimension[i].rSetParentRNode(this);
        }
    }

    /**
     * Gets number of the IXbcSubArrayDimensionChoice property <b>subArrayDimension</b>.
     *
     * @return int
     */
    public final int sizeSubArrayDimension() {
        return (subArrayDimension_.size());
    }

    /**
     * Gets the IXbcSubArrayDimensionChoice property <b>subArrayDimension</b> by index.
     *
     * @param index
     * @return IXbcSubArrayDimensionChoice
     */
    public final IXbcSubArrayDimensionChoice getSubArrayDimension(int index) {
        return ((IXbcSubArrayDimensionChoice)subArrayDimension_.get(index));
    }

    /**
     * Sets the IXbcSubArrayDimensionChoice property <b>subArrayDimension</b> by index.
     *
     * @param index
     * @param subArrayDimension
     */
    public final void setSubArrayDimension(int index, IXbcSubArrayDimensionChoice subArrayDimension) {
        this.subArrayDimension_.set(index, subArrayDimension);
        if (subArrayDimension != null) {
            subArrayDimension.rSetParentRNode(this);
        }
    }

    /**
     * Adds the IXbcSubArrayDimensionChoice property <b>subArrayDimension</b> by index.
     *
     * @param index
     * @param subArrayDimension
     */
    public final void addSubArrayDimension(int index, IXbcSubArrayDimensionChoice subArrayDimension) {
        this.subArrayDimension_.add(index, subArrayDimension);
        if (subArrayDimension != null) {
            subArrayDimension.rSetParentRNode(this);
        }
    }

    /**
     * Remove the IXbcSubArrayDimensionChoice property <b>subArrayDimension</b> by index.
     *
     * @param index
     */
    public final void removeSubArrayDimension(int index) {
        this.subArrayDimension_.remove(index);
    }

    /**
     * Remove the IXbcSubArrayDimensionChoice property <b>subArrayDimension</b> by object.
     *
     * @param subArrayDimension
     */
    public final void removeSubArrayDimension(IXbcSubArrayDimensionChoice subArrayDimension) {
        this.subArrayDimension_.remove(subArrayDimension);
    }

    /**
     * Clear the IXbcSubArrayDimensionChoice property <b>subArrayDimension</b>.
     *
     */
    public final void clearSubArrayDimension() {
        this.subArrayDimension_.clear();
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
        buffer.append("<subArrayRef");
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
        size = this.subArrayDimension_.size();
        for (int i = 0;i < size;i++) {
            IXbcSubArrayDimensionChoice value = (IXbcSubArrayDimensionChoice)this.subArrayDimension_.get(i);
            value.makeTextAttribute(buffer);
        }
        buffer.append(">");
        arrayAddr_.makeTextElement(buffer);
        size = this.subArrayDimension_.size();
        for (int i = 0;i < size;i++) {
            IXbcSubArrayDimensionChoice value = (IXbcSubArrayDimensionChoice)this.subArrayDimension_.get(i);
            value.makeTextElement(buffer);
        }
        buffer.append("</subArrayRef>");
    }

    /**
     * Makes an XML text representation.
     *
     * @param buffer
     * @exception IOException
     */
    public void makeTextElement(Writer buffer) throws IOException {
        int size;
        buffer.write("<subArrayRef");
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
        size = this.subArrayDimension_.size();
        for (int i = 0;i < size;i++) {
            IXbcSubArrayDimensionChoice value = (IXbcSubArrayDimensionChoice)this.subArrayDimension_.get(i);
            value.makeTextAttribute(buffer);
        }
        buffer.write(">");
        arrayAddr_.makeTextElement(buffer);
        size = this.subArrayDimension_.size();
        for (int i = 0;i < size;i++) {
            IXbcSubArrayDimensionChoice value = (IXbcSubArrayDimensionChoice)this.subArrayDimension_.get(i);
            value.makeTextElement(buffer);
        }
        buffer.write("</subArrayRef>");
    }

    /**
     * Makes an XML text representation.
     *
     * @param buffer
     */
    public void makeTextElement(PrintWriter buffer) {
        int size;
        buffer.print("<subArrayRef");
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
        size = this.subArrayDimension_.size();
        for (int i = 0;i < size;i++) {
            IXbcSubArrayDimensionChoice value = (IXbcSubArrayDimensionChoice)this.subArrayDimension_.get(i);
            value.makeTextAttribute(buffer);
        }
        buffer.print(">");
        arrayAddr_.makeTextElement(buffer);
        size = this.subArrayDimension_.size();
        for (int i = 0;i < size;i++) {
            IXbcSubArrayDimensionChoice value = (IXbcSubArrayDimensionChoice)this.subArrayDimension_.get(i);
            value.makeTextElement(buffer);
        }
        buffer.print("</subArrayRef>");
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
        classNodes.addAll(subArrayDimension_);
        IRNode[] nodes = new IRNode[classNodes.size()];
        return ((IRNode[])classNodes.toArray(nodes));
    }

    /**
     * Tests if a Element <code>element</code> is valid
     * for the <code>XbcSubArrayRef</code>.
     *
     * @param element
     * @return boolean
     */
    public static boolean isMatch(Element element) {
        if (!URelaxer.isTargetElement(element, "subArrayRef")) {
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
        } else if (XbcArrayRef.isMatchHungry(target)) {
            $match$ = true;
        } else if (XbcFunctionCall.isMatchHungry(target)) {
            $match$ = true;
        } else if (XbcGccCompoundExpr.isMatchHungry(target)) {
            $match$ = true;
        } else if (XbcSubArrayRef.isMatchHungry(target)) {
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
        } else if (XbcXmpDescOf.isMatchHungry(target)) {
            $match$ = true;
        } else if (XbcCoArrayAssignExpr.isMatchHungry(target)) {
            $match$ = true;
        } else if (XbcIndexRange.isMatchHungry(target)) {
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
        } else if (XbcSizeOfExpr.isMatchHungry(target)) {
            $match$ = true;
        } else if (XbcAddrOfExpr.isMatchHungry(target)) {
            $match$ = true;
        } else if (XbcGccAlignOfExpr.isMatchHungry(target)) {
            $match$ = true;
        } else if (XbcGccLabelAddr.isMatchHungry(target)) {
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
            } else if (XbcArrayRef.isMatchHungry(target)) {
                $match$ = true;
            } else if (XbcFunctionCall.isMatchHungry(target)) {
                $match$ = true;
            } else if (XbcGccCompoundExpr.isMatchHungry(target)) {
                $match$ = true;
            } else if (XbcSubArrayRef.isMatchHungry(target)) {
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
            } else if (XbcXmpDescOf.isMatchHungry(target)) {
                $match$ = true;
            } else if (XbcCoArrayAssignExpr.isMatchHungry(target)) {
                $match$ = true;
            } else if (XbcIndexRange.isMatchHungry(target)) {
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
            } else if (XbcSizeOfExpr.isMatchHungry(target)) {
                $match$ = true;
            } else if (XbcAddrOfExpr.isMatchHungry(target)) {
                $match$ = true;
            } else if (XbcGccAlignOfExpr.isMatchHungry(target)) {
                $match$ = true;
            } else if (XbcGccLabelAddr.isMatchHungry(target)) {
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
     * is valid for the <code>XbcSubArrayRef</code>.
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
     * is valid for the <code>XbcSubArrayRef</code>.
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
