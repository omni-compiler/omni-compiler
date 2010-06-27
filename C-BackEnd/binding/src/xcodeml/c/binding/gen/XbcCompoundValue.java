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
 * <b>XbcCompoundValue</b> is generated from XcodeML_C.rng by Relaxer.
 * This class is derived from:
 * 
 * <!-- for programmer
 * <element java:extends="xcodeml.c.obj.XmcObj" java:implements="xcodeml.c.binding.IXbcAnnotation" name="value">
 *   <ref name="annotation"/>
 *   <ref name="compoundLiteral"/>
 * </element>
 * -->
 * <!-- for javadoc -->
 * <pre> &lt;element java:extends="xcodeml.c.obj.XmcObj" java:implements="xcodeml.c.binding.IXbcAnnotation" name="value"&gt;
 *   &lt;ref name="annotation"/&gt;
 *   &lt;ref name="compoundLiteral"/&gt;
 * &lt;/element&gt;
 * </pre>
 *
 * @version XcodeML_C.rng (Thu Sep 24 16:30:19 JST 2009)
 * @author  Relaxer 1.0 (http://www.relaxer.org)
 */
public class XbcCompoundValue extends xcodeml.c.obj.XmcObj implements java.io.Serializable, Cloneable, IXbcCompoundLiteralChoice, xcodeml.c.binding.IXbcAnnotation, IRVisitable, IRNode, IXbcDesignatedValueChoice, IXbcValueChoice {
    public static final String ISGCCSYNTAX_0 = "0";
    public static final String ISGCCSYNTAX_1 = "1";
    public static final String ISGCCSYNTAX_TRUE = "true";
    public static final String ISGCCSYNTAX_FALSE = "false";
    public static final String ISMODIFIED_0 = "0";
    public static final String ISMODIFIED_1 = "1";
    public static final String ISMODIFIED_TRUE = "true";
    public static final String ISMODIFIED_FALSE = "false";

    private String isGccSyntax_;
    private String isModified_;
    // List<IXbcCompoundLiteralChoice>
    private java.util.List compoundLiteral_ = new java.util.ArrayList();
    private IRNode parentRNode_;

    /**
     * Creates a <code>XbcCompoundValue</code>.
     *
     */
    public XbcCompoundValue() {
    }

    /**
     * Creates a <code>XbcCompoundValue</code>.
     *
     * @param source
     */
    public XbcCompoundValue(XbcCompoundValue source) {
        setup(source);
    }

    /**
     * Creates a <code>XbcCompoundValue</code> by the Stack <code>stack</code>
     * that contains Elements.
     * This constructor is supposed to be used internally
     * by the Relaxer system.
     *
     * @param stack
     */
    public XbcCompoundValue(RStack stack) {
        setup(stack);
    }

    /**
     * Creates a <code>XbcCompoundValue</code> by the Document <code>doc</code>.
     *
     * @param doc
     */
    public XbcCompoundValue(Document doc) {
        setup(doc.getDocumentElement());
    }

    /**
     * Creates a <code>XbcCompoundValue</code> by the Element <code>element</code>.
     *
     * @param element
     */
    public XbcCompoundValue(Element element) {
        setup(element);
    }

    /**
     * Creates a <code>XbcCompoundValue</code> by the File <code>file</code>.
     *
     * @param file
     * @exception IOException
     * @exception SAXException
     * @exception ParserConfigurationException
     */
    public XbcCompoundValue(File file) throws IOException, SAXException, ParserConfigurationException {
        setup(file);
    }

    /**
     * Creates a <code>XbcCompoundValue</code>
     * by the String representation of URI <code>uri</code>.
     *
     * @param uri
     * @exception IOException
     * @exception SAXException
     * @exception ParserConfigurationException
     */
    public XbcCompoundValue(String uri) throws IOException, SAXException, ParserConfigurationException {
        setup(uri);
    }

    /**
     * Creates a <code>XbcCompoundValue</code> by the URL <code>url</code>.
     *
     * @param url
     * @exception IOException
     * @exception SAXException
     * @exception ParserConfigurationException
     */
    public XbcCompoundValue(URL url) throws IOException, SAXException, ParserConfigurationException {
        setup(url);
    }

    /**
     * Creates a <code>XbcCompoundValue</code> by the InputStream <code>in</code>.
     *
     * @param in
     * @exception IOException
     * @exception SAXException
     * @exception ParserConfigurationException
     */
    public XbcCompoundValue(InputStream in) throws IOException, SAXException, ParserConfigurationException {
        setup(in);
    }

    /**
     * Creates a <code>XbcCompoundValue</code> by the InputSource <code>is</code>.
     *
     * @param is
     * @exception IOException
     * @exception SAXException
     * @exception ParserConfigurationException
     */
    public XbcCompoundValue(InputSource is) throws IOException, SAXException, ParserConfigurationException {
        setup(is);
    }

    /**
     * Creates a <code>XbcCompoundValue</code> by the Reader <code>reader</code>.
     *
     * @param reader
     * @exception IOException
     * @exception SAXException
     * @exception ParserConfigurationException
     */
    public XbcCompoundValue(Reader reader) throws IOException, SAXException, ParserConfigurationException {
        setup(reader);
    }

    /**
     * Initializes the <code>XbcCompoundValue</code> by the XbcCompoundValue <code>source</code>.
     *
     * @param source
     */
    public void setup(XbcCompoundValue source) {
        int size;
        setIsGccSyntax(source.getIsGccSyntax());
        setIsModified(source.getIsModified());
        this.compoundLiteral_.clear();
        size = source.compoundLiteral_.size();
        for (int i = 0;i < size;i++) {
            addCompoundLiteral((IXbcCompoundLiteralChoice)source.getCompoundLiteral(i).clone());
        }
    }

    /**
     * Initializes the <code>XbcCompoundValue</code> by the Document <code>doc</code>.
     *
     * @param doc
     */
    public void setup(Document doc) {
        setup(doc.getDocumentElement());
    }

    /**
     * Initializes the <code>XbcCompoundValue</code> by the Element <code>element</code>.
     *
     * @param element
     */
    public void setup(Element element) {
        init(element);
    }

    /**
     * Initializes the <code>XbcCompoundValue</code> by the Stack <code>stack</code>
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
        isGccSyntax_ = URelaxer.getAttributePropertyAsString(element, "is_gccSyntax");
        isModified_ = URelaxer.getAttributePropertyAsString(element, "is_modified");
        compoundLiteral_.clear();
        while (true) {
            if (XbcBuiltinOp.isMatch(stack)) {
                addCompoundLiteral(factory.createXbcBuiltinOp(stack));
            } else if (XbcSubArrayRef.isMatch(stack)) {
                addCompoundLiteral(factory.createXbcSubArrayRef(stack));
            } else if (XbcCoArrayRef.isMatch(stack)) {
                addCompoundLiteral(factory.createXbcCoArrayRef(stack));
            } else if (XbcCondExpr.isMatch(stack)) {
                addCompoundLiteral(factory.createXbcCondExpr(stack));
            } else if (XbcFunctionCall.isMatch(stack)) {
                addCompoundLiteral(factory.createXbcFunctionCall(stack));
            } else if (XbcGccCompoundExpr.isMatch(stack)) {
                addCompoundLiteral(factory.createXbcGccCompoundExpr(stack));
            } else if (XbcCastExpr.isMatch(stack)) {
                addCompoundLiteral(factory.createXbcCastExpr(stack));
            } else if (XbcMemberRef.isMatch(stack)) {
                addCompoundLiteral(factory.createXbcMemberRef(stack));
            } else if (XbcMemberArrayRef.isMatch(stack)) {
                addCompoundLiteral(factory.createXbcMemberArrayRef(stack));
            } else if (XbcMemberArrayAddr.isMatch(stack)) {
                addCompoundLiteral(factory.createXbcMemberArrayAddr(stack));
            } else if (XbcBitXorExpr.isMatch(stack)) {
                addCompoundLiteral(factory.createXbcBitXorExpr(stack));
            } else if (XbcAssignExpr.isMatch(stack)) {
                addCompoundLiteral(factory.createXbcAssignExpr(stack));
            } else if (XbcPlusExpr.isMatch(stack)) {
                addCompoundLiteral(factory.createXbcPlusExpr(stack));
            } else if (XbcModExpr.isMatch(stack)) {
                addCompoundLiteral(factory.createXbcModExpr(stack));
            } else if (XbcRshiftExpr.isMatch(stack)) {
                addCompoundLiteral(factory.createXbcRshiftExpr(stack));
            } else if (XbcBitAndExpr.isMatch(stack)) {
                addCompoundLiteral(factory.createXbcBitAndExpr(stack));
            } else if (XbcAsgPlusExpr.isMatch(stack)) {
                addCompoundLiteral(factory.createXbcAsgPlusExpr(stack));
            } else if (XbcAsgLshiftExpr.isMatch(stack)) {
                addCompoundLiteral(factory.createXbcAsgLshiftExpr(stack));
            } else if (XbcAsgBitAndExpr.isMatch(stack)) {
                addCompoundLiteral(factory.createXbcAsgBitAndExpr(stack));
            } else if (XbcAsgBitXorExpr.isMatch(stack)) {
                addCompoundLiteral(factory.createXbcAsgBitXorExpr(stack));
            } else if (XbcLogEQExpr.isMatch(stack)) {
                addCompoundLiteral(factory.createXbcLogEQExpr(stack));
            } else if (XbcLogNEQExpr.isMatch(stack)) {
                addCompoundLiteral(factory.createXbcLogNEQExpr(stack));
            } else if (XbcLogGEExpr.isMatch(stack)) {
                addCompoundLiteral(factory.createXbcLogGEExpr(stack));
            } else if (XbcLogGTExpr.isMatch(stack)) {
                addCompoundLiteral(factory.createXbcLogGTExpr(stack));
            } else if (XbcLogLEExpr.isMatch(stack)) {
                addCompoundLiteral(factory.createXbcLogLEExpr(stack));
            } else if (XbcLogAndExpr.isMatch(stack)) {
                addCompoundLiteral(factory.createXbcLogAndExpr(stack));
            } else if (XbcLogOrExpr.isMatch(stack)) {
                addCompoundLiteral(factory.createXbcLogOrExpr(stack));
            } else if (XbcStringConstant.isMatch(stack)) {
                addCompoundLiteral(factory.createXbcStringConstant(stack));
            } else if (XbcVar.isMatch(stack)) {
                addCompoundLiteral(factory.createXbcVar(stack));
            } else if (XbcArrayRef.isMatch(stack)) {
                addCompoundLiteral(factory.createXbcArrayRef(stack));
            } else if (XbcArrayAddr.isMatch(stack)) {
                addCompoundLiteral(factory.createXbcArrayAddr(stack));
            } else if (XbcCompoundValueExpr.isMatch(stack)) {
                addCompoundLiteral(factory.createXbcCompoundValueExpr(stack));
            } else if (XbcCompoundValueAddrExpr.isMatch(stack)) {
                addCompoundLiteral(factory.createXbcCompoundValueAddrExpr(stack));
            } else if (XbcSizeOfExpr.isMatch(stack)) {
                addCompoundLiteral(factory.createXbcSizeOfExpr(stack));
            } else if (XbcVarAddr.isMatch(stack)) {
                addCompoundLiteral(factory.createXbcVarAddr(stack));
            } else if (XbcPointerRef.isMatch(stack)) {
                addCompoundLiteral(factory.createXbcPointerRef(stack));
            } else if (XbcBitNotExpr.isMatch(stack)) {
                addCompoundLiteral(factory.createXbcBitNotExpr(stack));
            } else if (XbcLogNotExpr.isMatch(stack)) {
                addCompoundLiteral(factory.createXbcLogNotExpr(stack));
            } else if (XbcCommaExpr.isMatch(stack)) {
                addCompoundLiteral(factory.createXbcCommaExpr(stack));
            } else if (XbcPostIncrExpr.isMatch(stack)) {
                addCompoundLiteral(factory.createXbcPostIncrExpr(stack));
            } else if (XbcPostDecrExpr.isMatch(stack)) {
                addCompoundLiteral(factory.createXbcPostDecrExpr(stack));
            } else if (XbcPreIncrExpr.isMatch(stack)) {
                addCompoundLiteral(factory.createXbcPreIncrExpr(stack));
            } else if (XbcPreDecrExpr.isMatch(stack)) {
                addCompoundLiteral(factory.createXbcPreDecrExpr(stack));
            } else if (XbcDesignatedValue.isMatch(stack)) {
                addCompoundLiteral(factory.createXbcDesignatedValue(stack));
            } else if (XbcGccAlignOfExpr.isMatch(stack)) {
                addCompoundLiteral(factory.createXbcGccAlignOfExpr(stack));
            } else if (XbcMinusExpr.isMatch(stack)) {
                addCompoundLiteral(factory.createXbcMinusExpr(stack));
            } else if (XbcMulExpr.isMatch(stack)) {
                addCompoundLiteral(factory.createXbcMulExpr(stack));
            } else if (XbcIntConstant.isMatch(stack)) {
                addCompoundLiteral(factory.createXbcIntConstant(stack));
            } else if (XbcFloatConstant.isMatch(stack)) {
                addCompoundLiteral(factory.createXbcFloatConstant(stack));
            } else if (XbcLonglongConstant.isMatch(stack)) {
                addCompoundLiteral(factory.createXbcLonglongConstant(stack));
            } else if (XbcMoeConstant.isMatch(stack)) {
                addCompoundLiteral(factory.createXbcMoeConstant(stack));
            } else if (XbcFuncAddr.isMatch(stack)) {
                addCompoundLiteral(factory.createXbcFuncAddr(stack));
            } else if (XbcGccLabelAddr.isMatch(stack)) {
                addCompoundLiteral(factory.createXbcGccLabelAddr(stack));
            } else if (XbcCoArrayAssignExpr.isMatch(stack)) {
                addCompoundLiteral(factory.createXbcCoArrayAssignExpr(stack));
            } else if (XbcDivExpr.isMatch(stack)) {
                addCompoundLiteral(factory.createXbcDivExpr(stack));
            } else if (XbcAsgDivExpr.isMatch(stack)) {
                addCompoundLiteral(factory.createXbcAsgDivExpr(stack));
            } else if (XbcAsgModExpr.isMatch(stack)) {
                addCompoundLiteral(factory.createXbcAsgModExpr(stack));
            } else if (XbcLogLTExpr.isMatch(stack)) {
                addCompoundLiteral(factory.createXbcLogLTExpr(stack));
            } else if (XbcUnaryMinusExpr.isMatch(stack)) {
                addCompoundLiteral(factory.createXbcUnaryMinusExpr(stack));
            } else if (XbcCompoundValue.isMatch(stack)) {
                addCompoundLiteral(factory.createXbcCompoundValue(stack));
            } else if (XbcMemberAddr.isMatch(stack)) {
                addCompoundLiteral(factory.createXbcMemberAddr(stack));
            } else if (XbcLshiftExpr.isMatch(stack)) {
                addCompoundLiteral(factory.createXbcLshiftExpr(stack));
            } else if (XbcBitOrExpr.isMatch(stack)) {
                addCompoundLiteral(factory.createXbcBitOrExpr(stack));
            } else if (XbcAsgMinusExpr.isMatch(stack)) {
                addCompoundLiteral(factory.createXbcAsgMinusExpr(stack));
            } else if (XbcAsgMulExpr.isMatch(stack)) {
                addCompoundLiteral(factory.createXbcAsgMulExpr(stack));
            } else if (XbcAsgRshiftExpr.isMatch(stack)) {
                addCompoundLiteral(factory.createXbcAsgRshiftExpr(stack));
            } else if (XbcAsgBitOrExpr.isMatch(stack)) {
                addCompoundLiteral(factory.createXbcAsgBitOrExpr(stack));
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
        return (factory.createXbcCompoundValue(this));
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
        Element element = doc.createElement("value");
        int size;
        if (this.isGccSyntax_ != null) {
            URelaxer.setAttributePropertyByString(element, "is_gccSyntax", this.isGccSyntax_);
        }
        if (this.isModified_ != null) {
            URelaxer.setAttributePropertyByString(element, "is_modified", this.isModified_);
        }
        size = this.compoundLiteral_.size();
        for (int i = 0;i < size;i++) {
            IXbcCompoundLiteralChoice value = (IXbcCompoundLiteralChoice)this.compoundLiteral_.get(i);
            value.makeElement(element);
        }
        parent.appendChild(element);
    }

    /**
     * Initializes the <code>XbcCompoundValue</code> by the File <code>file</code>.
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
     * Initializes the <code>XbcCompoundValue</code>
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
     * Initializes the <code>XbcCompoundValue</code> by the URL <code>url</code>.
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
     * Initializes the <code>XbcCompoundValue</code> by the InputStream <code>in</code>.
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
     * Initializes the <code>XbcCompoundValue</code> by the InputSource <code>is</code>.
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
     * Initializes the <code>XbcCompoundValue</code> by the Reader <code>reader</code>.
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
     * Gets the IXbcCompoundLiteralChoice property <b>compoundLiteral</b>.
     *
     * @return IXbcCompoundLiteralChoice[]
     */
    public final IXbcCompoundLiteralChoice[] getCompoundLiteral() {
        IXbcCompoundLiteralChoice[] array = new IXbcCompoundLiteralChoice[compoundLiteral_.size()];
        return ((IXbcCompoundLiteralChoice[])compoundLiteral_.toArray(array));
    }

    /**
     * Sets the IXbcCompoundLiteralChoice property <b>compoundLiteral</b>.
     *
     * @param compoundLiteral
     */
    public final void setCompoundLiteral(IXbcCompoundLiteralChoice[] compoundLiteral) {
        this.compoundLiteral_.clear();
        for (int i = 0;i < compoundLiteral.length;i++) {
            addCompoundLiteral(compoundLiteral[i]);
        }
        for (int i = 0;i < compoundLiteral.length;i++) {
            compoundLiteral[i].rSetParentRNode(this);
        }
    }

    /**
     * Sets the IXbcCompoundLiteralChoice property <b>compoundLiteral</b>.
     *
     * @param compoundLiteral
     */
    public final void setCompoundLiteral(IXbcCompoundLiteralChoice compoundLiteral) {
        this.compoundLiteral_.clear();
        addCompoundLiteral(compoundLiteral);
        if (compoundLiteral != null) {
            compoundLiteral.rSetParentRNode(this);
        }
    }

    /**
     * Adds the IXbcCompoundLiteralChoice property <b>compoundLiteral</b>.
     *
     * @param compoundLiteral
     */
    public final void addCompoundLiteral(IXbcCompoundLiteralChoice compoundLiteral) {
        this.compoundLiteral_.add(compoundLiteral);
        if (compoundLiteral != null) {
            compoundLiteral.rSetParentRNode(this);
        }
    }

    /**
     * Adds the IXbcCompoundLiteralChoice property <b>compoundLiteral</b>.
     *
     * @param compoundLiteral
     */
    public final void addCompoundLiteral(IXbcCompoundLiteralChoice[] compoundLiteral) {
        for (int i = 0;i < compoundLiteral.length;i++) {
            addCompoundLiteral(compoundLiteral[i]);
        }
        for (int i = 0;i < compoundLiteral.length;i++) {
            compoundLiteral[i].rSetParentRNode(this);
        }
    }

    /**
     * Gets number of the IXbcCompoundLiteralChoice property <b>compoundLiteral</b>.
     *
     * @return int
     */
    public final int sizeCompoundLiteral() {
        return (compoundLiteral_.size());
    }

    /**
     * Gets the IXbcCompoundLiteralChoice property <b>compoundLiteral</b> by index.
     *
     * @param index
     * @return IXbcCompoundLiteralChoice
     */
    public final IXbcCompoundLiteralChoice getCompoundLiteral(int index) {
        return ((IXbcCompoundLiteralChoice)compoundLiteral_.get(index));
    }

    /**
     * Sets the IXbcCompoundLiteralChoice property <b>compoundLiteral</b> by index.
     *
     * @param index
     * @param compoundLiteral
     */
    public final void setCompoundLiteral(int index, IXbcCompoundLiteralChoice compoundLiteral) {
        this.compoundLiteral_.set(index, compoundLiteral);
        if (compoundLiteral != null) {
            compoundLiteral.rSetParentRNode(this);
        }
    }

    /**
     * Adds the IXbcCompoundLiteralChoice property <b>compoundLiteral</b> by index.
     *
     * @param index
     * @param compoundLiteral
     */
    public final void addCompoundLiteral(int index, IXbcCompoundLiteralChoice compoundLiteral) {
        this.compoundLiteral_.add(index, compoundLiteral);
        if (compoundLiteral != null) {
            compoundLiteral.rSetParentRNode(this);
        }
    }

    /**
     * Remove the IXbcCompoundLiteralChoice property <b>compoundLiteral</b> by index.
     *
     * @param index
     */
    public final void removeCompoundLiteral(int index) {
        this.compoundLiteral_.remove(index);
    }

    /**
     * Remove the IXbcCompoundLiteralChoice property <b>compoundLiteral</b> by object.
     *
     * @param compoundLiteral
     */
    public final void removeCompoundLiteral(IXbcCompoundLiteralChoice compoundLiteral) {
        this.compoundLiteral_.remove(compoundLiteral);
    }

    /**
     * Clear the IXbcCompoundLiteralChoice property <b>compoundLiteral</b>.
     *
     */
    public final void clearCompoundLiteral() {
        this.compoundLiteral_.clear();
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
        buffer.append("<value");
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
        size = this.compoundLiteral_.size();
        for (int i = 0;i < size;i++) {
            IXbcCompoundLiteralChoice value = (IXbcCompoundLiteralChoice)this.compoundLiteral_.get(i);
            value.makeTextAttribute(buffer);
        }
        buffer.append(">");
        size = this.compoundLiteral_.size();
        for (int i = 0;i < size;i++) {
            IXbcCompoundLiteralChoice value = (IXbcCompoundLiteralChoice)this.compoundLiteral_.get(i);
            value.makeTextElement(buffer);
        }
        buffer.append("</value>");
    }

    /**
     * Makes an XML text representation.
     *
     * @param buffer
     * @exception IOException
     */
    public void makeTextElement(Writer buffer) throws IOException {
        int size;
        buffer.write("<value");
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
        size = this.compoundLiteral_.size();
        for (int i = 0;i < size;i++) {
            IXbcCompoundLiteralChoice value = (IXbcCompoundLiteralChoice)this.compoundLiteral_.get(i);
            value.makeTextAttribute(buffer);
        }
        buffer.write(">");
        size = this.compoundLiteral_.size();
        for (int i = 0;i < size;i++) {
            IXbcCompoundLiteralChoice value = (IXbcCompoundLiteralChoice)this.compoundLiteral_.get(i);
            value.makeTextElement(buffer);
        }
        buffer.write("</value>");
    }

    /**
     * Makes an XML text representation.
     *
     * @param buffer
     */
    public void makeTextElement(PrintWriter buffer) {
        int size;
        buffer.print("<value");
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
        size = this.compoundLiteral_.size();
        for (int i = 0;i < size;i++) {
            IXbcCompoundLiteralChoice value = (IXbcCompoundLiteralChoice)this.compoundLiteral_.get(i);
            value.makeTextAttribute(buffer);
        }
        buffer.print(">");
        size = this.compoundLiteral_.size();
        for (int i = 0;i < size;i++) {
            IXbcCompoundLiteralChoice value = (IXbcCompoundLiteralChoice)this.compoundLiteral_.get(i);
            value.makeTextElement(buffer);
        }
        buffer.print("</value>");
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
        classNodes.addAll(compoundLiteral_);
        IRNode[] nodes = new IRNode[classNodes.size()];
        return ((IRNode[])classNodes.toArray(nodes));
    }

    /**
     * Tests if a Element <code>element</code> is valid
     * for the <code>XbcCompoundValue</code>.
     *
     * @param element
     * @return boolean
     */
    public static boolean isMatch(Element element) {
        if (!URelaxer.isTargetElement(element, "value")) {
            return (false);
        }
        RStack target = new RStack(element);
        boolean $match$ = false;
        Element child;
        while (true) {
            if (XbcBuiltinOp.isMatchHungry(target)) {
                $match$ = true;
            } else if (XbcSubArrayRef.isMatchHungry(target)) {
                $match$ = true;
            } else if (XbcCoArrayRef.isMatchHungry(target)) {
                $match$ = true;
            } else if (XbcCondExpr.isMatchHungry(target)) {
                $match$ = true;
            } else if (XbcFunctionCall.isMatchHungry(target)) {
                $match$ = true;
            } else if (XbcGccCompoundExpr.isMatchHungry(target)) {
                $match$ = true;
            } else if (XbcCastExpr.isMatchHungry(target)) {
                $match$ = true;
            } else if (XbcMemberRef.isMatchHungry(target)) {
                $match$ = true;
            } else if (XbcMemberArrayRef.isMatchHungry(target)) {
                $match$ = true;
            } else if (XbcMemberArrayAddr.isMatchHungry(target)) {
                $match$ = true;
            } else if (XbcBitXorExpr.isMatchHungry(target)) {
                $match$ = true;
            } else if (XbcAssignExpr.isMatchHungry(target)) {
                $match$ = true;
            } else if (XbcPlusExpr.isMatchHungry(target)) {
                $match$ = true;
            } else if (XbcModExpr.isMatchHungry(target)) {
                $match$ = true;
            } else if (XbcRshiftExpr.isMatchHungry(target)) {
                $match$ = true;
            } else if (XbcBitAndExpr.isMatchHungry(target)) {
                $match$ = true;
            } else if (XbcAsgPlusExpr.isMatchHungry(target)) {
                $match$ = true;
            } else if (XbcAsgLshiftExpr.isMatchHungry(target)) {
                $match$ = true;
            } else if (XbcAsgBitAndExpr.isMatchHungry(target)) {
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
            } else if (XbcLogAndExpr.isMatchHungry(target)) {
                $match$ = true;
            } else if (XbcLogOrExpr.isMatchHungry(target)) {
                $match$ = true;
            } else if (XbcStringConstant.isMatchHungry(target)) {
                $match$ = true;
            } else if (XbcVar.isMatchHungry(target)) {
                $match$ = true;
            } else if (XbcArrayRef.isMatchHungry(target)) {
                $match$ = true;
            } else if (XbcArrayAddr.isMatchHungry(target)) {
                $match$ = true;
            } else if (XbcCompoundValueExpr.isMatchHungry(target)) {
                $match$ = true;
            } else if (XbcCompoundValueAddrExpr.isMatchHungry(target)) {
                $match$ = true;
            } else if (XbcSizeOfExpr.isMatchHungry(target)) {
                $match$ = true;
            } else if (XbcVarAddr.isMatchHungry(target)) {
                $match$ = true;
            } else if (XbcPointerRef.isMatchHungry(target)) {
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
            } else if (XbcDesignatedValue.isMatchHungry(target)) {
                $match$ = true;
            } else if (XbcGccAlignOfExpr.isMatchHungry(target)) {
                $match$ = true;
            } else if (XbcMinusExpr.isMatchHungry(target)) {
                $match$ = true;
            } else if (XbcMulExpr.isMatchHungry(target)) {
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
            } else if (XbcGccLabelAddr.isMatchHungry(target)) {
                $match$ = true;
            } else if (XbcCoArrayAssignExpr.isMatchHungry(target)) {
                $match$ = true;
            } else if (XbcDivExpr.isMatchHungry(target)) {
                $match$ = true;
            } else if (XbcAsgDivExpr.isMatchHungry(target)) {
                $match$ = true;
            } else if (XbcAsgModExpr.isMatchHungry(target)) {
                $match$ = true;
            } else if (XbcLogLTExpr.isMatchHungry(target)) {
                $match$ = true;
            } else if (XbcUnaryMinusExpr.isMatchHungry(target)) {
                $match$ = true;
            } else if (XbcCompoundValue.isMatchHungry(target)) {
                $match$ = true;
            } else if (XbcMemberAddr.isMatchHungry(target)) {
                $match$ = true;
            } else if (XbcLshiftExpr.isMatchHungry(target)) {
                $match$ = true;
            } else if (XbcBitOrExpr.isMatchHungry(target)) {
                $match$ = true;
            } else if (XbcAsgMinusExpr.isMatchHungry(target)) {
                $match$ = true;
            } else if (XbcAsgMulExpr.isMatchHungry(target)) {
                $match$ = true;
            } else if (XbcAsgRshiftExpr.isMatchHungry(target)) {
                $match$ = true;
            } else if (XbcAsgBitOrExpr.isMatchHungry(target)) {
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
     * is valid for the <code>XbcCompoundValue</code>.
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
     * is valid for the <code>XbcCompoundValue</code>.
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
