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
 * <b>XbcBuiltinOp</b> is generated from XcodeML_C.rng by Relaxer.
 * This class is derived from:
 * 
 * <!-- for programmer
 * <element java:extends="xcodeml.c.obj.XmcObj" java:implements="xcodeml.c.binding.IXbcTypedExpr" name="builtin_op">
 *   <ref name="BaseExpression"/>
 *   <attribute name="name">
 *     <data type="string"/>
 *   </attribute>
 *   <optional>
 *     <attribute name="is_id">
 *       <choice>
 *         <value>0</value>
 *         <value>1</value>
 *         <value>true</value>
 *         <value>false</value>
 *       </choice>
 *     </attribute>
 *   </optional>
 *   <optional>
 *     <attribute name="is_addrOf">
 *       <choice>
 *         <value>0</value>
 *         <value>1</value>
 *         <value>true</value>
 *         <value>false</value>
 *       </choice>
 *     </attribute>
 *   </optional>
 *   <zeroOrMore>
 *     <choice>
 *       <ref name="expressions"/>
 *       <ref name="typeName"/>
 *       <ref name="gccMemberDesignator"/>
 *     </choice>
 *   </zeroOrMore>
 * </element>
 * -->
 * <!-- for javadoc -->
 * <pre> &lt;element java:extends="xcodeml.c.obj.XmcObj" java:implements="xcodeml.c.binding.IXbcTypedExpr" name="builtin_op"&gt;
 *   &lt;ref name="BaseExpression"/&gt;
 *   &lt;attribute name="name"&gt;
 *     &lt;data type="string"/&gt;
 *   &lt;/attribute&gt;
 *   &lt;optional&gt;
 *     &lt;attribute name="is_id"&gt;
 *       &lt;choice&gt;
 *         &lt;value&gt;0&lt;/value&gt;
 *         &lt;value&gt;1&lt;/value&gt;
 *         &lt;value&gt;true&lt;/value&gt;
 *         &lt;value&gt;false&lt;/value&gt;
 *       &lt;/choice&gt;
 *     &lt;/attribute&gt;
 *   &lt;/optional&gt;
 *   &lt;optional&gt;
 *     &lt;attribute name="is_addrOf"&gt;
 *       &lt;choice&gt;
 *         &lt;value&gt;0&lt;/value&gt;
 *         &lt;value&gt;1&lt;/value&gt;
 *         &lt;value&gt;true&lt;/value&gt;
 *         &lt;value&gt;false&lt;/value&gt;
 *       &lt;/choice&gt;
 *     &lt;/attribute&gt;
 *   &lt;/optional&gt;
 *   &lt;zeroOrMore&gt;
 *     &lt;choice&gt;
 *       &lt;ref name="expressions"/&gt;
 *       &lt;ref name="typeName"/&gt;
 *       &lt;ref name="gccMemberDesignator"/&gt;
 *     &lt;/choice&gt;
 *   &lt;/zeroOrMore&gt;
 * &lt;/element&gt;
 * </pre>
 *
 * @version XcodeML_C.rng (Mon Aug 15 15:55:17 JST 2011)
 * @author  Relaxer 1.0 (http://www.relaxer.org)
 */
public class XbcBuiltinOp extends xcodeml.c.obj.XmcObj implements java.io.Serializable, Cloneable, IXbcBuiltinOpChoice, IXbcSubArrayDimensionChoice, IXbcValueChoice, IXbcCastExprChoice, IXbcExprOrTypeChoice, IXbcCompoundLiteralChoice, IXbcDesignatedValueChoice, IXbcGotoStatementChoice, xcodeml.c.binding.IXbcTypedExpr, IRVisitable, IRNode, IXbcExpressionsChoice {
    public static final String ISID_0 = "0";
    public static final String ISID_1 = "1";
    public static final String ISID_TRUE = "true";
    public static final String ISID_FALSE = "false";
    public static final String ISADDROF_0 = "0";
    public static final String ISADDROF_1 = "1";
    public static final String ISADDROF_TRUE = "true";
    public static final String ISADDROF_FALSE = "false";
    public static final String ISGCCSYNTAX_0 = "0";
    public static final String ISGCCSYNTAX_1 = "1";
    public static final String ISGCCSYNTAX_TRUE = "true";
    public static final String ISGCCSYNTAX_FALSE = "false";
    public static final String ISMODIFIED_0 = "0";
    public static final String ISMODIFIED_1 = "1";
    public static final String ISMODIFIED_TRUE = "true";
    public static final String ISMODIFIED_FALSE = "false";

    private String name_;
    private String isId_;
    private String isAddrOf_;
    private String type_;
    private String isGccSyntax_;
    private String isModified_;
    // List<IXbcBuiltinOpChoice>
    private java.util.List content_ = new java.util.ArrayList();
    private IRNode parentRNode_;

    /**
     * Creates a <code>XbcBuiltinOp</code>.
     *
     */
    public XbcBuiltinOp() {
        name_ = "";
    }

    /**
     * Creates a <code>XbcBuiltinOp</code>.
     *
     * @param source
     */
    public XbcBuiltinOp(XbcBuiltinOp source) {
        setup(source);
    }

    /**
     * Creates a <code>XbcBuiltinOp</code> by the Stack <code>stack</code>
     * that contains Elements.
     * This constructor is supposed to be used internally
     * by the Relaxer system.
     *
     * @param stack
     */
    public XbcBuiltinOp(RStack stack) {
        setup(stack);
    }

    /**
     * Creates a <code>XbcBuiltinOp</code> by the Document <code>doc</code>.
     *
     * @param doc
     */
    public XbcBuiltinOp(Document doc) {
        setup(doc.getDocumentElement());
    }

    /**
     * Creates a <code>XbcBuiltinOp</code> by the Element <code>element</code>.
     *
     * @param element
     */
    public XbcBuiltinOp(Element element) {
        setup(element);
    }

    /**
     * Creates a <code>XbcBuiltinOp</code> by the File <code>file</code>.
     *
     * @param file
     * @exception IOException
     * @exception SAXException
     * @exception ParserConfigurationException
     */
    public XbcBuiltinOp(File file) throws IOException, SAXException, ParserConfigurationException {
        setup(file);
    }

    /**
     * Creates a <code>XbcBuiltinOp</code>
     * by the String representation of URI <code>uri</code>.
     *
     * @param uri
     * @exception IOException
     * @exception SAXException
     * @exception ParserConfigurationException
     */
    public XbcBuiltinOp(String uri) throws IOException, SAXException, ParserConfigurationException {
        setup(uri);
    }

    /**
     * Creates a <code>XbcBuiltinOp</code> by the URL <code>url</code>.
     *
     * @param url
     * @exception IOException
     * @exception SAXException
     * @exception ParserConfigurationException
     */
    public XbcBuiltinOp(URL url) throws IOException, SAXException, ParserConfigurationException {
        setup(url);
    }

    /**
     * Creates a <code>XbcBuiltinOp</code> by the InputStream <code>in</code>.
     *
     * @param in
     * @exception IOException
     * @exception SAXException
     * @exception ParserConfigurationException
     */
    public XbcBuiltinOp(InputStream in) throws IOException, SAXException, ParserConfigurationException {
        setup(in);
    }

    /**
     * Creates a <code>XbcBuiltinOp</code> by the InputSource <code>is</code>.
     *
     * @param is
     * @exception IOException
     * @exception SAXException
     * @exception ParserConfigurationException
     */
    public XbcBuiltinOp(InputSource is) throws IOException, SAXException, ParserConfigurationException {
        setup(is);
    }

    /**
     * Creates a <code>XbcBuiltinOp</code> by the Reader <code>reader</code>.
     *
     * @param reader
     * @exception IOException
     * @exception SAXException
     * @exception ParserConfigurationException
     */
    public XbcBuiltinOp(Reader reader) throws IOException, SAXException, ParserConfigurationException {
        setup(reader);
    }

    /**
     * Initializes the <code>XbcBuiltinOp</code> by the XbcBuiltinOp <code>source</code>.
     *
     * @param source
     */
    public void setup(XbcBuiltinOp source) {
        int size;
        setName(source.getName());
        setIsId(source.getIsId());
        setIsAddrOf(source.getIsAddrOf());
        setType(source.getType());
        setIsGccSyntax(source.getIsGccSyntax());
        setIsModified(source.getIsModified());
        this.content_.clear();
        size = source.content_.size();
        for (int i = 0;i < size;i++) {
            addContent((IXbcBuiltinOpChoice)source.getContent(i).clone());
        }
    }

    /**
     * Initializes the <code>XbcBuiltinOp</code> by the Document <code>doc</code>.
     *
     * @param doc
     */
    public void setup(Document doc) {
        setup(doc.getDocumentElement());
    }

    /**
     * Initializes the <code>XbcBuiltinOp</code> by the Element <code>element</code>.
     *
     * @param element
     */
    public void setup(Element element) {
        init(element);
    }

    /**
     * Initializes the <code>XbcBuiltinOp</code> by the Stack <code>stack</code>
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
        name_ = URelaxer.getAttributePropertyAsString(element, "name");
        isId_ = URelaxer.getAttributePropertyAsString(element, "is_id");
        isAddrOf_ = URelaxer.getAttributePropertyAsString(element, "is_addrOf");
        type_ = URelaxer.getAttributePropertyAsString(element, "type");
        isGccSyntax_ = URelaxer.getAttributePropertyAsString(element, "is_gccSyntax");
        isModified_ = URelaxer.getAttributePropertyAsString(element, "is_modified");
        content_.clear();
        while (true) {
            if (XbcBuiltinOp.isMatch(stack)) {
                addContent(factory.createXbcBuiltinOp(stack));
            } else if (XbcArrayRef.isMatch(stack)) {
                addContent(factory.createXbcArrayRef(stack));
            } else if (XbcSubArrayRef.isMatch(stack)) {
                addContent(factory.createXbcSubArrayRef(stack));
            } else if (XbcCoArrayRef.isMatch(stack)) {
                addContent(factory.createXbcCoArrayRef(stack));
            } else if (XbcFunctionCall.isMatch(stack)) {
                addContent(factory.createXbcFunctionCall(stack));
            } else if (XbcGccCompoundExpr.isMatch(stack)) {
                addContent(factory.createXbcGccCompoundExpr(stack));
            } else if (XbcCastExpr.isMatch(stack)) {
                addContent(factory.createXbcCastExpr(stack));
            } else if (XbcAsgPlusExpr.isMatch(stack)) {
                addContent(factory.createXbcAsgPlusExpr(stack));
            } else if (XbcAsgBitOrExpr.isMatch(stack)) {
                addContent(factory.createXbcAsgBitOrExpr(stack));
            } else if (XbcLogGEExpr.isMatch(stack)) {
                addContent(factory.createXbcLogGEExpr(stack));
            } else if (XbcStringConstant.isMatch(stack)) {
                addContent(factory.createXbcStringConstant(stack));
            } else if (XbcVar.isMatch(stack)) {
                addContent(factory.createXbcVar(stack));
            } else if (XbcVarAddr.isMatch(stack)) {
                addContent(factory.createXbcVarAddr(stack));
            } else if (XbcArrayAddr.isMatch(stack)) {
                addContent(factory.createXbcArrayAddr(stack));
            } else if (XbcCompoundValueExpr.isMatch(stack)) {
                addContent(factory.createXbcCompoundValueExpr(stack));
            } else if (XbcCompoundValueAddrExpr.isMatch(stack)) {
                addContent(factory.createXbcCompoundValueAddrExpr(stack));
            } else if (XbcSizeOfExpr.isMatch(stack)) {
                addContent(factory.createXbcSizeOfExpr(stack));
            } else if (XbcGccAlignOfExpr.isMatch(stack)) {
                addContent(factory.createXbcGccAlignOfExpr(stack));
            } else if (XbcGccMemberDesignator.isMatch(stack)) {
                addContent(factory.createXbcGccMemberDesignator(stack));
            } else if (XbcIntConstant.isMatch(stack)) {
                addContent(factory.createXbcIntConstant(stack));
            } else if (XbcFloatConstant.isMatch(stack)) {
                addContent(factory.createXbcFloatConstant(stack));
            } else if (XbcLonglongConstant.isMatch(stack)) {
                addContent(factory.createXbcLonglongConstant(stack));
            } else if (XbcMoeConstant.isMatch(stack)) {
                addContent(factory.createXbcMoeConstant(stack));
            } else if (XbcFuncAddr.isMatch(stack)) {
                addContent(factory.createXbcFuncAddr(stack));
            } else if (XbcGccLabelAddr.isMatch(stack)) {
                addContent(factory.createXbcGccLabelAddr(stack));
            } else if (XbcCoArrayAssignExpr.isMatch(stack)) {
                addContent(factory.createXbcCoArrayAssignExpr(stack));
            } else if (XbcMemberAddr.isMatch(stack)) {
                addContent(factory.createXbcMemberAddr(stack));
            } else if (XbcMemberRef.isMatch(stack)) {
                addContent(factory.createXbcMemberRef(stack));
            } else if (XbcMemberArrayRef.isMatch(stack)) {
                addContent(factory.createXbcMemberArrayRef(stack));
            } else if (XbcMemberArrayAddr.isMatch(stack)) {
                addContent(factory.createXbcMemberArrayAddr(stack));
            } else if (XbcUnaryMinusExpr.isMatch(stack)) {
                addContent(factory.createXbcUnaryMinusExpr(stack));
            } else if (XbcTypeName.isMatch(stack)) {
                addContent(factory.createXbcTypeName(stack));
            } else if (XbcPointerRef.isMatch(stack)) {
                addContent(factory.createXbcPointerRef(stack));
            } else if (XbcAssignExpr.isMatch(stack)) {
                addContent(factory.createXbcAssignExpr(stack));
            } else if (XbcPlusExpr.isMatch(stack)) {
                addContent(factory.createXbcPlusExpr(stack));
            } else if (XbcMinusExpr.isMatch(stack)) {
                addContent(factory.createXbcMinusExpr(stack));
            } else if (XbcMulExpr.isMatch(stack)) {
                addContent(factory.createXbcMulExpr(stack));
            } else if (XbcDivExpr.isMatch(stack)) {
                addContent(factory.createXbcDivExpr(stack));
            } else if (XbcModExpr.isMatch(stack)) {
                addContent(factory.createXbcModExpr(stack));
            } else if (XbcLshiftExpr.isMatch(stack)) {
                addContent(factory.createXbcLshiftExpr(stack));
            } else if (XbcRshiftExpr.isMatch(stack)) {
                addContent(factory.createXbcRshiftExpr(stack));
            } else if (XbcBitAndExpr.isMatch(stack)) {
                addContent(factory.createXbcBitAndExpr(stack));
            } else if (XbcBitOrExpr.isMatch(stack)) {
                addContent(factory.createXbcBitOrExpr(stack));
            } else if (XbcBitXorExpr.isMatch(stack)) {
                addContent(factory.createXbcBitXorExpr(stack));
            } else if (XbcAsgMinusExpr.isMatch(stack)) {
                addContent(factory.createXbcAsgMinusExpr(stack));
            } else if (XbcAsgMulExpr.isMatch(stack)) {
                addContent(factory.createXbcAsgMulExpr(stack));
            } else if (XbcAsgDivExpr.isMatch(stack)) {
                addContent(factory.createXbcAsgDivExpr(stack));
            } else if (XbcAsgModExpr.isMatch(stack)) {
                addContent(factory.createXbcAsgModExpr(stack));
            } else if (XbcAsgLshiftExpr.isMatch(stack)) {
                addContent(factory.createXbcAsgLshiftExpr(stack));
            } else if (XbcAsgRshiftExpr.isMatch(stack)) {
                addContent(factory.createXbcAsgRshiftExpr(stack));
            } else if (XbcAsgBitAndExpr.isMatch(stack)) {
                addContent(factory.createXbcAsgBitAndExpr(stack));
            } else if (XbcAsgBitXorExpr.isMatch(stack)) {
                addContent(factory.createXbcAsgBitXorExpr(stack));
            } else if (XbcLogEQExpr.isMatch(stack)) {
                addContent(factory.createXbcLogEQExpr(stack));
            } else if (XbcLogNEQExpr.isMatch(stack)) {
                addContent(factory.createXbcLogNEQExpr(stack));
            } else if (XbcLogGTExpr.isMatch(stack)) {
                addContent(factory.createXbcLogGTExpr(stack));
            } else if (XbcLogLEExpr.isMatch(stack)) {
                addContent(factory.createXbcLogLEExpr(stack));
            } else if (XbcLogLTExpr.isMatch(stack)) {
                addContent(factory.createXbcLogLTExpr(stack));
            } else if (XbcLogAndExpr.isMatch(stack)) {
                addContent(factory.createXbcLogAndExpr(stack));
            } else if (XbcLogOrExpr.isMatch(stack)) {
                addContent(factory.createXbcLogOrExpr(stack));
            } else if (XbcBitNotExpr.isMatch(stack)) {
                addContent(factory.createXbcBitNotExpr(stack));
            } else if (XbcLogNotExpr.isMatch(stack)) {
                addContent(factory.createXbcLogNotExpr(stack));
            } else if (XbcCommaExpr.isMatch(stack)) {
                addContent(factory.createXbcCommaExpr(stack));
            } else if (XbcPostIncrExpr.isMatch(stack)) {
                addContent(factory.createXbcPostIncrExpr(stack));
            } else if (XbcPostDecrExpr.isMatch(stack)) {
                addContent(factory.createXbcPostDecrExpr(stack));
            } else if (XbcPreIncrExpr.isMatch(stack)) {
                addContent(factory.createXbcPreIncrExpr(stack));
            } else if (XbcPreDecrExpr.isMatch(stack)) {
                addContent(factory.createXbcPreDecrExpr(stack));
            } else if (XbcCondExpr.isMatch(stack)) {
                addContent(factory.createXbcCondExpr(stack));
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
        return (factory.createXbcBuiltinOp(this));
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
        Element element = doc.createElement("builtin_op");
        int size;
        if (this.name_ != null) {
            URelaxer.setAttributePropertyByString(element, "name", this.name_);
        }
        if (this.isId_ != null) {
            URelaxer.setAttributePropertyByString(element, "is_id", this.isId_);
        }
        if (this.isAddrOf_ != null) {
            URelaxer.setAttributePropertyByString(element, "is_addrOf", this.isAddrOf_);
        }
        if (this.type_ != null) {
            URelaxer.setAttributePropertyByString(element, "type", this.type_);
        }
        if (this.isGccSyntax_ != null) {
            URelaxer.setAttributePropertyByString(element, "is_gccSyntax", this.isGccSyntax_);
        }
        if (this.isModified_ != null) {
            URelaxer.setAttributePropertyByString(element, "is_modified", this.isModified_);
        }
        size = this.content_.size();
        for (int i = 0;i < size;i++) {
            IXbcBuiltinOpChoice value = (IXbcBuiltinOpChoice)this.content_.get(i);
            value.makeElement(element);
        }
        parent.appendChild(element);
    }

    /**
     * Initializes the <code>XbcBuiltinOp</code> by the File <code>file</code>.
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
     * Initializes the <code>XbcBuiltinOp</code>
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
     * Initializes the <code>XbcBuiltinOp</code> by the URL <code>url</code>.
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
     * Initializes the <code>XbcBuiltinOp</code> by the InputStream <code>in</code>.
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
     * Initializes the <code>XbcBuiltinOp</code> by the InputSource <code>is</code>.
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
     * Initializes the <code>XbcBuiltinOp</code> by the Reader <code>reader</code>.
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
     * Gets the String property <b>isId</b>.
     *
     * @return String
     */
    public final String getIsId() {
        return (isId_);
    }

    /**
     * Sets the String property <b>isId</b>.
     *
     * @param isId
     */
    public final void setIsId(String isId) {
        this.isId_ = isId;
    }

    /**
     * Gets the String property <b>isAddrOf</b>.
     *
     * @return String
     */
    public final String getIsAddrOf() {
        return (isAddrOf_);
    }

    /**
     * Sets the String property <b>isAddrOf</b>.
     *
     * @param isAddrOf
     */
    public final void setIsAddrOf(String isAddrOf) {
        this.isAddrOf_ = isAddrOf;
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
     * Gets the IXbcBuiltinOpChoice property <b>content</b>.
     *
     * @return IXbcBuiltinOpChoice[]
     */
    public final IXbcBuiltinOpChoice[] getContent() {
        IXbcBuiltinOpChoice[] array = new IXbcBuiltinOpChoice[content_.size()];
        return ((IXbcBuiltinOpChoice[])content_.toArray(array));
    }

    /**
     * Sets the IXbcBuiltinOpChoice property <b>content</b>.
     *
     * @param content
     */
    public final void setContent(IXbcBuiltinOpChoice[] content) {
        this.content_.clear();
        for (int i = 0;i < content.length;i++) {
            addContent(content[i]);
        }
        for (int i = 0;i < content.length;i++) {
            content[i].rSetParentRNode(this);
        }
    }

    /**
     * Sets the IXbcBuiltinOpChoice property <b>content</b>.
     *
     * @param content
     */
    public final void setContent(IXbcBuiltinOpChoice content) {
        this.content_.clear();
        addContent(content);
        if (content != null) {
            content.rSetParentRNode(this);
        }
    }

    /**
     * Adds the IXbcBuiltinOpChoice property <b>content</b>.
     *
     * @param content
     */
    public final void addContent(IXbcBuiltinOpChoice content) {
        this.content_.add(content);
        if (content != null) {
            content.rSetParentRNode(this);
        }
    }

    /**
     * Adds the IXbcBuiltinOpChoice property <b>content</b>.
     *
     * @param content
     */
    public final void addContent(IXbcBuiltinOpChoice[] content) {
        for (int i = 0;i < content.length;i++) {
            addContent(content[i]);
        }
        for (int i = 0;i < content.length;i++) {
            content[i].rSetParentRNode(this);
        }
    }

    /**
     * Gets number of the IXbcBuiltinOpChoice property <b>content</b>.
     *
     * @return int
     */
    public final int sizeContent() {
        return (content_.size());
    }

    /**
     * Gets the IXbcBuiltinOpChoice property <b>content</b> by index.
     *
     * @param index
     * @return IXbcBuiltinOpChoice
     */
    public final IXbcBuiltinOpChoice getContent(int index) {
        return ((IXbcBuiltinOpChoice)content_.get(index));
    }

    /**
     * Sets the IXbcBuiltinOpChoice property <b>content</b> by index.
     *
     * @param index
     * @param content
     */
    public final void setContent(int index, IXbcBuiltinOpChoice content) {
        this.content_.set(index, content);
        if (content != null) {
            content.rSetParentRNode(this);
        }
    }

    /**
     * Adds the IXbcBuiltinOpChoice property <b>content</b> by index.
     *
     * @param index
     * @param content
     */
    public final void addContent(int index, IXbcBuiltinOpChoice content) {
        this.content_.add(index, content);
        if (content != null) {
            content.rSetParentRNode(this);
        }
    }

    /**
     * Remove the IXbcBuiltinOpChoice property <b>content</b> by index.
     *
     * @param index
     */
    public final void removeContent(int index) {
        this.content_.remove(index);
    }

    /**
     * Remove the IXbcBuiltinOpChoice property <b>content</b> by object.
     *
     * @param content
     */
    public final void removeContent(IXbcBuiltinOpChoice content) {
        this.content_.remove(content);
    }

    /**
     * Clear the IXbcBuiltinOpChoice property <b>content</b>.
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
        buffer.append("<builtin_op");
        if (name_ != null) {
            buffer.append(" name=\"");
            buffer.append(URelaxer.escapeAttrQuot(URelaxer.getString(getName())));
            buffer.append("\"");
        }
        if (isId_ != null) {
            buffer.append(" is_id=\"");
            buffer.append(URelaxer.escapeAttrQuot(URelaxer.getString(getIsId())));
            buffer.append("\"");
        }
        if (isAddrOf_ != null) {
            buffer.append(" is_addrOf=\"");
            buffer.append(URelaxer.escapeAttrQuot(URelaxer.getString(getIsAddrOf())));
            buffer.append("\"");
        }
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
        size = this.content_.size();
        for (int i = 0;i < size;i++) {
            IXbcBuiltinOpChoice value = (IXbcBuiltinOpChoice)this.content_.get(i);
            value.makeTextAttribute(buffer);
        }
        buffer.append(">");
        size = this.content_.size();
        for (int i = 0;i < size;i++) {
            IXbcBuiltinOpChoice value = (IXbcBuiltinOpChoice)this.content_.get(i);
            value.makeTextElement(buffer);
        }
        buffer.append("</builtin_op>");
    }

    /**
     * Makes an XML text representation.
     *
     * @param buffer
     * @exception IOException
     */
    public void makeTextElement(Writer buffer) throws IOException {
        int size;
        buffer.write("<builtin_op");
        if (name_ != null) {
            buffer.write(" name=\"");
            buffer.write(URelaxer.escapeAttrQuot(URelaxer.getString(getName())));
            buffer.write("\"");
        }
        if (isId_ != null) {
            buffer.write(" is_id=\"");
            buffer.write(URelaxer.escapeAttrQuot(URelaxer.getString(getIsId())));
            buffer.write("\"");
        }
        if (isAddrOf_ != null) {
            buffer.write(" is_addrOf=\"");
            buffer.write(URelaxer.escapeAttrQuot(URelaxer.getString(getIsAddrOf())));
            buffer.write("\"");
        }
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
        size = this.content_.size();
        for (int i = 0;i < size;i++) {
            IXbcBuiltinOpChoice value = (IXbcBuiltinOpChoice)this.content_.get(i);
            value.makeTextAttribute(buffer);
        }
        buffer.write(">");
        size = this.content_.size();
        for (int i = 0;i < size;i++) {
            IXbcBuiltinOpChoice value = (IXbcBuiltinOpChoice)this.content_.get(i);
            value.makeTextElement(buffer);
        }
        buffer.write("</builtin_op>");
    }

    /**
     * Makes an XML text representation.
     *
     * @param buffer
     */
    public void makeTextElement(PrintWriter buffer) {
        int size;
        buffer.print("<builtin_op");
        if (name_ != null) {
            buffer.print(" name=\"");
            buffer.print(URelaxer.escapeAttrQuot(URelaxer.getString(getName())));
            buffer.print("\"");
        }
        if (isId_ != null) {
            buffer.print(" is_id=\"");
            buffer.print(URelaxer.escapeAttrQuot(URelaxer.getString(getIsId())));
            buffer.print("\"");
        }
        if (isAddrOf_ != null) {
            buffer.print(" is_addrOf=\"");
            buffer.print(URelaxer.escapeAttrQuot(URelaxer.getString(getIsAddrOf())));
            buffer.print("\"");
        }
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
        size = this.content_.size();
        for (int i = 0;i < size;i++) {
            IXbcBuiltinOpChoice value = (IXbcBuiltinOpChoice)this.content_.get(i);
            value.makeTextAttribute(buffer);
        }
        buffer.print(">");
        size = this.content_.size();
        for (int i = 0;i < size;i++) {
            IXbcBuiltinOpChoice value = (IXbcBuiltinOpChoice)this.content_.get(i);
            value.makeTextElement(buffer);
        }
        buffer.print("</builtin_op>");
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
    public String getIsIdAsString() {
        return (URelaxer.getString(getIsId()));
    }

    /**
     * Gets the property value as String.
     *
     * @return String
     */
    public String getIsAddrOfAsString() {
        return (URelaxer.getString(getIsAddrOf()));
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
    public void setIsIdByString(String string) {
        setIsId(string);
    }

    /**
     * Sets the property value by String.
     *
     * @param string
     */
    public void setIsAddrOfByString(String string) {
        setIsAddrOf(string);
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
     * for the <code>XbcBuiltinOp</code>.
     *
     * @param element
     * @return boolean
     */
    public static boolean isMatch(Element element) {
        if (!URelaxer.isTargetElement(element, "builtin_op")) {
            return (false);
        }
        RStack target = new RStack(element);
        boolean $match$ = false;
        Element child;
        if (!URelaxer.hasAttributeHungry(target, "name")) {
            return (false);
        }
        $match$ = true;
        while (true) {
            if (XbcBuiltinOp.isMatchHungry(target)) {
                $match$ = true;
            } else if (XbcArrayRef.isMatchHungry(target)) {
                $match$ = true;
            } else if (XbcSubArrayRef.isMatchHungry(target)) {
                $match$ = true;
            } else if (XbcCoArrayRef.isMatchHungry(target)) {
                $match$ = true;
            } else if (XbcFunctionCall.isMatchHungry(target)) {
                $match$ = true;
            } else if (XbcGccCompoundExpr.isMatchHungry(target)) {
                $match$ = true;
            } else if (XbcCastExpr.isMatchHungry(target)) {
                $match$ = true;
            } else if (XbcAsgPlusExpr.isMatchHungry(target)) {
                $match$ = true;
            } else if (XbcAsgBitOrExpr.isMatchHungry(target)) {
                $match$ = true;
            } else if (XbcLogGEExpr.isMatchHungry(target)) {
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
            } else if (XbcGccAlignOfExpr.isMatchHungry(target)) {
                $match$ = true;
            } else if (XbcGccMemberDesignator.isMatchHungry(target)) {
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
            } else if (XbcMemberAddr.isMatchHungry(target)) {
                $match$ = true;
            } else if (XbcMemberRef.isMatchHungry(target)) {
                $match$ = true;
            } else if (XbcMemberArrayRef.isMatchHungry(target)) {
                $match$ = true;
            } else if (XbcMemberArrayAddr.isMatchHungry(target)) {
                $match$ = true;
            } else if (XbcUnaryMinusExpr.isMatchHungry(target)) {
                $match$ = true;
            } else if (XbcTypeName.isMatchHungry(target)) {
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
            } else if (XbcAsgBitXorExpr.isMatchHungry(target)) {
                $match$ = true;
            } else if (XbcLogEQExpr.isMatchHungry(target)) {
                $match$ = true;
            } else if (XbcLogNEQExpr.isMatchHungry(target)) {
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
     * is valid for the <code>XbcBuiltinOp</code>.
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
     * is valid for the <code>XbcBuiltinOp</code>.
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
