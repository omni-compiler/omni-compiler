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
 * <b>XbcDesignatedValue</b> is generated from XcodeML_C.rng by Relaxer.
 * This class is derived from:
 * 
 * <!-- for programmer
 * <element java:extends="xcodeml.c.obj.XmcObj" java:implements="xcodeml.c.binding.IXbcAnnotation" name="designatedValue">
 *   <ref name="annotation"/>
 *   <attribute name="member">
 *     <data type="string"/>
 *   </attribute>
 *   <optional>
 *     <choice>
 *       <ref name="expressions"/>
 *       <ref name="compoundValue"/>
 *       <ref name="designatedValue"/>
 *     </choice>
 *   </optional>
 * </element>
 * -->
 * <!-- for javadoc -->
 * <pre> &lt;element java:extends="xcodeml.c.obj.XmcObj" java:implements="xcodeml.c.binding.IXbcAnnotation" name="designatedValue"&gt;
 *   &lt;ref name="annotation"/&gt;
 *   &lt;attribute name="member"&gt;
 *     &lt;data type="string"/&gt;
 *   &lt;/attribute&gt;
 *   &lt;optional&gt;
 *     &lt;choice&gt;
 *       &lt;ref name="expressions"/&gt;
 *       &lt;ref name="compoundValue"/&gt;
 *       &lt;ref name="designatedValue"/&gt;
 *     &lt;/choice&gt;
 *   &lt;/optional&gt;
 * &lt;/element&gt;
 * </pre>
 *
 * @version XcodeML_C.rng (Fri Oct 07 17:51:13 JST 2011)
 * @author  Relaxer 1.0 (http://www.relaxer.org)
 */
public class XbcDesignatedValue extends xcodeml.c.obj.XmcObj implements java.io.Serializable, Cloneable, IXbcDesignatedValueChoice, IXbcCompoundLiteralChoice, xcodeml.c.binding.IXbcAnnotation, IRVisitable, IRNode, IXbcValueChoice {
    public static final String ISGCCSYNTAX_0 = "0";
    public static final String ISGCCSYNTAX_1 = "1";
    public static final String ISGCCSYNTAX_TRUE = "true";
    public static final String ISGCCSYNTAX_FALSE = "false";
    public static final String ISMODIFIED_0 = "0";
    public static final String ISMODIFIED_1 = "1";
    public static final String ISMODIFIED_TRUE = "true";
    public static final String ISMODIFIED_FALSE = "false";

    private String member_;
    private String isGccSyntax_;
    private String isModified_;
    private IXbcDesignatedValueChoice content_;
    private IRNode parentRNode_;

    /**
     * Creates a <code>XbcDesignatedValue</code>.
     *
     */
    public XbcDesignatedValue() {
        member_ = "";
    }

    /**
     * Creates a <code>XbcDesignatedValue</code>.
     *
     * @param source
     */
    public XbcDesignatedValue(XbcDesignatedValue source) {
        setup(source);
    }

    /**
     * Creates a <code>XbcDesignatedValue</code> by the Stack <code>stack</code>
     * that contains Elements.
     * This constructor is supposed to be used internally
     * by the Relaxer system.
     *
     * @param stack
     */
    public XbcDesignatedValue(RStack stack) {
        setup(stack);
    }

    /**
     * Creates a <code>XbcDesignatedValue</code> by the Document <code>doc</code>.
     *
     * @param doc
     */
    public XbcDesignatedValue(Document doc) {
        setup(doc.getDocumentElement());
    }

    /**
     * Creates a <code>XbcDesignatedValue</code> by the Element <code>element</code>.
     *
     * @param element
     */
    public XbcDesignatedValue(Element element) {
        setup(element);
    }

    /**
     * Creates a <code>XbcDesignatedValue</code> by the File <code>file</code>.
     *
     * @param file
     * @exception IOException
     * @exception SAXException
     * @exception ParserConfigurationException
     */
    public XbcDesignatedValue(File file) throws IOException, SAXException, ParserConfigurationException {
        setup(file);
    }

    /**
     * Creates a <code>XbcDesignatedValue</code>
     * by the String representation of URI <code>uri</code>.
     *
     * @param uri
     * @exception IOException
     * @exception SAXException
     * @exception ParserConfigurationException
     */
    public XbcDesignatedValue(String uri) throws IOException, SAXException, ParserConfigurationException {
        setup(uri);
    }

    /**
     * Creates a <code>XbcDesignatedValue</code> by the URL <code>url</code>.
     *
     * @param url
     * @exception IOException
     * @exception SAXException
     * @exception ParserConfigurationException
     */
    public XbcDesignatedValue(URL url) throws IOException, SAXException, ParserConfigurationException {
        setup(url);
    }

    /**
     * Creates a <code>XbcDesignatedValue</code> by the InputStream <code>in</code>.
     *
     * @param in
     * @exception IOException
     * @exception SAXException
     * @exception ParserConfigurationException
     */
    public XbcDesignatedValue(InputStream in) throws IOException, SAXException, ParserConfigurationException {
        setup(in);
    }

    /**
     * Creates a <code>XbcDesignatedValue</code> by the InputSource <code>is</code>.
     *
     * @param is
     * @exception IOException
     * @exception SAXException
     * @exception ParserConfigurationException
     */
    public XbcDesignatedValue(InputSource is) throws IOException, SAXException, ParserConfigurationException {
        setup(is);
    }

    /**
     * Creates a <code>XbcDesignatedValue</code> by the Reader <code>reader</code>.
     *
     * @param reader
     * @exception IOException
     * @exception SAXException
     * @exception ParserConfigurationException
     */
    public XbcDesignatedValue(Reader reader) throws IOException, SAXException, ParserConfigurationException {
        setup(reader);
    }

    /**
     * Initializes the <code>XbcDesignatedValue</code> by the XbcDesignatedValue <code>source</code>.
     *
     * @param source
     */
    public void setup(XbcDesignatedValue source) {
        int size;
        setMember(source.getMember());
        setIsGccSyntax(source.getIsGccSyntax());
        setIsModified(source.getIsModified());
        if (source.content_ != null) {
            setContent((IXbcDesignatedValueChoice)source.getContent().clone());
        }
    }

    /**
     * Initializes the <code>XbcDesignatedValue</code> by the Document <code>doc</code>.
     *
     * @param doc
     */
    public void setup(Document doc) {
        setup(doc.getDocumentElement());
    }

    /**
     * Initializes the <code>XbcDesignatedValue</code> by the Element <code>element</code>.
     *
     * @param element
     */
    public void setup(Element element) {
        init(element);
    }

    /**
     * Initializes the <code>XbcDesignatedValue</code> by the Stack <code>stack</code>
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
        member_ = URelaxer.getAttributePropertyAsString(element, "member");
        isGccSyntax_ = URelaxer.getAttributePropertyAsString(element, "is_gccSyntax");
        isModified_ = URelaxer.getAttributePropertyAsString(element, "is_modified");
        if (XbcBuiltinOp.isMatch(stack)) {
            setContent(factory.createXbcBuiltinOp(stack));
        } else if (XbcArrayRef.isMatch(stack)) {
            setContent(factory.createXbcArrayRef(stack));
        } else if (XbcFunctionCall.isMatch(stack)) {
            setContent(factory.createXbcFunctionCall(stack));
        } else if (XbcGccCompoundExpr.isMatch(stack)) {
            setContent(factory.createXbcGccCompoundExpr(stack));
        } else if (XbcSubArrayRef.isMatch(stack)) {
            setContent(factory.createXbcSubArrayRef(stack));
        } else if (XbcCoArrayRef.isMatch(stack)) {
            setContent(factory.createXbcCoArrayRef(stack));
        } else if (XbcStringConstant.isMatch(stack)) {
            setContent(factory.createXbcStringConstant(stack));
        } else if (XbcVar.isMatch(stack)) {
            setContent(factory.createXbcVar(stack));
        } else if (XbcVarAddr.isMatch(stack)) {
            setContent(factory.createXbcVarAddr(stack));
        } else if (XbcArrayAddr.isMatch(stack)) {
            setContent(factory.createXbcArrayAddr(stack));
        } else if (XbcCastExpr.isMatch(stack)) {
            setContent(factory.createXbcCastExpr(stack));
        } else if (XbcCompoundValueExpr.isMatch(stack)) {
            setContent(factory.createXbcCompoundValueExpr(stack));
        } else if (XbcCompoundValueAddrExpr.isMatch(stack)) {
            setContent(factory.createXbcCompoundValueAddrExpr(stack));
        } else if (XbcCoArrayAssignExpr.isMatch(stack)) {
            setContent(factory.createXbcCoArrayAssignExpr(stack));
        } else if (XbcAddrOfExpr.isMatch(stack)) {
            setContent(factory.createXbcAddrOfExpr(stack));
        } else if (XbcIntConstant.isMatch(stack)) {
            setContent(factory.createXbcIntConstant(stack));
        } else if (XbcFloatConstant.isMatch(stack)) {
            setContent(factory.createXbcFloatConstant(stack));
        } else if (XbcLonglongConstant.isMatch(stack)) {
            setContent(factory.createXbcLonglongConstant(stack));
        } else if (XbcMoeConstant.isMatch(stack)) {
            setContent(factory.createXbcMoeConstant(stack));
        } else if (XbcFuncAddr.isMatch(stack)) {
            setContent(factory.createXbcFuncAddr(stack));
        } else if (XbcSizeOfExpr.isMatch(stack)) {
            setContent(factory.createXbcSizeOfExpr(stack));
        } else if (XbcGccAlignOfExpr.isMatch(stack)) {
            setContent(factory.createXbcGccAlignOfExpr(stack));
        } else if (XbcGccLabelAddr.isMatch(stack)) {
            setContent(factory.createXbcGccLabelAddr(stack));
        } else if (XbcDesignatedValue.isMatch(stack)) {
            setContent(factory.createXbcDesignatedValue(stack));
        } else if (XbcCondExpr.isMatch(stack)) {
            setContent(factory.createXbcCondExpr(stack));
        } else if (XbcCommaExpr.isMatch(stack)) {
            setContent(factory.createXbcCommaExpr(stack));
        } else if (XbcCompoundValue.isMatch(stack)) {
            setContent(factory.createXbcCompoundValue(stack));
        } else if (XbcMemberAddr.isMatch(stack)) {
            setContent(factory.createXbcMemberAddr(stack));
        } else if (XbcMemberRef.isMatch(stack)) {
            setContent(factory.createXbcMemberRef(stack));
        } else if (XbcMemberArrayRef.isMatch(stack)) {
            setContent(factory.createXbcMemberArrayRef(stack));
        } else if (XbcMemberArrayAddr.isMatch(stack)) {
            setContent(factory.createXbcMemberArrayAddr(stack));
        } else if (XbcPointerRef.isMatch(stack)) {
            setContent(factory.createXbcPointerRef(stack));
        } else if (XbcAssignExpr.isMatch(stack)) {
            setContent(factory.createXbcAssignExpr(stack));
        } else if (XbcPlusExpr.isMatch(stack)) {
            setContent(factory.createXbcPlusExpr(stack));
        } else if (XbcMinusExpr.isMatch(stack)) {
            setContent(factory.createXbcMinusExpr(stack));
        } else if (XbcMulExpr.isMatch(stack)) {
            setContent(factory.createXbcMulExpr(stack));
        } else if (XbcDivExpr.isMatch(stack)) {
            setContent(factory.createXbcDivExpr(stack));
        } else if (XbcModExpr.isMatch(stack)) {
            setContent(factory.createXbcModExpr(stack));
        } else if (XbcLshiftExpr.isMatch(stack)) {
            setContent(factory.createXbcLshiftExpr(stack));
        } else if (XbcRshiftExpr.isMatch(stack)) {
            setContent(factory.createXbcRshiftExpr(stack));
        } else if (XbcBitAndExpr.isMatch(stack)) {
            setContent(factory.createXbcBitAndExpr(stack));
        } else if (XbcBitOrExpr.isMatch(stack)) {
            setContent(factory.createXbcBitOrExpr(stack));
        } else if (XbcBitXorExpr.isMatch(stack)) {
            setContent(factory.createXbcBitXorExpr(stack));
        } else if (XbcAsgPlusExpr.isMatch(stack)) {
            setContent(factory.createXbcAsgPlusExpr(stack));
        } else if (XbcAsgMinusExpr.isMatch(stack)) {
            setContent(factory.createXbcAsgMinusExpr(stack));
        } else if (XbcAsgMulExpr.isMatch(stack)) {
            setContent(factory.createXbcAsgMulExpr(stack));
        } else if (XbcAsgDivExpr.isMatch(stack)) {
            setContent(factory.createXbcAsgDivExpr(stack));
        } else if (XbcAsgModExpr.isMatch(stack)) {
            setContent(factory.createXbcAsgModExpr(stack));
        } else if (XbcAsgLshiftExpr.isMatch(stack)) {
            setContent(factory.createXbcAsgLshiftExpr(stack));
        } else if (XbcAsgRshiftExpr.isMatch(stack)) {
            setContent(factory.createXbcAsgRshiftExpr(stack));
        } else if (XbcAsgBitAndExpr.isMatch(stack)) {
            setContent(factory.createXbcAsgBitAndExpr(stack));
        } else if (XbcAsgBitOrExpr.isMatch(stack)) {
            setContent(factory.createXbcAsgBitOrExpr(stack));
        } else if (XbcAsgBitXorExpr.isMatch(stack)) {
            setContent(factory.createXbcAsgBitXorExpr(stack));
        } else if (XbcLogEQExpr.isMatch(stack)) {
            setContent(factory.createXbcLogEQExpr(stack));
        } else if (XbcLogNEQExpr.isMatch(stack)) {
            setContent(factory.createXbcLogNEQExpr(stack));
        } else if (XbcLogGEExpr.isMatch(stack)) {
            setContent(factory.createXbcLogGEExpr(stack));
        } else if (XbcLogGTExpr.isMatch(stack)) {
            setContent(factory.createXbcLogGTExpr(stack));
        } else if (XbcLogLEExpr.isMatch(stack)) {
            setContent(factory.createXbcLogLEExpr(stack));
        } else if (XbcLogLTExpr.isMatch(stack)) {
            setContent(factory.createXbcLogLTExpr(stack));
        } else if (XbcLogAndExpr.isMatch(stack)) {
            setContent(factory.createXbcLogAndExpr(stack));
        } else if (XbcLogOrExpr.isMatch(stack)) {
            setContent(factory.createXbcLogOrExpr(stack));
        } else if (XbcUnaryMinusExpr.isMatch(stack)) {
            setContent(factory.createXbcUnaryMinusExpr(stack));
        } else if (XbcBitNotExpr.isMatch(stack)) {
            setContent(factory.createXbcBitNotExpr(stack));
        } else if (XbcLogNotExpr.isMatch(stack)) {
            setContent(factory.createXbcLogNotExpr(stack));
        } else if (XbcPostIncrExpr.isMatch(stack)) {
            setContent(factory.createXbcPostIncrExpr(stack));
        } else if (XbcPostDecrExpr.isMatch(stack)) {
            setContent(factory.createXbcPostDecrExpr(stack));
        } else if (XbcPreIncrExpr.isMatch(stack)) {
            setContent(factory.createXbcPreIncrExpr(stack));
        } else if (XbcPreDecrExpr.isMatch(stack)) {
            setContent(factory.createXbcPreDecrExpr(stack));
        } else {
        }
    }

    /**
     * @return Object
     */
    public Object clone() {
        IXcodeML_CFactory factory = XcodeML_CFactory.getFactory();
        return (factory.createXbcDesignatedValue(this));
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
        Element element = doc.createElement("designatedValue");
        int size;
        if (this.member_ != null) {
            URelaxer.setAttributePropertyByString(element, "member", this.member_);
        }
        if (this.isGccSyntax_ != null) {
            URelaxer.setAttributePropertyByString(element, "is_gccSyntax", this.isGccSyntax_);
        }
        if (this.isModified_ != null) {
            URelaxer.setAttributePropertyByString(element, "is_modified", this.isModified_);
        }
        if (this.content_ != null) {
            this.content_.makeElement(element);
        }
        parent.appendChild(element);
    }

    /**
     * Initializes the <code>XbcDesignatedValue</code> by the File <code>file</code>.
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
     * Initializes the <code>XbcDesignatedValue</code>
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
     * Initializes the <code>XbcDesignatedValue</code> by the URL <code>url</code>.
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
     * Initializes the <code>XbcDesignatedValue</code> by the InputStream <code>in</code>.
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
     * Initializes the <code>XbcDesignatedValue</code> by the InputSource <code>is</code>.
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
     * Initializes the <code>XbcDesignatedValue</code> by the Reader <code>reader</code>.
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
     * Gets the String property <b>member</b>.
     *
     * @return String
     */
    public final String getMember() {
        return (member_);
    }

    /**
     * Sets the String property <b>member</b>.
     *
     * @param member
     */
    public final void setMember(String member) {
        this.member_ = member;
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
     * Gets the IXbcDesignatedValueChoice property <b>content</b>.
     *
     * @return IXbcDesignatedValueChoice
     */
    public final IXbcDesignatedValueChoice getContent() {
        return (content_);
    }

    /**
     * Sets the IXbcDesignatedValueChoice property <b>content</b>.
     *
     * @param content
     */
    public final void setContent(IXbcDesignatedValueChoice content) {
        this.content_ = content;
        if (content != null) {
            content.rSetParentRNode(this);
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
        buffer.append("<designatedValue");
        if (member_ != null) {
            buffer.append(" member=\"");
            buffer.append(URelaxer.escapeAttrQuot(URelaxer.getString(getMember())));
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
        if (content_ != null) {
            content_.makeTextAttribute(buffer);
        }
        buffer.append(">");
        if (content_ != null) {
            content_.makeTextElement(buffer);
        }
        buffer.append("</designatedValue>");
    }

    /**
     * Makes an XML text representation.
     *
     * @param buffer
     * @exception IOException
     */
    public void makeTextElement(Writer buffer) throws IOException {
        int size;
        buffer.write("<designatedValue");
        if (member_ != null) {
            buffer.write(" member=\"");
            buffer.write(URelaxer.escapeAttrQuot(URelaxer.getString(getMember())));
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
        if (content_ != null) {
            content_.makeTextAttribute(buffer);
        }
        buffer.write(">");
        if (content_ != null) {
            content_.makeTextElement(buffer);
        }
        buffer.write("</designatedValue>");
    }

    /**
     * Makes an XML text representation.
     *
     * @param buffer
     */
    public void makeTextElement(PrintWriter buffer) {
        int size;
        buffer.print("<designatedValue");
        if (member_ != null) {
            buffer.print(" member=\"");
            buffer.print(URelaxer.escapeAttrQuot(URelaxer.getString(getMember())));
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
        if (content_ != null) {
            content_.makeTextAttribute(buffer);
        }
        buffer.print(">");
        if (content_ != null) {
            content_.makeTextElement(buffer);
        }
        buffer.print("</designatedValue>");
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
    public String getMemberAsString() {
        return (URelaxer.getString(getMember()));
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
    public void setMemberByString(String string) {
        setMember(string);
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
        if (content_ != null) {
            classNodes.add(content_);
        }
        IRNode[] nodes = new IRNode[classNodes.size()];
        return ((IRNode[])classNodes.toArray(nodes));
    }

    /**
     * Tests if a Element <code>element</code> is valid
     * for the <code>XbcDesignatedValue</code>.
     *
     * @param element
     * @return boolean
     */
    public static boolean isMatch(Element element) {
        if (!URelaxer.isTargetElement(element, "designatedValue")) {
            return (false);
        }
        RStack target = new RStack(element);
        boolean $match$ = false;
        Element child;
        if (!URelaxer.hasAttributeHungry(target, "member")) {
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
        } else if (XbcStringConstant.isMatchHungry(target)) {
            $match$ = true;
        } else if (XbcVar.isMatchHungry(target)) {
            $match$ = true;
        } else if (XbcVarAddr.isMatchHungry(target)) {
            $match$ = true;
        } else if (XbcArrayAddr.isMatchHungry(target)) {
            $match$ = true;
        } else if (XbcCastExpr.isMatchHungry(target)) {
            $match$ = true;
        } else if (XbcCompoundValueExpr.isMatchHungry(target)) {
            $match$ = true;
        } else if (XbcCompoundValueAddrExpr.isMatchHungry(target)) {
            $match$ = true;
        } else if (XbcCoArrayAssignExpr.isMatchHungry(target)) {
            $match$ = true;
        } else if (XbcAddrOfExpr.isMatchHungry(target)) {
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
        } else if (XbcGccAlignOfExpr.isMatchHungry(target)) {
            $match$ = true;
        } else if (XbcGccLabelAddr.isMatchHungry(target)) {
            $match$ = true;
        } else if (XbcDesignatedValue.isMatchHungry(target)) {
            $match$ = true;
        } else if (XbcCondExpr.isMatchHungry(target)) {
            $match$ = true;
        } else if (XbcCommaExpr.isMatchHungry(target)) {
            $match$ = true;
        } else if (XbcCompoundValue.isMatchHungry(target)) {
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
        } else if (XbcPostIncrExpr.isMatchHungry(target)) {
            $match$ = true;
        } else if (XbcPostDecrExpr.isMatchHungry(target)) {
            $match$ = true;
        } else if (XbcPreIncrExpr.isMatchHungry(target)) {
            $match$ = true;
        } else if (XbcPreDecrExpr.isMatchHungry(target)) {
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
     * is valid for the <code>XbcDesignatedValue</code>.
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
     * is valid for the <code>XbcDesignatedValue</code>.
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
