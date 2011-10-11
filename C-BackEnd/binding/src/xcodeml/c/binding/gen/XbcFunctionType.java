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
 * <b>XbcFunctionType</b> is generated from XcodeML_C.rng by Relaxer.
 * This class is derived from:
 * 
 * <!-- for programmer
 * <element java:extends="xcodeml.c.obj.XmcObj" java:implements="xcodeml.c.binding.IXbcType" name="functionType">
 *   <ref name="BaseType"/>
 *   <ref name="type"/>
 *   <attribute name="return_type">
 *     <data type="string"/>
 *   </attribute>
 *   <optional>
 *     <attribute name="is_inline">
 *       <choice>
 *         <value>0</value>
 *         <value>1</value>
 *         <value>true</value>
 *         <value>false</value>
 *       </choice>
 *     </attribute>
 *   </optional>
 *   <optional>
 *     <attribute name="is_static">
 *       <choice>
 *         <value>0</value>
 *         <value>1</value>
 *         <value>true</value>
 *         <value>false</value>
 *       </choice>
 *     </attribute>
 *   </optional>
 *   <optional>
 *     <ref name="params"/>
 *   </optional>
 * </element>
 * -->
 * <!-- for javadoc -->
 * <pre> &lt;element java:extends="xcodeml.c.obj.XmcObj" java:implements="xcodeml.c.binding.IXbcType" name="functionType"&gt;
 *   &lt;ref name="BaseType"/&gt;
 *   &lt;ref name="type"/&gt;
 *   &lt;attribute name="return_type"&gt;
 *     &lt;data type="string"/&gt;
 *   &lt;/attribute&gt;
 *   &lt;optional&gt;
 *     &lt;attribute name="is_inline"&gt;
 *       &lt;choice&gt;
 *         &lt;value&gt;0&lt;/value&gt;
 *         &lt;value&gt;1&lt;/value&gt;
 *         &lt;value&gt;true&lt;/value&gt;
 *         &lt;value&gt;false&lt;/value&gt;
 *       &lt;/choice&gt;
 *     &lt;/attribute&gt;
 *   &lt;/optional&gt;
 *   &lt;optional&gt;
 *     &lt;attribute name="is_static"&gt;
 *       &lt;choice&gt;
 *         &lt;value&gt;0&lt;/value&gt;
 *         &lt;value&gt;1&lt;/value&gt;
 *         &lt;value&gt;true&lt;/value&gt;
 *         &lt;value&gt;false&lt;/value&gt;
 *       &lt;/choice&gt;
 *     &lt;/attribute&gt;
 *   &lt;/optional&gt;
 *   &lt;optional&gt;
 *     &lt;ref name="params"/&gt;
 *   &lt;/optional&gt;
 * &lt;/element&gt;
 * </pre>
 *
 * @version XcodeML_C.rng (Fri Oct 07 17:51:13 JST 2011)
 * @author  Relaxer 1.0 (http://www.relaxer.org)
 */
public class XbcFunctionType extends xcodeml.c.obj.XmcObj implements java.io.Serializable, Cloneable, xcodeml.c.binding.IXbcType, IRVisitable, IRNode, IXbcTypesChoice {
    public static final String ISINLINE_0 = "0";
    public static final String ISINLINE_1 = "1";
    public static final String ISINLINE_TRUE = "true";
    public static final String ISINLINE_FALSE = "false";
    public static final String ISSTATIC_0 = "0";
    public static final String ISSTATIC_1 = "1";
    public static final String ISSTATIC_TRUE = "true";
    public static final String ISSTATIC_FALSE = "false";
    public static final String ISVOLATILE_0 = "0";
    public static final String ISVOLATILE_1 = "1";
    public static final String ISVOLATILE_TRUE = "true";
    public static final String ISVOLATILE_FALSE = "false";
    public static final String ISCONST_0 = "0";
    public static final String ISCONST_1 = "1";
    public static final String ISCONST_TRUE = "true";
    public static final String ISCONST_FALSE = "false";
    public static final String ISRESTRICT_0 = "0";
    public static final String ISRESTRICT_1 = "1";
    public static final String ISRESTRICT_TRUE = "true";
    public static final String ISRESTRICT_FALSE = "false";

    private String returnType_;
    private String isInline_;
    private String isStatic_;
    private String isVolatile_;
    private String isConst_;
    private String isRestrict_;
    private String type_;
    private XbcGccAttributes gccAttributes_;
    private XbcParams params_;
    private IRNode parentRNode_;

    /**
     * Creates a <code>XbcFunctionType</code>.
     *
     */
    public XbcFunctionType() {
        returnType_ = "";
        type_ = "";
    }

    /**
     * Creates a <code>XbcFunctionType</code>.
     *
     * @param source
     */
    public XbcFunctionType(XbcFunctionType source) {
        setup(source);
    }

    /**
     * Creates a <code>XbcFunctionType</code> by the Stack <code>stack</code>
     * that contains Elements.
     * This constructor is supposed to be used internally
     * by the Relaxer system.
     *
     * @param stack
     */
    public XbcFunctionType(RStack stack) {
        setup(stack);
    }

    /**
     * Creates a <code>XbcFunctionType</code> by the Document <code>doc</code>.
     *
     * @param doc
     */
    public XbcFunctionType(Document doc) {
        setup(doc.getDocumentElement());
    }

    /**
     * Creates a <code>XbcFunctionType</code> by the Element <code>element</code>.
     *
     * @param element
     */
    public XbcFunctionType(Element element) {
        setup(element);
    }

    /**
     * Creates a <code>XbcFunctionType</code> by the File <code>file</code>.
     *
     * @param file
     * @exception IOException
     * @exception SAXException
     * @exception ParserConfigurationException
     */
    public XbcFunctionType(File file) throws IOException, SAXException, ParserConfigurationException {
        setup(file);
    }

    /**
     * Creates a <code>XbcFunctionType</code>
     * by the String representation of URI <code>uri</code>.
     *
     * @param uri
     * @exception IOException
     * @exception SAXException
     * @exception ParserConfigurationException
     */
    public XbcFunctionType(String uri) throws IOException, SAXException, ParserConfigurationException {
        setup(uri);
    }

    /**
     * Creates a <code>XbcFunctionType</code> by the URL <code>url</code>.
     *
     * @param url
     * @exception IOException
     * @exception SAXException
     * @exception ParserConfigurationException
     */
    public XbcFunctionType(URL url) throws IOException, SAXException, ParserConfigurationException {
        setup(url);
    }

    /**
     * Creates a <code>XbcFunctionType</code> by the InputStream <code>in</code>.
     *
     * @param in
     * @exception IOException
     * @exception SAXException
     * @exception ParserConfigurationException
     */
    public XbcFunctionType(InputStream in) throws IOException, SAXException, ParserConfigurationException {
        setup(in);
    }

    /**
     * Creates a <code>XbcFunctionType</code> by the InputSource <code>is</code>.
     *
     * @param is
     * @exception IOException
     * @exception SAXException
     * @exception ParserConfigurationException
     */
    public XbcFunctionType(InputSource is) throws IOException, SAXException, ParserConfigurationException {
        setup(is);
    }

    /**
     * Creates a <code>XbcFunctionType</code> by the Reader <code>reader</code>.
     *
     * @param reader
     * @exception IOException
     * @exception SAXException
     * @exception ParserConfigurationException
     */
    public XbcFunctionType(Reader reader) throws IOException, SAXException, ParserConfigurationException {
        setup(reader);
    }

    /**
     * Initializes the <code>XbcFunctionType</code> by the XbcFunctionType <code>source</code>.
     *
     * @param source
     */
    public void setup(XbcFunctionType source) {
        int size;
        setReturnType(source.getReturnType());
        setIsInline(source.getIsInline());
        setIsStatic(source.getIsStatic());
        setIsVolatile(source.getIsVolatile());
        setIsConst(source.getIsConst());
        setIsRestrict(source.getIsRestrict());
        setType(source.getType());
        if (source.gccAttributes_ != null) {
            setGccAttributes((XbcGccAttributes)source.getGccAttributes().clone());
        }
        if (source.params_ != null) {
            setParams((XbcParams)source.getParams().clone());
        }
    }

    /**
     * Initializes the <code>XbcFunctionType</code> by the Document <code>doc</code>.
     *
     * @param doc
     */
    public void setup(Document doc) {
        setup(doc.getDocumentElement());
    }

    /**
     * Initializes the <code>XbcFunctionType</code> by the Element <code>element</code>.
     *
     * @param element
     */
    public void setup(Element element) {
        init(element);
    }

    /**
     * Initializes the <code>XbcFunctionType</code> by the Stack <code>stack</code>
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
        returnType_ = URelaxer.getAttributePropertyAsString(element, "return_type");
        isInline_ = URelaxer.getAttributePropertyAsString(element, "is_inline");
        isStatic_ = URelaxer.getAttributePropertyAsString(element, "is_static");
        isVolatile_ = URelaxer.getAttributePropertyAsString(element, "is_volatile");
        isConst_ = URelaxer.getAttributePropertyAsString(element, "is_const");
        isRestrict_ = URelaxer.getAttributePropertyAsString(element, "is_restrict");
        type_ = URelaxer.getAttributePropertyAsString(element, "type");
        if (XbcGccAttributes.isMatch(stack)) {
            setGccAttributes(factory.createXbcGccAttributes(stack));
        }
        if (XbcParams.isMatch(stack)) {
            setParams(factory.createXbcParams(stack));
        }
    }

    /**
     * @return Object
     */
    public Object clone() {
        IXcodeML_CFactory factory = XcodeML_CFactory.getFactory();
        return (factory.createXbcFunctionType(this));
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
        Element element = doc.createElement("functionType");
        int size;
        if (this.returnType_ != null) {
            URelaxer.setAttributePropertyByString(element, "return_type", this.returnType_);
        }
        if (this.isInline_ != null) {
            URelaxer.setAttributePropertyByString(element, "is_inline", this.isInline_);
        }
        if (this.isStatic_ != null) {
            URelaxer.setAttributePropertyByString(element, "is_static", this.isStatic_);
        }
        if (this.isVolatile_ != null) {
            URelaxer.setAttributePropertyByString(element, "is_volatile", this.isVolatile_);
        }
        if (this.isConst_ != null) {
            URelaxer.setAttributePropertyByString(element, "is_const", this.isConst_);
        }
        if (this.isRestrict_ != null) {
            URelaxer.setAttributePropertyByString(element, "is_restrict", this.isRestrict_);
        }
        if (this.type_ != null) {
            URelaxer.setAttributePropertyByString(element, "type", this.type_);
        }
        if (this.gccAttributes_ != null) {
            this.gccAttributes_.makeElement(element);
        }
        if (this.params_ != null) {
            this.params_.makeElement(element);
        }
        parent.appendChild(element);
    }

    /**
     * Initializes the <code>XbcFunctionType</code> by the File <code>file</code>.
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
     * Initializes the <code>XbcFunctionType</code>
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
     * Initializes the <code>XbcFunctionType</code> by the URL <code>url</code>.
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
     * Initializes the <code>XbcFunctionType</code> by the InputStream <code>in</code>.
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
     * Initializes the <code>XbcFunctionType</code> by the InputSource <code>is</code>.
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
     * Initializes the <code>XbcFunctionType</code> by the Reader <code>reader</code>.
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
     * Gets the String property <b>returnType</b>.
     *
     * @return String
     */
    public final String getReturnType() {
        return (returnType_);
    }

    /**
     * Sets the String property <b>returnType</b>.
     *
     * @param returnType
     */
    public final void setReturnType(String returnType) {
        this.returnType_ = returnType;
    }

    /**
     * Gets the String property <b>isInline</b>.
     *
     * @return String
     */
    public final String getIsInline() {
        return (isInline_);
    }

    /**
     * Sets the String property <b>isInline</b>.
     *
     * @param isInline
     */
    public final void setIsInline(String isInline) {
        this.isInline_ = isInline;
    }

    /**
     * Gets the String property <b>isStatic</b>.
     *
     * @return String
     */
    public final String getIsStatic() {
        return (isStatic_);
    }

    /**
     * Sets the String property <b>isStatic</b>.
     *
     * @param isStatic
     */
    public final void setIsStatic(String isStatic) {
        this.isStatic_ = isStatic;
    }

    /**
     * Gets the String property <b>isVolatile</b>.
     *
     * @return String
     */
    public final String getIsVolatile() {
        return (isVolatile_);
    }

    /**
     * Sets the String property <b>isVolatile</b>.
     *
     * @param isVolatile
     */
    public final void setIsVolatile(String isVolatile) {
        this.isVolatile_ = isVolatile;
    }

    /**
     * Gets the String property <b>isConst</b>.
     *
     * @return String
     */
    public final String getIsConst() {
        return (isConst_);
    }

    /**
     * Sets the String property <b>isConst</b>.
     *
     * @param isConst
     */
    public final void setIsConst(String isConst) {
        this.isConst_ = isConst;
    }

    /**
     * Gets the String property <b>isRestrict</b>.
     *
     * @return String
     */
    public final String getIsRestrict() {
        return (isRestrict_);
    }

    /**
     * Sets the String property <b>isRestrict</b>.
     *
     * @param isRestrict
     */
    public final void setIsRestrict(String isRestrict) {
        this.isRestrict_ = isRestrict;
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
     * Gets the XbcGccAttributes property <b>gccAttributes</b>.
     *
     * @return XbcGccAttributes
     */
    public final XbcGccAttributes getGccAttributes() {
        return (gccAttributes_);
    }

    /**
     * Sets the XbcGccAttributes property <b>gccAttributes</b>.
     *
     * @param gccAttributes
     */
    public final void setGccAttributes(XbcGccAttributes gccAttributes) {
        this.gccAttributes_ = gccAttributes;
        if (gccAttributes != null) {
            gccAttributes.rSetParentRNode(this);
        }
    }

    /**
     * Gets the XbcParams property <b>params</b>.
     *
     * @return XbcParams
     */
    public final XbcParams getParams() {
        return (params_);
    }

    /**
     * Sets the XbcParams property <b>params</b>.
     *
     * @param params
     */
    public final void setParams(XbcParams params) {
        this.params_ = params;
        if (params != null) {
            params.rSetParentRNode(this);
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
        buffer.append("<functionType");
        if (returnType_ != null) {
            buffer.append(" return_type=\"");
            buffer.append(URelaxer.escapeAttrQuot(URelaxer.getString(getReturnType())));
            buffer.append("\"");
        }
        if (isInline_ != null) {
            buffer.append(" is_inline=\"");
            buffer.append(URelaxer.escapeAttrQuot(URelaxer.getString(getIsInline())));
            buffer.append("\"");
        }
        if (isStatic_ != null) {
            buffer.append(" is_static=\"");
            buffer.append(URelaxer.escapeAttrQuot(URelaxer.getString(getIsStatic())));
            buffer.append("\"");
        }
        if (isVolatile_ != null) {
            buffer.append(" is_volatile=\"");
            buffer.append(URelaxer.escapeAttrQuot(URelaxer.getString(getIsVolatile())));
            buffer.append("\"");
        }
        if (isConst_ != null) {
            buffer.append(" is_const=\"");
            buffer.append(URelaxer.escapeAttrQuot(URelaxer.getString(getIsConst())));
            buffer.append("\"");
        }
        if (isRestrict_ != null) {
            buffer.append(" is_restrict=\"");
            buffer.append(URelaxer.escapeAttrQuot(URelaxer.getString(getIsRestrict())));
            buffer.append("\"");
        }
        if (type_ != null) {
            buffer.append(" type=\"");
            buffer.append(URelaxer.escapeAttrQuot(URelaxer.getString(getType())));
            buffer.append("\"");
        }
        buffer.append(">");
        if (gccAttributes_ != null) {
            gccAttributes_.makeTextElement(buffer);
        }
        if (params_ != null) {
            params_.makeTextElement(buffer);
        }
        buffer.append("</functionType>");
    }

    /**
     * Makes an XML text representation.
     *
     * @param buffer
     * @exception IOException
     */
    public void makeTextElement(Writer buffer) throws IOException {
        int size;
        buffer.write("<functionType");
        if (returnType_ != null) {
            buffer.write(" return_type=\"");
            buffer.write(URelaxer.escapeAttrQuot(URelaxer.getString(getReturnType())));
            buffer.write("\"");
        }
        if (isInline_ != null) {
            buffer.write(" is_inline=\"");
            buffer.write(URelaxer.escapeAttrQuot(URelaxer.getString(getIsInline())));
            buffer.write("\"");
        }
        if (isStatic_ != null) {
            buffer.write(" is_static=\"");
            buffer.write(URelaxer.escapeAttrQuot(URelaxer.getString(getIsStatic())));
            buffer.write("\"");
        }
        if (isVolatile_ != null) {
            buffer.write(" is_volatile=\"");
            buffer.write(URelaxer.escapeAttrQuot(URelaxer.getString(getIsVolatile())));
            buffer.write("\"");
        }
        if (isConst_ != null) {
            buffer.write(" is_const=\"");
            buffer.write(URelaxer.escapeAttrQuot(URelaxer.getString(getIsConst())));
            buffer.write("\"");
        }
        if (isRestrict_ != null) {
            buffer.write(" is_restrict=\"");
            buffer.write(URelaxer.escapeAttrQuot(URelaxer.getString(getIsRestrict())));
            buffer.write("\"");
        }
        if (type_ != null) {
            buffer.write(" type=\"");
            buffer.write(URelaxer.escapeAttrQuot(URelaxer.getString(getType())));
            buffer.write("\"");
        }
        buffer.write(">");
        if (gccAttributes_ != null) {
            gccAttributes_.makeTextElement(buffer);
        }
        if (params_ != null) {
            params_.makeTextElement(buffer);
        }
        buffer.write("</functionType>");
    }

    /**
     * Makes an XML text representation.
     *
     * @param buffer
     */
    public void makeTextElement(PrintWriter buffer) {
        int size;
        buffer.print("<functionType");
        if (returnType_ != null) {
            buffer.print(" return_type=\"");
            buffer.print(URelaxer.escapeAttrQuot(URelaxer.getString(getReturnType())));
            buffer.print("\"");
        }
        if (isInline_ != null) {
            buffer.print(" is_inline=\"");
            buffer.print(URelaxer.escapeAttrQuot(URelaxer.getString(getIsInline())));
            buffer.print("\"");
        }
        if (isStatic_ != null) {
            buffer.print(" is_static=\"");
            buffer.print(URelaxer.escapeAttrQuot(URelaxer.getString(getIsStatic())));
            buffer.print("\"");
        }
        if (isVolatile_ != null) {
            buffer.print(" is_volatile=\"");
            buffer.print(URelaxer.escapeAttrQuot(URelaxer.getString(getIsVolatile())));
            buffer.print("\"");
        }
        if (isConst_ != null) {
            buffer.print(" is_const=\"");
            buffer.print(URelaxer.escapeAttrQuot(URelaxer.getString(getIsConst())));
            buffer.print("\"");
        }
        if (isRestrict_ != null) {
            buffer.print(" is_restrict=\"");
            buffer.print(URelaxer.escapeAttrQuot(URelaxer.getString(getIsRestrict())));
            buffer.print("\"");
        }
        if (type_ != null) {
            buffer.print(" type=\"");
            buffer.print(URelaxer.escapeAttrQuot(URelaxer.getString(getType())));
            buffer.print("\"");
        }
        buffer.print(">");
        if (gccAttributes_ != null) {
            gccAttributes_.makeTextElement(buffer);
        }
        if (params_ != null) {
            params_.makeTextElement(buffer);
        }
        buffer.print("</functionType>");
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
    public String getReturnTypeAsString() {
        return (URelaxer.getString(getReturnType()));
    }

    /**
     * Gets the property value as String.
     *
     * @return String
     */
    public String getIsInlineAsString() {
        return (URelaxer.getString(getIsInline()));
    }

    /**
     * Gets the property value as String.
     *
     * @return String
     */
    public String getIsStaticAsString() {
        return (URelaxer.getString(getIsStatic()));
    }

    /**
     * Gets the property value as String.
     *
     * @return String
     */
    public String getIsVolatileAsString() {
        return (URelaxer.getString(getIsVolatile()));
    }

    /**
     * Gets the property value as String.
     *
     * @return String
     */
    public String getIsConstAsString() {
        return (URelaxer.getString(getIsConst()));
    }

    /**
     * Gets the property value as String.
     *
     * @return String
     */
    public String getIsRestrictAsString() {
        return (URelaxer.getString(getIsRestrict()));
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
    public void setReturnTypeByString(String string) {
        setReturnType(string);
    }

    /**
     * Sets the property value by String.
     *
     * @param string
     */
    public void setIsInlineByString(String string) {
        setIsInline(string);
    }

    /**
     * Sets the property value by String.
     *
     * @param string
     */
    public void setIsStaticByString(String string) {
        setIsStatic(string);
    }

    /**
     * Sets the property value by String.
     *
     * @param string
     */
    public void setIsVolatileByString(String string) {
        setIsVolatile(string);
    }

    /**
     * Sets the property value by String.
     *
     * @param string
     */
    public void setIsConstByString(String string) {
        setIsConst(string);
    }

    /**
     * Sets the property value by String.
     *
     * @param string
     */
    public void setIsRestrictByString(String string) {
        setIsRestrict(string);
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
        if (gccAttributes_ != null) {
            classNodes.add(gccAttributes_);
        }
        if (params_ != null) {
            classNodes.add(params_);
        }
        IRNode[] nodes = new IRNode[classNodes.size()];
        return ((IRNode[])classNodes.toArray(nodes));
    }

    /**
     * Tests if a Element <code>element</code> is valid
     * for the <code>XbcFunctionType</code>.
     *
     * @param element
     * @return boolean
     */
    public static boolean isMatch(Element element) {
        if (!URelaxer.isTargetElement(element, "functionType")) {
            return (false);
        }
        RStack target = new RStack(element);
        boolean $match$ = false;
        Element child;
        if (!URelaxer.hasAttributeHungry(target, "return_type")) {
            return (false);
        }
        $match$ = true;
        if (!URelaxer.hasAttributeHungry(target, "type")) {
            return (false);
        }
        $match$ = true;
        if (XbcGccAttributes.isMatchHungry(target)) {
        }
        if (XbcParams.isMatchHungry(target)) {
        }
        if (!target.isEmptyElement()) {
            return (false);
        }
        return (true);
    }

    /**
     * Tests if elements contained in a Stack <code>stack</code>
     * is valid for the <code>XbcFunctionType</code>.
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
     * is valid for the <code>XbcFunctionType</code>.
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
