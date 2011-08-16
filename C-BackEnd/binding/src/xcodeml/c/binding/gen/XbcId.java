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
 * <b>XbcId</b> is generated from XcodeML_C.rng by Relaxer.
 * This class is derived from:
 * 
 * <!-- for programmer
 * <element java:extends="xcodeml.c.obj.XmcObj" java:implements="xcodeml.c.binding.IXbcHasGccExtension" name="id">
 *   <optional>
 *     <ref name="type"/>
 *   </optional>
 *   <optional>
 *     <attribute name="sclass">
 *       <choice>
 *         <value>auto</value>
 *         <value>param</value>
 *         <value>extern</value>
 *         <value>extern_def</value>
 *         <value>static</value>
 *         <value>register</value>
 *         <value>label</value>
 *         <value>tagname</value>
 *         <value>moe</value>
 *         <value>typedef_name</value>
 *         <value>gccLabel</value>
 *       </choice>
 *     </attribute>
 *   </optional>
 *   <ref name="name"/>
 *   <optional>
 *     <ref name="value"/>
 *   </optional>
 *   <optional>
 *     <ref name="is_gccThread"/>
 *   </optional>
 *   <optional>
 *     <attribute name="bit_field">
 *       <data type="string"/>
 *     </attribute>
 *   </optional>
 *   <optional>
 *     <ref name="bitField"/>
 *   </optional>
 *   <optional>
 *     <ref name="gccAttributes"/>
 *   </optional>
 *   <ref name="gccExtendable"/>
 * </element>
 * -->
 * <!-- for javadoc -->
 * <pre> &lt;element java:extends="xcodeml.c.obj.XmcObj" java:implements="xcodeml.c.binding.IXbcHasGccExtension" name="id"&gt;
 *   &lt;optional&gt;
 *     &lt;ref name="type"/&gt;
 *   &lt;/optional&gt;
 *   &lt;optional&gt;
 *     &lt;attribute name="sclass"&gt;
 *       &lt;choice&gt;
 *         &lt;value&gt;auto&lt;/value&gt;
 *         &lt;value&gt;param&lt;/value&gt;
 *         &lt;value&gt;extern&lt;/value&gt;
 *         &lt;value&gt;extern_def&lt;/value&gt;
 *         &lt;value&gt;static&lt;/value&gt;
 *         &lt;value&gt;register&lt;/value&gt;
 *         &lt;value&gt;label&lt;/value&gt;
 *         &lt;value&gt;tagname&lt;/value&gt;
 *         &lt;value&gt;moe&lt;/value&gt;
 *         &lt;value&gt;typedef_name&lt;/value&gt;
 *         &lt;value&gt;gccLabel&lt;/value&gt;
 *       &lt;/choice&gt;
 *     &lt;/attribute&gt;
 *   &lt;/optional&gt;
 *   &lt;ref name="name"/&gt;
 *   &lt;optional&gt;
 *     &lt;ref name="value"/&gt;
 *   &lt;/optional&gt;
 *   &lt;optional&gt;
 *     &lt;ref name="is_gccThread"/&gt;
 *   &lt;/optional&gt;
 *   &lt;optional&gt;
 *     &lt;attribute name="bit_field"&gt;
 *       &lt;data type="string"/&gt;
 *     &lt;/attribute&gt;
 *   &lt;/optional&gt;
 *   &lt;optional&gt;
 *     &lt;ref name="bitField"/&gt;
 *   &lt;/optional&gt;
 *   &lt;optional&gt;
 *     &lt;ref name="gccAttributes"/&gt;
 *   &lt;/optional&gt;
 *   &lt;ref name="gccExtendable"/&gt;
 * &lt;/element&gt;
 * </pre>
 *
 * @version XcodeML_C.rng (Mon Aug 15 15:55:17 JST 2011)
 * @author  Relaxer 1.0 (http://www.relaxer.org)
 */
public class XbcId extends xcodeml.c.obj.XmcObj implements java.io.Serializable, Cloneable, xcodeml.c.binding.IXbcHasGccExtension, IRVisitable, IRNode, IXbcSymbolsChoice {
    public static final String SCLASS_AUTO = "auto";
    public static final String SCLASS_PARAM = "param";
    public static final String SCLASS_EXTERN = "extern";
    public static final String SCLASS_EXTERN_DEF = "extern_def";
    public static final String SCLASS_STATIC = "static";
    public static final String SCLASS_REGISTER = "register";
    public static final String SCLASS_LABEL = "label";
    public static final String SCLASS_TAGNAME = "tagname";
    public static final String SCLASS_MOE = "moe";
    public static final String SCLASS_TYPEDEF_NAME = "typedef_name";
    public static final String SCLASS_GCCLABEL = "gccLabel";
    public static final String ISGCCTHREAD_0 = "0";
    public static final String ISGCCTHREAD_1 = "1";
    public static final String ISGCCTHREAD_TRUE = "true";
    public static final String ISGCCTHREAD_FALSE = "false";
    public static final String ISGCCEXTENSION_0 = "0";
    public static final String ISGCCEXTENSION_1 = "1";
    public static final String ISGCCEXTENSION_TRUE = "true";
    public static final String ISGCCEXTENSION_FALSE = "false";

    private String sclass_;
    private String bitField1_;
    private String type_;
    private String isGccThread_;
    private String isGccExtension_;
    private XbcName name_;
    private XbcValue value_;
    private XbcBitField bitField2_;
    private XbcGccAttributes gccAttributes_;
    private IRNode parentRNode_;

    /**
     * Creates a <code>XbcId</code>.
     *
     */
    public XbcId() {
    }

    /**
     * Creates a <code>XbcId</code>.
     *
     * @param source
     */
    public XbcId(XbcId source) {
        setup(source);
    }

    /**
     * Creates a <code>XbcId</code> by the Stack <code>stack</code>
     * that contains Elements.
     * This constructor is supposed to be used internally
     * by the Relaxer system.
     *
     * @param stack
     */
    public XbcId(RStack stack) {
        setup(stack);
    }

    /**
     * Creates a <code>XbcId</code> by the Document <code>doc</code>.
     *
     * @param doc
     */
    public XbcId(Document doc) {
        setup(doc.getDocumentElement());
    }

    /**
     * Creates a <code>XbcId</code> by the Element <code>element</code>.
     *
     * @param element
     */
    public XbcId(Element element) {
        setup(element);
    }

    /**
     * Creates a <code>XbcId</code> by the File <code>file</code>.
     *
     * @param file
     * @exception IOException
     * @exception SAXException
     * @exception ParserConfigurationException
     */
    public XbcId(File file) throws IOException, SAXException, ParserConfigurationException {
        setup(file);
    }

    /**
     * Creates a <code>XbcId</code>
     * by the String representation of URI <code>uri</code>.
     *
     * @param uri
     * @exception IOException
     * @exception SAXException
     * @exception ParserConfigurationException
     */
    public XbcId(String uri) throws IOException, SAXException, ParserConfigurationException {
        setup(uri);
    }

    /**
     * Creates a <code>XbcId</code> by the URL <code>url</code>.
     *
     * @param url
     * @exception IOException
     * @exception SAXException
     * @exception ParserConfigurationException
     */
    public XbcId(URL url) throws IOException, SAXException, ParserConfigurationException {
        setup(url);
    }

    /**
     * Creates a <code>XbcId</code> by the InputStream <code>in</code>.
     *
     * @param in
     * @exception IOException
     * @exception SAXException
     * @exception ParserConfigurationException
     */
    public XbcId(InputStream in) throws IOException, SAXException, ParserConfigurationException {
        setup(in);
    }

    /**
     * Creates a <code>XbcId</code> by the InputSource <code>is</code>.
     *
     * @param is
     * @exception IOException
     * @exception SAXException
     * @exception ParserConfigurationException
     */
    public XbcId(InputSource is) throws IOException, SAXException, ParserConfigurationException {
        setup(is);
    }

    /**
     * Creates a <code>XbcId</code> by the Reader <code>reader</code>.
     *
     * @param reader
     * @exception IOException
     * @exception SAXException
     * @exception ParserConfigurationException
     */
    public XbcId(Reader reader) throws IOException, SAXException, ParserConfigurationException {
        setup(reader);
    }

    /**
     * Initializes the <code>XbcId</code> by the XbcId <code>source</code>.
     *
     * @param source
     */
    public void setup(XbcId source) {
        int size;
        setSclass(source.getSclass());
        setBitField1(source.getBitField1());
        setType(source.getType());
        setIsGccThread(source.getIsGccThread());
        setIsGccExtension(source.getIsGccExtension());
        if (source.name_ != null) {
            setName((XbcName)source.getName().clone());
        }
        if (source.value_ != null) {
            setValue((XbcValue)source.getValue().clone());
        }
        if (source.bitField2_ != null) {
            setBitField2((XbcBitField)source.getBitField2().clone());
        }
        if (source.gccAttributes_ != null) {
            setGccAttributes((XbcGccAttributes)source.getGccAttributes().clone());
        }
    }

    /**
     * Initializes the <code>XbcId</code> by the Document <code>doc</code>.
     *
     * @param doc
     */
    public void setup(Document doc) {
        setup(doc.getDocumentElement());
    }

    /**
     * Initializes the <code>XbcId</code> by the Element <code>element</code>.
     *
     * @param element
     */
    public void setup(Element element) {
        init(element);
    }

    /**
     * Initializes the <code>XbcId</code> by the Stack <code>stack</code>
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
        sclass_ = URelaxer.getAttributePropertyAsString(element, "sclass");
        bitField1_ = URelaxer.getAttributePropertyAsString(element, "bit_field");
        type_ = URelaxer.getAttributePropertyAsString(element, "type");
        isGccThread_ = URelaxer.getAttributePropertyAsString(element, "is_gccThread");
        isGccExtension_ = URelaxer.getAttributePropertyAsString(element, "is_gccExtension");
        setName(factory.createXbcName(stack));
        if (XbcValue.isMatch(stack)) {
            setValue(factory.createXbcValue(stack));
        }
        if (XbcBitField.isMatch(stack)) {
            setBitField2(factory.createXbcBitField(stack));
        }
        if (XbcGccAttributes.isMatch(stack)) {
            setGccAttributes(factory.createXbcGccAttributes(stack));
        }
    }

    /**
     * @return Object
     */
    public Object clone() {
        IXcodeML_CFactory factory = XcodeML_CFactory.getFactory();
        return (factory.createXbcId(this));
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
        Element element = doc.createElement("id");
        int size;
        if (this.sclass_ != null) {
            URelaxer.setAttributePropertyByString(element, "sclass", this.sclass_);
        }
        if (this.bitField1_ != null) {
            URelaxer.setAttributePropertyByString(element, "bit_field", this.bitField1_);
        }
        if (this.type_ != null) {
            URelaxer.setAttributePropertyByString(element, "type", this.type_);
        }
        if (this.isGccThread_ != null) {
            URelaxer.setAttributePropertyByString(element, "is_gccThread", this.isGccThread_);
        }
        if (this.isGccExtension_ != null) {
            URelaxer.setAttributePropertyByString(element, "is_gccExtension", this.isGccExtension_);
        }
        this.name_.makeElement(element);
        if (this.value_ != null) {
            this.value_.makeElement(element);
        }
        if (this.bitField2_ != null) {
            this.bitField2_.makeElement(element);
        }
        if (this.gccAttributes_ != null) {
            this.gccAttributes_.makeElement(element);
        }
        parent.appendChild(element);
    }

    /**
     * Initializes the <code>XbcId</code> by the File <code>file</code>.
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
     * Initializes the <code>XbcId</code>
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
     * Initializes the <code>XbcId</code> by the URL <code>url</code>.
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
     * Initializes the <code>XbcId</code> by the InputStream <code>in</code>.
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
     * Initializes the <code>XbcId</code> by the InputSource <code>is</code>.
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
     * Initializes the <code>XbcId</code> by the Reader <code>reader</code>.
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
     * Gets the String property <b>sclass</b>.
     *
     * @return String
     */
    public final String getSclass() {
        return (sclass_);
    }

    /**
     * Sets the String property <b>sclass</b>.
     *
     * @param sclass
     */
    public final void setSclass(String sclass) {
        this.sclass_ = sclass;
    }

    /**
     * Gets the String property <b>bitField1</b>.
     *
     * @return String
     */
    public final String getBitField1() {
        return (bitField1_);
    }

    /**
     * Sets the String property <b>bitField1</b>.
     *
     * @param bitField1
     */
    public final void setBitField1(String bitField1) {
        this.bitField1_ = bitField1;
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
     * Gets the String property <b>isGccThread</b>.
     *
     * @return String
     */
    public final String getIsGccThread() {
        return (isGccThread_);
    }

    /**
     * Sets the String property <b>isGccThread</b>.
     *
     * @param isGccThread
     */
    public final void setIsGccThread(String isGccThread) {
        this.isGccThread_ = isGccThread;
    }

    /**
     * Gets the String property <b>isGccExtension</b>.
     *
     * @return String
     */
    public final String getIsGccExtension() {
        return (isGccExtension_);
    }

    /**
     * Sets the String property <b>isGccExtension</b>.
     *
     * @param isGccExtension
     */
    public final void setIsGccExtension(String isGccExtension) {
        this.isGccExtension_ = isGccExtension;
    }

    /**
     * Gets the XbcName property <b>name</b>.
     *
     * @return XbcName
     */
    public final XbcName getName() {
        return (name_);
    }

    /**
     * Sets the XbcName property <b>name</b>.
     *
     * @param name
     */
    public final void setName(XbcName name) {
        this.name_ = name;
        if (name != null) {
            name.rSetParentRNode(this);
        }
    }

    /**
     * Gets the XbcValue property <b>value</b>.
     *
     * @return XbcValue
     */
    public final XbcValue getValue() {
        return (value_);
    }

    /**
     * Sets the XbcValue property <b>value</b>.
     *
     * @param value
     */
    public final void setValue(XbcValue value) {
        this.value_ = value;
        if (value != null) {
            value.rSetParentRNode(this);
        }
    }

    /**
     * Gets the XbcBitField property <b>bitField2</b>.
     *
     * @return XbcBitField
     */
    public final XbcBitField getBitField2() {
        return (bitField2_);
    }

    /**
     * Sets the XbcBitField property <b>bitField2</b>.
     *
     * @param bitField2
     */
    public final void setBitField2(XbcBitField bitField2) {
        this.bitField2_ = bitField2;
        if (bitField2 != null) {
            bitField2.rSetParentRNode(this);
        }
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
        buffer.append("<id");
        if (sclass_ != null) {
            buffer.append(" sclass=\"");
            buffer.append(URelaxer.escapeAttrQuot(URelaxer.getString(getSclass())));
            buffer.append("\"");
        }
        if (bitField1_ != null) {
            buffer.append(" bit_field=\"");
            buffer.append(URelaxer.escapeAttrQuot(URelaxer.getString(getBitField1())));
            buffer.append("\"");
        }
        if (type_ != null) {
            buffer.append(" type=\"");
            buffer.append(URelaxer.escapeAttrQuot(URelaxer.getString(getType())));
            buffer.append("\"");
        }
        if (isGccThread_ != null) {
            buffer.append(" is_gccThread=\"");
            buffer.append(URelaxer.escapeAttrQuot(URelaxer.getString(getIsGccThread())));
            buffer.append("\"");
        }
        if (isGccExtension_ != null) {
            buffer.append(" is_gccExtension=\"");
            buffer.append(URelaxer.escapeAttrQuot(URelaxer.getString(getIsGccExtension())));
            buffer.append("\"");
        }
        buffer.append(">");
        name_.makeTextElement(buffer);
        if (value_ != null) {
            value_.makeTextElement(buffer);
        }
        if (bitField2_ != null) {
            bitField2_.makeTextElement(buffer);
        }
        if (gccAttributes_ != null) {
            gccAttributes_.makeTextElement(buffer);
        }
        buffer.append("</id>");
    }

    /**
     * Makes an XML text representation.
     *
     * @param buffer
     * @exception IOException
     */
    public void makeTextElement(Writer buffer) throws IOException {
        int size;
        buffer.write("<id");
        if (sclass_ != null) {
            buffer.write(" sclass=\"");
            buffer.write(URelaxer.escapeAttrQuot(URelaxer.getString(getSclass())));
            buffer.write("\"");
        }
        if (bitField1_ != null) {
            buffer.write(" bit_field=\"");
            buffer.write(URelaxer.escapeAttrQuot(URelaxer.getString(getBitField1())));
            buffer.write("\"");
        }
        if (type_ != null) {
            buffer.write(" type=\"");
            buffer.write(URelaxer.escapeAttrQuot(URelaxer.getString(getType())));
            buffer.write("\"");
        }
        if (isGccThread_ != null) {
            buffer.write(" is_gccThread=\"");
            buffer.write(URelaxer.escapeAttrQuot(URelaxer.getString(getIsGccThread())));
            buffer.write("\"");
        }
        if (isGccExtension_ != null) {
            buffer.write(" is_gccExtension=\"");
            buffer.write(URelaxer.escapeAttrQuot(URelaxer.getString(getIsGccExtension())));
            buffer.write("\"");
        }
        buffer.write(">");
        name_.makeTextElement(buffer);
        if (value_ != null) {
            value_.makeTextElement(buffer);
        }
        if (bitField2_ != null) {
            bitField2_.makeTextElement(buffer);
        }
        if (gccAttributes_ != null) {
            gccAttributes_.makeTextElement(buffer);
        }
        buffer.write("</id>");
    }

    /**
     * Makes an XML text representation.
     *
     * @param buffer
     */
    public void makeTextElement(PrintWriter buffer) {
        int size;
        buffer.print("<id");
        if (sclass_ != null) {
            buffer.print(" sclass=\"");
            buffer.print(URelaxer.escapeAttrQuot(URelaxer.getString(getSclass())));
            buffer.print("\"");
        }
        if (bitField1_ != null) {
            buffer.print(" bit_field=\"");
            buffer.print(URelaxer.escapeAttrQuot(URelaxer.getString(getBitField1())));
            buffer.print("\"");
        }
        if (type_ != null) {
            buffer.print(" type=\"");
            buffer.print(URelaxer.escapeAttrQuot(URelaxer.getString(getType())));
            buffer.print("\"");
        }
        if (isGccThread_ != null) {
            buffer.print(" is_gccThread=\"");
            buffer.print(URelaxer.escapeAttrQuot(URelaxer.getString(getIsGccThread())));
            buffer.print("\"");
        }
        if (isGccExtension_ != null) {
            buffer.print(" is_gccExtension=\"");
            buffer.print(URelaxer.escapeAttrQuot(URelaxer.getString(getIsGccExtension())));
            buffer.print("\"");
        }
        buffer.print(">");
        name_.makeTextElement(buffer);
        if (value_ != null) {
            value_.makeTextElement(buffer);
        }
        if (bitField2_ != null) {
            bitField2_.makeTextElement(buffer);
        }
        if (gccAttributes_ != null) {
            gccAttributes_.makeTextElement(buffer);
        }
        buffer.print("</id>");
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
    public String getSclassAsString() {
        return (URelaxer.getString(getSclass()));
    }

    /**
     * Gets the property value as String.
     *
     * @return String
     */
    public String getBitField1AsString() {
        return (URelaxer.getString(getBitField1()));
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
    public String getIsGccThreadAsString() {
        return (URelaxer.getString(getIsGccThread()));
    }

    /**
     * Gets the property value as String.
     *
     * @return String
     */
    public String getIsGccExtensionAsString() {
        return (URelaxer.getString(getIsGccExtension()));
    }

    /**
     * Sets the property value by String.
     *
     * @param string
     */
    public void setSclassByString(String string) {
        setSclass(string);
    }

    /**
     * Sets the property value by String.
     *
     * @param string
     */
    public void setBitField1ByString(String string) {
        setBitField1(string);
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
    public void setIsGccThreadByString(String string) {
        setIsGccThread(string);
    }

    /**
     * Sets the property value by String.
     *
     * @param string
     */
    public void setIsGccExtensionByString(String string) {
        setIsGccExtension(string);
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
        if (name_ != null) {
            classNodes.add(name_);
        }
        if (value_ != null) {
            classNodes.add(value_);
        }
        if (bitField2_ != null) {
            classNodes.add(bitField2_);
        }
        if (gccAttributes_ != null) {
            classNodes.add(gccAttributes_);
        }
        IRNode[] nodes = new IRNode[classNodes.size()];
        return ((IRNode[])classNodes.toArray(nodes));
    }

    /**
     * Tests if a Element <code>element</code> is valid
     * for the <code>XbcId</code>.
     *
     * @param element
     * @return boolean
     */
    public static boolean isMatch(Element element) {
        if (!URelaxer.isTargetElement(element, "id")) {
            return (false);
        }
        RStack target = new RStack(element);
        boolean $match$ = false;
        Element child;
        if (!XbcName.isMatchHungry(target)) {
            return (false);
        }
        $match$ = true;
        if (XbcValue.isMatchHungry(target)) {
        }
        if (XbcBitField.isMatchHungry(target)) {
        }
        if (XbcGccAttributes.isMatchHungry(target)) {
        }
        if (!target.isEmptyElement()) {
            return (false);
        }
        return (true);
    }

    /**
     * Tests if elements contained in a Stack <code>stack</code>
     * is valid for the <code>XbcId</code>.
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
     * is valid for the <code>XbcId</code>.
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
