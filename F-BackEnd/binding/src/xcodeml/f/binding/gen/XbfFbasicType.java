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
 * <b>XbfFbasicType</b> is generated from XcodeML_F.rng by Relaxer.
 * This class is derived from:
 * 
 * <!-- for programmer
 * <element java:extends="xcodeml.f.XmfObj" java:implements="xcodeml.f.binding.IXbfType" name="FbasicType">
 *   <ref name="defAttrTypeName"/>
 *   <ref name="defAttrRefTypeName"/>
 *   <ref name="defAttrBasicTypeModifier"/>
 *   <optional>
 *     <ref name="kind"/>
 *   </optional>
 *   <optional>
 *     <choice>
 *       <ref name="len"/>
 *       <ref name="defModelArraySubscriptSequence1"/>
 *     </choice>
 *   </optional>
 *   <optional>
 *     <ref name="coShape"/>
 *   </optional>
 * </element>
 * -->
 * <!-- for javadoc -->
 * <pre> &lt;element java:extends="xcodeml.f.XmfObj" java:implements="xcodeml.f.binding.IXbfType" name="FbasicType"&gt;
 *   &lt;ref name="defAttrTypeName"/&gt;
 *   &lt;ref name="defAttrRefTypeName"/&gt;
 *   &lt;ref name="defAttrBasicTypeModifier"/&gt;
 *   &lt;optional&gt;
 *     &lt;ref name="kind"/&gt;
 *   &lt;/optional&gt;
 *   &lt;optional&gt;
 *     &lt;choice&gt;
 *       &lt;ref name="len"/&gt;
 *       &lt;ref name="defModelArraySubscriptSequence1"/&gt;
 *     &lt;/choice&gt;
 *   &lt;/optional&gt;
 *   &lt;optional&gt;
 *     &lt;ref name="coShape"/&gt;
 *   &lt;/optional&gt;
 * &lt;/element&gt;
 * </pre>
 *
 * @version XcodeML_F.rng (Mon Jan 23 20:53:32 JST 2012)
 * @author  Relaxer 1.0 (http://www.relaxer.org)
 */
public class XbfFbasicType extends xcodeml.f.XmfObj implements java.io.Serializable, Cloneable, xcodeml.f.binding.IXbfType, IRVisitable, IRNode, IXbfTypeTableChoice {
    public static final String INTENT_IN = "in";
    public static final String INTENT_OUT = "out";
    public static final String INTENT_INOUT = "inout";

    private String type_;
    private String ref_;
    private Boolean isPublic_;
    private Boolean isPrivate_;
    private Boolean isPointer_;
    private Boolean isTarget_;
    private Boolean isOptional_;
    private Boolean isSave_;
    private Boolean isParameter_;
    private Boolean isAllocatable_;
    private String intent_;
    private XbfKind kind_;
    private IXbfFbasicTypeChoice content_;
    private XbfCoShape coShape_;
    private IRNode parentRNode_;

    /**
     * Creates a <code>XbfFbasicType</code>.
     *
     */
    public XbfFbasicType() {
    }

    /**
     * Creates a <code>XbfFbasicType</code>.
     *
     * @param source
     */
    public XbfFbasicType(XbfFbasicType source) {
        setup(source);
    }

    /**
     * Creates a <code>XbfFbasicType</code> by the Stack <code>stack</code>
     * that contains Elements.
     * This constructor is supposed to be used internally
     * by the Relaxer system.
     *
     * @param stack
     */
    public XbfFbasicType(RStack stack) {
        setup(stack);
    }

    /**
     * Creates a <code>XbfFbasicType</code> by the Document <code>doc</code>.
     *
     * @param doc
     */
    public XbfFbasicType(Document doc) {
        setup(doc.getDocumentElement());
    }

    /**
     * Creates a <code>XbfFbasicType</code> by the Element <code>element</code>.
     *
     * @param element
     */
    public XbfFbasicType(Element element) {
        setup(element);
    }

    /**
     * Creates a <code>XbfFbasicType</code> by the File <code>file</code>.
     *
     * @param file
     * @exception IOException
     * @exception SAXException
     * @exception ParserConfigurationException
     */
    public XbfFbasicType(File file) throws IOException, SAXException, ParserConfigurationException {
        setup(file);
    }

    /**
     * Creates a <code>XbfFbasicType</code>
     * by the String representation of URI <code>uri</code>.
     *
     * @param uri
     * @exception IOException
     * @exception SAXException
     * @exception ParserConfigurationException
     */
    public XbfFbasicType(String uri) throws IOException, SAXException, ParserConfigurationException {
        setup(uri);
    }

    /**
     * Creates a <code>XbfFbasicType</code> by the URL <code>url</code>.
     *
     * @param url
     * @exception IOException
     * @exception SAXException
     * @exception ParserConfigurationException
     */
    public XbfFbasicType(URL url) throws IOException, SAXException, ParserConfigurationException {
        setup(url);
    }

    /**
     * Creates a <code>XbfFbasicType</code> by the InputStream <code>in</code>.
     *
     * @param in
     * @exception IOException
     * @exception SAXException
     * @exception ParserConfigurationException
     */
    public XbfFbasicType(InputStream in) throws IOException, SAXException, ParserConfigurationException {
        setup(in);
    }

    /**
     * Creates a <code>XbfFbasicType</code> by the InputSource <code>is</code>.
     *
     * @param is
     * @exception IOException
     * @exception SAXException
     * @exception ParserConfigurationException
     */
    public XbfFbasicType(InputSource is) throws IOException, SAXException, ParserConfigurationException {
        setup(is);
    }

    /**
     * Creates a <code>XbfFbasicType</code> by the Reader <code>reader</code>.
     *
     * @param reader
     * @exception IOException
     * @exception SAXException
     * @exception ParserConfigurationException
     */
    public XbfFbasicType(Reader reader) throws IOException, SAXException, ParserConfigurationException {
        setup(reader);
    }

    /**
     * Initializes the <code>XbfFbasicType</code> by the XbfFbasicType <code>source</code>.
     *
     * @param source
     */
    public void setup(XbfFbasicType source) {
        int size;
        setType(source.getType());
        setRef(source.getRef());
        setIsPublic(source.getIsPublic());
        setIsPrivate(source.getIsPrivate());
        setIsPointer(source.getIsPointer());
        setIsTarget(source.getIsTarget());
        setIsOptional(source.getIsOptional());
        setIsSave(source.getIsSave());
        setIsParameter(source.getIsParameter());
        setIsAllocatable(source.getIsAllocatable());
        setIntent(source.getIntent());
        if (source.kind_ != null) {
            setKind((XbfKind)source.getKind().clone());
        }
        if (source.content_ != null) {
            setContent((IXbfFbasicTypeChoice)source.getContent().clone());
        }
        if (source.coShape_ != null) {
            setCoShape((XbfCoShape)source.getCoShape().clone());
        }
    }

    /**
     * Initializes the <code>XbfFbasicType</code> by the Document <code>doc</code>.
     *
     * @param doc
     */
    public void setup(Document doc) {
        setup(doc.getDocumentElement());
    }

    /**
     * Initializes the <code>XbfFbasicType</code> by the Element <code>element</code>.
     *
     * @param element
     */
    public void setup(Element element) {
        init(element);
    }

    /**
     * Initializes the <code>XbfFbasicType</code> by the Stack <code>stack</code>
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
        ref_ = URelaxer.getAttributePropertyAsString(element, "ref");
        isPublic_ = URelaxer.getAttributePropertyAsBooleanObject(element, "is_public");
        isPrivate_ = URelaxer.getAttributePropertyAsBooleanObject(element, "is_private");
        isPointer_ = URelaxer.getAttributePropertyAsBooleanObject(element, "is_pointer");
        isTarget_ = URelaxer.getAttributePropertyAsBooleanObject(element, "is_target");
        isOptional_ = URelaxer.getAttributePropertyAsBooleanObject(element, "is_optional");
        isSave_ = URelaxer.getAttributePropertyAsBooleanObject(element, "is_save");
        isParameter_ = URelaxer.getAttributePropertyAsBooleanObject(element, "is_parameter");
        isAllocatable_ = URelaxer.getAttributePropertyAsBooleanObject(element, "is_allocatable");
        intent_ = URelaxer.getAttributePropertyAsString(element, "intent");
        if (XbfKind.isMatch(stack)) {
            setKind(factory.createXbfKind(stack));
        }
        if (XbfLen.isMatch(stack)) {
            setContent(factory.createXbfLen(stack));
        } else if (XbfDefModelArraySubscriptSequence1.isMatch(stack)) {
            setContent(factory.createXbfDefModelArraySubscriptSequence1(stack));
        } else {
        }
        if (XbfCoShape.isMatch(stack)) {
            setCoShape(factory.createXbfCoShape(stack));
        }
    }

    /**
     * @return Object
     */
    public Object clone() {
        IXcodeML_FFactory factory = XcodeML_FFactory.getFactory();
        return (factory.createXbfFbasicType(this));
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
        Element element = doc.createElement("FbasicType");
        int size;
        if (this.type_ != null) {
            URelaxer.setAttributePropertyByString(element, "type", this.type_);
        }
        if (this.ref_ != null) {
            URelaxer.setAttributePropertyByString(element, "ref", this.ref_);
        }
        if (this.isPublic_ != null) {
            URelaxer.setAttributePropertyByBoolean(element, "is_public", this.isPublic_);
        }
        if (this.isPrivate_ != null) {
            URelaxer.setAttributePropertyByBoolean(element, "is_private", this.isPrivate_);
        }
        if (this.isPointer_ != null) {
            URelaxer.setAttributePropertyByBoolean(element, "is_pointer", this.isPointer_);
        }
        if (this.isTarget_ != null) {
            URelaxer.setAttributePropertyByBoolean(element, "is_target", this.isTarget_);
        }
        if (this.isOptional_ != null) {
            URelaxer.setAttributePropertyByBoolean(element, "is_optional", this.isOptional_);
        }
        if (this.isSave_ != null) {
            URelaxer.setAttributePropertyByBoolean(element, "is_save", this.isSave_);
        }
        if (this.isParameter_ != null) {
            URelaxer.setAttributePropertyByBoolean(element, "is_parameter", this.isParameter_);
        }
        if (this.isAllocatable_ != null) {
            URelaxer.setAttributePropertyByBoolean(element, "is_allocatable", this.isAllocatable_);
        }
        if (this.intent_ != null) {
            URelaxer.setAttributePropertyByString(element, "intent", this.intent_);
        }
        if (this.kind_ != null) {
            this.kind_.makeElement(element);
        }
        if (this.content_ != null) {
            this.content_.makeElement(element);
        }
        if (this.coShape_ != null) {
            this.coShape_.makeElement(element);
        }
        parent.appendChild(element);
    }

    /**
     * Initializes the <code>XbfFbasicType</code> by the File <code>file</code>.
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
     * Initializes the <code>XbfFbasicType</code>
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
     * Initializes the <code>XbfFbasicType</code> by the URL <code>url</code>.
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
     * Initializes the <code>XbfFbasicType</code> by the InputStream <code>in</code>.
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
     * Initializes the <code>XbfFbasicType</code> by the InputSource <code>is</code>.
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
     * Initializes the <code>XbfFbasicType</code> by the Reader <code>reader</code>.
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
     * Gets the String property <b>ref</b>.
     *
     * @return String
     */
    public final String getRef() {
        return (ref_);
    }

    /**
     * Sets the String property <b>ref</b>.
     *
     * @param ref
     */
    public final void setRef(String ref) {
        this.ref_ = ref;
    }

    /**
     * Gets the boolean property <b>isPublic</b>.
     *
     * @return boolean
     */
    public boolean getIsPublic() {
        if (isPublic_ == null) {
            return(false);
        }
        return (isPublic_.booleanValue());
    }

    /**
     * Gets the boolean property <b>isPublic</b>.
     *
     * @param isPublic
     * @return boolean
     */
    public boolean getIsPublic(boolean isPublic) {
        if (isPublic_ == null) {
            return(isPublic);
        }
        return (this.isPublic_.booleanValue());
    }

    /**
     * Gets the boolean property <b>isPublic</b>.
     *
     * @return Boolean
     */
    public Boolean getIsPublicAsBoolean() {
        return (isPublic_);
    }

    /**
     * Check the boolean property <b>isPublic</b>.
     *
     * @return boolean
     */
    public boolean checkIsPublic() {
        return (isPublic_ != null);
    }

    /**
     * Sets the boolean property <b>isPublic</b>.
     *
     * @param isPublic
     */
    public void setIsPublic(boolean isPublic) {
        this.isPublic_ = new Boolean(isPublic);
    }

    /**
     * Sets the boolean property <b>isPublic</b>.
     *
     * @param isPublic
     */
    public void setIsPublic(Boolean isPublic) {
        this.isPublic_ = isPublic;
    }

    /**
     * Gets the boolean property <b>isPrivate</b>.
     *
     * @return boolean
     */
    public boolean getIsPrivate() {
        if (isPrivate_ == null) {
            return(false);
        }
        return (isPrivate_.booleanValue());
    }

    /**
     * Gets the boolean property <b>isPrivate</b>.
     *
     * @param isPrivate
     * @return boolean
     */
    public boolean getIsPrivate(boolean isPrivate) {
        if (isPrivate_ == null) {
            return(isPrivate);
        }
        return (this.isPrivate_.booleanValue());
    }

    /**
     * Gets the boolean property <b>isPrivate</b>.
     *
     * @return Boolean
     */
    public Boolean getIsPrivateAsBoolean() {
        return (isPrivate_);
    }

    /**
     * Check the boolean property <b>isPrivate</b>.
     *
     * @return boolean
     */
    public boolean checkIsPrivate() {
        return (isPrivate_ != null);
    }

    /**
     * Sets the boolean property <b>isPrivate</b>.
     *
     * @param isPrivate
     */
    public void setIsPrivate(boolean isPrivate) {
        this.isPrivate_ = new Boolean(isPrivate);
    }

    /**
     * Sets the boolean property <b>isPrivate</b>.
     *
     * @param isPrivate
     */
    public void setIsPrivate(Boolean isPrivate) {
        this.isPrivate_ = isPrivate;
    }

    /**
     * Gets the boolean property <b>isPointer</b>.
     *
     * @return boolean
     */
    public boolean getIsPointer() {
        if (isPointer_ == null) {
            return(false);
        }
        return (isPointer_.booleanValue());
    }

    /**
     * Gets the boolean property <b>isPointer</b>.
     *
     * @param isPointer
     * @return boolean
     */
    public boolean getIsPointer(boolean isPointer) {
        if (isPointer_ == null) {
            return(isPointer);
        }
        return (this.isPointer_.booleanValue());
    }

    /**
     * Gets the boolean property <b>isPointer</b>.
     *
     * @return Boolean
     */
    public Boolean getIsPointerAsBoolean() {
        return (isPointer_);
    }

    /**
     * Check the boolean property <b>isPointer</b>.
     *
     * @return boolean
     */
    public boolean checkIsPointer() {
        return (isPointer_ != null);
    }

    /**
     * Sets the boolean property <b>isPointer</b>.
     *
     * @param isPointer
     */
    public void setIsPointer(boolean isPointer) {
        this.isPointer_ = new Boolean(isPointer);
    }

    /**
     * Sets the boolean property <b>isPointer</b>.
     *
     * @param isPointer
     */
    public void setIsPointer(Boolean isPointer) {
        this.isPointer_ = isPointer;
    }

    /**
     * Gets the boolean property <b>isTarget</b>.
     *
     * @return boolean
     */
    public boolean getIsTarget() {
        if (isTarget_ == null) {
            return(false);
        }
        return (isTarget_.booleanValue());
    }

    /**
     * Gets the boolean property <b>isTarget</b>.
     *
     * @param isTarget
     * @return boolean
     */
    public boolean getIsTarget(boolean isTarget) {
        if (isTarget_ == null) {
            return(isTarget);
        }
        return (this.isTarget_.booleanValue());
    }

    /**
     * Gets the boolean property <b>isTarget</b>.
     *
     * @return Boolean
     */
    public Boolean getIsTargetAsBoolean() {
        return (isTarget_);
    }

    /**
     * Check the boolean property <b>isTarget</b>.
     *
     * @return boolean
     */
    public boolean checkIsTarget() {
        return (isTarget_ != null);
    }

    /**
     * Sets the boolean property <b>isTarget</b>.
     *
     * @param isTarget
     */
    public void setIsTarget(boolean isTarget) {
        this.isTarget_ = new Boolean(isTarget);
    }

    /**
     * Sets the boolean property <b>isTarget</b>.
     *
     * @param isTarget
     */
    public void setIsTarget(Boolean isTarget) {
        this.isTarget_ = isTarget;
    }

    /**
     * Gets the boolean property <b>isOptional</b>.
     *
     * @return boolean
     */
    public boolean getIsOptional() {
        if (isOptional_ == null) {
            return(false);
        }
        return (isOptional_.booleanValue());
    }

    /**
     * Gets the boolean property <b>isOptional</b>.
     *
     * @param isOptional
     * @return boolean
     */
    public boolean getIsOptional(boolean isOptional) {
        if (isOptional_ == null) {
            return(isOptional);
        }
        return (this.isOptional_.booleanValue());
    }

    /**
     * Gets the boolean property <b>isOptional</b>.
     *
     * @return Boolean
     */
    public Boolean getIsOptionalAsBoolean() {
        return (isOptional_);
    }

    /**
     * Check the boolean property <b>isOptional</b>.
     *
     * @return boolean
     */
    public boolean checkIsOptional() {
        return (isOptional_ != null);
    }

    /**
     * Sets the boolean property <b>isOptional</b>.
     *
     * @param isOptional
     */
    public void setIsOptional(boolean isOptional) {
        this.isOptional_ = new Boolean(isOptional);
    }

    /**
     * Sets the boolean property <b>isOptional</b>.
     *
     * @param isOptional
     */
    public void setIsOptional(Boolean isOptional) {
        this.isOptional_ = isOptional;
    }

    /**
     * Gets the boolean property <b>isSave</b>.
     *
     * @return boolean
     */
    public boolean getIsSave() {
        if (isSave_ == null) {
            return(false);
        }
        return (isSave_.booleanValue());
    }

    /**
     * Gets the boolean property <b>isSave</b>.
     *
     * @param isSave
     * @return boolean
     */
    public boolean getIsSave(boolean isSave) {
        if (isSave_ == null) {
            return(isSave);
        }
        return (this.isSave_.booleanValue());
    }

    /**
     * Gets the boolean property <b>isSave</b>.
     *
     * @return Boolean
     */
    public Boolean getIsSaveAsBoolean() {
        return (isSave_);
    }

    /**
     * Check the boolean property <b>isSave</b>.
     *
     * @return boolean
     */
    public boolean checkIsSave() {
        return (isSave_ != null);
    }

    /**
     * Sets the boolean property <b>isSave</b>.
     *
     * @param isSave
     */
    public void setIsSave(boolean isSave) {
        this.isSave_ = new Boolean(isSave);
    }

    /**
     * Sets the boolean property <b>isSave</b>.
     *
     * @param isSave
     */
    public void setIsSave(Boolean isSave) {
        this.isSave_ = isSave;
    }

    /**
     * Gets the boolean property <b>isParameter</b>.
     *
     * @return boolean
     */
    public boolean getIsParameter() {
        if (isParameter_ == null) {
            return(false);
        }
        return (isParameter_.booleanValue());
    }

    /**
     * Gets the boolean property <b>isParameter</b>.
     *
     * @param isParameter
     * @return boolean
     */
    public boolean getIsParameter(boolean isParameter) {
        if (isParameter_ == null) {
            return(isParameter);
        }
        return (this.isParameter_.booleanValue());
    }

    /**
     * Gets the boolean property <b>isParameter</b>.
     *
     * @return Boolean
     */
    public Boolean getIsParameterAsBoolean() {
        return (isParameter_);
    }

    /**
     * Check the boolean property <b>isParameter</b>.
     *
     * @return boolean
     */
    public boolean checkIsParameter() {
        return (isParameter_ != null);
    }

    /**
     * Sets the boolean property <b>isParameter</b>.
     *
     * @param isParameter
     */
    public void setIsParameter(boolean isParameter) {
        this.isParameter_ = new Boolean(isParameter);
    }

    /**
     * Sets the boolean property <b>isParameter</b>.
     *
     * @param isParameter
     */
    public void setIsParameter(Boolean isParameter) {
        this.isParameter_ = isParameter;
    }

    /**
     * Gets the boolean property <b>isAllocatable</b>.
     *
     * @return boolean
     */
    public boolean getIsAllocatable() {
        if (isAllocatable_ == null) {
            return(false);
        }
        return (isAllocatable_.booleanValue());
    }

    /**
     * Gets the boolean property <b>isAllocatable</b>.
     *
     * @param isAllocatable
     * @return boolean
     */
    public boolean getIsAllocatable(boolean isAllocatable) {
        if (isAllocatable_ == null) {
            return(isAllocatable);
        }
        return (this.isAllocatable_.booleanValue());
    }

    /**
     * Gets the boolean property <b>isAllocatable</b>.
     *
     * @return Boolean
     */
    public Boolean getIsAllocatableAsBoolean() {
        return (isAllocatable_);
    }

    /**
     * Check the boolean property <b>isAllocatable</b>.
     *
     * @return boolean
     */
    public boolean checkIsAllocatable() {
        return (isAllocatable_ != null);
    }

    /**
     * Sets the boolean property <b>isAllocatable</b>.
     *
     * @param isAllocatable
     */
    public void setIsAllocatable(boolean isAllocatable) {
        this.isAllocatable_ = new Boolean(isAllocatable);
    }

    /**
     * Sets the boolean property <b>isAllocatable</b>.
     *
     * @param isAllocatable
     */
    public void setIsAllocatable(Boolean isAllocatable) {
        this.isAllocatable_ = isAllocatable;
    }

    /**
     * Gets the String property <b>intent</b>.
     *
     * @return String
     */
    public final String getIntent() {
        return (intent_);
    }

    /**
     * Sets the String property <b>intent</b>.
     *
     * @param intent
     */
    public final void setIntent(String intent) {
        this.intent_ = intent;
    }

    /**
     * Gets the XbfKind property <b>kind</b>.
     *
     * @return XbfKind
     */
    public final XbfKind getKind() {
        return (kind_);
    }

    /**
     * Sets the XbfKind property <b>kind</b>.
     *
     * @param kind
     */
    public final void setKind(XbfKind kind) {
        this.kind_ = kind;
        if (kind != null) {
            kind.rSetParentRNode(this);
        }
    }

    /**
     * Gets the IXbfFbasicTypeChoice property <b>content</b>.
     *
     * @return IXbfFbasicTypeChoice
     */
    public final IXbfFbasicTypeChoice getContent() {
        return (content_);
    }

    /**
     * Sets the IXbfFbasicTypeChoice property <b>content</b>.
     *
     * @param content
     */
    public final void setContent(IXbfFbasicTypeChoice content) {
        this.content_ = content;
        if (content != null) {
            content.rSetParentRNode(this);
        }
    }

    /**
     * Gets the XbfCoShape property <b>coShape</b>.
     *
     * @return XbfCoShape
     */
    public final XbfCoShape getCoShape() {
        return (coShape_);
    }

    /**
     * Sets the XbfCoShape property <b>coShape</b>.
     *
     * @param coShape
     */
    public final void setCoShape(XbfCoShape coShape) {
        this.coShape_ = coShape;
        if (coShape != null) {
            coShape.rSetParentRNode(this);
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
        buffer.append("<FbasicType");
        if (type_ != null) {
            buffer.append(" type=\"");
            buffer.append(URelaxer.escapeAttrQuot(URelaxer.getString(getType())));
            buffer.append("\"");
        }
        if (ref_ != null) {
            buffer.append(" ref=\"");
            buffer.append(URelaxer.escapeAttrQuot(URelaxer.getString(getRef())));
            buffer.append("\"");
        }
        if (isPublic_ != null) {
            buffer.append(" is_public=\"");
            buffer.append(URelaxer.getString(getIsPublic()));
            buffer.append("\"");
        }
        if (isPrivate_ != null) {
            buffer.append(" is_private=\"");
            buffer.append(URelaxer.getString(getIsPrivate()));
            buffer.append("\"");
        }
        if (isPointer_ != null) {
            buffer.append(" is_pointer=\"");
            buffer.append(URelaxer.getString(getIsPointer()));
            buffer.append("\"");
        }
        if (isTarget_ != null) {
            buffer.append(" is_target=\"");
            buffer.append(URelaxer.getString(getIsTarget()));
            buffer.append("\"");
        }
        if (isOptional_ != null) {
            buffer.append(" is_optional=\"");
            buffer.append(URelaxer.getString(getIsOptional()));
            buffer.append("\"");
        }
        if (isSave_ != null) {
            buffer.append(" is_save=\"");
            buffer.append(URelaxer.getString(getIsSave()));
            buffer.append("\"");
        }
        if (isParameter_ != null) {
            buffer.append(" is_parameter=\"");
            buffer.append(URelaxer.getString(getIsParameter()));
            buffer.append("\"");
        }
        if (isAllocatable_ != null) {
            buffer.append(" is_allocatable=\"");
            buffer.append(URelaxer.getString(getIsAllocatable()));
            buffer.append("\"");
        }
        if (intent_ != null) {
            buffer.append(" intent=\"");
            buffer.append(URelaxer.escapeAttrQuot(URelaxer.getString(getIntent())));
            buffer.append("\"");
        }
        if (content_ != null) {
            content_.makeTextAttribute(buffer);
        }
        buffer.append(">");
        if (kind_ != null) {
            kind_.makeTextElement(buffer);
        }
        if (content_ != null) {
            content_.makeTextElement(buffer);
        }
        if (coShape_ != null) {
            coShape_.makeTextElement(buffer);
        }
        buffer.append("</FbasicType>");
    }

    /**
     * Makes an XML text representation.
     *
     * @param buffer
     * @exception IOException
     */
    public void makeTextElement(Writer buffer) throws IOException {
        int size;
        buffer.write("<FbasicType");
        if (type_ != null) {
            buffer.write(" type=\"");
            buffer.write(URelaxer.escapeAttrQuot(URelaxer.getString(getType())));
            buffer.write("\"");
        }
        if (ref_ != null) {
            buffer.write(" ref=\"");
            buffer.write(URelaxer.escapeAttrQuot(URelaxer.getString(getRef())));
            buffer.write("\"");
        }
        if (isPublic_ != null) {
            buffer.write(" is_public=\"");
            buffer.write(URelaxer.getString(getIsPublic()));
            buffer.write("\"");
        }
        if (isPrivate_ != null) {
            buffer.write(" is_private=\"");
            buffer.write(URelaxer.getString(getIsPrivate()));
            buffer.write("\"");
        }
        if (isPointer_ != null) {
            buffer.write(" is_pointer=\"");
            buffer.write(URelaxer.getString(getIsPointer()));
            buffer.write("\"");
        }
        if (isTarget_ != null) {
            buffer.write(" is_target=\"");
            buffer.write(URelaxer.getString(getIsTarget()));
            buffer.write("\"");
        }
        if (isOptional_ != null) {
            buffer.write(" is_optional=\"");
            buffer.write(URelaxer.getString(getIsOptional()));
            buffer.write("\"");
        }
        if (isSave_ != null) {
            buffer.write(" is_save=\"");
            buffer.write(URelaxer.getString(getIsSave()));
            buffer.write("\"");
        }
        if (isParameter_ != null) {
            buffer.write(" is_parameter=\"");
            buffer.write(URelaxer.getString(getIsParameter()));
            buffer.write("\"");
        }
        if (isAllocatable_ != null) {
            buffer.write(" is_allocatable=\"");
            buffer.write(URelaxer.getString(getIsAllocatable()));
            buffer.write("\"");
        }
        if (intent_ != null) {
            buffer.write(" intent=\"");
            buffer.write(URelaxer.escapeAttrQuot(URelaxer.getString(getIntent())));
            buffer.write("\"");
        }
        if (content_ != null) {
            content_.makeTextAttribute(buffer);
        }
        buffer.write(">");
        if (kind_ != null) {
            kind_.makeTextElement(buffer);
        }
        if (content_ != null) {
            content_.makeTextElement(buffer);
        }
        if (coShape_ != null) {
            coShape_.makeTextElement(buffer);
        }
        buffer.write("</FbasicType>");
    }

    /**
     * Makes an XML text representation.
     *
     * @param buffer
     */
    public void makeTextElement(PrintWriter buffer) {
        int size;
        buffer.print("<FbasicType");
        if (type_ != null) {
            buffer.print(" type=\"");
            buffer.print(URelaxer.escapeAttrQuot(URelaxer.getString(getType())));
            buffer.print("\"");
        }
        if (ref_ != null) {
            buffer.print(" ref=\"");
            buffer.print(URelaxer.escapeAttrQuot(URelaxer.getString(getRef())));
            buffer.print("\"");
        }
        if (isPublic_ != null) {
            buffer.print(" is_public=\"");
            buffer.print(URelaxer.getString(getIsPublic()));
            buffer.print("\"");
        }
        if (isPrivate_ != null) {
            buffer.print(" is_private=\"");
            buffer.print(URelaxer.getString(getIsPrivate()));
            buffer.print("\"");
        }
        if (isPointer_ != null) {
            buffer.print(" is_pointer=\"");
            buffer.print(URelaxer.getString(getIsPointer()));
            buffer.print("\"");
        }
        if (isTarget_ != null) {
            buffer.print(" is_target=\"");
            buffer.print(URelaxer.getString(getIsTarget()));
            buffer.print("\"");
        }
        if (isOptional_ != null) {
            buffer.print(" is_optional=\"");
            buffer.print(URelaxer.getString(getIsOptional()));
            buffer.print("\"");
        }
        if (isSave_ != null) {
            buffer.print(" is_save=\"");
            buffer.print(URelaxer.getString(getIsSave()));
            buffer.print("\"");
        }
        if (isParameter_ != null) {
            buffer.print(" is_parameter=\"");
            buffer.print(URelaxer.getString(getIsParameter()));
            buffer.print("\"");
        }
        if (isAllocatable_ != null) {
            buffer.print(" is_allocatable=\"");
            buffer.print(URelaxer.getString(getIsAllocatable()));
            buffer.print("\"");
        }
        if (intent_ != null) {
            buffer.print(" intent=\"");
            buffer.print(URelaxer.escapeAttrQuot(URelaxer.getString(getIntent())));
            buffer.print("\"");
        }
        if (content_ != null) {
            content_.makeTextAttribute(buffer);
        }
        buffer.print(">");
        if (kind_ != null) {
            kind_.makeTextElement(buffer);
        }
        if (content_ != null) {
            content_.makeTextElement(buffer);
        }
        if (coShape_ != null) {
            coShape_.makeTextElement(buffer);
        }
        buffer.print("</FbasicType>");
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
    public String getRefAsString() {
        return (URelaxer.getString(getRef()));
    }

    /**
     * Gets the property value as String.
     *
     * @return String
     */
    public String getIsPublicAsString() {
        return (URelaxer.getString(getIsPublic()));
    }

    /**
     * Gets the property value as String.
     *
     * @return String
     */
    public String getIsPrivateAsString() {
        return (URelaxer.getString(getIsPrivate()));
    }

    /**
     * Gets the property value as String.
     *
     * @return String
     */
    public String getIsPointerAsString() {
        return (URelaxer.getString(getIsPointer()));
    }

    /**
     * Gets the property value as String.
     *
     * @return String
     */
    public String getIsTargetAsString() {
        return (URelaxer.getString(getIsTarget()));
    }

    /**
     * Gets the property value as String.
     *
     * @return String
     */
    public String getIsOptionalAsString() {
        return (URelaxer.getString(getIsOptional()));
    }

    /**
     * Gets the property value as String.
     *
     * @return String
     */
    public String getIsSaveAsString() {
        return (URelaxer.getString(getIsSave()));
    }

    /**
     * Gets the property value as String.
     *
     * @return String
     */
    public String getIsParameterAsString() {
        return (URelaxer.getString(getIsParameter()));
    }

    /**
     * Gets the property value as String.
     *
     * @return String
     */
    public String getIsAllocatableAsString() {
        return (URelaxer.getString(getIsAllocatable()));
    }

    /**
     * Gets the property value as String.
     *
     * @return String
     */
    public String getIntentAsString() {
        return (URelaxer.getString(getIntent()));
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
    public void setRefByString(String string) {
        setRef(string);
    }

    /**
     * Sets the property value by String.
     *
     * @param string
     */
    public void setIsPublicByString(String string) {
        setIsPublic(new Boolean(string).booleanValue());
    }

    /**
     * Sets the property value by String.
     *
     * @param string
     */
    public void setIsPrivateByString(String string) {
        setIsPrivate(new Boolean(string).booleanValue());
    }

    /**
     * Sets the property value by String.
     *
     * @param string
     */
    public void setIsPointerByString(String string) {
        setIsPointer(new Boolean(string).booleanValue());
    }

    /**
     * Sets the property value by String.
     *
     * @param string
     */
    public void setIsTargetByString(String string) {
        setIsTarget(new Boolean(string).booleanValue());
    }

    /**
     * Sets the property value by String.
     *
     * @param string
     */
    public void setIsOptionalByString(String string) {
        setIsOptional(new Boolean(string).booleanValue());
    }

    /**
     * Sets the property value by String.
     *
     * @param string
     */
    public void setIsSaveByString(String string) {
        setIsSave(new Boolean(string).booleanValue());
    }

    /**
     * Sets the property value by String.
     *
     * @param string
     */
    public void setIsParameterByString(String string) {
        setIsParameter(new Boolean(string).booleanValue());
    }

    /**
     * Sets the property value by String.
     *
     * @param string
     */
    public void setIsAllocatableByString(String string) {
        setIsAllocatable(new Boolean(string).booleanValue());
    }

    /**
     * Sets the property value by String.
     *
     * @param string
     */
    public void setIntentByString(String string) {
        setIntent(string);
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
        if (kind_ != null) {
            classNodes.add(kind_);
        }
        if (content_ != null) {
            classNodes.add(content_);
        }
        if (coShape_ != null) {
            classNodes.add(coShape_);
        }
        IRNode[] nodes = new IRNode[classNodes.size()];
        return ((IRNode[])classNodes.toArray(nodes));
    }

    /**
     * Tests if a Element <code>element</code> is valid
     * for the <code>XbfFbasicType</code>.
     *
     * @param element
     * @return boolean
     */
    public static boolean isMatch(Element element) {
        if (!URelaxer.isTargetElement(element, "FbasicType")) {
            return (false);
        }
        RStack target = new RStack(element);
        boolean $match$ = false;
        Element child;
        if (XbfKind.isMatchHungry(target)) {
        }
        if (XbfLen.isMatchHungry(target)) {
            $match$ = true;
        } else if (XbfDefModelArraySubscriptSequence1.isMatchHungry(target)) {
            $match$ = true;
        } else {
        }
        if (XbfCoShape.isMatchHungry(target)) {
        }
        if (!target.isEmptyElement()) {
            return (false);
        }
        return (true);
    }

    /**
     * Tests if elements contained in a Stack <code>stack</code>
     * is valid for the <code>XbfFbasicType</code>.
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
     * is valid for the <code>XbfFbasicType</code>.
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
