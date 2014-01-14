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
 * <b>XbfFfunctionType</b> is generated from XcodeML_F.rng by Relaxer.
 * This class is derived from:
 * 
 * <!-- for programmer
 * <element java:extends="xcodeml.f.XmfObj" java:implements="xcodeml.f.binding.IXbfType" name="FfunctionType">
 *   <ref name="defAttrTypeName"/>
 *   <ref name="defAttrReturnTypeName"/>
 *   <ref name="defAttrResultName"/>
 *   <ref name="defAttrFunctionTypeModifier"/>
 *   <optional>
 *     <ref name="params"/>
 *   </optional>
 * </element>
 * -->
 * <!-- for javadoc -->
 * <pre> &lt;element java:extends="xcodeml.f.XmfObj" java:implements="xcodeml.f.binding.IXbfType" name="FfunctionType"&gt;
 *   &lt;ref name="defAttrTypeName"/&gt;
 *   &lt;ref name="defAttrReturnTypeName"/&gt;
 *   &lt;ref name="defAttrResultName"/&gt;
 *   &lt;ref name="defAttrFunctionTypeModifier"/&gt;
 *   &lt;optional&gt;
 *     &lt;ref name="params"/&gt;
 *   &lt;/optional&gt;
 * &lt;/element&gt;
 * </pre>
 *
 * @version XcodeML_F.rng (Mon Jan 23 20:53:32 JST 2012)
 * @author  Relaxer 1.0 (http://www.relaxer.org)
 */
public class XbfFfunctionType extends xcodeml.f.XmfObj implements java.io.Serializable, Cloneable, xcodeml.f.binding.IXbfType, IRVisitable, IRNode, IXbfTypeTableChoice {
    private String type_;
    private String returnType_;
    private String resultName_;
    private Boolean isRecursive_;
    private Boolean isProgram_;
    private Boolean isInternal_;
    private Boolean isIntrinsic_;
    private Boolean isExternal_;
    private Boolean isPublic_;
    private Boolean isPrivate_;
    private Boolean isPure_;
    private Boolean isElemental_;
    private XbfParams params_;
    private IRNode parentRNode_;

    /**
     * Creates a <code>XbfFfunctionType</code>.
     *
     */
    public XbfFfunctionType() {
    }

    /**
     * Creates a <code>XbfFfunctionType</code>.
     *
     * @param source
     */
    public XbfFfunctionType(XbfFfunctionType source) {
        setup(source);
    }

    /**
     * Creates a <code>XbfFfunctionType</code> by the Stack <code>stack</code>
     * that contains Elements.
     * This constructor is supposed to be used internally
     * by the Relaxer system.
     *
     * @param stack
     */
    public XbfFfunctionType(RStack stack) {
        setup(stack);
    }

    /**
     * Creates a <code>XbfFfunctionType</code> by the Document <code>doc</code>.
     *
     * @param doc
     */
    public XbfFfunctionType(Document doc) {
        setup(doc.getDocumentElement());
    }

    /**
     * Creates a <code>XbfFfunctionType</code> by the Element <code>element</code>.
     *
     * @param element
     */
    public XbfFfunctionType(Element element) {
        setup(element);
    }

    /**
     * Creates a <code>XbfFfunctionType</code> by the File <code>file</code>.
     *
     * @param file
     * @exception IOException
     * @exception SAXException
     * @exception ParserConfigurationException
     */
    public XbfFfunctionType(File file) throws IOException, SAXException, ParserConfigurationException {
        setup(file);
    }

    /**
     * Creates a <code>XbfFfunctionType</code>
     * by the String representation of URI <code>uri</code>.
     *
     * @param uri
     * @exception IOException
     * @exception SAXException
     * @exception ParserConfigurationException
     */
    public XbfFfunctionType(String uri) throws IOException, SAXException, ParserConfigurationException {
        setup(uri);
    }

    /**
     * Creates a <code>XbfFfunctionType</code> by the URL <code>url</code>.
     *
     * @param url
     * @exception IOException
     * @exception SAXException
     * @exception ParserConfigurationException
     */
    public XbfFfunctionType(URL url) throws IOException, SAXException, ParserConfigurationException {
        setup(url);
    }

    /**
     * Creates a <code>XbfFfunctionType</code> by the InputStream <code>in</code>.
     *
     * @param in
     * @exception IOException
     * @exception SAXException
     * @exception ParserConfigurationException
     */
    public XbfFfunctionType(InputStream in) throws IOException, SAXException, ParserConfigurationException {
        setup(in);
    }

    /**
     * Creates a <code>XbfFfunctionType</code> by the InputSource <code>is</code>.
     *
     * @param is
     * @exception IOException
     * @exception SAXException
     * @exception ParserConfigurationException
     */
    public XbfFfunctionType(InputSource is) throws IOException, SAXException, ParserConfigurationException {
        setup(is);
    }

    /**
     * Creates a <code>XbfFfunctionType</code> by the Reader <code>reader</code>.
     *
     * @param reader
     * @exception IOException
     * @exception SAXException
     * @exception ParserConfigurationException
     */
    public XbfFfunctionType(Reader reader) throws IOException, SAXException, ParserConfigurationException {
        setup(reader);
    }

    /**
     * Initializes the <code>XbfFfunctionType</code> by the XbfFfunctionType <code>source</code>.
     *
     * @param source
     */
    public void setup(XbfFfunctionType source) {
        int size;
        setType(source.getType());
        setReturnType(source.getReturnType());
        setResultName(source.getResultName());
        setIsRecursive(source.getIsRecursive());
        setIsProgram(source.getIsProgram());
        setIsInternal(source.getIsInternal());
        setIsIntrinsic(source.getIsIntrinsic());
        setIsExternal(source.getIsExternal());
        setIsPublic(source.getIsPublic());
        setIsPrivate(source.getIsPrivate());
        setIsPure(source.getIsPure());
        setIsElemental(source.getIsElemental());
        if (source.params_ != null) {
            setParams((XbfParams)source.getParams().clone());
        }
    }

    /**
     * Initializes the <code>XbfFfunctionType</code> by the Document <code>doc</code>.
     *
     * @param doc
     */
    public void setup(Document doc) {
        setup(doc.getDocumentElement());
    }

    /**
     * Initializes the <code>XbfFfunctionType</code> by the Element <code>element</code>.
     *
     * @param element
     */
    public void setup(Element element) {
        init(element);
    }

    /**
     * Initializes the <code>XbfFfunctionType</code> by the Stack <code>stack</code>
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
        returnType_ = URelaxer.getAttributePropertyAsString(element, "return_type");
        resultName_ = URelaxer.getAttributePropertyAsString(element, "result_name");
        isRecursive_ = URelaxer.getAttributePropertyAsBooleanObject(element, "is_recursive");
        isProgram_ = URelaxer.getAttributePropertyAsBooleanObject(element, "is_program");
        isInternal_ = URelaxer.getAttributePropertyAsBooleanObject(element, "is_internal");
        isIntrinsic_ = URelaxer.getAttributePropertyAsBooleanObject(element, "is_intrinsic");
        isExternal_ = URelaxer.getAttributePropertyAsBooleanObject(element, "is_external");
        isPublic_ = URelaxer.getAttributePropertyAsBooleanObject(element, "is_public");
        isPrivate_ = URelaxer.getAttributePropertyAsBooleanObject(element, "is_private");
        isPure_ = URelaxer.getAttributePropertyAsBooleanObject(element, "is_pure");
        isElemental_ = URelaxer.getAttributePropertyAsBooleanObject(element, "is_elemental");
        if (XbfParams.isMatch(stack)) {
            setParams(factory.createXbfParams(stack));
        }
    }

    /**
     * @return Object
     */
    public Object clone() {
        IXcodeML_FFactory factory = XcodeML_FFactory.getFactory();
        return (factory.createXbfFfunctionType(this));
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
        Element element = doc.createElement("FfunctionType");
        int size;
        if (this.type_ != null) {
            URelaxer.setAttributePropertyByString(element, "type", this.type_);
        }
        if (this.returnType_ != null) {
            URelaxer.setAttributePropertyByString(element, "return_type", this.returnType_);
        }
        if (this.resultName_ != null) {
            URelaxer.setAttributePropertyByString(element, "result_name", this.resultName_);
        }
        if (this.isRecursive_ != null) {
            URelaxer.setAttributePropertyByBoolean(element, "is_recursive", this.isRecursive_);
        }
        if (this.isProgram_ != null) {
            URelaxer.setAttributePropertyByBoolean(element, "is_program", this.isProgram_);
        }
        if (this.isInternal_ != null) {
            URelaxer.setAttributePropertyByBoolean(element, "is_internal", this.isInternal_);
        }
        if (this.isIntrinsic_ != null) {
            URelaxer.setAttributePropertyByBoolean(element, "is_intrinsic", this.isIntrinsic_);
        }
        if (this.isExternal_ != null) {
            URelaxer.setAttributePropertyByBoolean(element, "is_external", this.isExternal_);
        }
        if (this.isPublic_ != null) {
            URelaxer.setAttributePropertyByBoolean(element, "is_public", this.isPublic_);
        }
        if (this.isPrivate_ != null) {
            URelaxer.setAttributePropertyByBoolean(element, "is_private", this.isPrivate_);
        }
        if (this.isPure_ != null) {
            URelaxer.setAttributePropertyByBoolean(element, "is_pure", this.isPure_);
        }
        if (this.isElemental_ != null) {
            URelaxer.setAttributePropertyByBoolean(element, "is_elemental", this.isElemental_);
        }
        if (this.params_ != null) {
            this.params_.makeElement(element);
        }
        parent.appendChild(element);
    }

    /**
     * Initializes the <code>XbfFfunctionType</code> by the File <code>file</code>.
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
     * Initializes the <code>XbfFfunctionType</code>
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
     * Initializes the <code>XbfFfunctionType</code> by the URL <code>url</code>.
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
     * Initializes the <code>XbfFfunctionType</code> by the InputStream <code>in</code>.
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
     * Initializes the <code>XbfFfunctionType</code> by the InputSource <code>is</code>.
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
     * Initializes the <code>XbfFfunctionType</code> by the Reader <code>reader</code>.
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
     * Gets the String property <b>resultName</b>.
     *
     * @return String
     */
    public final String getResultName() {
        return (resultName_);
    }

    /**
     * Sets the String property <b>resultName</b>.
     *
     * @param resultName
     */
    public final void setResultName(String resultName) {
        this.resultName_ = resultName;
    }

    /**
     * Gets the boolean property <b>isRecursive</b>.
     *
     * @return boolean
     */
    public boolean getIsRecursive() {
        if (isRecursive_ == null) {
            return(false);
        }
        return (isRecursive_.booleanValue());
    }

    /**
     * Gets the boolean property <b>isRecursive</b>.
     *
     * @param isRecursive
     * @return boolean
     */
    public boolean getIsRecursive(boolean isRecursive) {
        if (isRecursive_ == null) {
            return(isRecursive);
        }
        return (this.isRecursive_.booleanValue());
    }

    /**
     * Gets the boolean property <b>isRecursive</b>.
     *
     * @return Boolean
     */
    public Boolean getIsRecursiveAsBoolean() {
        return (isRecursive_);
    }

    /**
     * Check the boolean property <b>isRecursive</b>.
     *
     * @return boolean
     */
    public boolean checkIsRecursive() {
        return (isRecursive_ != null);
    }

    /**
     * Sets the boolean property <b>isRecursive</b>.
     *
     * @param isRecursive
     */
    public void setIsRecursive(boolean isRecursive) {
        this.isRecursive_ = new Boolean(isRecursive);
    }

    /**
     * Sets the boolean property <b>isRecursive</b>.
     *
     * @param isRecursive
     */
    public void setIsRecursive(Boolean isRecursive) {
        this.isRecursive_ = isRecursive;
    }

    /**
     * Gets the boolean property <b>isProgram</b>.
     *
     * @return boolean
     */
    public boolean getIsProgram() {
        if (isProgram_ == null) {
            return(false);
        }
        return (isProgram_.booleanValue());
    }

    /**
     * Gets the boolean property <b>isProgram</b>.
     *
     * @param isProgram
     * @return boolean
     */
    public boolean getIsProgram(boolean isProgram) {
        if (isProgram_ == null) {
            return(isProgram);
        }
        return (this.isProgram_.booleanValue());
    }

    /**
     * Gets the boolean property <b>isProgram</b>.
     *
     * @return Boolean
     */
    public Boolean getIsProgramAsBoolean() {
        return (isProgram_);
    }

    /**
     * Check the boolean property <b>isProgram</b>.
     *
     * @return boolean
     */
    public boolean checkIsProgram() {
        return (isProgram_ != null);
    }

    /**
     * Sets the boolean property <b>isProgram</b>.
     *
     * @param isProgram
     */
    public void setIsProgram(boolean isProgram) {
        this.isProgram_ = new Boolean(isProgram);
    }

    /**
     * Sets the boolean property <b>isProgram</b>.
     *
     * @param isProgram
     */
    public void setIsProgram(Boolean isProgram) {
        this.isProgram_ = isProgram;
    }

    /**
     * Gets the boolean property <b>isInternal</b>.
     *
     * @return boolean
     */
    public boolean getIsInternal() {
        if (isInternal_ == null) {
            return(false);
        }
        return (isInternal_.booleanValue());
    }

    /**
     * Gets the boolean property <b>isInternal</b>.
     *
     * @param isInternal
     * @return boolean
     */
    public boolean getIsInternal(boolean isInternal) {
        if (isInternal_ == null) {
            return(isInternal);
        }
        return (this.isInternal_.booleanValue());
    }

    /**
     * Gets the boolean property <b>isInternal</b>.
     *
     * @return Boolean
     */
    public Boolean getIsInternalAsBoolean() {
        return (isInternal_);
    }

    /**
     * Check the boolean property <b>isInternal</b>.
     *
     * @return boolean
     */
    public boolean checkIsInternal() {
        return (isInternal_ != null);
    }

    /**
     * Sets the boolean property <b>isInternal</b>.
     *
     * @param isInternal
     */
    public void setIsInternal(boolean isInternal) {
        this.isInternal_ = new Boolean(isInternal);
    }

    /**
     * Sets the boolean property <b>isInternal</b>.
     *
     * @param isInternal
     */
    public void setIsInternal(Boolean isInternal) {
        this.isInternal_ = isInternal;
    }

    /**
     * Gets the boolean property <b>isIntrinsic</b>.
     *
     * @return boolean
     */
    public boolean getIsIntrinsic() {
        if (isIntrinsic_ == null) {
            return(false);
        }
        return (isIntrinsic_.booleanValue());
    }

    /**
     * Gets the boolean property <b>isIntrinsic</b>.
     *
     * @param isIntrinsic
     * @return boolean
     */
    public boolean getIsIntrinsic(boolean isIntrinsic) {
        if (isIntrinsic_ == null) {
            return(isIntrinsic);
        }
        return (this.isIntrinsic_.booleanValue());
    }

    /**
     * Gets the boolean property <b>isIntrinsic</b>.
     *
     * @return Boolean
     */
    public Boolean getIsIntrinsicAsBoolean() {
        return (isIntrinsic_);
    }

    /**
     * Check the boolean property <b>isIntrinsic</b>.
     *
     * @return boolean
     */
    public boolean checkIsIntrinsic() {
        return (isIntrinsic_ != null);
    }

    /**
     * Sets the boolean property <b>isIntrinsic</b>.
     *
     * @param isIntrinsic
     */
    public void setIsIntrinsic(boolean isIntrinsic) {
        this.isIntrinsic_ = new Boolean(isIntrinsic);
    }

    /**
     * Sets the boolean property <b>isIntrinsic</b>.
     *
     * @param isIntrinsic
     */
    public void setIsIntrinsic(Boolean isIntrinsic) {
        this.isIntrinsic_ = isIntrinsic;
    }

    /**
     * Gets the boolean property <b>isExternal</b>.
     *
     * @return boolean
     */
    public boolean getIsExternal() {
        if (isExternal_ == null) {
            return(false);
        }
        return (isExternal_.booleanValue());
    }

    /**
     * Gets the boolean property <b>isExternal</b>.
     *
     * @param isExternal
     * @return boolean
     */
    public boolean getIsExternal(boolean isExternal) {
        if (isExternal_ == null) {
            return(isExternal);
        }
        return (this.isExternal_.booleanValue());
    }

    /**
     * Gets the boolean property <b>isExternal</b>.
     *
     * @return Boolean
     */
    public Boolean getIsExternalAsBoolean() {
        return (isExternal_);
    }

    /**
     * Check the boolean property <b>isExternal</b>.
     *
     * @return boolean
     */
    public boolean checkIsExternal() {
        return (isExternal_ != null);
    }

    /**
     * Sets the boolean property <b>isExternal</b>.
     *
     * @param isExternal
     */
    public void setIsExternal(boolean isExternal) {
        this.isExternal_ = new Boolean(isExternal);
    }

    /**
     * Sets the boolean property <b>isExternal</b>.
     *
     * @param isExternal
     */
    public void setIsExternal(Boolean isExternal) {
        this.isExternal_ = isExternal;
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
     * Gets the boolean property <b>isPure</b>.
     *
     * @return boolean
     */
    public boolean getIsPure() {
        if (isPure_ == null) {
            return(false);
        }
        return (isPure_.booleanValue());
    }

    /**
     * Gets the boolean property <b>isPure</b>.
     *
     * @param isPure
     * @return boolean
     */
    public boolean getIsPure(boolean isPure) {
        if (isPure_ == null) {
            return(isPure);
        }
        return (this.isPure_.booleanValue());
    }

    /**
     * Gets the boolean property <b>isPure</b>.
     *
     * @return Boolean
     */
    public Boolean getIsPureAsBoolean() {
        return (isPure_);
    }

    /**
     * Check the boolean property <b>isPure</b>.
     *
     * @return boolean
     */
    public boolean checkIsPure() {
        return (isPure_ != null);
    }

    /**
     * Sets the boolean property <b>isPure</b>.
     *
     * @param isPure
     */
    public void setIsPure(boolean isPure) {
        this.isPure_ = new Boolean(isPure);
    }

    /**
     * Sets the boolean property <b>isPure</b>.
     *
     * @param isPure
     */
    public void setIsPure(Boolean isPure) {
        this.isPure_ = isPure;
    }

    /**
     * Gets the boolean property <b>isElemental</b>.
     *
     * @return boolean
     */
    public boolean getIsElemental() {
        if (isElemental_ == null) {
            return(false);
        }
        return (isElemental_.booleanValue());
    }

    /**
     * Gets the boolean property <b>isElemental</b>.
     *
     * @param isElemental
     * @return boolean
     */
    public boolean getIsElemental(boolean isElemental) {
        if (isElemental_ == null) {
            return(isElemental);
        }
        return (this.isElemental_.booleanValue());
    }

    /**
     * Gets the boolean property <b>isElemental</b>.
     *
     * @return Boolean
     */
    public Boolean getIsElementalAsBoolean() {
        return (isElemental_);
    }

    /**
     * Check the boolean property <b>isElemental</b>.
     *
     * @return boolean
     */
    public boolean checkIsElemental() {
        return (isElemental_ != null);
    }

    /**
     * Sets the boolean property <b>isElemental</b>.
     *
     * @param isElemental
     */
    public void setIsElemental(boolean isElemental) {
        this.isElemental_ = new Boolean(isElemental);
    }

    /**
     * Sets the boolean property <b>isElemental</b>.
     *
     * @param isElemental
     */
    public void setIsElemental(Boolean isElemental) {
        this.isElemental_ = isElemental;
    }

    /**
     * Gets the XbfParams property <b>params</b>.
     *
     * @return XbfParams
     */
    public final XbfParams getParams() {
        return (params_);
    }

    /**
     * Sets the XbfParams property <b>params</b>.
     *
     * @param params
     */
    public final void setParams(XbfParams params) {
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
        buffer.append("<FfunctionType");
        if (type_ != null) {
            buffer.append(" type=\"");
            buffer.append(URelaxer.escapeAttrQuot(URelaxer.getString(getType())));
            buffer.append("\"");
        }
        if (returnType_ != null) {
            buffer.append(" return_type=\"");
            buffer.append(URelaxer.escapeAttrQuot(URelaxer.getString(getReturnType())));
            buffer.append("\"");
        }
        if (resultName_ != null) {
            buffer.append(" result_name=\"");
            buffer.append(URelaxer.escapeAttrQuot(URelaxer.getString(getResultName())));
            buffer.append("\"");
        }
        if (isRecursive_ != null) {
            buffer.append(" is_recursive=\"");
            buffer.append(URelaxer.getString(getIsRecursive()));
            buffer.append("\"");
        }
        if (isProgram_ != null) {
            buffer.append(" is_program=\"");
            buffer.append(URelaxer.getString(getIsProgram()));
            buffer.append("\"");
        }
        if (isInternal_ != null) {
            buffer.append(" is_internal=\"");
            buffer.append(URelaxer.getString(getIsInternal()));
            buffer.append("\"");
        }
        if (isIntrinsic_ != null) {
            buffer.append(" is_intrinsic=\"");
            buffer.append(URelaxer.getString(getIsIntrinsic()));
            buffer.append("\"");
        }
        if (isExternal_ != null) {
            buffer.append(" is_external=\"");
            buffer.append(URelaxer.getString(getIsExternal()));
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
        if (isPure_ != null) {
            buffer.append(" is_pure=\"");
            buffer.append(URelaxer.getString(getIsPure()));
            buffer.append("\"");
        }
        if (isElemental_ != null) {
            buffer.append(" is_elemental=\"");
            buffer.append(URelaxer.getString(getIsElemental()));
            buffer.append("\"");
        }
        buffer.append(">");
        if (params_ != null) {
            params_.makeTextElement(buffer);
        }
        buffer.append("</FfunctionType>");
    }

    /**
     * Makes an XML text representation.
     *
     * @param buffer
     * @exception IOException
     */
    public void makeTextElement(Writer buffer) throws IOException {
        int size;
        buffer.write("<FfunctionType");
        if (type_ != null) {
            buffer.write(" type=\"");
            buffer.write(URelaxer.escapeAttrQuot(URelaxer.getString(getType())));
            buffer.write("\"");
        }
        if (returnType_ != null) {
            buffer.write(" return_type=\"");
            buffer.write(URelaxer.escapeAttrQuot(URelaxer.getString(getReturnType())));
            buffer.write("\"");
        }
        if (resultName_ != null) {
            buffer.write(" result_name=\"");
            buffer.write(URelaxer.escapeAttrQuot(URelaxer.getString(getResultName())));
            buffer.write("\"");
        }
        if (isRecursive_ != null) {
            buffer.write(" is_recursive=\"");
            buffer.write(URelaxer.getString(getIsRecursive()));
            buffer.write("\"");
        }
        if (isProgram_ != null) {
            buffer.write(" is_program=\"");
            buffer.write(URelaxer.getString(getIsProgram()));
            buffer.write("\"");
        }
        if (isInternal_ != null) {
            buffer.write(" is_internal=\"");
            buffer.write(URelaxer.getString(getIsInternal()));
            buffer.write("\"");
        }
        if (isIntrinsic_ != null) {
            buffer.write(" is_intrinsic=\"");
            buffer.write(URelaxer.getString(getIsIntrinsic()));
            buffer.write("\"");
        }
        if (isExternal_ != null) {
            buffer.write(" is_external=\"");
            buffer.write(URelaxer.getString(getIsExternal()));
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
        if (isPure_ != null) {
            buffer.write(" is_pure=\"");
            buffer.write(URelaxer.getString(getIsPure()));
            buffer.write("\"");
        }
        if (isElemental_ != null) {
            buffer.write(" is_elemental=\"");
            buffer.write(URelaxer.getString(getIsElemental()));
            buffer.write("\"");
        }
        buffer.write(">");
        if (params_ != null) {
            params_.makeTextElement(buffer);
        }
        buffer.write("</FfunctionType>");
    }

    /**
     * Makes an XML text representation.
     *
     * @param buffer
     */
    public void makeTextElement(PrintWriter buffer) {
        int size;
        buffer.print("<FfunctionType");
        if (type_ != null) {
            buffer.print(" type=\"");
            buffer.print(URelaxer.escapeAttrQuot(URelaxer.getString(getType())));
            buffer.print("\"");
        }
        if (returnType_ != null) {
            buffer.print(" return_type=\"");
            buffer.print(URelaxer.escapeAttrQuot(URelaxer.getString(getReturnType())));
            buffer.print("\"");
        }
        if (resultName_ != null) {
            buffer.print(" result_name=\"");
            buffer.print(URelaxer.escapeAttrQuot(URelaxer.getString(getResultName())));
            buffer.print("\"");
        }
        if (isRecursive_ != null) {
            buffer.print(" is_recursive=\"");
            buffer.print(URelaxer.getString(getIsRecursive()));
            buffer.print("\"");
        }
        if (isProgram_ != null) {
            buffer.print(" is_program=\"");
            buffer.print(URelaxer.getString(getIsProgram()));
            buffer.print("\"");
        }
        if (isInternal_ != null) {
            buffer.print(" is_internal=\"");
            buffer.print(URelaxer.getString(getIsInternal()));
            buffer.print("\"");
        }
        if (isIntrinsic_ != null) {
            buffer.print(" is_intrinsic=\"");
            buffer.print(URelaxer.getString(getIsIntrinsic()));
            buffer.print("\"");
        }
        if (isExternal_ != null) {
            buffer.print(" is_external=\"");
            buffer.print(URelaxer.getString(getIsExternal()));
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
        if (isPure_ != null) {
            buffer.print(" is_pure=\"");
            buffer.print(URelaxer.getString(getIsPure()));
            buffer.print("\"");
        }
        if (isElemental_ != null) {
            buffer.print(" is_elemental=\"");
            buffer.print(URelaxer.getString(getIsElemental()));
            buffer.print("\"");
        }
        buffer.print(">");
        if (params_ != null) {
            params_.makeTextElement(buffer);
        }
        buffer.print("</FfunctionType>");
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
    public String getReturnTypeAsString() {
        return (URelaxer.getString(getReturnType()));
    }

    /**
     * Gets the property value as String.
     *
     * @return String
     */
    public String getResultNameAsString() {
        return (URelaxer.getString(getResultName()));
    }

    /**
     * Gets the property value as String.
     *
     * @return String
     */
    public String getIsRecursiveAsString() {
        return (URelaxer.getString(getIsRecursive()));
    }

    /**
     * Gets the property value as String.
     *
     * @return String
     */
    public String getIsProgramAsString() {
        return (URelaxer.getString(getIsProgram()));
    }

    /**
     * Gets the property value as String.
     *
     * @return String
     */
    public String getIsInternalAsString() {
        return (URelaxer.getString(getIsInternal()));
    }

    /**
     * Gets the property value as String.
     *
     * @return String
     */
    public String getIsIntrinsicAsString() {
        return (URelaxer.getString(getIsIntrinsic()));
    }

    /**
     * Gets the property value as String.
     *
     * @return String
     */
    public String getIsExternalAsString() {
        return (URelaxer.getString(getIsExternal()));
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
    public String getIsPureAsString() {
        return (URelaxer.getString(getIsPure()));
    }

    /**
     * Gets the property value as String.
     *
     * @return String
     */
    public String getIsElementalAsString() {
        return (URelaxer.getString(getIsElemental()));
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
    public void setReturnTypeByString(String string) {
        setReturnType(string);
    }

    /**
     * Sets the property value by String.
     *
     * @param string
     */
    public void setResultNameByString(String string) {
        setResultName(string);
    }

    /**
     * Sets the property value by String.
     *
     * @param string
     */
    public void setIsRecursiveByString(String string) {
        setIsRecursive(new Boolean(string).booleanValue());
    }

    /**
     * Sets the property value by String.
     *
     * @param string
     */
    public void setIsProgramByString(String string) {
        setIsProgram(new Boolean(string).booleanValue());
    }

    /**
     * Sets the property value by String.
     *
     * @param string
     */
    public void setIsInternalByString(String string) {
        setIsInternal(new Boolean(string).booleanValue());
    }

    /**
     * Sets the property value by String.
     *
     * @param string
     */
    public void setIsIntrinsicByString(String string) {
        setIsIntrinsic(new Boolean(string).booleanValue());
    }

    /**
     * Sets the property value by String.
     *
     * @param string
     */
    public void setIsExternalByString(String string) {
        setIsExternal(new Boolean(string).booleanValue());
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
        if (params_ != null) {
            classNodes.add(params_);
        }
        IRNode[] nodes = new IRNode[classNodes.size()];
        return ((IRNode[])classNodes.toArray(nodes));
    }

    /**
     * Tests if a Element <code>element</code> is valid
     * for the <code>XbfFfunctionType</code>.
     *
     * @param element
     * @return boolean
     */
    public static boolean isMatch(Element element) {
        if (!URelaxer.isTargetElement(element, "FfunctionType")) {
            return (false);
        }
        RStack target = new RStack(element);
        boolean $match$ = false;
        Element child;
        if (XbfParams.isMatchHungry(target)) {
        }
        if (!target.isEmptyElement()) {
            return (false);
        }
        return (true);
    }

    /**
     * Tests if elements contained in a Stack <code>stack</code>
     * is valid for the <code>XbfFfunctionType</code>.
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
     * is valid for the <code>XbfFfunctionType</code>.
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
