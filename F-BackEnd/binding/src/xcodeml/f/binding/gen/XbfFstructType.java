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
 * <b>XbfFstructType</b> is generated from XcodeML_F.rng by Relaxer.
 * This class is derived from:
 * 
 * <!-- for programmer
 * <element java:extends="xcodeml.f.XmfObj" java:implements="xcodeml.f.binding.IXbfType" name="FstructType">
 *   <ref name="defAttrTypeName"/>
 *   <ref name="defAttrStructTypeModifier"/>
 *   <ref name="symbols"/>
 * </element>
 * -->
 * <!-- for javadoc -->
 * <pre> &lt;element java:extends="xcodeml.f.XmfObj" java:implements="xcodeml.f.binding.IXbfType" name="FstructType"&gt;
 *   &lt;ref name="defAttrTypeName"/&gt;
 *   &lt;ref name="defAttrStructTypeModifier"/&gt;
 *   &lt;ref name="symbols"/&gt;
 * &lt;/element&gt;
 * </pre>
 *
 * @version XcodeML_F.rng (Fri Dec 04 19:18:15 JST 2009)
 * @author  Relaxer 1.0 (http://www.relaxer.org)
 */
public class XbfFstructType extends xcodeml.f.XmfObj implements java.io.Serializable, Cloneable, xcodeml.f.binding.IXbfType, IRVisitable, IRNode, IXbfTypeTableChoice {
    private String type_;
    private Boolean isPublic_;
    private Boolean isPrivate_;
    private Boolean isSequence_;
    private Boolean isInternalPrivate_;
    private XbfSymbols symbols_;
    private IRNode parentRNode_;

    /**
     * Creates a <code>XbfFstructType</code>.
     *
     */
    public XbfFstructType() {
    }

    /**
     * Creates a <code>XbfFstructType</code>.
     *
     * @param source
     */
    public XbfFstructType(XbfFstructType source) {
        setup(source);
    }

    /**
     * Creates a <code>XbfFstructType</code> by the Stack <code>stack</code>
     * that contains Elements.
     * This constructor is supposed to be used internally
     * by the Relaxer system.
     *
     * @param stack
     */
    public XbfFstructType(RStack stack) {
        setup(stack);
    }

    /**
     * Creates a <code>XbfFstructType</code> by the Document <code>doc</code>.
     *
     * @param doc
     */
    public XbfFstructType(Document doc) {
        setup(doc.getDocumentElement());
    }

    /**
     * Creates a <code>XbfFstructType</code> by the Element <code>element</code>.
     *
     * @param element
     */
    public XbfFstructType(Element element) {
        setup(element);
    }

    /**
     * Creates a <code>XbfFstructType</code> by the File <code>file</code>.
     *
     * @param file
     * @exception IOException
     * @exception SAXException
     * @exception ParserConfigurationException
     */
    public XbfFstructType(File file) throws IOException, SAXException, ParserConfigurationException {
        setup(file);
    }

    /**
     * Creates a <code>XbfFstructType</code>
     * by the String representation of URI <code>uri</code>.
     *
     * @param uri
     * @exception IOException
     * @exception SAXException
     * @exception ParserConfigurationException
     */
    public XbfFstructType(String uri) throws IOException, SAXException, ParserConfigurationException {
        setup(uri);
    }

    /**
     * Creates a <code>XbfFstructType</code> by the URL <code>url</code>.
     *
     * @param url
     * @exception IOException
     * @exception SAXException
     * @exception ParserConfigurationException
     */
    public XbfFstructType(URL url) throws IOException, SAXException, ParserConfigurationException {
        setup(url);
    }

    /**
     * Creates a <code>XbfFstructType</code> by the InputStream <code>in</code>.
     *
     * @param in
     * @exception IOException
     * @exception SAXException
     * @exception ParserConfigurationException
     */
    public XbfFstructType(InputStream in) throws IOException, SAXException, ParserConfigurationException {
        setup(in);
    }

    /**
     * Creates a <code>XbfFstructType</code> by the InputSource <code>is</code>.
     *
     * @param is
     * @exception IOException
     * @exception SAXException
     * @exception ParserConfigurationException
     */
    public XbfFstructType(InputSource is) throws IOException, SAXException, ParserConfigurationException {
        setup(is);
    }

    /**
     * Creates a <code>XbfFstructType</code> by the Reader <code>reader</code>.
     *
     * @param reader
     * @exception IOException
     * @exception SAXException
     * @exception ParserConfigurationException
     */
    public XbfFstructType(Reader reader) throws IOException, SAXException, ParserConfigurationException {
        setup(reader);
    }

    /**
     * Initializes the <code>XbfFstructType</code> by the XbfFstructType <code>source</code>.
     *
     * @param source
     */
    public void setup(XbfFstructType source) {
        int size;
        setType(source.getType());
        setIsPublic(source.getIsPublic());
        setIsPrivate(source.getIsPrivate());
        setIsSequence(source.getIsSequence());
        setIsInternalPrivate(source.getIsInternalPrivate());
        if (source.symbols_ != null) {
            setSymbols((XbfSymbols)source.getSymbols().clone());
        }
    }

    /**
     * Initializes the <code>XbfFstructType</code> by the Document <code>doc</code>.
     *
     * @param doc
     */
    public void setup(Document doc) {
        setup(doc.getDocumentElement());
    }

    /**
     * Initializes the <code>XbfFstructType</code> by the Element <code>element</code>.
     *
     * @param element
     */
    public void setup(Element element) {
        init(element);
    }

    /**
     * Initializes the <code>XbfFstructType</code> by the Stack <code>stack</code>
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
        isPublic_ = URelaxer.getAttributePropertyAsBooleanObject(element, "is_public");
        isPrivate_ = URelaxer.getAttributePropertyAsBooleanObject(element, "is_private");
        isSequence_ = URelaxer.getAttributePropertyAsBooleanObject(element, "is_sequence");
        isInternalPrivate_ = URelaxer.getAttributePropertyAsBooleanObject(element, "is_internal_private");
        setSymbols(factory.createXbfSymbols(stack));
    }

    /**
     * @return Object
     */
    public Object clone() {
        IXcodeML_FFactory factory = XcodeML_FFactory.getFactory();
        return (factory.createXbfFstructType(this));
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
        Element element = doc.createElement("FstructType");
        int size;
        if (this.type_ != null) {
            URelaxer.setAttributePropertyByString(element, "type", this.type_);
        }
        if (this.isPublic_ != null) {
            URelaxer.setAttributePropertyByBoolean(element, "is_public", this.isPublic_);
        }
        if (this.isPrivate_ != null) {
            URelaxer.setAttributePropertyByBoolean(element, "is_private", this.isPrivate_);
        }
        if (this.isSequence_ != null) {
            URelaxer.setAttributePropertyByBoolean(element, "is_sequence", this.isSequence_);
        }
        if (this.isInternalPrivate_ != null) {
            URelaxer.setAttributePropertyByBoolean(element, "is_internal_private", this.isInternalPrivate_);
        }
        this.symbols_.makeElement(element);
        parent.appendChild(element);
    }

    /**
     * Initializes the <code>XbfFstructType</code> by the File <code>file</code>.
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
     * Initializes the <code>XbfFstructType</code>
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
     * Initializes the <code>XbfFstructType</code> by the URL <code>url</code>.
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
     * Initializes the <code>XbfFstructType</code> by the InputStream <code>in</code>.
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
     * Initializes the <code>XbfFstructType</code> by the InputSource <code>is</code>.
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
     * Initializes the <code>XbfFstructType</code> by the Reader <code>reader</code>.
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
     * Gets the boolean property <b>isSequence</b>.
     *
     * @return boolean
     */
    public boolean getIsSequence() {
        if (isSequence_ == null) {
            return(false);
        }
        return (isSequence_.booleanValue());
    }

    /**
     * Gets the boolean property <b>isSequence</b>.
     *
     * @param isSequence
     * @return boolean
     */
    public boolean getIsSequence(boolean isSequence) {
        if (isSequence_ == null) {
            return(isSequence);
        }
        return (this.isSequence_.booleanValue());
    }

    /**
     * Gets the boolean property <b>isSequence</b>.
     *
     * @return Boolean
     */
    public Boolean getIsSequenceAsBoolean() {
        return (isSequence_);
    }

    /**
     * Check the boolean property <b>isSequence</b>.
     *
     * @return boolean
     */
    public boolean checkIsSequence() {
        return (isSequence_ != null);
    }

    /**
     * Sets the boolean property <b>isSequence</b>.
     *
     * @param isSequence
     */
    public void setIsSequence(boolean isSequence) {
        this.isSequence_ = new Boolean(isSequence);
    }

    /**
     * Sets the boolean property <b>isSequence</b>.
     *
     * @param isSequence
     */
    public void setIsSequence(Boolean isSequence) {
        this.isSequence_ = isSequence;
    }

    /**
     * Gets the boolean property <b>isInternalPrivate</b>.
     *
     * @return boolean
     */
    public boolean getIsInternalPrivate() {
        if (isInternalPrivate_ == null) {
            return(false);
        }
        return (isInternalPrivate_.booleanValue());
    }

    /**
     * Gets the boolean property <b>isInternalPrivate</b>.
     *
     * @param isInternalPrivate
     * @return boolean
     */
    public boolean getIsInternalPrivate(boolean isInternalPrivate) {
        if (isInternalPrivate_ == null) {
            return(isInternalPrivate);
        }
        return (this.isInternalPrivate_.booleanValue());
    }

    /**
     * Gets the boolean property <b>isInternalPrivate</b>.
     *
     * @return Boolean
     */
    public Boolean getIsInternalPrivateAsBoolean() {
        return (isInternalPrivate_);
    }

    /**
     * Check the boolean property <b>isInternalPrivate</b>.
     *
     * @return boolean
     */
    public boolean checkIsInternalPrivate() {
        return (isInternalPrivate_ != null);
    }

    /**
     * Sets the boolean property <b>isInternalPrivate</b>.
     *
     * @param isInternalPrivate
     */
    public void setIsInternalPrivate(boolean isInternalPrivate) {
        this.isInternalPrivate_ = new Boolean(isInternalPrivate);
    }

    /**
     * Sets the boolean property <b>isInternalPrivate</b>.
     *
     * @param isInternalPrivate
     */
    public void setIsInternalPrivate(Boolean isInternalPrivate) {
        this.isInternalPrivate_ = isInternalPrivate;
    }

    /**
     * Gets the XbfSymbols property <b>symbols</b>.
     *
     * @return XbfSymbols
     */
    public final XbfSymbols getSymbols() {
        return (symbols_);
    }

    /**
     * Sets the XbfSymbols property <b>symbols</b>.
     *
     * @param symbols
     */
    public final void setSymbols(XbfSymbols symbols) {
        this.symbols_ = symbols;
        if (symbols != null) {
            symbols.rSetParentRNode(this);
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
        buffer.append("<FstructType");
        if (type_ != null) {
            buffer.append(" type=\"");
            buffer.append(URelaxer.escapeAttrQuot(URelaxer.getString(getType())));
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
        if (isSequence_ != null) {
            buffer.append(" is_sequence=\"");
            buffer.append(URelaxer.getString(getIsSequence()));
            buffer.append("\"");
        }
        if (isInternalPrivate_ != null) {
            buffer.append(" is_internal_private=\"");
            buffer.append(URelaxer.getString(getIsInternalPrivate()));
            buffer.append("\"");
        }
        buffer.append(">");
        symbols_.makeTextElement(buffer);
        buffer.append("</FstructType>");
    }

    /**
     * Makes an XML text representation.
     *
     * @param buffer
     * @exception IOException
     */
    public void makeTextElement(Writer buffer) throws IOException {
        int size;
        buffer.write("<FstructType");
        if (type_ != null) {
            buffer.write(" type=\"");
            buffer.write(URelaxer.escapeAttrQuot(URelaxer.getString(getType())));
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
        if (isSequence_ != null) {
            buffer.write(" is_sequence=\"");
            buffer.write(URelaxer.getString(getIsSequence()));
            buffer.write("\"");
        }
        if (isInternalPrivate_ != null) {
            buffer.write(" is_internal_private=\"");
            buffer.write(URelaxer.getString(getIsInternalPrivate()));
            buffer.write("\"");
        }
        buffer.write(">");
        symbols_.makeTextElement(buffer);
        buffer.write("</FstructType>");
    }

    /**
     * Makes an XML text representation.
     *
     * @param buffer
     */
    public void makeTextElement(PrintWriter buffer) {
        int size;
        buffer.print("<FstructType");
        if (type_ != null) {
            buffer.print(" type=\"");
            buffer.print(URelaxer.escapeAttrQuot(URelaxer.getString(getType())));
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
        if (isSequence_ != null) {
            buffer.print(" is_sequence=\"");
            buffer.print(URelaxer.getString(getIsSequence()));
            buffer.print("\"");
        }
        if (isInternalPrivate_ != null) {
            buffer.print(" is_internal_private=\"");
            buffer.print(URelaxer.getString(getIsInternalPrivate()));
            buffer.print("\"");
        }
        buffer.print(">");
        symbols_.makeTextElement(buffer);
        buffer.print("</FstructType>");
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
    public String getIsSequenceAsString() {
        return (URelaxer.getString(getIsSequence()));
    }

    /**
     * Gets the property value as String.
     *
     * @return String
     */
    public String getIsInternalPrivateAsString() {
        return (URelaxer.getString(getIsInternalPrivate()));
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
    public void setIsSequenceByString(String string) {
        setIsSequence(new Boolean(string).booleanValue());
    }

    /**
     * Sets the property value by String.
     *
     * @param string
     */
    public void setIsInternalPrivateByString(String string) {
        setIsInternalPrivate(new Boolean(string).booleanValue());
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
        if (symbols_ != null) {
            classNodes.add(symbols_);
        }
        IRNode[] nodes = new IRNode[classNodes.size()];
        return ((IRNode[])classNodes.toArray(nodes));
    }

    /**
     * Tests if a Element <code>element</code> is valid
     * for the <code>XbfFstructType</code>.
     *
     * @param element
     * @return boolean
     */
    public static boolean isMatch(Element element) {
        if (!URelaxer.isTargetElement(element, "FstructType")) {
            return (false);
        }
        RStack target = new RStack(element);
        boolean $match$ = false;
        Element child;
        if (!XbfSymbols.isMatchHungry(target)) {
            return (false);
        }
        $match$ = true;
        if (!target.isEmptyElement()) {
            return (false);
        }
        return (true);
    }

    /**
     * Tests if elements contained in a Stack <code>stack</code>
     * is valid for the <code>XbfFstructType</code>.
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
     * is valid for the <code>XbfFstructType</code>.
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
