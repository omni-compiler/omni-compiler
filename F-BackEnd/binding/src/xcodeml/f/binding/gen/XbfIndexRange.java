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
 * <b>XbfIndexRange</b> is generated from XcodeML_F.rng by Relaxer.
 * This class is derived from:
 * 
 * <!-- for programmer
 * <element java:extends="xcodeml.f.XmfObj" name="indexRange">
 *   <ref name="defAttrIndexRange"/>
 *   <optional>
 *     <ref name="lowerBound"/>
 *   </optional>
 *   <optional>
 *     <ref name="upperBound"/>
 *   </optional>
 *   <optional>
 *     <ref name="step"/>
 *   </optional>
 * </element>
 * -->
 * <!-- for javadoc -->
 * <pre> &lt;element java:extends="xcodeml.f.XmfObj" name="indexRange"&gt;
 *   &lt;ref name="defAttrIndexRange"/&gt;
 *   &lt;optional&gt;
 *     &lt;ref name="lowerBound"/&gt;
 *   &lt;/optional&gt;
 *   &lt;optional&gt;
 *     &lt;ref name="upperBound"/&gt;
 *   &lt;/optional&gt;
 *   &lt;optional&gt;
 *     &lt;ref name="step"/&gt;
 *   &lt;/optional&gt;
 * &lt;/element&gt;
 * </pre>
 *
 * @version XcodeML_F.rng (Fri Dec 04 19:18:15 JST 2009)
 * @author  Relaxer 1.0 (http://www.relaxer.org)
 */
public class XbfIndexRange extends xcodeml.f.XmfObj implements java.io.Serializable, Cloneable, IRVisitable, IRNode, IXbfFarrayRefChoice, IXbfDefModelArraySubscriptChoice, IXbfFcaseLabelChoice {
    private Boolean isAssumedShape_;
    private Boolean isAssumedSize_;
    private XbfLowerBound lowerBound_;
    private XbfUpperBound upperBound_;
    private XbfStep step_;
    private IRNode parentRNode_;

    /**
     * Creates a <code>XbfIndexRange</code>.
     *
     */
    public XbfIndexRange() {
    }

    /**
     * Creates a <code>XbfIndexRange</code>.
     *
     * @param source
     */
    public XbfIndexRange(XbfIndexRange source) {
        setup(source);
    }

    /**
     * Creates a <code>XbfIndexRange</code> by the Stack <code>stack</code>
     * that contains Elements.
     * This constructor is supposed to be used internally
     * by the Relaxer system.
     *
     * @param stack
     */
    public XbfIndexRange(RStack stack) {
        setup(stack);
    }

    /**
     * Creates a <code>XbfIndexRange</code> by the Document <code>doc</code>.
     *
     * @param doc
     */
    public XbfIndexRange(Document doc) {
        setup(doc.getDocumentElement());
    }

    /**
     * Creates a <code>XbfIndexRange</code> by the Element <code>element</code>.
     *
     * @param element
     */
    public XbfIndexRange(Element element) {
        setup(element);
    }

    /**
     * Creates a <code>XbfIndexRange</code> by the File <code>file</code>.
     *
     * @param file
     * @exception IOException
     * @exception SAXException
     * @exception ParserConfigurationException
     */
    public XbfIndexRange(File file) throws IOException, SAXException, ParserConfigurationException {
        setup(file);
    }

    /**
     * Creates a <code>XbfIndexRange</code>
     * by the String representation of URI <code>uri</code>.
     *
     * @param uri
     * @exception IOException
     * @exception SAXException
     * @exception ParserConfigurationException
     */
    public XbfIndexRange(String uri) throws IOException, SAXException, ParserConfigurationException {
        setup(uri);
    }

    /**
     * Creates a <code>XbfIndexRange</code> by the URL <code>url</code>.
     *
     * @param url
     * @exception IOException
     * @exception SAXException
     * @exception ParserConfigurationException
     */
    public XbfIndexRange(URL url) throws IOException, SAXException, ParserConfigurationException {
        setup(url);
    }

    /**
     * Creates a <code>XbfIndexRange</code> by the InputStream <code>in</code>.
     *
     * @param in
     * @exception IOException
     * @exception SAXException
     * @exception ParserConfigurationException
     */
    public XbfIndexRange(InputStream in) throws IOException, SAXException, ParserConfigurationException {
        setup(in);
    }

    /**
     * Creates a <code>XbfIndexRange</code> by the InputSource <code>is</code>.
     *
     * @param is
     * @exception IOException
     * @exception SAXException
     * @exception ParserConfigurationException
     */
    public XbfIndexRange(InputSource is) throws IOException, SAXException, ParserConfigurationException {
        setup(is);
    }

    /**
     * Creates a <code>XbfIndexRange</code> by the Reader <code>reader</code>.
     *
     * @param reader
     * @exception IOException
     * @exception SAXException
     * @exception ParserConfigurationException
     */
    public XbfIndexRange(Reader reader) throws IOException, SAXException, ParserConfigurationException {
        setup(reader);
    }

    /**
     * Initializes the <code>XbfIndexRange</code> by the XbfIndexRange <code>source</code>.
     *
     * @param source
     */
    public void setup(XbfIndexRange source) {
        int size;
        setIsAssumedShape(source.getIsAssumedShape());
        setIsAssumedSize(source.getIsAssumedSize());
        if (source.lowerBound_ != null) {
            setLowerBound((XbfLowerBound)source.getLowerBound().clone());
        }
        if (source.upperBound_ != null) {
            setUpperBound((XbfUpperBound)source.getUpperBound().clone());
        }
        if (source.step_ != null) {
            setStep((XbfStep)source.getStep().clone());
        }
    }

    /**
     * Initializes the <code>XbfIndexRange</code> by the Document <code>doc</code>.
     *
     * @param doc
     */
    public void setup(Document doc) {
        setup(doc.getDocumentElement());
    }

    /**
     * Initializes the <code>XbfIndexRange</code> by the Element <code>element</code>.
     *
     * @param element
     */
    public void setup(Element element) {
        init(element);
    }

    /**
     * Initializes the <code>XbfIndexRange</code> by the Stack <code>stack</code>
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
        isAssumedShape_ = URelaxer.getAttributePropertyAsBooleanObject(element, "is_assumed_shape");
        isAssumedSize_ = URelaxer.getAttributePropertyAsBooleanObject(element, "is_assumed_size");
        if (XbfLowerBound.isMatch(stack)) {
            setLowerBound(factory.createXbfLowerBound(stack));
        }
        if (XbfUpperBound.isMatch(stack)) {
            setUpperBound(factory.createXbfUpperBound(stack));
        }
        if (XbfStep.isMatch(stack)) {
            setStep(factory.createXbfStep(stack));
        }
    }

    /**
     * @return Object
     */
    public Object clone() {
        IXcodeML_FFactory factory = XcodeML_FFactory.getFactory();
        return (factory.createXbfIndexRange(this));
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
        Element element = doc.createElement("indexRange");
        int size;
        if (this.isAssumedShape_ != null) {
            URelaxer.setAttributePropertyByBoolean(element, "is_assumed_shape", this.isAssumedShape_);
        }
        if (this.isAssumedSize_ != null) {
            URelaxer.setAttributePropertyByBoolean(element, "is_assumed_size", this.isAssumedSize_);
        }
        if (this.lowerBound_ != null) {
            this.lowerBound_.makeElement(element);
        }
        if (this.upperBound_ != null) {
            this.upperBound_.makeElement(element);
        }
        if (this.step_ != null) {
            this.step_.makeElement(element);
        }
        parent.appendChild(element);
    }

    /**
     * Initializes the <code>XbfIndexRange</code> by the File <code>file</code>.
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
     * Initializes the <code>XbfIndexRange</code>
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
     * Initializes the <code>XbfIndexRange</code> by the URL <code>url</code>.
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
     * Initializes the <code>XbfIndexRange</code> by the InputStream <code>in</code>.
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
     * Initializes the <code>XbfIndexRange</code> by the InputSource <code>is</code>.
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
     * Initializes the <code>XbfIndexRange</code> by the Reader <code>reader</code>.
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
     * Gets the boolean property <b>isAssumedShape</b>.
     *
     * @return boolean
     */
    public boolean getIsAssumedShape() {
        if (isAssumedShape_ == null) {
            return(false);
        }
        return (isAssumedShape_.booleanValue());
    }

    /**
     * Gets the boolean property <b>isAssumedShape</b>.
     *
     * @param isAssumedShape
     * @return boolean
     */
    public boolean getIsAssumedShape(boolean isAssumedShape) {
        if (isAssumedShape_ == null) {
            return(isAssumedShape);
        }
        return (this.isAssumedShape_.booleanValue());
    }

    /**
     * Gets the boolean property <b>isAssumedShape</b>.
     *
     * @return Boolean
     */
    public Boolean getIsAssumedShapeAsBoolean() {
        return (isAssumedShape_);
    }

    /**
     * Check the boolean property <b>isAssumedShape</b>.
     *
     * @return boolean
     */
    public boolean checkIsAssumedShape() {
        return (isAssumedShape_ != null);
    }

    /**
     * Sets the boolean property <b>isAssumedShape</b>.
     *
     * @param isAssumedShape
     */
    public void setIsAssumedShape(boolean isAssumedShape) {
        this.isAssumedShape_ = new Boolean(isAssumedShape);
    }

    /**
     * Sets the boolean property <b>isAssumedShape</b>.
     *
     * @param isAssumedShape
     */
    public void setIsAssumedShape(Boolean isAssumedShape) {
        this.isAssumedShape_ = isAssumedShape;
    }

    /**
     * Gets the boolean property <b>isAssumedSize</b>.
     *
     * @return boolean
     */
    public boolean getIsAssumedSize() {
        if (isAssumedSize_ == null) {
            return(false);
        }
        return (isAssumedSize_.booleanValue());
    }

    /**
     * Gets the boolean property <b>isAssumedSize</b>.
     *
     * @param isAssumedSize
     * @return boolean
     */
    public boolean getIsAssumedSize(boolean isAssumedSize) {
        if (isAssumedSize_ == null) {
            return(isAssumedSize);
        }
        return (this.isAssumedSize_.booleanValue());
    }

    /**
     * Gets the boolean property <b>isAssumedSize</b>.
     *
     * @return Boolean
     */
    public Boolean getIsAssumedSizeAsBoolean() {
        return (isAssumedSize_);
    }

    /**
     * Check the boolean property <b>isAssumedSize</b>.
     *
     * @return boolean
     */
    public boolean checkIsAssumedSize() {
        return (isAssumedSize_ != null);
    }

    /**
     * Sets the boolean property <b>isAssumedSize</b>.
     *
     * @param isAssumedSize
     */
    public void setIsAssumedSize(boolean isAssumedSize) {
        this.isAssumedSize_ = new Boolean(isAssumedSize);
    }

    /**
     * Sets the boolean property <b>isAssumedSize</b>.
     *
     * @param isAssumedSize
     */
    public void setIsAssumedSize(Boolean isAssumedSize) {
        this.isAssumedSize_ = isAssumedSize;
    }

    /**
     * Gets the XbfLowerBound property <b>lowerBound</b>.
     *
     * @return XbfLowerBound
     */
    public final XbfLowerBound getLowerBound() {
        return (lowerBound_);
    }

    /**
     * Sets the XbfLowerBound property <b>lowerBound</b>.
     *
     * @param lowerBound
     */
    public final void setLowerBound(XbfLowerBound lowerBound) {
        this.lowerBound_ = lowerBound;
        if (lowerBound != null) {
            lowerBound.rSetParentRNode(this);
        }
    }

    /**
     * Gets the XbfUpperBound property <b>upperBound</b>.
     *
     * @return XbfUpperBound
     */
    public final XbfUpperBound getUpperBound() {
        return (upperBound_);
    }

    /**
     * Sets the XbfUpperBound property <b>upperBound</b>.
     *
     * @param upperBound
     */
    public final void setUpperBound(XbfUpperBound upperBound) {
        this.upperBound_ = upperBound;
        if (upperBound != null) {
            upperBound.rSetParentRNode(this);
        }
    }

    /**
     * Gets the XbfStep property <b>step</b>.
     *
     * @return XbfStep
     */
    public final XbfStep getStep() {
        return (step_);
    }

    /**
     * Sets the XbfStep property <b>step</b>.
     *
     * @param step
     */
    public final void setStep(XbfStep step) {
        this.step_ = step;
        if (step != null) {
            step.rSetParentRNode(this);
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
        buffer.append("<indexRange");
        if (isAssumedShape_ != null) {
            buffer.append(" is_assumed_shape=\"");
            buffer.append(URelaxer.getString(getIsAssumedShape()));
            buffer.append("\"");
        }
        if (isAssumedSize_ != null) {
            buffer.append(" is_assumed_size=\"");
            buffer.append(URelaxer.getString(getIsAssumedSize()));
            buffer.append("\"");
        }
        buffer.append(">");
        if (lowerBound_ != null) {
            lowerBound_.makeTextElement(buffer);
        }
        if (upperBound_ != null) {
            upperBound_.makeTextElement(buffer);
        }
        if (step_ != null) {
            step_.makeTextElement(buffer);
        }
        buffer.append("</indexRange>");
    }

    /**
     * Makes an XML text representation.
     *
     * @param buffer
     * @exception IOException
     */
    public void makeTextElement(Writer buffer) throws IOException {
        int size;
        buffer.write("<indexRange");
        if (isAssumedShape_ != null) {
            buffer.write(" is_assumed_shape=\"");
            buffer.write(URelaxer.getString(getIsAssumedShape()));
            buffer.write("\"");
        }
        if (isAssumedSize_ != null) {
            buffer.write(" is_assumed_size=\"");
            buffer.write(URelaxer.getString(getIsAssumedSize()));
            buffer.write("\"");
        }
        buffer.write(">");
        if (lowerBound_ != null) {
            lowerBound_.makeTextElement(buffer);
        }
        if (upperBound_ != null) {
            upperBound_.makeTextElement(buffer);
        }
        if (step_ != null) {
            step_.makeTextElement(buffer);
        }
        buffer.write("</indexRange>");
    }

    /**
     * Makes an XML text representation.
     *
     * @param buffer
     */
    public void makeTextElement(PrintWriter buffer) {
        int size;
        buffer.print("<indexRange");
        if (isAssumedShape_ != null) {
            buffer.print(" is_assumed_shape=\"");
            buffer.print(URelaxer.getString(getIsAssumedShape()));
            buffer.print("\"");
        }
        if (isAssumedSize_ != null) {
            buffer.print(" is_assumed_size=\"");
            buffer.print(URelaxer.getString(getIsAssumedSize()));
            buffer.print("\"");
        }
        buffer.print(">");
        if (lowerBound_ != null) {
            lowerBound_.makeTextElement(buffer);
        }
        if (upperBound_ != null) {
            upperBound_.makeTextElement(buffer);
        }
        if (step_ != null) {
            step_.makeTextElement(buffer);
        }
        buffer.print("</indexRange>");
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
    public String getIsAssumedShapeAsString() {
        return (URelaxer.getString(getIsAssumedShape()));
    }

    /**
     * Gets the property value as String.
     *
     * @return String
     */
    public String getIsAssumedSizeAsString() {
        return (URelaxer.getString(getIsAssumedSize()));
    }

    /**
     * Sets the property value by String.
     *
     * @param string
     */
    public void setIsAssumedShapeByString(String string) {
        setIsAssumedShape(new Boolean(string).booleanValue());
    }

    /**
     * Sets the property value by String.
     *
     * @param string
     */
    public void setIsAssumedSizeByString(String string) {
        setIsAssumedSize(new Boolean(string).booleanValue());
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
        if (lowerBound_ != null) {
            classNodes.add(lowerBound_);
        }
        if (upperBound_ != null) {
            classNodes.add(upperBound_);
        }
        if (step_ != null) {
            classNodes.add(step_);
        }
        IRNode[] nodes = new IRNode[classNodes.size()];
        return ((IRNode[])classNodes.toArray(nodes));
    }

    /**
     * Tests if a Element <code>element</code> is valid
     * for the <code>XbfIndexRange</code>.
     *
     * @param element
     * @return boolean
     */
    public static boolean isMatch(Element element) {
        if (!URelaxer.isTargetElement(element, "indexRange")) {
            return (false);
        }
        RStack target = new RStack(element);
        boolean $match$ = false;
        Element child;
        if (XbfLowerBound.isMatchHungry(target)) {
        }
        if (XbfUpperBound.isMatchHungry(target)) {
        }
        if (XbfStep.isMatchHungry(target)) {
        }
        if (!target.isEmptyElement()) {
            return (false);
        }
        return (true);
    }

    /**
     * Tests if elements contained in a Stack <code>stack</code>
     * is valid for the <code>XbfIndexRange</code>.
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
     * is valid for the <code>XbfIndexRange</code>.
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
