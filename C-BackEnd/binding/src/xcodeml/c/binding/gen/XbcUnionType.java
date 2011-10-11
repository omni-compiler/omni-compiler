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
 * <b>XbcUnionType</b> is generated from XcodeML_C.rng by Relaxer.
 * This class is derived from:
 * 
 * <!-- for programmer
 * <element java:extends="xcodeml.c.obj.XmcObj" java:implements="xcodeml.c.binding.IXbcCompositeType" name="unionType">
 *   <ref name="BaseType"/>
 *   <ref name="type"/>
 *   <optional>
 *     <ref name="symbols"/>
 *   </optional>
 * </element>
 * -->
 * <!-- for javadoc -->
 * <pre> &lt;element java:extends="xcodeml.c.obj.XmcObj" java:implements="xcodeml.c.binding.IXbcCompositeType" name="unionType"&gt;
 *   &lt;ref name="BaseType"/&gt;
 *   &lt;ref name="type"/&gt;
 *   &lt;optional&gt;
 *     &lt;ref name="symbols"/&gt;
 *   &lt;/optional&gt;
 * &lt;/element&gt;
 * </pre>
 *
 * @version XcodeML_C.rng (Fri Oct 07 17:51:13 JST 2011)
 * @author  Relaxer 1.0 (http://www.relaxer.org)
 */
public class XbcUnionType extends xcodeml.c.obj.XmcObj implements java.io.Serializable, Cloneable, xcodeml.c.binding.IXbcCompositeType, IRVisitable, IRNode, IXbcTypesChoice {
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

    private String isVolatile_;
    private String isConst_;
    private String isRestrict_;
    private String type_;
    private XbcGccAttributes gccAttributes_;
    private XbcSymbols symbols_;
    private IRNode parentRNode_;

    /**
     * Creates a <code>XbcUnionType</code>.
     *
     */
    public XbcUnionType() {
        type_ = "";
    }

    /**
     * Creates a <code>XbcUnionType</code>.
     *
     * @param source
     */
    public XbcUnionType(XbcUnionType source) {
        setup(source);
    }

    /**
     * Creates a <code>XbcUnionType</code> by the Stack <code>stack</code>
     * that contains Elements.
     * This constructor is supposed to be used internally
     * by the Relaxer system.
     *
     * @param stack
     */
    public XbcUnionType(RStack stack) {
        setup(stack);
    }

    /**
     * Creates a <code>XbcUnionType</code> by the Document <code>doc</code>.
     *
     * @param doc
     */
    public XbcUnionType(Document doc) {
        setup(doc.getDocumentElement());
    }

    /**
     * Creates a <code>XbcUnionType</code> by the Element <code>element</code>.
     *
     * @param element
     */
    public XbcUnionType(Element element) {
        setup(element);
    }

    /**
     * Creates a <code>XbcUnionType</code> by the File <code>file</code>.
     *
     * @param file
     * @exception IOException
     * @exception SAXException
     * @exception ParserConfigurationException
     */
    public XbcUnionType(File file) throws IOException, SAXException, ParserConfigurationException {
        setup(file);
    }

    /**
     * Creates a <code>XbcUnionType</code>
     * by the String representation of URI <code>uri</code>.
     *
     * @param uri
     * @exception IOException
     * @exception SAXException
     * @exception ParserConfigurationException
     */
    public XbcUnionType(String uri) throws IOException, SAXException, ParserConfigurationException {
        setup(uri);
    }

    /**
     * Creates a <code>XbcUnionType</code> by the URL <code>url</code>.
     *
     * @param url
     * @exception IOException
     * @exception SAXException
     * @exception ParserConfigurationException
     */
    public XbcUnionType(URL url) throws IOException, SAXException, ParserConfigurationException {
        setup(url);
    }

    /**
     * Creates a <code>XbcUnionType</code> by the InputStream <code>in</code>.
     *
     * @param in
     * @exception IOException
     * @exception SAXException
     * @exception ParserConfigurationException
     */
    public XbcUnionType(InputStream in) throws IOException, SAXException, ParserConfigurationException {
        setup(in);
    }

    /**
     * Creates a <code>XbcUnionType</code> by the InputSource <code>is</code>.
     *
     * @param is
     * @exception IOException
     * @exception SAXException
     * @exception ParserConfigurationException
     */
    public XbcUnionType(InputSource is) throws IOException, SAXException, ParserConfigurationException {
        setup(is);
    }

    /**
     * Creates a <code>XbcUnionType</code> by the Reader <code>reader</code>.
     *
     * @param reader
     * @exception IOException
     * @exception SAXException
     * @exception ParserConfigurationException
     */
    public XbcUnionType(Reader reader) throws IOException, SAXException, ParserConfigurationException {
        setup(reader);
    }

    /**
     * Initializes the <code>XbcUnionType</code> by the XbcUnionType <code>source</code>.
     *
     * @param source
     */
    public void setup(XbcUnionType source) {
        int size;
        setIsVolatile(source.getIsVolatile());
        setIsConst(source.getIsConst());
        setIsRestrict(source.getIsRestrict());
        setType(source.getType());
        if (source.gccAttributes_ != null) {
            setGccAttributes((XbcGccAttributes)source.getGccAttributes().clone());
        }
        if (source.symbols_ != null) {
            setSymbols((XbcSymbols)source.getSymbols().clone());
        }
    }

    /**
     * Initializes the <code>XbcUnionType</code> by the Document <code>doc</code>.
     *
     * @param doc
     */
    public void setup(Document doc) {
        setup(doc.getDocumentElement());
    }

    /**
     * Initializes the <code>XbcUnionType</code> by the Element <code>element</code>.
     *
     * @param element
     */
    public void setup(Element element) {
        init(element);
    }

    /**
     * Initializes the <code>XbcUnionType</code> by the Stack <code>stack</code>
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
        isVolatile_ = URelaxer.getAttributePropertyAsString(element, "is_volatile");
        isConst_ = URelaxer.getAttributePropertyAsString(element, "is_const");
        isRestrict_ = URelaxer.getAttributePropertyAsString(element, "is_restrict");
        type_ = URelaxer.getAttributePropertyAsString(element, "type");
        if (XbcGccAttributes.isMatch(stack)) {
            setGccAttributes(factory.createXbcGccAttributes(stack));
        }
        if (XbcSymbols.isMatch(stack)) {
            setSymbols(factory.createXbcSymbols(stack));
        }
    }

    /**
     * @return Object
     */
    public Object clone() {
        IXcodeML_CFactory factory = XcodeML_CFactory.getFactory();
        return (factory.createXbcUnionType(this));
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
        Element element = doc.createElement("unionType");
        int size;
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
        if (this.symbols_ != null) {
            this.symbols_.makeElement(element);
        }
        parent.appendChild(element);
    }

    /**
     * Initializes the <code>XbcUnionType</code> by the File <code>file</code>.
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
     * Initializes the <code>XbcUnionType</code>
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
     * Initializes the <code>XbcUnionType</code> by the URL <code>url</code>.
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
     * Initializes the <code>XbcUnionType</code> by the InputStream <code>in</code>.
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
     * Initializes the <code>XbcUnionType</code> by the InputSource <code>is</code>.
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
     * Initializes the <code>XbcUnionType</code> by the Reader <code>reader</code>.
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
     * Gets the XbcSymbols property <b>symbols</b>.
     *
     * @return XbcSymbols
     */
    public final XbcSymbols getSymbols() {
        return (symbols_);
    }

    /**
     * Sets the XbcSymbols property <b>symbols</b>.
     *
     * @param symbols
     */
    public final void setSymbols(XbcSymbols symbols) {
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
        buffer.append("<unionType");
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
        if (symbols_ != null) {
            symbols_.makeTextElement(buffer);
        }
        buffer.append("</unionType>");
    }

    /**
     * Makes an XML text representation.
     *
     * @param buffer
     * @exception IOException
     */
    public void makeTextElement(Writer buffer) throws IOException {
        int size;
        buffer.write("<unionType");
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
        if (symbols_ != null) {
            symbols_.makeTextElement(buffer);
        }
        buffer.write("</unionType>");
    }

    /**
     * Makes an XML text representation.
     *
     * @param buffer
     */
    public void makeTextElement(PrintWriter buffer) {
        int size;
        buffer.print("<unionType");
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
        if (symbols_ != null) {
            symbols_.makeTextElement(buffer);
        }
        buffer.print("</unionType>");
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
        if (symbols_ != null) {
            classNodes.add(symbols_);
        }
        IRNode[] nodes = new IRNode[classNodes.size()];
        return ((IRNode[])classNodes.toArray(nodes));
    }

    /**
     * Tests if a Element <code>element</code> is valid
     * for the <code>XbcUnionType</code>.
     *
     * @param element
     * @return boolean
     */
    public static boolean isMatch(Element element) {
        if (!URelaxer.isTargetElement(element, "unionType")) {
            return (false);
        }
        RStack target = new RStack(element);
        boolean $match$ = false;
        Element child;
        if (!URelaxer.hasAttributeHungry(target, "type")) {
            return (false);
        }
        $match$ = true;
        if (XbcGccAttributes.isMatchHungry(target)) {
        }
        if (XbcSymbols.isMatchHungry(target)) {
        }
        if (!target.isEmptyElement()) {
            return (false);
        }
        return (true);
    }

    /**
     * Tests if elements contained in a Stack <code>stack</code>
     * is valid for the <code>XbcUnionType</code>.
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
     * is valid for the <code>XbcUnionType</code>.
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
