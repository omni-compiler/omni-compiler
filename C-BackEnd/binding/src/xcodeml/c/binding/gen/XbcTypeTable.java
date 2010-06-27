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
 * <b>XbcTypeTable</b> is generated from XcodeML_C.rng by Relaxer.
 * This class is derived from:
 * 
 * <!-- for programmer
 * <element java:extends="xcodeml.c.obj.XmcObj" name="typeTable">
 *   <zeroOrMore>
 *     <ref name="types"/>
 *   </zeroOrMore>
 * </element>
 * -->
 * <!-- for javadoc -->
 * <pre> &lt;element java:extends="xcodeml.c.obj.XmcObj" name="typeTable"&gt;
 *   &lt;zeroOrMore&gt;
 *     &lt;ref name="types"/&gt;
 *   &lt;/zeroOrMore&gt;
 * &lt;/element&gt;
 * </pre>
 *
 * @version XcodeML_C.rng (Thu Sep 24 16:30:18 JST 2009)
 * @author  Relaxer 1.0 (http://www.relaxer.org)
 */
public class XbcTypeTable extends xcodeml.c.obj.XmcObj implements java.io.Serializable, Cloneable, IRVisitable, IRNode {
    // List<IXbcTypesChoice>
    private java.util.List types_ = new java.util.ArrayList();
    private IRNode parentRNode_;

    /**
     * Creates a <code>XbcTypeTable</code>.
     *
     */
    public XbcTypeTable() {
    }

    /**
     * Creates a <code>XbcTypeTable</code>.
     *
     * @param source
     */
    public XbcTypeTable(XbcTypeTable source) {
        setup(source);
    }

    /**
     * Creates a <code>XbcTypeTable</code> by the Stack <code>stack</code>
     * that contains Elements.
     * This constructor is supposed to be used internally
     * by the Relaxer system.
     *
     * @param stack
     */
    public XbcTypeTable(RStack stack) {
        setup(stack);
    }

    /**
     * Creates a <code>XbcTypeTable</code> by the Document <code>doc</code>.
     *
     * @param doc
     */
    public XbcTypeTable(Document doc) {
        setup(doc.getDocumentElement());
    }

    /**
     * Creates a <code>XbcTypeTable</code> by the Element <code>element</code>.
     *
     * @param element
     */
    public XbcTypeTable(Element element) {
        setup(element);
    }

    /**
     * Creates a <code>XbcTypeTable</code> by the File <code>file</code>.
     *
     * @param file
     * @exception IOException
     * @exception SAXException
     * @exception ParserConfigurationException
     */
    public XbcTypeTable(File file) throws IOException, SAXException, ParserConfigurationException {
        setup(file);
    }

    /**
     * Creates a <code>XbcTypeTable</code>
     * by the String representation of URI <code>uri</code>.
     *
     * @param uri
     * @exception IOException
     * @exception SAXException
     * @exception ParserConfigurationException
     */
    public XbcTypeTable(String uri) throws IOException, SAXException, ParserConfigurationException {
        setup(uri);
    }

    /**
     * Creates a <code>XbcTypeTable</code> by the URL <code>url</code>.
     *
     * @param url
     * @exception IOException
     * @exception SAXException
     * @exception ParserConfigurationException
     */
    public XbcTypeTable(URL url) throws IOException, SAXException, ParserConfigurationException {
        setup(url);
    }

    /**
     * Creates a <code>XbcTypeTable</code> by the InputStream <code>in</code>.
     *
     * @param in
     * @exception IOException
     * @exception SAXException
     * @exception ParserConfigurationException
     */
    public XbcTypeTable(InputStream in) throws IOException, SAXException, ParserConfigurationException {
        setup(in);
    }

    /**
     * Creates a <code>XbcTypeTable</code> by the InputSource <code>is</code>.
     *
     * @param is
     * @exception IOException
     * @exception SAXException
     * @exception ParserConfigurationException
     */
    public XbcTypeTable(InputSource is) throws IOException, SAXException, ParserConfigurationException {
        setup(is);
    }

    /**
     * Creates a <code>XbcTypeTable</code> by the Reader <code>reader</code>.
     *
     * @param reader
     * @exception IOException
     * @exception SAXException
     * @exception ParserConfigurationException
     */
    public XbcTypeTable(Reader reader) throws IOException, SAXException, ParserConfigurationException {
        setup(reader);
    }

    /**
     * Initializes the <code>XbcTypeTable</code> by the XbcTypeTable <code>source</code>.
     *
     * @param source
     */
    public void setup(XbcTypeTable source) {
        int size;
        this.types_.clear();
        size = source.types_.size();
        for (int i = 0;i < size;i++) {
            addTypes((IXbcTypesChoice)source.getTypes(i).clone());
        }
    }

    /**
     * Initializes the <code>XbcTypeTable</code> by the Document <code>doc</code>.
     *
     * @param doc
     */
    public void setup(Document doc) {
        setup(doc.getDocumentElement());
    }

    /**
     * Initializes the <code>XbcTypeTable</code> by the Element <code>element</code>.
     *
     * @param element
     */
    public void setup(Element element) {
        init(element);
    }

    /**
     * Initializes the <code>XbcTypeTable</code> by the Stack <code>stack</code>
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
        types_.clear();
        while (true) {
            if (XbcArrayType.isMatch(stack)) {
                addTypes(factory.createXbcArrayType(stack));
            } else if (XbcFunctionType.isMatch(stack)) {
                addTypes(factory.createXbcFunctionType(stack));
            } else if (XbcBasicType.isMatch(stack)) {
                addTypes(factory.createXbcBasicType(stack));
            } else if (XbcPointerType.isMatch(stack)) {
                addTypes(factory.createXbcPointerType(stack));
            } else if (XbcUnionType.isMatch(stack)) {
                addTypes(factory.createXbcUnionType(stack));
            } else if (XbcStructType.isMatch(stack)) {
                addTypes(factory.createXbcStructType(stack));
            } else if (XbcEnumType.isMatch(stack)) {
                addTypes(factory.createXbcEnumType(stack));
            } else if (XbcCoArrayType.isMatch(stack)) {
                addTypes(factory.createXbcCoArrayType(stack));
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
        return (factory.createXbcTypeTable(this));
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
        Element element = doc.createElement("typeTable");
        int size;
        size = this.types_.size();
        for (int i = 0;i < size;i++) {
            IXbcTypesChoice value = (IXbcTypesChoice)this.types_.get(i);
            value.makeElement(element);
        }
        parent.appendChild(element);
    }

    /**
     * Initializes the <code>XbcTypeTable</code> by the File <code>file</code>.
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
     * Initializes the <code>XbcTypeTable</code>
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
     * Initializes the <code>XbcTypeTable</code> by the URL <code>url</code>.
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
     * Initializes the <code>XbcTypeTable</code> by the InputStream <code>in</code>.
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
     * Initializes the <code>XbcTypeTable</code> by the InputSource <code>is</code>.
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
     * Initializes the <code>XbcTypeTable</code> by the Reader <code>reader</code>.
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
     * Gets the IXbcTypesChoice property <b>types</b>.
     *
     * @return IXbcTypesChoice[]
     */
    public final IXbcTypesChoice[] getTypes() {
        IXbcTypesChoice[] array = new IXbcTypesChoice[types_.size()];
        return ((IXbcTypesChoice[])types_.toArray(array));
    }

    /**
     * Sets the IXbcTypesChoice property <b>types</b>.
     *
     * @param types
     */
    public final void setTypes(IXbcTypesChoice[] types) {
        this.types_.clear();
        for (int i = 0;i < types.length;i++) {
            addTypes(types[i]);
        }
        for (int i = 0;i < types.length;i++) {
            types[i].rSetParentRNode(this);
        }
    }

    /**
     * Sets the IXbcTypesChoice property <b>types</b>.
     *
     * @param types
     */
    public final void setTypes(IXbcTypesChoice types) {
        this.types_.clear();
        addTypes(types);
        if (types != null) {
            types.rSetParentRNode(this);
        }
    }

    /**
     * Adds the IXbcTypesChoice property <b>types</b>.
     *
     * @param types
     */
    public final void addTypes(IXbcTypesChoice types) {
        this.types_.add(types);
        if (types != null) {
            types.rSetParentRNode(this);
        }
    }

    /**
     * Adds the IXbcTypesChoice property <b>types</b>.
     *
     * @param types
     */
    public final void addTypes(IXbcTypesChoice[] types) {
        for (int i = 0;i < types.length;i++) {
            addTypes(types[i]);
        }
        for (int i = 0;i < types.length;i++) {
            types[i].rSetParentRNode(this);
        }
    }

    /**
     * Gets number of the IXbcTypesChoice property <b>types</b>.
     *
     * @return int
     */
    public final int sizeTypes() {
        return (types_.size());
    }

    /**
     * Gets the IXbcTypesChoice property <b>types</b> by index.
     *
     * @param index
     * @return IXbcTypesChoice
     */
    public final IXbcTypesChoice getTypes(int index) {
        return ((IXbcTypesChoice)types_.get(index));
    }

    /**
     * Sets the IXbcTypesChoice property <b>types</b> by index.
     *
     * @param index
     * @param types
     */
    public final void setTypes(int index, IXbcTypesChoice types) {
        this.types_.set(index, types);
        if (types != null) {
            types.rSetParentRNode(this);
        }
    }

    /**
     * Adds the IXbcTypesChoice property <b>types</b> by index.
     *
     * @param index
     * @param types
     */
    public final void addTypes(int index, IXbcTypesChoice types) {
        this.types_.add(index, types);
        if (types != null) {
            types.rSetParentRNode(this);
        }
    }

    /**
     * Remove the IXbcTypesChoice property <b>types</b> by index.
     *
     * @param index
     */
    public final void removeTypes(int index) {
        this.types_.remove(index);
    }

    /**
     * Remove the IXbcTypesChoice property <b>types</b> by object.
     *
     * @param types
     */
    public final void removeTypes(IXbcTypesChoice types) {
        this.types_.remove(types);
    }

    /**
     * Clear the IXbcTypesChoice property <b>types</b>.
     *
     */
    public final void clearTypes() {
        this.types_.clear();
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
        buffer.append("<typeTable");
        size = this.types_.size();
        for (int i = 0;i < size;i++) {
            IXbcTypesChoice value = (IXbcTypesChoice)this.types_.get(i);
            value.makeTextAttribute(buffer);
        }
        buffer.append(">");
        size = this.types_.size();
        for (int i = 0;i < size;i++) {
            IXbcTypesChoice value = (IXbcTypesChoice)this.types_.get(i);
            value.makeTextElement(buffer);
        }
        buffer.append("</typeTable>");
    }

    /**
     * Makes an XML text representation.
     *
     * @param buffer
     * @exception IOException
     */
    public void makeTextElement(Writer buffer) throws IOException {
        int size;
        buffer.write("<typeTable");
        size = this.types_.size();
        for (int i = 0;i < size;i++) {
            IXbcTypesChoice value = (IXbcTypesChoice)this.types_.get(i);
            value.makeTextAttribute(buffer);
        }
        buffer.write(">");
        size = this.types_.size();
        for (int i = 0;i < size;i++) {
            IXbcTypesChoice value = (IXbcTypesChoice)this.types_.get(i);
            value.makeTextElement(buffer);
        }
        buffer.write("</typeTable>");
    }

    /**
     * Makes an XML text representation.
     *
     * @param buffer
     */
    public void makeTextElement(PrintWriter buffer) {
        int size;
        buffer.print("<typeTable");
        size = this.types_.size();
        for (int i = 0;i < size;i++) {
            IXbcTypesChoice value = (IXbcTypesChoice)this.types_.get(i);
            value.makeTextAttribute(buffer);
        }
        buffer.print(">");
        size = this.types_.size();
        for (int i = 0;i < size;i++) {
            IXbcTypesChoice value = (IXbcTypesChoice)this.types_.get(i);
            value.makeTextElement(buffer);
        }
        buffer.print("</typeTable>");
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
        classNodes.addAll(types_);
        IRNode[] nodes = new IRNode[classNodes.size()];
        return ((IRNode[])classNodes.toArray(nodes));
    }

    /**
     * Tests if a Element <code>element</code> is valid
     * for the <code>XbcTypeTable</code>.
     *
     * @param element
     * @return boolean
     */
    public static boolean isMatch(Element element) {
        if (!URelaxer.isTargetElement(element, "typeTable")) {
            return (false);
        }
        RStack target = new RStack(element);
        boolean $match$ = false;
        Element child;
        while (true) {
            if (XbcArrayType.isMatchHungry(target)) {
                $match$ = true;
            } else if (XbcFunctionType.isMatchHungry(target)) {
                $match$ = true;
            } else if (XbcBasicType.isMatchHungry(target)) {
                $match$ = true;
            } else if (XbcPointerType.isMatchHungry(target)) {
                $match$ = true;
            } else if (XbcUnionType.isMatchHungry(target)) {
                $match$ = true;
            } else if (XbcStructType.isMatchHungry(target)) {
                $match$ = true;
            } else if (XbcEnumType.isMatchHungry(target)) {
                $match$ = true;
            } else if (XbcCoArrayType.isMatchHungry(target)) {
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
     * is valid for the <code>XbcTypeTable</code>.
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
     * is valid for the <code>XbcTypeTable</code>.
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
