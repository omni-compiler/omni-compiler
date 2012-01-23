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
 * <b>XbfXcodeProgram</b> is generated from XcodeML_F.rng by Relaxer.
 * This class is derived from:
 * 
 * <!-- for programmer
 * <element java:extends="xcodeml.f.XmfObj" java:implements="xcodeml.binding.XmXcodeProgram" name="XcodeProgram">
 *   <ref name="defAttrProgram"/>
 *   <ref name="typeTable"/>
 *   <ref name="globalSymbols"/>
 *   <ref name="globalDeclarations"/>
 * </element>
 * -->
 * <!-- for javadoc -->
 * <pre> &lt;element java:extends="xcodeml.f.XmfObj" java:implements="xcodeml.binding.XmXcodeProgram" name="XcodeProgram"&gt;
 *   &lt;ref name="defAttrProgram"/&gt;
 *   &lt;ref name="typeTable"/&gt;
 *   &lt;ref name="globalSymbols"/&gt;
 *   &lt;ref name="globalDeclarations"/&gt;
 * &lt;/element&gt;
 * </pre>
 *
 * @version XcodeML_F.rng (Mon Jan 23 20:53:32 JST 2012)
 * @author  Relaxer 1.0 (http://www.relaxer.org)
 */
public class XbfXcodeProgram extends xcodeml.f.XmfObj implements java.io.Serializable, Cloneable, xcodeml.binding.XmXcodeProgram, IRVisitable, IRNode {
    private String compilerInfo_;
    private String version_;
    private String time_;
    private String language_;
    private String source_;
    private XbfTypeTable typeTable_;
    private XbfGlobalSymbols globalSymbols_;
    private XbfGlobalDeclarations globalDeclarations_;
    private IRNode parentRNode_;

    /**
     * Creates a <code>XbfXcodeProgram</code>.
     *
     */
    public XbfXcodeProgram() {
    }

    /**
     * Creates a <code>XbfXcodeProgram</code>.
     *
     * @param source
     */
    public XbfXcodeProgram(XbfXcodeProgram source) {
        setup(source);
    }

    /**
     * Creates a <code>XbfXcodeProgram</code> by the Stack <code>stack</code>
     * that contains Elements.
     * This constructor is supposed to be used internally
     * by the Relaxer system.
     *
     * @param stack
     */
    public XbfXcodeProgram(RStack stack) {
        setup(stack);
    }

    /**
     * Creates a <code>XbfXcodeProgram</code> by the Document <code>doc</code>.
     *
     * @param doc
     */
    public XbfXcodeProgram(Document doc) {
        setup(doc.getDocumentElement());
    }

    /**
     * Creates a <code>XbfXcodeProgram</code> by the Element <code>element</code>.
     *
     * @param element
     */
    public XbfXcodeProgram(Element element) {
        setup(element);
    }

    /**
     * Creates a <code>XbfXcodeProgram</code> by the File <code>file</code>.
     *
     * @param file
     * @exception IOException
     * @exception SAXException
     * @exception ParserConfigurationException
     */
    public XbfXcodeProgram(File file) throws IOException, SAXException, ParserConfigurationException {
        setup(file);
    }

    /**
     * Creates a <code>XbfXcodeProgram</code>
     * by the String representation of URI <code>uri</code>.
     *
     * @param uri
     * @exception IOException
     * @exception SAXException
     * @exception ParserConfigurationException
     */
    public XbfXcodeProgram(String uri) throws IOException, SAXException, ParserConfigurationException {
        setup(uri);
    }

    /**
     * Creates a <code>XbfXcodeProgram</code> by the URL <code>url</code>.
     *
     * @param url
     * @exception IOException
     * @exception SAXException
     * @exception ParserConfigurationException
     */
    public XbfXcodeProgram(URL url) throws IOException, SAXException, ParserConfigurationException {
        setup(url);
    }

    /**
     * Creates a <code>XbfXcodeProgram</code> by the InputStream <code>in</code>.
     *
     * @param in
     * @exception IOException
     * @exception SAXException
     * @exception ParserConfigurationException
     */
    public XbfXcodeProgram(InputStream in) throws IOException, SAXException, ParserConfigurationException {
        setup(in);
    }

    /**
     * Creates a <code>XbfXcodeProgram</code> by the InputSource <code>is</code>.
     *
     * @param is
     * @exception IOException
     * @exception SAXException
     * @exception ParserConfigurationException
     */
    public XbfXcodeProgram(InputSource is) throws IOException, SAXException, ParserConfigurationException {
        setup(is);
    }

    /**
     * Creates a <code>XbfXcodeProgram</code> by the Reader <code>reader</code>.
     *
     * @param reader
     * @exception IOException
     * @exception SAXException
     * @exception ParserConfigurationException
     */
    public XbfXcodeProgram(Reader reader) throws IOException, SAXException, ParserConfigurationException {
        setup(reader);
    }

    /**
     * Initializes the <code>XbfXcodeProgram</code> by the XbfXcodeProgram <code>source</code>.
     *
     * @param source
     */
    public void setup(XbfXcodeProgram source) {
        int size;
        setCompilerInfo(source.getCompilerInfo());
        setVersion(source.getVersion());
        setTime(source.getTime());
        setLanguage(source.getLanguage());
        setSource(source.getSource());
        if (source.typeTable_ != null) {
            setTypeTable((XbfTypeTable)source.getTypeTable().clone());
        }
        if (source.globalSymbols_ != null) {
            setGlobalSymbols((XbfGlobalSymbols)source.getGlobalSymbols().clone());
        }
        if (source.globalDeclarations_ != null) {
            setGlobalDeclarations((XbfGlobalDeclarations)source.getGlobalDeclarations().clone());
        }
    }

    /**
     * Initializes the <code>XbfXcodeProgram</code> by the Document <code>doc</code>.
     *
     * @param doc
     */
    public void setup(Document doc) {
        setup(doc.getDocumentElement());
    }

    /**
     * Initializes the <code>XbfXcodeProgram</code> by the Element <code>element</code>.
     *
     * @param element
     */
    public void setup(Element element) {
        init(element);
    }

    /**
     * Initializes the <code>XbfXcodeProgram</code> by the Stack <code>stack</code>
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
        compilerInfo_ = URelaxer.getAttributePropertyAsString(element, "compiler-info");
        version_ = URelaxer.getAttributePropertyAsString(element, "version");
        time_ = URelaxer.getAttributePropertyAsString(element, "time");
        language_ = URelaxer.getAttributePropertyAsString(element, "language");
        source_ = URelaxer.getAttributePropertyAsString(element, "source");
        setTypeTable(factory.createXbfTypeTable(stack));
        setGlobalSymbols(factory.createXbfGlobalSymbols(stack));
        setGlobalDeclarations(factory.createXbfGlobalDeclarations(stack));
    }

    /**
     * @return Object
     */
    public Object clone() {
        IXcodeML_FFactory factory = XcodeML_FFactory.getFactory();
        return (factory.createXbfXcodeProgram(this));
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
        Element element = doc.createElement("XcodeProgram");
        int size;
        if (this.compilerInfo_ != null) {
            URelaxer.setAttributePropertyByString(element, "compiler-info", this.compilerInfo_);
        }
        if (this.version_ != null) {
            URelaxer.setAttributePropertyByString(element, "version", this.version_);
        }
        if (this.time_ != null) {
            URelaxer.setAttributePropertyByString(element, "time", this.time_);
        }
        if (this.language_ != null) {
            URelaxer.setAttributePropertyByString(element, "language", this.language_);
        }
        if (this.source_ != null) {
            URelaxer.setAttributePropertyByString(element, "source", this.source_);
        }
        this.typeTable_.makeElement(element);
        this.globalSymbols_.makeElement(element);
        this.globalDeclarations_.makeElement(element);
        parent.appendChild(element);
    }

    /**
     * Initializes the <code>XbfXcodeProgram</code> by the File <code>file</code>.
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
     * Initializes the <code>XbfXcodeProgram</code>
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
     * Initializes the <code>XbfXcodeProgram</code> by the URL <code>url</code>.
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
     * Initializes the <code>XbfXcodeProgram</code> by the InputStream <code>in</code>.
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
     * Initializes the <code>XbfXcodeProgram</code> by the InputSource <code>is</code>.
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
     * Initializes the <code>XbfXcodeProgram</code> by the Reader <code>reader</code>.
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
     * Gets the String property <b>compilerInfo</b>.
     *
     * @return String
     */
    public final String getCompilerInfo() {
        return (compilerInfo_);
    }

    /**
     * Sets the String property <b>compilerInfo</b>.
     *
     * @param compilerInfo
     */
    public final void setCompilerInfo(String compilerInfo) {
        this.compilerInfo_ = compilerInfo;
    }

    /**
     * Gets the String property <b>version</b>.
     *
     * @return String
     */
    public final String getVersion() {
        return (version_);
    }

    /**
     * Sets the String property <b>version</b>.
     *
     * @param version
     */
    public final void setVersion(String version) {
        this.version_ = version;
    }

    /**
     * Gets the String property <b>time</b>.
     *
     * @return String
     */
    public final String getTime() {
        return (time_);
    }

    /**
     * Sets the String property <b>time</b>.
     *
     * @param time
     */
    public final void setTime(String time) {
        this.time_ = time;
    }

    /**
     * Gets the String property <b>language</b>.
     *
     * @return String
     */
    public final String getLanguage() {
        return (language_);
    }

    /**
     * Sets the String property <b>language</b>.
     *
     * @param language
     */
    public final void setLanguage(String language) {
        this.language_ = language;
    }

    /**
     * Gets the String property <b>source</b>.
     *
     * @return String
     */
    public final String getSource() {
        return (source_);
    }

    /**
     * Sets the String property <b>source</b>.
     *
     * @param source
     */
    public final void setSource(String source) {
        this.source_ = source;
    }

    /**
     * Gets the XbfTypeTable property <b>typeTable</b>.
     *
     * @return XbfTypeTable
     */
    public final XbfTypeTable getTypeTable() {
        return (typeTable_);
    }

    /**
     * Sets the XbfTypeTable property <b>typeTable</b>.
     *
     * @param typeTable
     */
    public final void setTypeTable(XbfTypeTable typeTable) {
        this.typeTable_ = typeTable;
        if (typeTable != null) {
            typeTable.rSetParentRNode(this);
        }
    }

    /**
     * Gets the XbfGlobalSymbols property <b>globalSymbols</b>.
     *
     * @return XbfGlobalSymbols
     */
    public final XbfGlobalSymbols getGlobalSymbols() {
        return (globalSymbols_);
    }

    /**
     * Sets the XbfGlobalSymbols property <b>globalSymbols</b>.
     *
     * @param globalSymbols
     */
    public final void setGlobalSymbols(XbfGlobalSymbols globalSymbols) {
        this.globalSymbols_ = globalSymbols;
        if (globalSymbols != null) {
            globalSymbols.rSetParentRNode(this);
        }
    }

    /**
     * Gets the XbfGlobalDeclarations property <b>globalDeclarations</b>.
     *
     * @return XbfGlobalDeclarations
     */
    public final XbfGlobalDeclarations getGlobalDeclarations() {
        return (globalDeclarations_);
    }

    /**
     * Sets the XbfGlobalDeclarations property <b>globalDeclarations</b>.
     *
     * @param globalDeclarations
     */
    public final void setGlobalDeclarations(XbfGlobalDeclarations globalDeclarations) {
        this.globalDeclarations_ = globalDeclarations;
        if (globalDeclarations != null) {
            globalDeclarations.rSetParentRNode(this);
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
        buffer.append("<XcodeProgram");
        if (compilerInfo_ != null) {
            buffer.append(" compiler-info=\"");
            buffer.append(URelaxer.escapeAttrQuot(URelaxer.getString(getCompilerInfo())));
            buffer.append("\"");
        }
        if (version_ != null) {
            buffer.append(" version=\"");
            buffer.append(URelaxer.escapeAttrQuot(URelaxer.getString(getVersion())));
            buffer.append("\"");
        }
        if (time_ != null) {
            buffer.append(" time=\"");
            buffer.append(URelaxer.escapeAttrQuot(URelaxer.getString(getTime())));
            buffer.append("\"");
        }
        if (language_ != null) {
            buffer.append(" language=\"");
            buffer.append(URelaxer.escapeAttrQuot(URelaxer.getString(getLanguage())));
            buffer.append("\"");
        }
        if (source_ != null) {
            buffer.append(" source=\"");
            buffer.append(URelaxer.escapeAttrQuot(URelaxer.getString(getSource())));
            buffer.append("\"");
        }
        buffer.append(">");
        typeTable_.makeTextElement(buffer);
        globalSymbols_.makeTextElement(buffer);
        globalDeclarations_.makeTextElement(buffer);
        buffer.append("</XcodeProgram>");
    }

    /**
     * Makes an XML text representation.
     *
     * @param buffer
     * @exception IOException
     */
    public void makeTextElement(Writer buffer) throws IOException {
        int size;
        buffer.write("<XcodeProgram");
        if (compilerInfo_ != null) {
            buffer.write(" compiler-info=\"");
            buffer.write(URelaxer.escapeAttrQuot(URelaxer.getString(getCompilerInfo())));
            buffer.write("\"");
        }
        if (version_ != null) {
            buffer.write(" version=\"");
            buffer.write(URelaxer.escapeAttrQuot(URelaxer.getString(getVersion())));
            buffer.write("\"");
        }
        if (time_ != null) {
            buffer.write(" time=\"");
            buffer.write(URelaxer.escapeAttrQuot(URelaxer.getString(getTime())));
            buffer.write("\"");
        }
        if (language_ != null) {
            buffer.write(" language=\"");
            buffer.write(URelaxer.escapeAttrQuot(URelaxer.getString(getLanguage())));
            buffer.write("\"");
        }
        if (source_ != null) {
            buffer.write(" source=\"");
            buffer.write(URelaxer.escapeAttrQuot(URelaxer.getString(getSource())));
            buffer.write("\"");
        }
        buffer.write(">");
        typeTable_.makeTextElement(buffer);
        globalSymbols_.makeTextElement(buffer);
        globalDeclarations_.makeTextElement(buffer);
        buffer.write("</XcodeProgram>");
    }

    /**
     * Makes an XML text representation.
     *
     * @param buffer
     */
    public void makeTextElement(PrintWriter buffer) {
        int size;
        buffer.print("<XcodeProgram");
        if (compilerInfo_ != null) {
            buffer.print(" compiler-info=\"");
            buffer.print(URelaxer.escapeAttrQuot(URelaxer.getString(getCompilerInfo())));
            buffer.print("\"");
        }
        if (version_ != null) {
            buffer.print(" version=\"");
            buffer.print(URelaxer.escapeAttrQuot(URelaxer.getString(getVersion())));
            buffer.print("\"");
        }
        if (time_ != null) {
            buffer.print(" time=\"");
            buffer.print(URelaxer.escapeAttrQuot(URelaxer.getString(getTime())));
            buffer.print("\"");
        }
        if (language_ != null) {
            buffer.print(" language=\"");
            buffer.print(URelaxer.escapeAttrQuot(URelaxer.getString(getLanguage())));
            buffer.print("\"");
        }
        if (source_ != null) {
            buffer.print(" source=\"");
            buffer.print(URelaxer.escapeAttrQuot(URelaxer.getString(getSource())));
            buffer.print("\"");
        }
        buffer.print(">");
        typeTable_.makeTextElement(buffer);
        globalSymbols_.makeTextElement(buffer);
        globalDeclarations_.makeTextElement(buffer);
        buffer.print("</XcodeProgram>");
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
    public String getCompilerInfoAsString() {
        return (URelaxer.getString(getCompilerInfo()));
    }

    /**
     * Gets the property value as String.
     *
     * @return String
     */
    public String getVersionAsString() {
        return (URelaxer.getString(getVersion()));
    }

    /**
     * Gets the property value as String.
     *
     * @return String
     */
    public String getTimeAsString() {
        return (URelaxer.getString(getTime()));
    }

    /**
     * Gets the property value as String.
     *
     * @return String
     */
    public String getLanguageAsString() {
        return (URelaxer.getString(getLanguage()));
    }

    /**
     * Gets the property value as String.
     *
     * @return String
     */
    public String getSourceAsString() {
        return (URelaxer.getString(getSource()));
    }

    /**
     * Sets the property value by String.
     *
     * @param string
     */
    public void setCompilerInfoByString(String string) {
        setCompilerInfo(string);
    }

    /**
     * Sets the property value by String.
     *
     * @param string
     */
    public void setVersionByString(String string) {
        setVersion(string);
    }

    /**
     * Sets the property value by String.
     *
     * @param string
     */
    public void setTimeByString(String string) {
        setTime(string);
    }

    /**
     * Sets the property value by String.
     *
     * @param string
     */
    public void setLanguageByString(String string) {
        setLanguage(string);
    }

    /**
     * Sets the property value by String.
     *
     * @param string
     */
    public void setSourceByString(String string) {
        setSource(string);
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
        if (typeTable_ != null) {
            classNodes.add(typeTable_);
        }
        if (globalSymbols_ != null) {
            classNodes.add(globalSymbols_);
        }
        if (globalDeclarations_ != null) {
            classNodes.add(globalDeclarations_);
        }
        IRNode[] nodes = new IRNode[classNodes.size()];
        return ((IRNode[])classNodes.toArray(nodes));
    }

    /**
     * Tests if a Element <code>element</code> is valid
     * for the <code>XbfXcodeProgram</code>.
     *
     * @param element
     * @return boolean
     */
    public static boolean isMatch(Element element) {
        if (!URelaxer.isTargetElement(element, "XcodeProgram")) {
            return (false);
        }
        RStack target = new RStack(element);
        boolean $match$ = false;
        Element child;
        if (!XbfTypeTable.isMatchHungry(target)) {
            return (false);
        }
        $match$ = true;
        if (!XbfGlobalSymbols.isMatchHungry(target)) {
            return (false);
        }
        $match$ = true;
        if (!XbfGlobalDeclarations.isMatchHungry(target)) {
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
     * is valid for the <code>XbfXcodeProgram</code>.
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
     * is valid for the <code>XbfXcodeProgram</code>.
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
