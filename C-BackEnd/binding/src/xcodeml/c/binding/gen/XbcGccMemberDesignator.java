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
 * <b>XbcGccMemberDesignator</b> is generated from XcodeML_C.rng by Relaxer.
 * This class is derived from:
 * 
 * <!-- for programmer
 * <element java:extends="xcodeml.c.obj.XmcObj" name="gccMemberDesignator">
 *   <ref name="ref"/>
 *   <optional>
 *     <attribute name="member">
 *       <data type="string"/>
 *     </attribute>
 *   </optional>
 *   <optional>
 *     <ref name="gccMemberDesignator"/>
 *   </optional>
 *   <optional>
 *     <ref name="expressions"/>
 *   </optional>
 * </element>
 * -->
 * <!-- for javadoc -->
 * <pre> &lt;element java:extends="xcodeml.c.obj.XmcObj" name="gccMemberDesignator"&gt;
 *   &lt;ref name="ref"/&gt;
 *   &lt;optional&gt;
 *     &lt;attribute name="member"&gt;
 *       &lt;data type="string"/&gt;
 *     &lt;/attribute&gt;
 *   &lt;/optional&gt;
 *   &lt;optional&gt;
 *     &lt;ref name="gccMemberDesignator"/&gt;
 *   &lt;/optional&gt;
 *   &lt;optional&gt;
 *     &lt;ref name="expressions"/&gt;
 *   &lt;/optional&gt;
 * &lt;/element&gt;
 * </pre>
 *
 * @version XcodeML_C.rng (Thu Sep 24 16:30:19 JST 2009)
 * @author  Relaxer 1.0 (http://www.relaxer.org)
 */
public class XbcGccMemberDesignator extends xcodeml.c.obj.XmcObj implements java.io.Serializable, Cloneable, IRVisitable, IRNode, IXbcBuiltinOpChoice {
    private String member_;
    private String ref_;
    private XbcGccMemberDesignator gccMemberDesignator_;
    private IXbcExpressionsChoice expressions_;
    private IRNode parentRNode_;

    /**
     * Creates a <code>XbcGccMemberDesignator</code>.
     *
     */
    public XbcGccMemberDesignator() {
        ref_ = "";
    }

    /**
     * Creates a <code>XbcGccMemberDesignator</code>.
     *
     * @param source
     */
    public XbcGccMemberDesignator(XbcGccMemberDesignator source) {
        setup(source);
    }

    /**
     * Creates a <code>XbcGccMemberDesignator</code> by the Stack <code>stack</code>
     * that contains Elements.
     * This constructor is supposed to be used internally
     * by the Relaxer system.
     *
     * @param stack
     */
    public XbcGccMemberDesignator(RStack stack) {
        setup(stack);
    }

    /**
     * Creates a <code>XbcGccMemberDesignator</code> by the Document <code>doc</code>.
     *
     * @param doc
     */
    public XbcGccMemberDesignator(Document doc) {
        setup(doc.getDocumentElement());
    }

    /**
     * Creates a <code>XbcGccMemberDesignator</code> by the Element <code>element</code>.
     *
     * @param element
     */
    public XbcGccMemberDesignator(Element element) {
        setup(element);
    }

    /**
     * Creates a <code>XbcGccMemberDesignator</code> by the File <code>file</code>.
     *
     * @param file
     * @exception IOException
     * @exception SAXException
     * @exception ParserConfigurationException
     */
    public XbcGccMemberDesignator(File file) throws IOException, SAXException, ParserConfigurationException {
        setup(file);
    }

    /**
     * Creates a <code>XbcGccMemberDesignator</code>
     * by the String representation of URI <code>uri</code>.
     *
     * @param uri
     * @exception IOException
     * @exception SAXException
     * @exception ParserConfigurationException
     */
    public XbcGccMemberDesignator(String uri) throws IOException, SAXException, ParserConfigurationException {
        setup(uri);
    }

    /**
     * Creates a <code>XbcGccMemberDesignator</code> by the URL <code>url</code>.
     *
     * @param url
     * @exception IOException
     * @exception SAXException
     * @exception ParserConfigurationException
     */
    public XbcGccMemberDesignator(URL url) throws IOException, SAXException, ParserConfigurationException {
        setup(url);
    }

    /**
     * Creates a <code>XbcGccMemberDesignator</code> by the InputStream <code>in</code>.
     *
     * @param in
     * @exception IOException
     * @exception SAXException
     * @exception ParserConfigurationException
     */
    public XbcGccMemberDesignator(InputStream in) throws IOException, SAXException, ParserConfigurationException {
        setup(in);
    }

    /**
     * Creates a <code>XbcGccMemberDesignator</code> by the InputSource <code>is</code>.
     *
     * @param is
     * @exception IOException
     * @exception SAXException
     * @exception ParserConfigurationException
     */
    public XbcGccMemberDesignator(InputSource is) throws IOException, SAXException, ParserConfigurationException {
        setup(is);
    }

    /**
     * Creates a <code>XbcGccMemberDesignator</code> by the Reader <code>reader</code>.
     *
     * @param reader
     * @exception IOException
     * @exception SAXException
     * @exception ParserConfigurationException
     */
    public XbcGccMemberDesignator(Reader reader) throws IOException, SAXException, ParserConfigurationException {
        setup(reader);
    }

    /**
     * Initializes the <code>XbcGccMemberDesignator</code> by the XbcGccMemberDesignator <code>source</code>.
     *
     * @param source
     */
    public void setup(XbcGccMemberDesignator source) {
        int size;
        setMember(source.getMember());
        setRef(source.getRef());
        if (source.gccMemberDesignator_ != null) {
            setGccMemberDesignator((XbcGccMemberDesignator)source.getGccMemberDesignator().clone());
        }
        if (source.expressions_ != null) {
            setExpressions((IXbcExpressionsChoice)source.getExpressions().clone());
        }
    }

    /**
     * Initializes the <code>XbcGccMemberDesignator</code> by the Document <code>doc</code>.
     *
     * @param doc
     */
    public void setup(Document doc) {
        setup(doc.getDocumentElement());
    }

    /**
     * Initializes the <code>XbcGccMemberDesignator</code> by the Element <code>element</code>.
     *
     * @param element
     */
    public void setup(Element element) {
        init(element);
    }

    /**
     * Initializes the <code>XbcGccMemberDesignator</code> by the Stack <code>stack</code>
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
        ref_ = URelaxer.getAttributePropertyAsString(element, "ref");
        if (XbcGccMemberDesignator.isMatch(stack)) {
            setGccMemberDesignator(factory.createXbcGccMemberDesignator(stack));
        }
        if (XbcSubArrayRef.isMatch(stack)) {
            setExpressions(factory.createXbcSubArrayRef(stack));
        } else if (XbcBuiltinOp.isMatch(stack)) {
            setExpressions(factory.createXbcBuiltinOp(stack));
        } else if (XbcFunctionCall.isMatch(stack)) {
            setExpressions(factory.createXbcFunctionCall(stack));
        } else if (XbcGccCompoundExpr.isMatch(stack)) {
            setExpressions(factory.createXbcGccCompoundExpr(stack));
        } else if (XbcCoArrayRef.isMatch(stack)) {
            setExpressions(factory.createXbcCoArrayRef(stack));
        } else if (XbcStringConstant.isMatch(stack)) {
            setExpressions(factory.createXbcStringConstant(stack));
        } else if (XbcVar.isMatch(stack)) {
            setExpressions(factory.createXbcVar(stack));
        } else if (XbcArrayRef.isMatch(stack)) {
            setExpressions(factory.createXbcArrayRef(stack));
        } else if (XbcArrayAddr.isMatch(stack)) {
            setExpressions(factory.createXbcArrayAddr(stack));
        } else if (XbcCastExpr.isMatch(stack)) {
            setExpressions(factory.createXbcCastExpr(stack));
        } else if (XbcCompoundValueExpr.isMatch(stack)) {
            setExpressions(factory.createXbcCompoundValueExpr(stack));
        } else if (XbcCompoundValueAddrExpr.isMatch(stack)) {
            setExpressions(factory.createXbcCompoundValueAddrExpr(stack));
        } else if (XbcVarAddr.isMatch(stack)) {
            setExpressions(factory.createXbcVarAddr(stack));
        } else if (XbcCommaExpr.isMatch(stack)) {
            setExpressions(factory.createXbcCommaExpr(stack));
        } else if (XbcIntConstant.isMatch(stack)) {
            setExpressions(factory.createXbcIntConstant(stack));
        } else if (XbcFloatConstant.isMatch(stack)) {
            setExpressions(factory.createXbcFloatConstant(stack));
        } else if (XbcLonglongConstant.isMatch(stack)) {
            setExpressions(factory.createXbcLonglongConstant(stack));
        } else if (XbcMoeConstant.isMatch(stack)) {
            setExpressions(factory.createXbcMoeConstant(stack));
        } else if (XbcFuncAddr.isMatch(stack)) {
            setExpressions(factory.createXbcFuncAddr(stack));
        } else if (XbcSizeOfExpr.isMatch(stack)) {
            setExpressions(factory.createXbcSizeOfExpr(stack));
        } else if (XbcGccAlignOfExpr.isMatch(stack)) {
            setExpressions(factory.createXbcGccAlignOfExpr(stack));
        } else if (XbcGccLabelAddr.isMatch(stack)) {
            setExpressions(factory.createXbcGccLabelAddr(stack));
        } else if (XbcCoArrayAssignExpr.isMatch(stack)) {
            setExpressions(factory.createXbcCoArrayAssignExpr(stack));
        } else if (XbcMemberAddr.isMatch(stack)) {
            setExpressions(factory.createXbcMemberAddr(stack));
        } else if (XbcMemberRef.isMatch(stack)) {
            setExpressions(factory.createXbcMemberRef(stack));
        } else if (XbcMemberArrayRef.isMatch(stack)) {
            setExpressions(factory.createXbcMemberArrayRef(stack));
        } else if (XbcMemberArrayAddr.isMatch(stack)) {
            setExpressions(factory.createXbcMemberArrayAddr(stack));
        } else if (XbcBitXorExpr.isMatch(stack)) {
            setExpressions(factory.createXbcBitXorExpr(stack));
        } else if (XbcPointerRef.isMatch(stack)) {
            setExpressions(factory.createXbcPointerRef(stack));
        } else if (XbcAssignExpr.isMatch(stack)) {
            setExpressions(factory.createXbcAssignExpr(stack));
        } else if (XbcPlusExpr.isMatch(stack)) {
            setExpressions(factory.createXbcPlusExpr(stack));
        } else if (XbcMinusExpr.isMatch(stack)) {
            setExpressions(factory.createXbcMinusExpr(stack));
        } else if (XbcMulExpr.isMatch(stack)) {
            setExpressions(factory.createXbcMulExpr(stack));
        } else if (XbcDivExpr.isMatch(stack)) {
            setExpressions(factory.createXbcDivExpr(stack));
        } else if (XbcModExpr.isMatch(stack)) {
            setExpressions(factory.createXbcModExpr(stack));
        } else if (XbcLshiftExpr.isMatch(stack)) {
            setExpressions(factory.createXbcLshiftExpr(stack));
        } else if (XbcRshiftExpr.isMatch(stack)) {
            setExpressions(factory.createXbcRshiftExpr(stack));
        } else if (XbcBitAndExpr.isMatch(stack)) {
            setExpressions(factory.createXbcBitAndExpr(stack));
        } else if (XbcBitOrExpr.isMatch(stack)) {
            setExpressions(factory.createXbcBitOrExpr(stack));
        } else if (XbcAsgPlusExpr.isMatch(stack)) {
            setExpressions(factory.createXbcAsgPlusExpr(stack));
        } else if (XbcAsgMinusExpr.isMatch(stack)) {
            setExpressions(factory.createXbcAsgMinusExpr(stack));
        } else if (XbcAsgMulExpr.isMatch(stack)) {
            setExpressions(factory.createXbcAsgMulExpr(stack));
        } else if (XbcAsgDivExpr.isMatch(stack)) {
            setExpressions(factory.createXbcAsgDivExpr(stack));
        } else if (XbcAsgModExpr.isMatch(stack)) {
            setExpressions(factory.createXbcAsgModExpr(stack));
        } else if (XbcAsgLshiftExpr.isMatch(stack)) {
            setExpressions(factory.createXbcAsgLshiftExpr(stack));
        } else if (XbcAsgRshiftExpr.isMatch(stack)) {
            setExpressions(factory.createXbcAsgRshiftExpr(stack));
        } else if (XbcAsgBitAndExpr.isMatch(stack)) {
            setExpressions(factory.createXbcAsgBitAndExpr(stack));
        } else if (XbcAsgBitOrExpr.isMatch(stack)) {
            setExpressions(factory.createXbcAsgBitOrExpr(stack));
        } else if (XbcAsgBitXorExpr.isMatch(stack)) {
            setExpressions(factory.createXbcAsgBitXorExpr(stack));
        } else if (XbcLogEQExpr.isMatch(stack)) {
            setExpressions(factory.createXbcLogEQExpr(stack));
        } else if (XbcLogNEQExpr.isMatch(stack)) {
            setExpressions(factory.createXbcLogNEQExpr(stack));
        } else if (XbcLogGEExpr.isMatch(stack)) {
            setExpressions(factory.createXbcLogGEExpr(stack));
        } else if (XbcLogGTExpr.isMatch(stack)) {
            setExpressions(factory.createXbcLogGTExpr(stack));
        } else if (XbcLogLEExpr.isMatch(stack)) {
            setExpressions(factory.createXbcLogLEExpr(stack));
        } else if (XbcLogLTExpr.isMatch(stack)) {
            setExpressions(factory.createXbcLogLTExpr(stack));
        } else if (XbcLogAndExpr.isMatch(stack)) {
            setExpressions(factory.createXbcLogAndExpr(stack));
        } else if (XbcLogOrExpr.isMatch(stack)) {
            setExpressions(factory.createXbcLogOrExpr(stack));
        } else if (XbcUnaryMinusExpr.isMatch(stack)) {
            setExpressions(factory.createXbcUnaryMinusExpr(stack));
        } else if (XbcBitNotExpr.isMatch(stack)) {
            setExpressions(factory.createXbcBitNotExpr(stack));
        } else if (XbcLogNotExpr.isMatch(stack)) {
            setExpressions(factory.createXbcLogNotExpr(stack));
        } else if (XbcPostIncrExpr.isMatch(stack)) {
            setExpressions(factory.createXbcPostIncrExpr(stack));
        } else if (XbcPostDecrExpr.isMatch(stack)) {
            setExpressions(factory.createXbcPostDecrExpr(stack));
        } else if (XbcPreIncrExpr.isMatch(stack)) {
            setExpressions(factory.createXbcPreIncrExpr(stack));
        } else if (XbcPreDecrExpr.isMatch(stack)) {
            setExpressions(factory.createXbcPreDecrExpr(stack));
        } else if (XbcCondExpr.isMatch(stack)) {
            setExpressions(factory.createXbcCondExpr(stack));
        } else {
        }
    }

    /**
     * @return Object
     */
    public Object clone() {
        IXcodeML_CFactory factory = XcodeML_CFactory.getFactory();
        return (factory.createXbcGccMemberDesignator(this));
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
        Element element = doc.createElement("gccMemberDesignator");
        int size;
        if (this.member_ != null) {
            URelaxer.setAttributePropertyByString(element, "member", this.member_);
        }
        if (this.ref_ != null) {
            URelaxer.setAttributePropertyByString(element, "ref", this.ref_);
        }
        if (this.gccMemberDesignator_ != null) {
            this.gccMemberDesignator_.makeElement(element);
        }
        if (this.expressions_ != null) {
            this.expressions_.makeElement(element);
        }
        parent.appendChild(element);
    }

    /**
     * Initializes the <code>XbcGccMemberDesignator</code> by the File <code>file</code>.
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
     * Initializes the <code>XbcGccMemberDesignator</code>
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
     * Initializes the <code>XbcGccMemberDesignator</code> by the URL <code>url</code>.
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
     * Initializes the <code>XbcGccMemberDesignator</code> by the InputStream <code>in</code>.
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
     * Initializes the <code>XbcGccMemberDesignator</code> by the InputSource <code>is</code>.
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
     * Initializes the <code>XbcGccMemberDesignator</code> by the Reader <code>reader</code>.
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
     * Gets the XbcGccMemberDesignator property <b>gccMemberDesignator</b>.
     *
     * @return XbcGccMemberDesignator
     */
    public final XbcGccMemberDesignator getGccMemberDesignator() {
        return (gccMemberDesignator_);
    }

    /**
     * Sets the XbcGccMemberDesignator property <b>gccMemberDesignator</b>.
     *
     * @param gccMemberDesignator
     */
    public final void setGccMemberDesignator(XbcGccMemberDesignator gccMemberDesignator) {
        this.gccMemberDesignator_ = gccMemberDesignator;
        if (gccMemberDesignator != null) {
            gccMemberDesignator.rSetParentRNode(this);
        }
    }

    /**
     * Gets the IXbcExpressionsChoice property <b>expressions</b>.
     *
     * @return IXbcExpressionsChoice
     */
    public final IXbcExpressionsChoice getExpressions() {
        return (expressions_);
    }

    /**
     * Sets the IXbcExpressionsChoice property <b>expressions</b>.
     *
     * @param expressions
     */
    public final void setExpressions(IXbcExpressionsChoice expressions) {
        this.expressions_ = expressions;
        if (expressions != null) {
            expressions.rSetParentRNode(this);
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
        buffer.append("<gccMemberDesignator");
        if (member_ != null) {
            buffer.append(" member=\"");
            buffer.append(URelaxer.escapeAttrQuot(URelaxer.getString(getMember())));
            buffer.append("\"");
        }
        if (ref_ != null) {
            buffer.append(" ref=\"");
            buffer.append(URelaxer.escapeAttrQuot(URelaxer.getString(getRef())));
            buffer.append("\"");
        }
        if (expressions_ != null) {
            expressions_.makeTextAttribute(buffer);
        }
        buffer.append(">");
        if (gccMemberDesignator_ != null) {
            gccMemberDesignator_.makeTextElement(buffer);
        }
        if (expressions_ != null) {
            expressions_.makeTextElement(buffer);
        }
        buffer.append("</gccMemberDesignator>");
    }

    /**
     * Makes an XML text representation.
     *
     * @param buffer
     * @exception IOException
     */
    public void makeTextElement(Writer buffer) throws IOException {
        int size;
        buffer.write("<gccMemberDesignator");
        if (member_ != null) {
            buffer.write(" member=\"");
            buffer.write(URelaxer.escapeAttrQuot(URelaxer.getString(getMember())));
            buffer.write("\"");
        }
        if (ref_ != null) {
            buffer.write(" ref=\"");
            buffer.write(URelaxer.escapeAttrQuot(URelaxer.getString(getRef())));
            buffer.write("\"");
        }
        if (expressions_ != null) {
            expressions_.makeTextAttribute(buffer);
        }
        buffer.write(">");
        if (gccMemberDesignator_ != null) {
            gccMemberDesignator_.makeTextElement(buffer);
        }
        if (expressions_ != null) {
            expressions_.makeTextElement(buffer);
        }
        buffer.write("</gccMemberDesignator>");
    }

    /**
     * Makes an XML text representation.
     *
     * @param buffer
     */
    public void makeTextElement(PrintWriter buffer) {
        int size;
        buffer.print("<gccMemberDesignator");
        if (member_ != null) {
            buffer.print(" member=\"");
            buffer.print(URelaxer.escapeAttrQuot(URelaxer.getString(getMember())));
            buffer.print("\"");
        }
        if (ref_ != null) {
            buffer.print(" ref=\"");
            buffer.print(URelaxer.escapeAttrQuot(URelaxer.getString(getRef())));
            buffer.print("\"");
        }
        if (expressions_ != null) {
            expressions_.makeTextAttribute(buffer);
        }
        buffer.print(">");
        if (gccMemberDesignator_ != null) {
            gccMemberDesignator_.makeTextElement(buffer);
        }
        if (expressions_ != null) {
            expressions_.makeTextElement(buffer);
        }
        buffer.print("</gccMemberDesignator>");
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
    public String getRefAsString() {
        return (URelaxer.getString(getRef()));
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
    public void setRefByString(String string) {
        setRef(string);
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
        if (gccMemberDesignator_ != null) {
            classNodes.add(gccMemberDesignator_);
        }
        if (expressions_ != null) {
            classNodes.add(expressions_);
        }
        IRNode[] nodes = new IRNode[classNodes.size()];
        return ((IRNode[])classNodes.toArray(nodes));
    }

    /**
     * Tests if a Element <code>element</code> is valid
     * for the <code>XbcGccMemberDesignator</code>.
     *
     * @param element
     * @return boolean
     */
    public static boolean isMatch(Element element) {
        if (!URelaxer.isTargetElement(element, "gccMemberDesignator")) {
            return (false);
        }
        RStack target = new RStack(element);
        boolean $match$ = false;
        Element child;
        if (!URelaxer.hasAttributeHungry(target, "ref")) {
            return (false);
        }
        $match$ = true;
        if (XbcGccMemberDesignator.isMatchHungry(target)) {
        }
        if (XbcSubArrayRef.isMatchHungry(target)) {
            $match$ = true;
        } else if (XbcBuiltinOp.isMatchHungry(target)) {
            $match$ = true;
        } else if (XbcFunctionCall.isMatchHungry(target)) {
            $match$ = true;
        } else if (XbcGccCompoundExpr.isMatchHungry(target)) {
            $match$ = true;
        } else if (XbcCoArrayRef.isMatchHungry(target)) {
            $match$ = true;
        } else if (XbcStringConstant.isMatchHungry(target)) {
            $match$ = true;
        } else if (XbcVar.isMatchHungry(target)) {
            $match$ = true;
        } else if (XbcArrayRef.isMatchHungry(target)) {
            $match$ = true;
        } else if (XbcArrayAddr.isMatchHungry(target)) {
            $match$ = true;
        } else if (XbcCastExpr.isMatchHungry(target)) {
            $match$ = true;
        } else if (XbcCompoundValueExpr.isMatchHungry(target)) {
            $match$ = true;
        } else if (XbcCompoundValueAddrExpr.isMatchHungry(target)) {
            $match$ = true;
        } else if (XbcVarAddr.isMatchHungry(target)) {
            $match$ = true;
        } else if (XbcCommaExpr.isMatchHungry(target)) {
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
        } else if (XbcCoArrayAssignExpr.isMatchHungry(target)) {
            $match$ = true;
        } else if (XbcMemberAddr.isMatchHungry(target)) {
            $match$ = true;
        } else if (XbcMemberRef.isMatchHungry(target)) {
            $match$ = true;
        } else if (XbcMemberArrayRef.isMatchHungry(target)) {
            $match$ = true;
        } else if (XbcMemberArrayAddr.isMatchHungry(target)) {
            $match$ = true;
        } else if (XbcBitXorExpr.isMatchHungry(target)) {
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
        } else if (XbcCondExpr.isMatchHungry(target)) {
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
     * is valid for the <code>XbcGccMemberDesignator</code>.
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
     * is valid for the <code>XbcGccMemberDesignator</code>.
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
