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

import java.io.IOException;
import java.io.PrintWriter;
import java.io.Writer;
import org.w3c.dom.*;

/**
 * <b>XbfGotoStatementSequence</b> is generated from XcodeML_F.rng by Relaxer.
 *
 * @version XcodeML_F.rng (Fri Dec 04 19:18:15 JST 2009)
 * @author  Relaxer 1.0 (http://www.relaxer.org)
 */
public class XbfGotoStatementSequence implements java.io.Serializable, Cloneable, IRVisitable, IRNode {
    private XbfParams params_;
    private XbfValue value_;
    private IRNode parentRNode_;

    /**
     * Creates a <code>XbfGotoStatementSequence</code>.
     *
     */
    public XbfGotoStatementSequence() {
    }

    /**
     * Creates a <code>XbfGotoStatementSequence</code>.
     *
     * @param source
     */
    public XbfGotoStatementSequence(XbfGotoStatementSequence source) {
        setup(source);
    }

    /**
     * Creates a <code>XbfGotoStatementSequence</code> by the Stack <code>stack</code>
     * that contains Elements.
     * This constructor is supposed to be used internally
     * by the Relaxer system.
     *
     * @param stack
     */
    public XbfGotoStatementSequence(RStack stack) {
        setup(stack);
    }

    /**
     * Initializes the <code>XbfGotoStatementSequence</code> by the XbfGotoStatementSequence <code>source</code>.
     *
     * @param source
     */
    public void setup(XbfGotoStatementSequence source) {
        int size;
        if (source.params_ != null) {
            setParams((XbfParams)source.getParams().clone());
        }
        if (source.value_ != null) {
            setValue((XbfValue)source.getValue().clone());
        }
    }

    /**
     * Initializes the <code>XbfGotoStatementSequence</code> by the Stack <code>stack</code>
     * that contains Elements.
     * This constructor is supposed to be used internally
     * by the Relaxer system.
     *
     * @param stack
     */
    public void setup(RStack stack) {
        Element element = stack.getContextElement();
        IXcodeML_FFactory factory = XcodeML_FFactory.getFactory();
        setParams(factory.createXbfParams(stack));
        setValue(factory.createXbfValue(stack));
    }

    /**
     * @return Object
     */
    public Object clone() {
        IXcodeML_FFactory factory = XcodeML_FFactory.getFactory();
        return (factory.createXbfGotoStatementSequence(this));
    }

    /**
     * Creates a DOM representation of the object.
     * Result is appended to the Node <code>parent</code>.
     *
     * @param parent
     */
    public void makeElement(Node parent) {
        Document doc = parent.getOwnerDocument();
        Element element = (Element)parent;
        int size;
        this.params_.makeElement(element);
        this.value_.makeElement(element);
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
     * Gets the XbfValue property <b>value</b>.
     *
     * @return XbfValue
     */
    public final XbfValue getValue() {
        return (value_);
    }

    /**
     * Sets the XbfValue property <b>value</b>.
     *
     * @param value
     */
    public final void setValue(XbfValue value) {
        this.value_ = value;
        if (value != null) {
            value.rSetParentRNode(this);
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
        params_.makeTextElement(buffer);
        value_.makeTextElement(buffer);
    }

    /**
     * Makes an XML text representation.
     *
     * @param buffer
     * @exception IOException
     */
    public void makeTextElement(Writer buffer) throws IOException {
        int size;
        params_.makeTextElement(buffer);
        value_.makeTextElement(buffer);
    }

    /**
     * Makes an XML text representation.
     *
     * @param buffer
     */
    public void makeTextElement(PrintWriter buffer) {
        int size;
        params_.makeTextElement(buffer);
        value_.makeTextElement(buffer);
    }

    /**
     * Makes an XML text representation.
     *
     * @param buffer
     */
    public void makeTextAttribute(StringBuffer buffer) {
        int size;
    }

    /**
     * Makes an XML text representation.
     *
     * @param buffer
     * @exception IOException
     */
    public void makeTextAttribute(Writer buffer) throws IOException {
        int size;
    }

    /**
     * Makes an XML text representation.
     *
     * @param buffer
     */
    public void makeTextAttribute(PrintWriter buffer) {
        int size;
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
        if (value_ != null) {
            classNodes.add(value_);
        }
        IRNode[] nodes = new IRNode[classNodes.size()];
        return ((IRNode[])classNodes.toArray(nodes));
    }

    /**
     * Tests if elements contained in a Stack <code>stack</code>
     * is valid for the <code>XbfGotoStatementSequence</code>.
     * This mehtod is supposed to be used internally
     * by the Relaxer system.
     *
     * @param stack
     * @return boolean
     */
    public static boolean isMatch(RStack stack) {
        return (isMatchHungry(stack.makeClone()));
    }

    /**
     * Tests if elements contained in a Stack <code>stack</code>
     * is valid for the <code>XbfGotoStatementSequence</code>.
     * This method consumes the stack contents during matching operation.
     * This mehtod is supposed to be used internally
     * by the Relaxer system.
     *
     * @param stack
     * @return boolean
     */
    public static boolean isMatchHungry(RStack stack) {
        RStack target = stack;
        boolean $match$ = false;
        Element element = stack.peekElement();
        Element child;
        if (!XbfParams.isMatchHungry(target)) {
            return (false);
        }
        $match$ = true;
        if (!XbfValue.isMatchHungry(target)) {
            return (false);
        }
        $match$ = true;
        return ($match$);
    }
}
