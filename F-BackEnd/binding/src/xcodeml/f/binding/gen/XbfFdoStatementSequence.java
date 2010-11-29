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
 * <b>XbfFdoStatementSequence</b> is generated from XcodeML_F.rng by Relaxer.
 *
 * @version XcodeML_F.rng (Mon Nov 29 15:25:56 JST 2010)
 * @author  Relaxer 1.0 (http://www.relaxer.org)
 */
public class XbfFdoStatementSequence implements java.io.Serializable, Cloneable, IRVisitable, IRNode {
    private XbfVar var_;
    private XbfIndexRange indexRange_;
    private IRNode parentRNode_;

    /**
     * Creates a <code>XbfFdoStatementSequence</code>.
     *
     */
    public XbfFdoStatementSequence() {
    }

    /**
     * Creates a <code>XbfFdoStatementSequence</code>.
     *
     * @param source
     */
    public XbfFdoStatementSequence(XbfFdoStatementSequence source) {
        setup(source);
    }

    /**
     * Creates a <code>XbfFdoStatementSequence</code> by the Stack <code>stack</code>
     * that contains Elements.
     * This constructor is supposed to be used internally
     * by the Relaxer system.
     *
     * @param stack
     */
    public XbfFdoStatementSequence(RStack stack) {
        setup(stack);
    }

    /**
     * Initializes the <code>XbfFdoStatementSequence</code> by the XbfFdoStatementSequence <code>source</code>.
     *
     * @param source
     */
    public void setup(XbfFdoStatementSequence source) {
        int size;
        if (source.var_ != null) {
            setVar((XbfVar)source.getVar().clone());
        }
        if (source.indexRange_ != null) {
            setIndexRange((XbfIndexRange)source.getIndexRange().clone());
        }
    }

    /**
     * Initializes the <code>XbfFdoStatementSequence</code> by the Stack <code>stack</code>
     * that contains Elements.
     * This constructor is supposed to be used internally
     * by the Relaxer system.
     *
     * @param stack
     */
    public void setup(RStack stack) {
        Element element = stack.getContextElement();
        IXcodeML_FFactory factory = XcodeML_FFactory.getFactory();
        setVar(factory.createXbfVar(stack));
        setIndexRange(factory.createXbfIndexRange(stack));
    }

    /**
     * @return Object
     */
    public Object clone() {
        IXcodeML_FFactory factory = XcodeML_FFactory.getFactory();
        return (factory.createXbfFdoStatementSequence(this));
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
        this.var_.makeElement(element);
        this.indexRange_.makeElement(element);
    }

    /**
     * Gets the XbfVar property <b>Var</b>.
     *
     * @return XbfVar
     */
    public final XbfVar getVar() {
        return (var_);
    }

    /**
     * Sets the XbfVar property <b>Var</b>.
     *
     * @param var
     */
    public final void setVar(XbfVar var) {
        this.var_ = var;
        if (var != null) {
            var.rSetParentRNode(this);
        }
    }

    /**
     * Gets the XbfIndexRange property <b>indexRange</b>.
     *
     * @return XbfIndexRange
     */
    public final XbfIndexRange getIndexRange() {
        return (indexRange_);
    }

    /**
     * Sets the XbfIndexRange property <b>indexRange</b>.
     *
     * @param indexRange
     */
    public final void setIndexRange(XbfIndexRange indexRange) {
        this.indexRange_ = indexRange;
        if (indexRange != null) {
            indexRange.rSetParentRNode(this);
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
        var_.makeTextElement(buffer);
        indexRange_.makeTextElement(buffer);
    }

    /**
     * Makes an XML text representation.
     *
     * @param buffer
     * @exception IOException
     */
    public void makeTextElement(Writer buffer) throws IOException {
        int size;
        var_.makeTextElement(buffer);
        indexRange_.makeTextElement(buffer);
    }

    /**
     * Makes an XML text representation.
     *
     * @param buffer
     */
    public void makeTextElement(PrintWriter buffer) {
        int size;
        var_.makeTextElement(buffer);
        indexRange_.makeTextElement(buffer);
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
        if (var_ != null) {
            classNodes.add(var_);
        }
        if (indexRange_ != null) {
            classNodes.add(indexRange_);
        }
        IRNode[] nodes = new IRNode[classNodes.size()];
        return ((IRNode[])classNodes.toArray(nodes));
    }

    /**
     * Tests if elements contained in a Stack <code>stack</code>
     * is valid for the <code>XbfFdoStatementSequence</code>.
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
     * is valid for the <code>XbfFdoStatementSequence</code>.
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
        if (!XbfVar.isMatchHungry(target)) {
            return (false);
        }
        $match$ = true;
        if (!XbfIndexRange.isMatchHungry(target)) {
            return (false);
        }
        $match$ = true;
        return ($match$);
    }
}
