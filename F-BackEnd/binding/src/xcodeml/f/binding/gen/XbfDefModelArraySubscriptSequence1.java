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
 * <b>XbfDefModelArraySubscriptSequence1</b> is generated from XcodeML_F.rng by Relaxer.
 * This class is derived from:
 * 
 * <!-- for programmer
 * <define name="defModelArraySubscriptSequence1">
 *   <oneOrMore>
 *     <ref name="defModelArraySubscript"/>
 *   </oneOrMore>
 * </define>
 * -->
 * <!-- for javadoc -->
 * <pre> &lt;define name="defModelArraySubscriptSequence1"&gt;
 *   &lt;oneOrMore&gt;
 *     &lt;ref name="defModelArraySubscript"/&gt;
 *   &lt;/oneOrMore&gt;
 * &lt;/define&gt;
 * </pre>
 *
 * @version XcodeML_F.rng (Fri Dec 04 19:18:15 JST 2009)
 * @author  Relaxer 1.0 (http://www.relaxer.org)
 */
public class XbfDefModelArraySubscriptSequence1 implements java.io.Serializable, Cloneable, IRVisitable, IRNode, IXbfFbasicTypeChoice {
    // List<IXbfDefModelArraySubscriptChoice>
    private java.util.List defModelArraySubscript_ = new java.util.ArrayList();
    private IRNode parentRNode_;

    /**
     * Creates a <code>XbfDefModelArraySubscriptSequence1</code>.
     *
     */
    public XbfDefModelArraySubscriptSequence1() {
    }

    /**
     * Creates a <code>XbfDefModelArraySubscriptSequence1</code>.
     *
     * @param source
     */
    public XbfDefModelArraySubscriptSequence1(XbfDefModelArraySubscriptSequence1 source) {
        setup(source);
    }

    /**
     * Creates a <code>XbfDefModelArraySubscriptSequence1</code> by the Stack <code>stack</code>
     * that contains Elements.
     * This constructor is supposed to be used internally
     * by the Relaxer system.
     *
     * @param stack
     */
    public XbfDefModelArraySubscriptSequence1(RStack stack) {
        setup(stack);
    }

    /**
     * Initializes the <code>XbfDefModelArraySubscriptSequence1</code> by the XbfDefModelArraySubscriptSequence1 <code>source</code>.
     *
     * @param source
     */
    public void setup(XbfDefModelArraySubscriptSequence1 source) {
        int size;
        this.defModelArraySubscript_.clear();
        size = source.defModelArraySubscript_.size();
        for (int i = 0;i < size;i++) {
            addDefModelArraySubscript((IXbfDefModelArraySubscriptChoice)source.getDefModelArraySubscript(i).clone());
        }
    }

    /**
     * Initializes the <code>XbfDefModelArraySubscriptSequence1</code> by the Stack <code>stack</code>
     * that contains Elements.
     * This constructor is supposed to be used internally
     * by the Relaxer system.
     *
     * @param stack
     */
    public void setup(RStack stack) {
        Element element = stack.getContextElement();
        IXcodeML_FFactory factory = XcodeML_FFactory.getFactory();
        defModelArraySubscript_.clear();
        while (true) {
            if (XbfIndexRange.isMatch(stack)) {
                addDefModelArraySubscript(factory.createXbfIndexRange(stack));
            } else if (XbfArrayIndex.isMatch(stack)) {
                addDefModelArraySubscript(factory.createXbfArrayIndex(stack));
            } else {
                break;
            }
        }
    }

    /**
     * @return Object
     */
    public Object clone() {
        IXcodeML_FFactory factory = XcodeML_FFactory.getFactory();
        return (factory.createXbfDefModelArraySubscriptSequence1(this));
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
        size = this.defModelArraySubscript_.size();
        for (int i = 0;i < size;i++) {
            IXbfDefModelArraySubscriptChoice value = (IXbfDefModelArraySubscriptChoice)this.defModelArraySubscript_.get(i);
            value.makeElement(element);
        }
    }

    /**
     * Gets the IXbfDefModelArraySubscriptChoice property <b>defModelArraySubscript</b>.
     *
     * @return IXbfDefModelArraySubscriptChoice[]
     */
    public final IXbfDefModelArraySubscriptChoice[] getDefModelArraySubscript() {
        IXbfDefModelArraySubscriptChoice[] array = new IXbfDefModelArraySubscriptChoice[defModelArraySubscript_.size()];
        return ((IXbfDefModelArraySubscriptChoice[])defModelArraySubscript_.toArray(array));
    }

    /**
     * Sets the IXbfDefModelArraySubscriptChoice property <b>defModelArraySubscript</b>.
     *
     * @param defModelArraySubscript
     */
    public final void setDefModelArraySubscript(IXbfDefModelArraySubscriptChoice[] defModelArraySubscript) {
        this.defModelArraySubscript_.clear();
        for (int i = 0;i < defModelArraySubscript.length;i++) {
            addDefModelArraySubscript(defModelArraySubscript[i]);
        }
        for (int i = 0;i < defModelArraySubscript.length;i++) {
            defModelArraySubscript[i].rSetParentRNode(this);
        }
    }

    /**
     * Sets the IXbfDefModelArraySubscriptChoice property <b>defModelArraySubscript</b>.
     *
     * @param defModelArraySubscript
     */
    public final void setDefModelArraySubscript(IXbfDefModelArraySubscriptChoice defModelArraySubscript) {
        this.defModelArraySubscript_.clear();
        addDefModelArraySubscript(defModelArraySubscript);
        if (defModelArraySubscript != null) {
            defModelArraySubscript.rSetParentRNode(this);
        }
    }

    /**
     * Adds the IXbfDefModelArraySubscriptChoice property <b>defModelArraySubscript</b>.
     *
     * @param defModelArraySubscript
     */
    public final void addDefModelArraySubscript(IXbfDefModelArraySubscriptChoice defModelArraySubscript) {
        this.defModelArraySubscript_.add(defModelArraySubscript);
        if (defModelArraySubscript != null) {
            defModelArraySubscript.rSetParentRNode(this);
        }
    }

    /**
     * Adds the IXbfDefModelArraySubscriptChoice property <b>defModelArraySubscript</b>.
     *
     * @param defModelArraySubscript
     */
    public final void addDefModelArraySubscript(IXbfDefModelArraySubscriptChoice[] defModelArraySubscript) {
        for (int i = 0;i < defModelArraySubscript.length;i++) {
            addDefModelArraySubscript(defModelArraySubscript[i]);
        }
        for (int i = 0;i < defModelArraySubscript.length;i++) {
            defModelArraySubscript[i].rSetParentRNode(this);
        }
    }

    /**
     * Gets number of the IXbfDefModelArraySubscriptChoice property <b>defModelArraySubscript</b>.
     *
     * @return int
     */
    public final int sizeDefModelArraySubscript() {
        return (defModelArraySubscript_.size());
    }

    /**
     * Gets the IXbfDefModelArraySubscriptChoice property <b>defModelArraySubscript</b> by index.
     *
     * @param index
     * @return IXbfDefModelArraySubscriptChoice
     */
    public final IXbfDefModelArraySubscriptChoice getDefModelArraySubscript(int index) {
        return ((IXbfDefModelArraySubscriptChoice)defModelArraySubscript_.get(index));
    }

    /**
     * Sets the IXbfDefModelArraySubscriptChoice property <b>defModelArraySubscript</b> by index.
     *
     * @param index
     * @param defModelArraySubscript
     */
    public final void setDefModelArraySubscript(int index, IXbfDefModelArraySubscriptChoice defModelArraySubscript) {
        this.defModelArraySubscript_.set(index, defModelArraySubscript);
        if (defModelArraySubscript != null) {
            defModelArraySubscript.rSetParentRNode(this);
        }
    }

    /**
     * Adds the IXbfDefModelArraySubscriptChoice property <b>defModelArraySubscript</b> by index.
     *
     * @param index
     * @param defModelArraySubscript
     */
    public final void addDefModelArraySubscript(int index, IXbfDefModelArraySubscriptChoice defModelArraySubscript) {
        this.defModelArraySubscript_.add(index, defModelArraySubscript);
        if (defModelArraySubscript != null) {
            defModelArraySubscript.rSetParentRNode(this);
        }
    }

    /**
     * Remove the IXbfDefModelArraySubscriptChoice property <b>defModelArraySubscript</b> by index.
     *
     * @param index
     */
    public final void removeDefModelArraySubscript(int index) {
        this.defModelArraySubscript_.remove(index);
    }

    /**
     * Remove the IXbfDefModelArraySubscriptChoice property <b>defModelArraySubscript</b> by object.
     *
     * @param defModelArraySubscript
     */
    public final void removeDefModelArraySubscript(IXbfDefModelArraySubscriptChoice defModelArraySubscript) {
        this.defModelArraySubscript_.remove(defModelArraySubscript);
    }

    /**
     * Clear the IXbfDefModelArraySubscriptChoice property <b>defModelArraySubscript</b>.
     *
     */
    public final void clearDefModelArraySubscript() {
        this.defModelArraySubscript_.clear();
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
        size = this.defModelArraySubscript_.size();
        for (int i = 0;i < size;i++) {
            IXbfDefModelArraySubscriptChoice value = (IXbfDefModelArraySubscriptChoice)this.defModelArraySubscript_.get(i);
            value.makeTextElement(buffer);
        }
    }

    /**
     * Makes an XML text representation.
     *
     * @param buffer
     * @exception IOException
     */
    public void makeTextElement(Writer buffer) throws IOException {
        int size;
        size = this.defModelArraySubscript_.size();
        for (int i = 0;i < size;i++) {
            IXbfDefModelArraySubscriptChoice value = (IXbfDefModelArraySubscriptChoice)this.defModelArraySubscript_.get(i);
            value.makeTextElement(buffer);
        }
    }

    /**
     * Makes an XML text representation.
     *
     * @param buffer
     */
    public void makeTextElement(PrintWriter buffer) {
        int size;
        size = this.defModelArraySubscript_.size();
        for (int i = 0;i < size;i++) {
            IXbfDefModelArraySubscriptChoice value = (IXbfDefModelArraySubscriptChoice)this.defModelArraySubscript_.get(i);
            value.makeTextElement(buffer);
        }
    }

    /**
     * Makes an XML text representation.
     *
     * @param buffer
     */
    public void makeTextAttribute(StringBuffer buffer) {
        int size;
        size = this.defModelArraySubscript_.size();
        for (int i = 0;i < size;i++) {
            IXbfDefModelArraySubscriptChoice value = (IXbfDefModelArraySubscriptChoice)this.defModelArraySubscript_.get(i);
            value.makeTextAttribute(buffer);
        }
    }

    /**
     * Makes an XML text representation.
     *
     * @param buffer
     * @exception IOException
     */
    public void makeTextAttribute(Writer buffer) throws IOException {
        int size;
        size = this.defModelArraySubscript_.size();
        for (int i = 0;i < size;i++) {
            IXbfDefModelArraySubscriptChoice value = (IXbfDefModelArraySubscriptChoice)this.defModelArraySubscript_.get(i);
            value.makeTextAttribute(buffer);
        }
    }

    /**
     * Makes an XML text representation.
     *
     * @param buffer
     */
    public void makeTextAttribute(PrintWriter buffer) {
        int size;
        size = this.defModelArraySubscript_.size();
        for (int i = 0;i < size;i++) {
            IXbfDefModelArraySubscriptChoice value = (IXbfDefModelArraySubscriptChoice)this.defModelArraySubscript_.get(i);
            value.makeTextAttribute(buffer);
        }
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
        classNodes.addAll(defModelArraySubscript_);
        IRNode[] nodes = new IRNode[classNodes.size()];
        return ((IRNode[])classNodes.toArray(nodes));
    }

    /**
     * Tests if elements contained in a Stack <code>stack</code>
     * is valid for the <code>XbfDefModelArraySubscriptSequence1</code>.
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
     * is valid for the <code>XbfDefModelArraySubscriptSequence1</code>.
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
        if (XbfIndexRange.isMatchHungry(target)) {
            $match$ = true;
        } else if (XbfArrayIndex.isMatchHungry(target)) {
            $match$ = true;
        } else {
            return (false);
        }
        while (true) {
            if (XbfIndexRange.isMatchHungry(target)) {
                $match$ = true;
            } else if (XbfArrayIndex.isMatchHungry(target)) {
                $match$ = true;
            } else {
                break;
            }
        }
        return ($match$);
    }
}
