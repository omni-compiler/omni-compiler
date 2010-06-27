/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
package xcodeml;

import java.io.IOException;
import java.io.Reader;
import java.io.Writer;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;

import javax.xml.parsers.ParserConfigurationException;

import org.xml.sax.SAXException;

import xcodeml.binding.IRNode;
import xcodeml.binding.IXmlElement;
import xcodeml.util.XmNodeIteratorFactory;

/**
 * Top class of binding objects
 */
public abstract class XmObj implements XmNode, IXmlElement, IRNode
{
    private Map<String, XmProperty> properties;

    /**
     * Creates XmObj.
     */
    public XmObj()
    {
        properties = new HashMap<String, XmProperty>();
    }

    @Override
    public final Iterator<XmNode> iterator()
    {
        XmNodeIteratorFactory factory = XmNodeIteratorFactory.getFactory();
        if (factory != null)
            return factory.createXmNodeIterator(this);
        else
            return null;
    }

    /**
     * Replaces XmObj children.
     *
     * @param from replaced by 'to'
     * @param to replaced with 'from' and to be child of the object.
     */
    public void replace(XmObj from, XmObj to)
    {
        throw new UnsupportedOperationException(
                "replace not supported by class " + getClass().toString());
    }

    /**
     * Replaces from the object's parent.
     *
     * @param to replaced with the object and to be child of the object's parent.
     */
    public final void replaceWith(XmObj to)
    {
        ((XmObj) getParent()).replace(this, to);
    }

    /**
     * Inserts before some child.
     *
     * @param at 'obj' inserted before this child.
     * @param obj inserted object.
     */
    public void insertBefore(XmObj at, XmObj obj)
    {
        throw new UnsupportedOperationException(
                "insertBefore not supported by class "
                        + getClass().toString());
    }

    /**
     * Inserts before the object.
     *
     * @param obj inserted to before the object.
     */
    public final void insertBefore(XmObj obj)
    {
        ((XmObj) getParent()).insertBefore(this, obj);
    }

    /**
     * Inserts after some child.
     *
     * @param at 'obj' inserted after this child.
     * @param obj inserted object.
     */
    public void insertAfter(XmObj at, XmObj obj)
    {
        throw new UnsupportedOperationException(
                "insertAfter not supported by class " + getClass().toString());
    }

    /**
     * Inserts after the object.
     *
     * @param obj inserted to after the object.
     */
    public final void insertAfter(XmObj obj)
    {
        ((XmObj) getParent()).insertAfter(this, obj);
    }

    /**
     * Tests the object has the property.
     *
     * @param name name of the property
     */
    public final boolean hasProperty(String name)
    {
        return properties.containsKey(name);
    }

    /**
     * Gets the property.
     *
     * @return the property of the name
     * @param name name of the property
     */
    public final XmProperty getProperty(String name)
    {
        if (properties.containsKey(name)) {
            return properties.get(name);
        } else {
            return null;
        }
    }

    /**
     * Sets the property.
     *
     * @param p the property
     */
    public final void setProperty(XmProperty p)
    {
        properties.put(p.getName(), p);
    }

    /**
     * Gets the iterator of the properties.
     *
     * @return iterator of the properties.
     */
    public final Iterator<XmProperty> getPropertyIterator()
    {
        return properties.values().iterator();
    }

    public final void setParent(XmNode parent)
    {
        rSetParentRNode((IRNode)parent);
    }

    public final XmNode getParent()
    {
        return (XmNode)rGetParentRNode();
    }

    @Override
    public final XmNode[] getChildren()
    {
        IRNode[] rnodes = rGetRNodes();

        XmNode[] nodes = new XmNode[rnodes.length];
        for (int i = 0; i < rnodes.length; i++)
            {
            nodes[i] = (XmNode) rnodes[i];
        }

        return nodes;
    }
    
    /**
     * Makes an XML text representation.
     *
     * @param buffer output
     */
    public abstract void makeTextElement(StringBuffer buffer);

    /**
     * Makes an XML text representation.
     *
     * @param buffer output
     * @exception IOException
     */
    public abstract void makeTextElement(Writer buffer) throws IOException;

    /**
     * Initializes by Reader
     *
     * @param reader
     * @exception IOException
     * @exception SAXException
     * @exception ParserConfigurationException
     */
    public abstract void setup(Reader reader) throws IOException, SAXException, ParserConfigurationException;
}
