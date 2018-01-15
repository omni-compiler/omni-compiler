package xcodeml.c.decompile;

import java.util.List;
import java.util.ArrayList;
import xcodeml.util.XmException;
import xcodeml.c.obj.XcNode;
import xcodeml.c.type.XcIdent;
import xcodeml.c.type.XcType;
import xcodeml.c.util.XmcWriter;

/**
 * Internal object represents arrayRef.
 */
public class XcArrayRefObj extends XcObj implements XcExprObj
{
    private XcExprObj _arrayAddr;
    private List<XcExprObj> _dimList = new ArrayList<XcExprObj>();
    private XcType _type;
    private XcType _elementType;

    @Override
    public void appendCode(XmcWriter w) throws XmException
    {
	w.add(_arrayAddr);
	for (XcExprObj i : _dimList){
	    w.add("[").add(i).add("]");
	}
    }

    @Override
    public void addChild(XcNode child)
    {
        if (child instanceof XcExprObj)
            _dimList.add((XcExprObj)child);
        else
            throw new IllegalArgumentException(child.getClass().getName());
    }

    @Override
    public void checkChild()
    {
    }

    @Override
    public XcNode[] getChild()
    {
        XcNode[] nodes = (_dimList.toArray(new XcNode[_dimList.size() + 1]));

        nodes[_dimList.size()] = _arrayAddr;

        return nodes;
    }

    @Override
    public void setChild(int index, XcNode child)
    {
    }

    /**
     * Sets an arrayAddr of the operator.
     * 
     * @param arrayAddr an arrayAddr of the operator.
     */
    public void setArrayAddr(XcExprObj arrayAddr)
    {
        _arrayAddr = arrayAddr;
    }

    /**
     * Gets an arrayAddr of the operator.
     * 
     * @return an arrayAddr of the operator.
     */
    public XcExprObj getArrayAddr()
    {
        return _arrayAddr;
    }

    /**
     * Sets an array type.
     * 
     * @param type a type of array the operator referred.
      */
    public void setType(XcType type)
    {
        _type = type;
    }

    /**
     * Gets an array type.
     * 
     * @return a type of array the operator referred.
     */
    public XcType getType()
    {
        return _type;
    }

    /**
     * Sets a type of array element.
     * 
     * @param elementType a type of array element.
     */
    public void setElementType(XcType elementType)
    {
        _elementType = elementType;
    }

    /**
     * Gets a type of array element.
     * 
     * @return an type of coarray element.
     */
    public XcType getElementType()
    {
        return _elementType;
    }

    /**
     * Gets array dimensions.
     * 
     * @return array dimensions.
     */
    public List<XcExprObj> getDimList()
    {
        return _dimList;
    }

}
