package xcodeml.c.decompile;

import java.util.List;
import java.util.ArrayList;

import xcodeml.util.XmException;
import xcodeml.c.obj.XcNode;
import xcodeml.c.type.XcIdent;
import xcodeml.c.type.XcType;
import xcodeml.c.util.XmcWriter;

/**
 * Internal object represents coArrayRef.
 */
public class XcXmpCoArrayRefObj extends XcObj implements XcExprObj
{
    private XcExprObj _content;

    private List<XcExprObj> _dimList = new ArrayList<XcExprObj>();

    private XcIdent _tempVar;

    private XcType _type;

    private XcType _elementType;

    private boolean _needAddr = false;

    private boolean _pointerRef = true;

    /**
     * Sets whether is the temporary value used as an address reference to true.
     */
    public void setNeedAddr()
    {
        _needAddr = true;
    }

    /**
     * Gets whether is the temporary value used as an address reference to false
     * 
     * @return true if the temporary value must be used an address reference. 
     */
    public boolean isNeedAddr()
    {
        return _needAddr;
    }

    /**
     * Sets is the parent of the operator to be XcPointerRef to false. 
     */
    public void unsetPointerRef()
    {
        _pointerRef = false;
    }

    /**
     * Gets if is  the parent of the operator to be XcPointerRef.
     * 
     * @return true if the parent of the operator must be XcPointerRef.
     */
    public boolean isNeedPointerRef()
    {
        return _pointerRef;
    }

    @Override
    public void appendCode(XmcWriter w) throws XmException
    {
	w.add(_content);
	w.add(":");
	for (XcExprObj i : _dimList){
	    w.add("[").add(i).add("]");
	}

//         if(_tempVar != null) {
//             XcExprObj obj = XcXmpFactory.createCoAGetFuncCall(this);
//             w.add(obj);
//         } else {
//             w.add(_content);
// 	    w.add(":");
//             for (XcExprObj i : _dimList){
// 		w.add("[").add(i).add("]");
// 	    }
//         }
    }

    @Override
    public void addChild(XcNode child)
    {
        if(child instanceof XcExprObj)
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

        nodes[_dimList.size()] = _content;

        return nodes;
    }

    @Override
    public void setChild(int index, XcNode child)
    {
    }

    /**
     * Sets temporary variable object which coarray get function store value.
     * 
     * @param ident a temporary variable.
     */
    public void setTempVar(XcIdent ident)
    {
        _tempVar = ident;
    }

    /**
     * Gets a temporary variable object which coarray get function store value.
     * 
     * @retrun a temporary variable.
     */
    public XcIdent getTempVar()
    {
        return _tempVar;
    }

    /**
     * Sets a content of the operator.
     * 
     * @param content a content of the operator.
     */
    public void setContent(XcExprObj content)
    {
        _content = content;
    }

    /**
     * Gets a content of the operator.
     * 
     * @return a content of the operator.
     */
    public XcExprObj getContent()
    {
        return _content;
    }

    /**
     * Sets a coarray type.
     * 
     * @param type a type of coarray the operator referred.
      */
    public void setType(XcType type)
    {
        _type = type;
    }

    /**
     * Gets a coarray type.
     * 
     * @return a type of coarray the operator referred.
     */
    public XcType getType()
    {
        return _type;
    }

    /**
     * Sets an type of coarray element.
     * 
     * @param elementType an type of coarray element.
     */
    public void setElementType(XcType elementType)
    {
        _elementType = elementType;
    }

    /**
     * Gets an type of coarray element.
     * 
     * @return an type of coarray element.
     */
    public XcType getElementType()
    {
        return _elementType;
    }

    /**
     * Gets coarray dimensions.
     * 
     * @return coarray dimensions.
     */
    public List<XcExprObj> getDimList()
    {
        return _dimList;
    }

    /**
     * Sets the original content to the first term of parent and
     * the parent to the new content.
     * 
     * @param parent the parent of the object.
     * @param type an type of the new content.
     */
    public void turnOver(XcXmpCoArrayParent parent, XcType type) {
        if((parent instanceof XcExprObj) == false)
            return;
        
        parent.setCoArrayContent(_content);
        _elementType = type;
        _content = (XcExprObj)parent;
    }
}
