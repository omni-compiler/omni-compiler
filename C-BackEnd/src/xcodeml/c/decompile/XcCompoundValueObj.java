package xcodeml.c.decompile;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

import xcodeml.util.XmException;
import xcodeml.c.obj.XcNode;
import xcodeml.c.type.XcType;
import xcodeml.c.util.XmcWriter;

/**
 * Internal object represents compound literal.
 */
public class XcCompoundValueObj extends XcObj implements XcExprObj
{
    private List<XcExprObj> _exprList;

    /**
     * Creates XcCompoundValueObj
     */
    public XcCompoundValueObj()
    {
        _exprList = new ArrayList<XcExprObj>();
    }

    /**
     * Creates XcCompoundValueObj
     *
     * @param exprList expressions inside compound value.
     */
    public XcCompoundValueObj(List<XcExprObj> exprList)
    {
        _exprList = exprList;
    }

    /**
     * Creates XcCompoundValueObj
     *
     * @param exprList expressions inside compound value.
     */
    public XcCompoundValueObj(XcExprObj... exprList)
    {
        _exprList = new ArrayList<XcExprObj>();

        for(XcExprObj expr : exprList) {
            _exprList.add(expr);
        }
    }

    @Override
    public final void addChild(XcNode child)
    {
        if(child instanceof XcExprObj)
            _exprList.add((XcExprObj)child);
        else
            throw new IllegalArgumentException(child.getClass().getName());
    }

    @Override
    public final void checkChild()
    {
    }

    @Override
    public XcNode[] getChild()
    {
        if(_exprList == null)
            return null;

        return _exprList.toArray(new XcNode[_exprList.size()]);
    }

    @Override
    public final void setChild(int index, XcNode child)
    {
        if((child instanceof XcExprObj) == false)
            throw new IllegalArgumentException(child.getClass().getName());

        if(index >= 0 && index < _exprList.size())
            _exprList.set(index, (XcExprObj)child);
        else
            throw new IllegalArgumentException(index + ":" + child.getClass().getName());
    }

    @Override
    public final void appendCode(XmcWriter w) throws XmException
    {
        w.add("{");
        for(Iterator<XcExprObj> iter = _exprList.iterator();iter.hasNext();) {
            w.add(iter.next());

            if(iter.hasNext())
                w.add(",");
        }
        w.add("}");
    }

    /**
     * Internal object represents compoundValue.
     */
    public static class Ref extends XcObj implements XcExprObj
    {
        private XcType _type;

        private XcExprObj _value;

        /**
         * Creates XcCompoundValue.Ref.
         */
        public Ref()
        {
        }

        /**
         * Creates XcCompoundValue.Ref
         * 
         * @param type casted to a content of the object.
         * @param expr a content of the object.
         */
        public Ref(XcType type, XcExprObj expr)
        {
            _type = type;
            _value = expr;
        }

        /**
         * Sets a type of a value.
         * 
         * @param type a type of a value.
         */
        public void setType(XcType type)
        {
            _type = type;
        }

        /**
        * Gets a type of a value.
         *
        * @return a type of a value.
         */
        public XcType getType()
        {
            return _type;
        }

        /**
         * Sets a value of object.
         * 
         * @param value a value of the object.
         */
        public void setValue(XcExprObj value)
        {
            _value = value;
        }

        /**
         * Gets a value of the object.
         * 
         * @return a value of the object.
         */
        public XcExprObj getValue()
        {
            return _value;
        }

        @Override
        public void appendCode(XmcWriter w) throws XmException
        {
            w.add("((").add(_type).add(")").add(_value).add(")");
        }

        @Override
        public void addChild(XcNode child)
        {
            if(child instanceof XcExprObj)
                _value = (XcExprObj)child;
            else
                throw new IllegalArgumentException(child.getClass().getName());
        }

        @Override
        public void checkChild()
        {
            if(_type == null || _value == null)
                throw new IllegalArgumentException();
        }

        @Override
        public XcNode[] getChild()
        {
            return toNodeArray(_type, _value);
        }

        @Override
        public void setChild(int index, XcNode child)
        {
            if(index != 0 && index != 1)
                throw new IllegalArgumentException(child.getClass().getName());

            if(child instanceof XcType)
                _type = (XcType) child;
            else if(child instanceof XcExprObj)
                _value = (XcExprObj) child;
            else
                throw new IllegalArgumentException(child.getClass().getName());
        }
    }

    /**
     * Internal object represents compoundValueAddr.
     */
    public static class AddrRef extends XcObj implements XcExprObj
    {
        private XcType _type;

        private XcExprObj _value;

        /**
         * Creates XcCompoundValue.Addr.
         */
        public AddrRef()
        {
        }

        /**
         * Creates XcCompoundValue.Addr
         * 
         * @param type casted to a content of the object.
         * @param expr a content of the object.
         */
        public AddrRef(XcType type, XcExprObj expr)
        {
            _type = type;
            _value = expr;
        }

        /**
         * Sets a type of a value.
         * 
         * @param type a type of a value.
         */
        public void setType(XcType type)
        {
            _type = type;
        }

        /**
        * Gets a type of a value.
         *
        * @return a type of a value.
         */
        public XcType getType()
        {
            return _type;
        }

        /**
         * Sets a value of object.
         * 
         * @param value a value of the object.
         */
        public void setValue(XcExprObj value)
        {
            _value = value;
        }

        /**
         * Gets a value of the object.
         * 
         * @return a value of the object.
         */
        public XcExprObj getValue()
        {
            return _value;
        }

        @Override
        public void appendCode(XmcWriter w) throws XmException
        {
            w.add("&((").add(_type).add(")").add(_value).add(")");
        }

        @Override
        public void addChild(XcNode child)
        {
            if(child instanceof XcExprObj)
                _value = (XcExprObj)child;
            else
                throw new IllegalArgumentException(child.getClass().getName());
        }

        @Override
        public void checkChild()
        {
            if(_type == null || _value == null)
                throw new IllegalArgumentException();
        }

        @Override
        public XcNode[] getChild()
        {
            return toNodeArray(_type, _value);
        }

        @Override
        public void setChild(int index, XcNode child)
        {
            if(index != 0 && index != 1)
                throw new IllegalArgumentException(child.getClass().getName());

            if(child instanceof XcType)
                _type = (XcType) child;
            else if(child instanceof XcExprObj)
                _value = (XcExprObj) child;
            else
                throw new IllegalArgumentException(child.getClass().getName());
        }
    }
}
