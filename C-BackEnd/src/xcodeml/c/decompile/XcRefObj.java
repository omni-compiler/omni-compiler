/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
package xcodeml.c.decompile;

import xcodeml.util.XmException;
import xcodeml.c.obj.XcNode;
import xcodeml.c.type.XcIdent;
import xcodeml.c.util.XmcWriter;

/**
 * Internal object represents following elements:
 *   pointerRef, arrayRef, memberRef
 */
public abstract class XcRefObj extends XcObj implements XcExprObj, XcXmpCoArrayParent
{
    private XcRefEnum _refEnum;

    private XcExprObj _expr;

    /**
     * Creates a XcRefObj.
     * 
     * @param refEnum indicates what the object is.
     */
    private XcRefObj(XcRefEnum refEnum)
    {
        _refEnum = refEnum;
    }

    /**
     * Creates a XcRefObj.
     * 
     * @param refEnum indicates what the object is.
     * @param expr the expression the object referred to.
     */
    private XcRefObj(XcRefEnum refEnum, XcExprObj expr)
    {
        _refEnum = refEnum;
        _expr = expr;
    }

    /**
     * Gets the XcRefEnum.
     * 
     * @return the XcRefEnum.
     */
    public final XcRefEnum getRefEnum()
    {
        return _refEnum;
    }

    /**
     * Gets the expression the object referred to.
     * 
     * @return the expression the object referred to.
     */
    public final XcExprObj getExpr()
    {
        return _expr;
    }

    /**
     * Sets the expression the object referred to.
     * 
     * @param expr the expression the object referred to.
     */
    public final void setExpr(XcExprObj expr)
    {
        _expr = expr;
    }

    @Override
    public void setCoArrayContent(XcExprObj expr)
    {
        _expr = expr;
    }

    /**
     * Internal object represents varAddr.
     */
    public static final class Addr extends XcRefObj
    {
        /**
         * Creates Addr.
         */
        public Addr()
        {
            super(XcRefEnum.ADDR);
        }

        /**
         * Creates Addr.
         * 
         * @param expr a expression referred address.
         */
        public Addr(XcExprObj expr)
        {
            super(XcRefEnum.ADDR, expr);
        }

        @Override
        public void addChild(XcNode child)
        {
            if(child instanceof XcExprObj) {
                if(getExpr() == null)
                    setExpr((XcExprObj)child);
                else
                    throw new IllegalArgumentException(child.getClass().getName());
            } else
                throw new IllegalArgumentException(child.getClass().getName());
        }

        @Override
        public void checkChild()
        {
            if(getExpr() == null)
                throw new IllegalArgumentException("no expression");
        }

        @Override
        public XcNode[] getChild()
        {
            return toNodeArray(getExpr());
        }

        @Override
        public final void setChild(int index, XcNode child)
        {
            switch(index) {
            case 0:
                setExpr((XcExprObj)child);
                break;
            default:
                throw new IllegalArgumentException(index + ":" + child.getClass().getName());
            }
        }

        @Override
        public final void appendCode(XmcWriter w) throws XmException
        {
            w.add("&").addBrace(getExpr());
        }
    }

    /**
     * Internal object represents pointerRef, pointerAddr.
     */
    public static final class PointerRef extends XcRefObj
    {
        /**
         * Creates PointerRef.
         */
        public PointerRef()
        {
            super(XcRefEnum.POINTER_REF);
        }

        /**
         * Creates PointerRef.
         * 
         * @param expr a expression referred by the object.
         */
        public PointerRef(XcExprObj expr)
        {
            super(XcRefEnum.POINTER_REF, expr);
        }

        @Override
        public void addChild(XcNode child)
        {
            if(child instanceof XcExprObj) {
                if(getExpr() == null)
                    setExpr((XcExprObj)child);
                else
                    throw new IllegalArgumentException(child.getClass().getName());
            } else
                throw new IllegalArgumentException(child.getClass().getName());
        }

        @Override
        public void checkChild()
        {
            if(getExpr() == null)
                throw new IllegalArgumentException("no expression");
        }

        @Override
        public XcNode[] getChild()
        {
            return toNodeArray(getExpr());
        }

        @Override
        public final void setChild(int index, XcNode child)
        {
            switch(index) {
            case 0:
                setExpr((XcExprObj)child);
                break;
            default:
                throw new IllegalArgumentException(index + ":" + child.getClass().getName());
            }
        }

        @Override
        public final void appendCode(XmcWriter w) throws XmException
        {
            w.add("*").addBrace(getExpr());
        }
    }
    
    /**
     * Internal object represents memberRef, memberArrayRef.
     */
    public static final class MemberRef extends XcRefObj
    {
        private XcIdent _member;
        
        /**
         * Creates MemberRef.
         * 
         * @param member an identifier of a member.
         */
        public MemberRef(XcIdent member)
        {
            this(null, member);
        }
        
        /**
         * Creates MemberRef.
         * 
         * @param expr an expression refferd a member by the object.
         * @param member an identifier of a member.
         */
        public MemberRef(XcExprObj expr, XcIdent member)
        {
            super(XcRefEnum.MEMBER_REF, expr);
            _member = member;
        }

        /**
         * Gets a identifier of a member.
         * 
         * @return an identifier of a member.
         */
        public XcIdent getMember()
        {
            return _member;
        }

        @Override
        public void addChild(XcNode child)
        {
            if(child instanceof XcExprObj) {
                if(getExpr() == null)
                    setExpr((XcExprObj)child);
                else
                    throw new IllegalArgumentException();
            } else if(child instanceof XcIdent)
                _member = (XcIdent)child;
            else
                throw new IllegalArgumentException();
        }

        @Override
        public void checkChild()
        {
            if(getExpr() == null)
                throw new IllegalArgumentException("no expression");
            if(_member == null)
                throw new IllegalArgumentException("no member");
        }

        @Override
        public XcNode[] getChild()
        {
            return toNodeArray(getExpr(), _member);
        }

        @Override
        public final void setChild(int index, XcNode child)
        {
            switch(index) {
            case 0:
                setExpr((XcExprObj)child);
                break;
            case 1:
                _member = (XcIdent)child;
                break;
            default:
                throw new IllegalArgumentException(index + ":" + child.getClass().getName());
            }
        }
        
        @Override
        public final void appendCode(XmcWriter w) throws XmException
        {
            w.addBraceIfNeeded(getExpr()).add("->").add(_member.getSymbol());
        }
    }

    /**
     * Internal object represents memberAddr, memberArrayAddr.
     */
    public static final class MemberAddr extends XcRefObj
    {
        private XcIdent _member;
        
        /**
         * Creates MemberAddr.
         * 
         * @param member an identifier of a member.
         */
        public MemberAddr(XcIdent member)
        {
            this(null, member);
        }
        
        /**
         * Creates MemberAddr.
         * 
         * @param expr an expression refferd a member by the object.
         * @param member an identifier of a member.
         */
        public MemberAddr(XcExprObj expr, XcIdent member)
        {
            super(XcRefEnum.MEMBER_ADDR, expr);
            _member = member;
        }

        /**
         * Gets a identifier of a member.
         * 
         * @return an identifier of a member.
         */
        public XcIdent getMember()
        {
            return _member;
        }

        @Override
        public void addChild(XcNode child)
        {
            if(child instanceof XcExprObj) {
                if(getExpr() == null)
                    setExpr((XcExprObj)child);
                else
                    throw new IllegalArgumentException();
            } else if(child instanceof XcIdent)
                _member = (XcIdent)child;
            else
                throw new IllegalArgumentException();
        }

        @Override
        public void checkChild()
        {
            if(getExpr() == null)
                throw new IllegalArgumentException("no expression");
            if(_member == null)
                throw new IllegalArgumentException("no member");
        }

        @Override
        public XcNode[] getChild()
        {
            return toNodeArray(getExpr(), _member);
        }

        @Override
        public final void setChild(int index, XcNode child)
        {
            switch(index) {
            case 0:
                setExpr((XcExprObj)child);
                break;
            case 1:
                _member = (XcIdent)child;
                break;
            default:
                throw new IllegalArgumentException(index + ":" + child.getClass().getName());
            }
        }
        
        @Override
        public final void appendCode(XmcWriter w) throws XmException
        {
            w.add("&").addBraceIfNeeded(getExpr()).add("->").add(_member.getSymbol());
        }
    }
    
//     /**
//     * no corresponding element in XcodeML
//     */
//     public static final class ArrayElem extends XcRefObj
//     {
//         private XcExprObj _indexExpr;
//
//         public ArrayElem()
//         {
//             super(XcRefEnum.ARRAY_ELEM);
//         }
//
//         public ArrayElem(XcExprObj expr, XcExprObj indexExpr)
//         {
//             super(XcRefEnum.ARRAY_ELEM, expr);
//             _indexExpr = indexExpr;
//         }
//
//         @Override
//         public void addChild(XcNode child)
//         {
//             if(child instanceof XcExprObj) {
//                 if(getExpr() == null)
//                     setExpr((XcExprObj)child);
//                 else if(_indexExpr == null)
//                     _indexExpr = (XcExprObj)child;
//                 else
//                     throw new IllegalArgumentException();
//             } else
//                 throw new IllegalArgumentException();
//         }
//
//         @Override
//         public void checkChild()
//         {
//             if(getExpr() == null)
//                 throw new IllegalArgumentException("no expression");
//             if(_indexExpr == null)
//                 throw new IllegalArgumentException("no index expression");
//         }
//
//         @Override
//         public XcNode[] getChild()
//         {
//             return toNodeArray(getExpr(), _indexExpr);
//         }
//
//         @Override
//         public final void setChild(int index, XcNode child)
//         {
//             switch(index) {
//             case 0:
//                 setExpr((XcExprObj)child);
//                 break;
//             case 1:
//                 _indexExpr = (XcExprObj)child;
//                 break;
//             default:
//                 throw new IllegalArgumentException(index + ":" + child.getClass().getName());
//             }
//         }
//         @Override
//         public final void appendCode(XmcWriter w) throws XmException
//         {
//             w.addBrace(getExpr()).add("[").add(_indexExpr).add("]");
//         }
//     }
}
