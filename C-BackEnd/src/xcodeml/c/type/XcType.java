package xcodeml.c.type;

import java.util.Stack;
import java.util.List;
import java.util.ArrayList;

import xcodeml.util.XmException;
import xcodeml.c.decompile.XcObj;
import xcodeml.c.util.XmcWriter;

/**
 * top class which represents type
 */
public abstract class XcType extends XcObj implements Cloneable, XcGccAttributable
{
    /* class id */
    private XcTypeEnum      _typeEnum;

    /* type name */
    private String          _typeId;

    /* type qualifier: const, volatile, restrict */
    private boolean         _isConst, _isVolatile, _isRestrict;

    /* reference type */
    private XcType          _refType;

    /* pre-resolved reference type name */
    private String          _tempRefTypeId;

    private boolean         _resolved;

    /* GCC attribute */
    private XcGccAttributeList  _gccAttrs;

    protected XcType()
    {
    }

    protected XcType(XcTypeEnum typeEnum, String typeId)
    {
        _typeEnum = typeEnum;
        _typeId = typeId;
    }

    protected abstract void resolveOverride(XcIdentTableStack itStack) throws XmException;

    public XcType copy()
    {
        try {
            return (XcType)clone();
        } catch(CloneNotSupportedException e) {
            throw new RuntimeException(e);
        }
    }

    public final XcTypeEnum getTypeEnum()
    {
        return _typeEnum;
    }

    public final String getTypeId()
    {
        return _typeId;
    }

    public final boolean isConst()
    {
        return _isConst;
    }

    public final void setIsConst(boolean enable)
    {
        _isConst = enable;
    }

    public final boolean isVolatile()
    {
        return _isVolatile;
    }

    public final void setIsVolatile(boolean enable)
    {
        _isVolatile = enable;
    }

    public final boolean isRestrict()
    {
        return _isRestrict;
    }

    public final void setIsRestrict(boolean enable)
    {
        _isRestrict = enable;
    }

    public final void copyTypeQualifiersFrom(XcType type)
    {
        _isConst = type._isConst;
        _isRestrict = type._isRestrict;
        _isVolatile = type._isVolatile;
    }

    public final void resetTypeQualifiers()
    {
        _isConst = false;
        _isRestrict = false;
        _isVolatile = false;
    }

    public final XcType getRefType()
    {
        return _refType;
    }

    public final void setRefType(XcType type)
    {
        _refType = type;
    }

    public final String getTempRefTypeId()
    {
        return _tempRefTypeId;
    }

    public final void setTempRefTypeId(String typeId)
    {
        _tempRefTypeId = typeId;
    }

    public final boolean canBeBottomType()
    {
        switch(getTypeEnum())
        {
        case BASETYPE:
        case BUILTIN:
        case STRUCT:
        case UNION:
        case ENUM:
            return true;
        }

        return false;
    }

    public final void appendTypeNameCode(XmcWriter w) throws XmException
    {
        switch(getTypeEnum())
        {
        case BASETYPE:
            w.addSpc(((XcBaseType)this).getBaseTypeEnum().getCCode());
            break;
        case BASICTYPE:
            break;
        case STRUCT:
        case UNION:
        case ENUM:
            {
                XcTaggedType tt = (XcTaggedType)this;
                w.addSpc(tt.getTypeNameHeader());
                if(tt.getTagName() != null) {
                    w.addSpc(tt.getTagName());
                }
            }
            break;
        default: // POINTER, ARRAY, FUNC
            throw new XmException("pointer/array/function type has no name");
        }
    }

    @Override
    public final void appendCode(XmcWriter w) throws XmException
    {
        appendDeclCode(w, null, true, false);
    }

    public final void appendCode(XmcWriter w, String symbol) throws XmException
    {
        appendDeclCode(w, symbol, true, false);
    }

    public final void appendBodyCode(XmcWriter w) throws XmException
    {
        appendBodyCode(w, false);
    }

    public final void appendBodyCode(XmcWriter w, boolean isIncomplete) throws XmException
    {
        switch(getTypeEnum()) {
        case STRUCT:
        case UNION:
        case ENUM:
            XcTaggedType tt = (XcTaggedType)this;
            w.addSpc(tt.getTypeNameHeader());

            appendGccAtrribute(w);

            if(tt.getTagName() != null) {
                w.addSpc(tt.getTagName());
            }

            if(this instanceof XcCompositeType) {
                XcMemberList memberList = (((XcCompositeType)tt).getMemberList());
                if(memberList != null && isIncomplete == false) {
                    memberList.appendCode(w);
                }
            } else {
                ((XcEnumType)this).getEnumeratorList().appendCode(w);
            }

            break;
        default:
            throw new IllegalArgumentException();
        }
    }

    public final void appendTypeDefCode(XmcWriter w, String symbol) throws XmException
    {
        if(this instanceof XcTaggedType) {
            appendDeclCode(w, null, true, false);
            w.addSpc(symbol);
        } else {
            appendDeclCode(w, symbol, true, false);
        }
    }

    public final void appendSizeOfCode(XmcWriter w) throws XmException
    {
        appendDeclCode(w, null, true, false, _gccAttrs);
    }

    public final void appendGccAtrribute(XmcWriter w) throws XmException
    {
        if (_gccAttrs != null)
            _gccAttrs.appendCode(w);
    }

    public final void appendDeclCode(XmcWriter w,
                                     String symbol,
                                     boolean withBaseType,
                                     boolean isPreDecl
                                     ) throws XmException
    {
        appendDeclCode(w, symbol, withBaseType, isPreDecl, null);
    }

  public final void appendDeclCode(XmcWriter w,
                                   String symbol,
                                   boolean withBaseType,
                                   boolean isPreDecl,
                                   XcGccAttributeList attr
                                   ) throws XmException
  {
    appendDeclCode(w, symbol, withBaseType, isPreDecl, attr, false);
  }
  
    public final void appendDeclCode(XmcWriter w,
                                     String symbol,
                                     boolean withBaseType,
                                     boolean isPreDecl,
                                     XcGccAttributeList attr,
                                     boolean isFromFunc
                                     ) throws XmException
    {
        Stack<XcType> decls = new Stack<XcType>();
        XcType t = this;

        while(t.getTypeEnum() == XcTypeEnum.COARRAY) {
            t = ((XcXmpCoArrayType)t).getRefType();
        }

        for(; t != null; t = t.getRefType()) {
            if(t instanceof XcExtType)
                decls.push(t);
            else
                break;
        }

        if(t.canBeBottomType() == false)
            throw new XmException("invalid type");

        if(getTypeEnum() == XcTypeEnum.FUNC) {
            XcFuncType ft = (XcFuncType)this;
            if(ft.isPrototype() == false) {
                if(ft.isStatic())
                    w.addSpc("static");
                if(ft.isInline())
                    w.addSpc("inline");
            }
        }

        /*
         * if type (or real type of basic type) has attributes
         * and this type referenced by pointer,
         * these type's attributes must be guard by typeof()
         * to avoid these token over by pointer type.
         *
         * ex)
         * if p declared as follows, then type of p, not *p,
         * has attribute 'aligned(8)'.
         *
         *   int __attribute__((aligned(8)) * p;
         *
         * On the other hands, if q declared as follows,
         * type of q has not attribute,
         * but type of *q has attribute 'aligned(8)'.
         *
         *   typeof(int __attribute__((aligned(8))) * q;
         *
         */
        boolean isAttrGuarded = false;
        if((symbol != null) && (_needAttrGuard(t, decls) == true)) {
            w.addSpc("typeof(");
            isAttrGuarded = true;
        }

        // XXX add w.addSpc(attr);
        w.addSpc(attr);
        t.appendTypeQualCode(w);
        t.appendTypeNameCode(w);

        if(decls.isEmpty() == false)
            w.spc();
        else if(symbol == null)
            return;

        // XXX change attr to null: _appendCode(w, decls, symbol, isPreDecl, isAttrGuarded, attr, null);
        _appendCode(w, decls, symbol, isPreDecl, isAttrGuarded, null, null, isFromFunc);
     }

     private static void _appendBasicType(XmcWriter w, Stack<XcType> decls, List<XcGccAttributeList> attrsList) throws XmException
     {
         if(decls.isEmpty() == false) {
             XcType bt = decls.pop();

             while(bt instanceof XcBasicType) {

                 if(attrsList == null) {
                     bt.appendGccAtrribute(w);
                 } else if(bt.getGccAttribute() != null) {
                     attrsList.add(bt.getGccAttribute());
                 }

                 bt.appendTypeQualCode(w);

                 if(decls.isEmpty())
                     break;

                 bt = decls.pop();
             }

             if((bt instanceof XcBasicType) == false)
                 decls.push(bt);
         }
     }

     private static XcType _peekRealType(Stack<XcType> decls)
     {
         XcType returnType = null;

         if(decls.isEmpty())
             return null;

         XcType type = decls.pop();

         if(type.getTypeEnum() == XcTypeEnum.BASICTYPE) {
             returnType = _peekRealType(decls);
         } else {
             returnType = type;
         }

         decls.push(type);

         return returnType;
     }

     private static boolean _recNeedAttrGuard(Stack<XcType> decls, boolean hasAttr)
     {
         if(decls.isEmpty())
             return false;

         boolean needAttrGuard = false;
         XcType type = decls.pop();
         XcTypeEnum te = type.getTypeEnum();

         switch (te) {
         case POINTER:
             if(hasAttr == true)
                 needAttrGuard = true;
            break;

        case ARRAY:
            if(decls.isEmpty() == true) {
                needAttrGuard = false;
            } else {
                if(hasAttr == true)
                    needAttrGuard = true;
            }
            break;

        default:
            if(hasAttr == false) {
                if(type.getGccAttribute() != null &&
                   type.getGccAttribute().isEmpty() == false)
                    hasAttr = true;
            }
            needAttrGuard = _recNeedAttrGuard(decls, hasAttr);
            break;
        }

        decls.push(type);

        return needAttrGuard;
    }

    private static boolean _needAttrGuard(XcType type, Stack<XcType> decls)
    {
        XcType rt = type.getRealType();

        if(rt.getTypeEnum() != XcTypeEnum.BASETYPE)
            return false;

        boolean needAttrGuard = type.getGccAttribute() != null &&
            type.getGccAttribute().isEmpty() == false;

        return _recNeedAttrGuard(decls, needAttrGuard);
    }


  private static final void _appendCode(XmcWriter w,
                                        Stack<XcType> decls,
                                        String symbol,
                                        boolean isPreDecl,
                                        boolean isAttrGuarded,
                                        XcGccAttributeList attr,
                                        List<XcGccAttributeList> attrOfArrayTypes
                                        ) throws XmException
  {
    _appendCode(w, decls, symbol, isPreDecl, isAttrGuarded, attr, attrOfArrayTypes, false);
  }
  
    private static final void _appendCode(XmcWriter w,
                                          Stack<XcType> decls,
                                          String symbol,
                                          boolean isPreDecl,
                                          boolean isAttrGuarded,
                                          XcGccAttributeList attr,
                                          List<XcGccAttributeList> attrOfArrayTypes,
                                          boolean isFromFunc
                                          ) throws XmException
    {
        if (decls.isEmpty()) {
            if (symbol != null) {
                if (attr != null) {
                    w.addSpc(attr);
                }

                w.addSpc(symbol);
            }
            return;
        }

        boolean brace = false;
        XcType t = decls.pop();

        XcTypeEnum te = t.getTypeEnum();

        /*
         * if type is functionType or array type
         * which refered by pointer (or basicType of pointer)
         * symbol description must be braced.
         *
         * ex)
         *    [in C]
         *    void (* funct_t)(...);
         *
         *    The above function pointer discription in C
         *    is taranslated in XcodeML as follow.
         *
         *    [in XcodeML]
         *    <functionType type="F0" return="void">
         *      ...
         *    </functionType>
         *    <pointerType type="P0" ref="F0>
         */
        XcType nt = _peekRealType(decls);
        if(nt != null && te != XcTypeEnum.POINTER &&
           (nt.getTypeEnum() == XcTypeEnum.POINTER)) {
            brace = true;
        }

        switch (te) {
        case BASICTYPE:
            {
              if(!isFromFunc)
                t.appendGccAtrribute(w);
                t.appendTypeQualCode(w);

                _appendCode(w, decls, symbol, isPreDecl, isAttrGuarded, attr, attrOfArrayTypes);
            }
            break;
        case POINTER:
            {
                if(isAttrGuarded)
                    w.add(")");
                isAttrGuarded = false;

                w.addSpc("*").spc();

                t.appendGccAtrribute(w);
                t.appendTypeQualCode(w);

                _appendCode(w, decls, symbol, isPreDecl, isAttrGuarded, attr, attrOfArrayTypes);
            }
            break;

        case FUNC:
            {
                t.appendGccAtrribute(w);

                XcFuncType ft = (XcFuncType)t;
                isPreDecl = ft.isPreDecl();

                _appendBasicType(w, decls, null);

                if(brace)
                    w.add("(");
                _appendCode(w, decls, symbol, isPreDecl, isAttrGuarded, attr, attrOfArrayTypes);
                if(brace)
                    w.add(")");

                if (ft.getParamList().isEmpty() == false) {
                    ft.getParamList().appendArgs(w, (brace == false) &&
                                                (ft.isPrototype() == false),
                                                 isPreDecl);
                } else {
                    w.add("()");
                }
            }
            break;

        case ARRAY:
            {
                if(attrOfArrayTypes == null)
                    attrOfArrayTypes = new ArrayList<XcGccAttributeList>();

                if(t.getGccAttribute() != null)
                    attrOfArrayTypes.add(t.getGccAttribute());

                XcArrayType at = (XcArrayType)t;

                _appendBasicType(w, decls, attrOfArrayTypes);

                if(decls.isEmpty() == false &&
                    decls.peek().getTypeEnum() == XcTypeEnum.FUNC) {
                    if(isAttrGuarded)
                        w.add(")");
                    isAttrGuarded = false;

                    w.add("*");
                    _appendCode(w, decls, symbol, isPreDecl, isAttrGuarded, attr, attrOfArrayTypes);

                } else {
                    boolean isFirstIndex = decls.isEmpty();

                    if(brace)
                        w.add("(");
                    _appendCode(w, decls, symbol, isPreDecl, isAttrGuarded, attr, attrOfArrayTypes);
                    if(brace)
                        w.add(")");

                    at.appendArraySpecCode(w, isPreDecl, isFirstIndex);

                    XcType elemType = at.getRefType().getRealType();

                    if(elemType.getTypeEnum() != XcTypeEnum.ARRAY
                       && attrOfArrayTypes != null) {
                        for(XcGccAttributeList attrs : attrOfArrayTypes) {
                            attrs.appendCode(w);
                        }
                    }
                }
            }
            break;

        default:
            throw new XmException();
        }
    }

    public final void appendTypeQualCode(XmcWriter w) throws XmException
    {
        if(_isConst)
            w.addSpc("const");
        if(_isVolatile)
            w.addSpc("volatile");
        if(_isRestrict)
            w.addSpc("restrict");
    }

    protected final StringBuilder commonToString(StringBuilder b)
    {
        b.append("type=").append(_typeEnum);
        b.append(",typeId=").append(_typeId);
        if(this instanceof XcTaggedType)
            b.append(",tagged=").append(((XcTaggedType)this).getTagName());
        return b;
    }

    @Override
    public String toString()
    {
        StringBuilder b = new StringBuilder(128);
        b.append("[");
        commonToString(b);
        b.append("]");
        return b.toString();
    }

    public void resolve(XcIdentTableStack itStack) throws XmException
    {
        if(_resolved || _refType != null)
            return;

        resolveOverride(itStack);

        if(canBeBottomType() == false) {
            _refType = XcBaseTypeEnum.createTypeByXcode(_tempRefTypeId);

            if(_refType == null) {
                _refType = itStack.getType(_tempRefTypeId);
            }

            _tempRefTypeId = null;
        }

        _resolved = true;
    }

    public XcType getRealType()
    {
        XcType type = this;
        while(type instanceof XcBasicType) {
            type = type.getRefType();
            if(type == null)
                break;
        }
        return type;
    }

    @Override
    public void setGccAttribute(XcGccAttributeList attrs)
    {
        _gccAttrs = attrs;
    }

    @Override
    public XcGccAttributeList getGccAttribute()
    {
        return _gccAttrs;
    }
}
