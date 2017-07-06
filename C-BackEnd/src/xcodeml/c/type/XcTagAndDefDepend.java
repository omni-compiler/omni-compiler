package xcodeml.c.type;

import java.util.HashSet;
import java.util.Set;
import xcodeml.c.decompile.XcRefObj;
import xcodeml.c.obj.XcNode;

/**
 * dependency of tagname and typedef name.
 */
public class XcTagAndDefDepend
{
    private Set<XcTagAndDefDepend> _dependOn = new HashSet<XcTagAndDefDepend>();

    private Set<XcTagAndDefDepend> _defineOf = new HashSet<XcTagAndDefDepend>();

    private XcTagAndDefSet         _set;

    private String                 _symbol;

    private String                 _typeDefName;

    private XcIdent                _ident;

    private boolean                _isReferenceType = false;

    public void dumpSymbol()
    {
        if(_typeDefName != null) {
            System.out.print("typedef_" + _typeDefName);

        } else {
            if(_isReferenceType)
                System.out.print("ref_");

            System.out.print(_symbol);
        }
    }

    public void dump()
    {
        dumpSymbol();
        System.out.println();
        System.out.print("depend:");
        for(XcTagAndDefDepend depend : _dependOn) {
            depend.dumpSymbol();
            System.out.print(",");
        }
        System.out.println();
        System.out.print("define:");
        for(XcTagAndDefDepend depend : _defineOf) {
            depend.dumpSymbol();
            System.out.print(",");
        }
        System.out.println();
    }

    public void setIsReferenceType(boolean enable)
    {
        _isReferenceType = enable;
    }

    public boolean isReferenceType()
    {
        return _isReferenceType;
    }


    public XcTagAndDefDepend()
    {
    }

    public XcTagAndDefDepend(String symbol)
    {
        _symbol = symbol;
    }

    public XcTagAndDefDepend(String symbol, XcTagAndDefSet set)
    {
        _set = set;
        _symbol = symbol;
    }

    public XcTagAndDefDepend(XcTagAndDefSet set)
    {
        _set = set;
    }

    public String getSymbol()
    {
        return _symbol;
    }

    public String getTypeDefName()
    {
        return _typeDefName;
    }

    public void setTypeDefName(String typeDefName)
    {
        _typeDefName = typeDefName;
    }

    public void addDepend(XcTagAndDefDepend depend)
    {
        if((depend == null) || (depend.getSymbol() == null))
            return;

        _dependOn.add(depend);
    }

    public void addDefine(XcTagAndDefDepend depend)
    {
        if((depend == null) || (depend.getSymbol() == null))
            return;

        _defineOf.add(depend);
    }

    public boolean containIdAsDepend(String typeId)
    {
        for(XcTagAndDefDepend depend : _dependOn)
            if(typeId.equals(depend.toString()))
               return true;

        return false;
    }

    public boolean containIdAsDefine(String typeId)
    {
        for(XcTagAndDefDepend define : _defineOf)
            if(typeId.equals(define.toString()))
               return true;

        return false;
    }

    public boolean containsDepend(XcTagAndDefDepend depend)
    {
        return _dependOn.contains(depend);
    }

    public boolean containsDependAll(Set<XcTagAndDefDepend> dependSet)
    {
        return _dependOn.containsAll(dependSet);
    }

    public void removeAll(Set<String> set)
    {
        _dependOn.removeAll(set);
    }

    @Override
    public String toString()
    {
        return _symbol;
    }

    public Set<XcTagAndDefDepend> getDependSet()
    {
        return _dependOn;
    }

    public Set<XcTagAndDefDepend> getDefineSet()
    {
        return _defineOf;
    }

    public XcIdent getIdent()
    {
        return _ident;
    }

    public void setIdent(XcIdent ident)
    {
        _ident = ident;
        _dependOn.clear();
    }

    public void tagnameCrowler()
    {
        _tagnameCrowler(_ident);
    }

    public void setMap(XcTagAndDefSet set)
    {
        _set = set;
    }

    private void _tagnameCrowler(XcIdent ident)
    {
        /*
         * crowl symbols of identifier (must be tagname or typedef),
         * and decide whether does this identifier define the symbol or use it.
         */

        /*
          insideParam indicate that struct/union of tagname appeared inside function parameter.

          ex)
          int function(struct a * a);

          In the above case, incomplete type of struct 'a'
          must be appeared before this function.
         */
        boolean insideParam = false;

        /*
          asPointer indicate that struct/union of tagname was used as pointer.

          ex)
          struct b {
              struct a * a;
          };

          In the above case, struct a may appeared after this declaration,
          and it declared incomplete type of struct a;
         */
        boolean asPointer = false;

        /*
          memberRef indicate that member of struct/union of tagname was refered.

          ex)
          int array[sizeof(((struct a *)0)->member)];

          In the above case,
          struct 'a' must be appeared before this function,
          otherwise this description ocurr error
          'reference to incomplete type'.
         */
        boolean memberRef = false;

        XcType type = ident.getType();

        switch(type.getTypeEnum()) {
        case STRUCT:
        case UNION:
            XcCompositeType xc = (XcCompositeType)type;

            if(xc.getMemberList() != null) {
                for(XcIdent child : xc.getMemberList()) {
                    _tagnameCrowler(insideParam, asPointer, memberRef, (XcNode)child);
                }
                addDefine(this);
            }

            addDefine(_set.getRef(_symbol));
            break;
        case ENUM:
            addDefine(this);
            addDefine(_set.getRef(_symbol));

            XcEnumType xe = (XcEnumType)type;

            if(xe.getEnumeratorList() != null) {
                for(XcIdent child : xe.getEnumeratorList()) {
                    _tagnameCrowler(insideParam, asPointer, memberRef, (XcNode)child.getValue());
                }
            }

            break;
        case FUNC:
            XcFuncType xf = (XcFuncType)type;

            XcType returnType = type.getRefType();

            _tagnameCrowler(insideParam, true, memberRef, returnType);

            insideParam = true;

            for(XcIdent child : xf.getParamList()) {
                _tagnameCrowler(insideParam, asPointer, memberRef, child);
            }

            if(xf.getParamList() != null || (xf.getParamList().isEmpty() == false))
                addDefine(this);

            addDefine(_set.getRef(_symbol));
        case ARRAY:
        case BASICTYPE:
            addDefine(this);

        case POINTER:
            addDefine(_set.getRef(_symbol));

            _tagnameCrowler(insideParam, asPointer, memberRef, type);
            break;
        case BASETYPE:
        case BUILTIN:
            break;
        default: /* type is unable to be tagged */
            break;
        }

        _tagnameCrowler(insideParam, asPointer, memberRef, ident.getGccAttribute());
    }

    private void _tagnameCrowler(boolean insideParam, boolean asPointer, boolean memberRef, XcIdent ident)
    {
        if(ident instanceof XcIdent.MoeConstant) {
            XcEnumType enumType = ((XcIdent.MoeConstant)ident).getEnumType();

            /* this statemet allows these moeConstant
             *
             * ex)
             * enum e {
             *     e1,
             *     e2 = sizeof(e1),
             *     e3 = sizeof(typeof(e2))
             * }
             */
            if(_defineOf.contains(_set.get(enumType.getTagName())) == false) {
                _tagnameCrowler(false, false, false, enumType);
            }

            return;
        }

        XcType type = ident.getType();

        _tagnameCrowler(insideParam, asPointer, memberRef, ident.getGccAttribute());
        _tagnameCrowler(insideParam, asPointer, memberRef, type);
    }

    private void _tagnameCrowler(boolean insideParam, boolean asPointer, boolean memberRef, XcType type)
    {
        if(type == null)
            return;

        while (type instanceof XcBasicType) {
            type = type.getRefType();
        }

        if(type instanceof XcBaseType) {
            return;
        } else if(type instanceof XcTaggedType) {
            XcTaggedType tt = (XcTaggedType)type;
            String tagname = tt.getTagName();

            if(memberRef) {
                addDepend(_set.get(tagname));
                memberRef = false;
            } else if(insideParam) {
                /*
                  tagname cannot be defined inside parameter list
                 */
                if(tagname != _symbol) {
                    XcTagAndDefDepend dep = _set.getRef(tagname);
                    addDepend(dep);
                    if(asPointer)
                        dep.setIsReferenceType(true);
                }

                addDefine(_set.getRef(tagname));

            } else if(asPointer) {
                addDepend(_set.getRef(tagname));
            } else {
                addDepend(_set.get(tagname));
            }
        } else if(type instanceof XcFuncType) {
            XcFuncType xf = (XcFuncType)type;

            asPointer = true;

            _tagnameCrowler(insideParam, asPointer, memberRef, ((XcFuncType)type).getRefType());

            asPointer = false;
            insideParam = true;

            for(XcIdent child : xf.getParamList()) {
                _tagnameCrowler(insideParam, asPointer, memberRef, child);
            }
        } else if(type instanceof XcArrayType) {
            _tagnameCrowler(insideParam, asPointer, memberRef, ((XcArrayType)type).getArraySizeExpr());

        } else if(type instanceof XcPointerType) {
            asPointer = true;
        }

        _tagnameCrowler(insideParam, asPointer, memberRef, type.getGccAttribute());
        _tagnameCrowler(insideParam, asPointer, memberRef, type.getRefType());
    }

    private void _tagnameCrowler(boolean insideParam, boolean asPointer, boolean memberRef, XcNode ... nodes)
    {
        if(nodes == null || nodes.length == 0) {
            return;
        }

        for(XcNode node : nodes) {
            if(node == null) continue;

            if(node instanceof XcIdent) {
                _tagnameCrowler(insideParam, asPointer, memberRef, (XcIdent)node);
            }

            if(node instanceof XcType) {
                _tagnameCrowler(insideParam, asPointer, memberRef, (XcType)node);

            }

            if((node instanceof XcRefObj.MemberRef) ||
               (node instanceof XcRefObj.MemberAddr))
                memberRef = true;

            _tagnameCrowler(insideParam, asPointer, memberRef, node.getChild());
        }
    }

    @Override
    public boolean equals(Object o)
    {
        if(o == null)
            return false;
        XcTagAndDefDepend dep = (XcTagAndDefDepend)o;
        if(_symbol == null || dep._symbol == null)
            return false;
        return _symbol.equals(dep._symbol) && _isReferenceType == dep._isReferenceType;
    }

    @Override
    public int hashCode()
    {
        if(_typeDefName != null)
            return _typeDefName.hashCode();

        return _symbol.hashCode();
    }
}
