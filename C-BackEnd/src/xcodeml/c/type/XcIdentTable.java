/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
package xcodeml.c.type;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.Map.Entry;

import xcodeml.XmException;
import xcodeml.c.decompile.XcBindingVisitor;
import xcodeml.c.decompile.XcDecAndDefObj;
import xcodeml.c.decompile.XcDeclObj;
import xcodeml.c.decompile.XcDeclsObj;
import xcodeml.c.decompile.XcFuncDefObj;
import xcodeml.c.util.XmcWriter;

/**
 * represents symbol table
 */
public final class XcIdentTable
{
    /* symbol table for variable name/function name/type name/enumerator name */
    private Map<String, XcIdent> _mainSymMap = new HashMap<String, XcIdent>();

    /* symbol table for tag name */
    private Map<String, XcIdent> _tagSymMap = new HashMap<String, XcIdent>();

    /* symbol table for label */
    private Map<String, XcIdent> _labelSymMap = new HashMap<String, XcIdent>();

    private List<XcIdent>  _tagAndTypeDefList = new ArrayList<XcIdent>();

    private XcTagAndDefSet _defineTypeSet = new XcTagAndDefSet();

    public XcIdentTable()
    {
    }

    private Map<String, XcIdent> _getSymMap(XcIdentTableEnum itEnum)
    {
        switch(itEnum)
        {
        case MAIN:
            return _mainSymMap;
        case TAGNAME:
            return _tagSymMap;
        case ANONYM:
            return null;
        default: // LABEL:
            return _labelSymMap;
        }
    }

    public void add(XcIdentTableEnum itEnum, XcIdent ident)
    {
        Map<String, XcIdent> map = _getSymMap(itEnum);

        switch(itEnum) {
        case MAIN:
            if(ident.isTypedef())
                _defineTypeSet.addIdent(ident);
            break;
        case TAGNAME:
            _defineTypeSet.addIdent(ident);
            break;
        default:
            break;
        }

        if(map != null)
            map.put(ident.getSymbol(), ident);
    }

    public void addAnonIdent(XcIdent ident)
    {
        _defineTypeSet.addAnonIdent(ident);
    }

    public void _lazyEval(XcBindingVisitor visitor, XcType type)
    {
        if(type == null)
            return;

        if(type instanceof XcArrayType) {
            do {
                visitor.lazyEnter((XcLazyEvalType) type);
                ((XcLazyEvalType)type).setIsLazyEvalType(false);

                type = type.getRefType();
            } while(type instanceof XcArrayType);
        }
    }

    public XcIdent getIdent(XcIdentTableEnum itEnum, String symbol)
    {
        Map<String, XcIdent> map = _getSymMap(itEnum);

        XcIdent ident = map.get(symbol);

        return ident;
    }

    public boolean containsIdent(XcIdentTableEnum itEnum, String symbol)
    {
        Map<String, XcIdent> map = _getSymMap(itEnum);
        return map.containsKey(symbol);
    }

    private void _removeExternVar(XcIdent ident)
    {
        Set<String> variables = ident.getDependVar();

        for(String var : variables) {
            if(_mainSymMap.containsKey(var) == false)
                variables.remove(var);
        }
    }

    public void resolveDependency(XcBindingVisitor visitor) throws XmException
    {
        Iterator<Entry<String, XcIdent>> iter = _mainSymMap.entrySet().iterator();

        while(iter.hasNext()) {
            XcIdent ident = iter.next().getValue();
            ident.lazyEval(visitor);
        }

        for(XcIdent ident : _defineTypeSet.getIdentList()) {
             ident.lazyEval(visitor);
             ident.gatherVar();
             _removeExternVar(ident);
        }

        _defineTypeSet.resolveDepend();

        _tagAndTypeDefList = _defineTypeSet.getIdentList();
    }

    public void appendCode(XmcWriter w) throws XmException
    {
        for(XcIdent ident : _tagAndTypeDefList) {
            if(ident != null && ident.isOutput() == false) {
                ident.appendTagOrTypedef(w);
                ident.setOutput();
            }
        }
    }

    public void appendCode(XmcWriter w, XcDeclsObj decls) throws XmException
    {
        if(decls == null)
            appendCode(w);
        else
            appendCode(w, decls.getDeclList());
    }

    public void appendCode(XmcWriter w, List<XcDecAndDefObj> declAndDefList) throws XmException
    {
        Set<String> variables = new HashSet<String>();

        Iterator<XcDecAndDefObj> iter = declAndDefList.iterator();

        for(XcIdent ident : _tagAndTypeDefList) {
            if(ident == null || ident.isOutput() == true)
                continue;

            if(ident.getDependVar().isEmpty() == false) {
                Set<String> depend = ident.getDependVar();

                while(variables.containsAll(depend) == false) {
                    String symbol = null;

                    XcDecAndDefObj decAndDef = iter.next();

                    if(decAndDef instanceof XcDeclObj) {
                        symbol = ((XcDeclObj)decAndDef).getSymbol();
                    }

                    if(decAndDef instanceof XcFuncDefObj) {
                        symbol = ((XcFuncDefObj)decAndDef).getSymbol();
                    }

                    if(symbol != null)
                        variables.add(symbol);

                    w.add(decAndDef).noLfOrLf();
                }
            }

            ident.appendTagOrTypedef(w);
            ident.setOutput();
        }

        while(iter.hasNext() == true) {
            XcDecAndDefObj decAndDef = iter.next();
            w.add(decAndDef).noLfOrLf();
        }
    }

    public void setTypeSet(XcTagAndDefSet definedTypeSet)
    {
        _defineTypeSet = definedTypeSet;
    }

    public void copyTypeSet(XcTagAndDefSet definedTypeSet)
    {
        if(definedTypeSet == null)
            return;

        _defineTypeSet.copyDefinedType(definedTypeSet);
    }

    public XcTagAndDefSet getDefinedTypeSet()
    {
        return _defineTypeSet;
    }
}
