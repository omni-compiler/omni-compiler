/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
package xcodeml.c.type;

import java.util.Iterator;
import java.util.List;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.Map;
import java.util.Set;

import xcodeml.util.XmException;

/**
 * Set of dependency of tagname and typedef name.
 */
public class XcTagAndDefSet
{
    /* strictly defined type     (ex : "struct a { ... };" defines a. */
    private Map<String, XcTagAndDefDepend> _typeDependMap           = new HashMap<String, XcTagAndDefDepend>();

    /* defined reference of type (ex : "struct a * * a;" defines reference a. */
    private Map<String, XcTagAndDefDepend> _refTypeDependMap        = new HashMap<String, XcTagAndDefDepend>();

    private LinkedList<XcTagAndDefDepend>  _preResolvedDependQueue  = new LinkedList<XcTagAndDefDepend>();

    /* list of tagname or typedef name with complete type.*/
    private List<XcTagAndDefDepend>        _resolvedDependList      = new ArrayList<XcTagAndDefDepend>();

    /* set of tagname or typedef name with complete/incomplete type.*/
    private Set<XcTagAndDefDepend>         _resolvedDependSet       = new HashSet<XcTagAndDefDepend>();

    /* identifier list with no tagname (ex anonymous enum type */
    private List<XcIdent>                  _anonIdentList           = new ArrayList<XcIdent>();

    /**
     * dump state.
     */
    public void dump()
    {
        System.out.println("/* dump tag and def set");

        for(XcTagAndDefDepend depend : _preResolvedDependQueue) {
            depend.dump();
        }

        System.out.println("*/");
    }

    /**
     * copy tagname or typedef name from other set.
     * @param typeSet other tagname and typedef name set.
     */
    public void copyDefinedType(XcTagAndDefSet typeSet)
    {
        _resolvedDependSet.addAll(typeSet.getResolvedDependSet());
    }

    /**
     * @return tagname or typedef name set.
     */
    public Set<XcTagAndDefDepend> getResolvedDependSet()
    {
        return _resolvedDependSet;
    }

    public void addIdent(XcIdent ident)
    {
        /* get tagname of identifier */
        String symbol = ident.getSymbol();

        XcTagAndDefDepend depend;

        if(ident.isTypedef()) {
            depend =  new XcTagAndDefDepend();
            depend.setMap(this);
            depend.setTypeDefName(symbol);
            depend.setIdent(ident);

        } else {
            depend = get(symbol);
            depend.setIdent(ident);

        }

        _preResolvedDependQueue.add(depend);
    }

    public void addAnonIdent(XcIdent ident)
    {
        String typeId = ident.getType().getTypeId();

        for(XcIdent anonIdent : _anonIdentList) {
            if(typeId.equals(anonIdent.getType().getTypeId()))
                return;
        }

        _anonIdentList.add(ident);
    }

    private List<XcTagAndDefDepend> _addableIncomp(XcTagAndDefDepend depend) {
        if(depend == null)
            return null;

        List<XcTagAndDefDepend> dependList = new ArrayList<XcTagAndDefDepend>();

        boolean needOnlyIncomp = true;

        for(XcTagAndDefDepend dependOn : depend.getDependSet()) {
            if(_resolvedDependSet.contains(dependOn))
                continue;

            if((_resolvedDependList.contains(dependOn)) == false &&
               dependOn.isReferenceType() == false) {
                needOnlyIncomp = false;
                break;
            }

            dependList.add(dependOn);
        }

        if(needOnlyIncomp)
                return dependList;

        return null;
    }

    private void _addAsIncomplete(List<XcTagAndDefDepend> dependList) throws XmException
    {
        for(XcTagAndDefDepend depend : dependList) {
            if(depend == null)
                continue;

            String tagname = depend.getSymbol();

            if(tagname != null && _resolvedDependSet.contains(depend) == false) {
                XcTagAndDefDepend dependTag = get(tagname);

                XcIdent ident = dependTag.getIdent();

                if(ident == null) {
                    throw new XmException("tagname " + tagname + " used but not defined.");
                }

                ident = ident.getIncomp();
                depend.setIdent(ident);

                _resolvedDependSet.add(depend);
                _resolvedDependList.add(depend);
            }
        }
    }

    private void _addAsComplete(XcTagAndDefDepend depend)
    {
        _resolvedDependSet.addAll(depend.getDefineSet());
        _resolvedDependList.add(depend);
    }

    private void _addAnonIdent()
    {
        for(XcIdent anonIdent : _anonIdentList) {
            boolean hasTagname = false;
            String typeId = anonIdent.getType().getTypeId();

            for(XcTagAndDefDepend depend : _preResolvedDependQueue) {
                if(typeId.equals(depend.getIdent().getType().getTypeId())) {
                    hasTagname = true;
                    break;
                }
            }

            if(hasTagname == false) {
                XcTagAndDefDepend anonDepend = new XcTagAndDefDepend();
                anonDepend.setIdent(anonIdent);
                anonDepend.setMap(this);

                _preResolvedDependQueue.addFirst(anonDepend);
            }
        }
        _anonIdentList.clear();
    }

    /**
     * resolve tagname and typedef name dependency.
     */
    public void resolveDepend() throws XmException
    {
        _addAnonIdent();

        for(XcTagAndDefDepend depend : _preResolvedDependQueue) {
            depend.tagnameCrowler();
        }

        LinkedList<XcTagAndDefDepend> preResolvedDependList = new LinkedList<XcTagAndDefDepend>();
        List<XcTagAndDefDepend> preResolvedVarDepend = new ArrayList<XcTagAndDefDepend>();
        List<XcTagAndDefDepend> notResolved = new ArrayList<XcTagAndDefDepend>();

        for(XcTagAndDefDepend depend : _preResolvedDependQueue) {
            XcIdent ident = depend.getIdent();

            if(ident.getDependVar().isEmpty()) {
                preResolvedDependList.add(depend);
            } else {
                preResolvedVarDepend.add(depend);
            }
        }
        _preResolvedDependQueue.clear();

        Iterator<XcTagAndDefDepend> varDependIter = preResolvedVarDepend.iterator();

        do {
            for(XcTagAndDefDepend dependSet : preResolvedDependList) {
                if(dependSet == null)
                    continue;

                if(_resolvedDependSet.containsAll(dependSet.getDependSet())) {
                    _addAsComplete(dependSet);

                } else {
                    notResolved.add(dependSet);

                }
            }
            preResolvedDependList.clear();

            boolean isAdded = true;
            List<XcTagAndDefDepend> rest = new ArrayList<XcTagAndDefDepend>();
            while(isAdded == true) {
                isAdded = false;

                for(XcTagAndDefDepend depend : notResolved) {
                    List<XcTagAndDefDepend> dependList = _addableIncomp(depend);

                    if(_resolvedDependList.containsAll(depend.getDependSet())) {
                        isAdded = true;
                        _addAsComplete(depend);

                    } else if(dependList != null) {
                        isAdded = true;

                        _addAsIncomplete(dependList);
                        _addAsComplete(depend);

                    } else {
                        rest.add(depend);
                    }
                }

                notResolved.clear();
                notResolved.addAll(rest);
                rest.clear();

            }

            if(notResolved.isEmpty())
                break;

            if(varDependIter.hasNext() == false)
                break;

            XcTagAndDefDepend depend = varDependIter.next();
            List<XcTagAndDefDepend> dependList = _addableIncomp(depend);

            if(_resolvedDependSet.containsAll(depend.getDependSet())) {
                _addAsComplete(depend);

            } else if(dependList != null) {
                _addAsIncomplete(dependList);
                _addAsComplete(depend);

            } else {
                break;

            }

            preResolvedDependList.addAll(notResolved);
            notResolved.clear();

        } while(notResolved.isEmpty());

        if(notResolved.isEmpty() == false) {
            Iterator<XcTagAndDefDepend> iter = notResolved.iterator();
            XcTagAndDefDepend dependSet = iter.next();
            String symbol = dependSet.getSymbol();

            if(symbol == null) {
                symbol = "'typedef'" + dependSet.getTypeDefName();
            }

            throw new XmException("cannot resolve tagname dependency " + symbol);
        }

        while(varDependIter.hasNext()) {
            XcTagAndDefDepend depend = varDependIter.next();
            _addAsComplete(depend);
        }
    }

    public ArrayList<XcIdent> getIdentList()
    {
        ArrayList<XcIdent> identList = new ArrayList<XcIdent>();

        if(_resolvedDependList.isEmpty()) {
            for(XcTagAndDefDepend depend : _preResolvedDependQueue) {
                XcIdent ident = depend.getIdent();
                if(ident != null)
                    identList.add(ident);
            }
            return identList;

        } else {
            for(XcTagAndDefDepend depend : _resolvedDependList) {
                XcIdent ident = depend.getIdent();
                if(ident != null)
                    identList.add(ident);
            }

            _resolvedDependList.clear();

            return identList;
        }
    }

    public XcTagAndDefDepend get(String symbol)
    {
        XcTagAndDefDepend _depend = null;

        if(symbol == null)
            return null;

        if(_typeDependMap.containsKey(symbol))
            _depend = _typeDependMap.get(symbol);

        else {
            _depend = new XcTagAndDefDepend(symbol, this);
            _typeDependMap.put(symbol, _depend);

        }

        return _depend;
    }

    public XcTagAndDefDepend getRef(String symbol)
    {
        XcTagAndDefDepend _depend = null;

        if(symbol == null) {
            _depend =  new XcTagAndDefDepend(this);
            _depend.setIsReferenceType(true);

        } else if(_refTypeDependMap.containsKey(symbol)) {
            _depend = _refTypeDependMap.get(symbol);
            _depend.setIsReferenceType(true);

        } else {
            _depend = new XcTagAndDefDepend(symbol, this);
            _depend.setIsReferenceType(true);
            _refTypeDependMap.put(symbol, _depend);

        }

        return _depend;
    }

    public void addSet(XcTagAndDefDepend set)
    {
        _preResolvedDependQueue.add(set);
    }
}
