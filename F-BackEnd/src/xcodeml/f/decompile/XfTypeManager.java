/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
package xcodeml.f.decompile;

import java.util.HashMap;
import java.util.LinkedList;
import java.util.Map;

import xcodeml.XmException;
import xcodeml.f.binding.gen.*;

/**
 * Type and Symbol manager.
 */
class XfTypeManager
{
    private TypeMap _typeMap;
    private SymbolMapStack _symbolMapStack;
    private AliasMapStack _aliasMapStack;
    private AliasMap _reverseBasicRefMap;

    // Deque< HashMap<String, IRNode> > _symbolMapStack;
    // NodeMap _typeMap;

    @SuppressWarnings("serial")
    private class TypeMap extends HashMap<String, IXbfTypeTableChoice>
    {
        @Override
        public String toString()
        {
            StringBuilder sb = new StringBuilder();
            sb.append("[TypeMap, key(type name) -> value(class)]\n");
            for (Map.Entry<String, IXbfTypeTableChoice> entry : entrySet()) {
                sb.append(entry.getKey());
                sb.append(" -> ");
                sb.append(entry.getValue().getClass().getSimpleName());
                sb.append("\n");
            }
            return sb.toString();
        }
    }

    @SuppressWarnings("serial")
    private class SymbolMap extends HashMap<String, XbfId>
    {
        @Override
        public String toString()
        {
            StringBuilder sb = new StringBuilder();
            sb.append("[SymbolMap, key(symbol name) -> value(type name)]\n");
            for (Map.Entry<String, XbfId> entry : entrySet()) {
                sb.append(entry.getKey());
                sb.append(" -> ");
                sb.append(entry.getValue().getType());
                sb.append("\n");
            }
            return sb.toString();
        }
    }

    @SuppressWarnings("serial")
    private class SymbolMapStack extends LinkedList<SymbolMap>
    {
        @Override
        public String toString()
        {
            StringBuilder sb = new StringBuilder();
            sb.append("[SymbolMapStack]\n");
            for (SymbolMap symbolMap : _symbolMapStack) {
                sb.append(symbolMap.toString());
            }
            return sb.toString();
        }
    }

    @SuppressWarnings("serial")
    private class AliasMap extends HashMap<String, String>
    {

        @Override
        public String toString()
        {
            StringBuilder sb = new StringBuilder();
            sb.append("[AliasMap, key(type) -> value(alias type)]\n");
            for (Map.Entry<String, String> entry : entrySet()) {
                sb.append(entry.getKey());
                sb.append(" -> ");
                sb.append(entry.getValue());
                sb.append("\n");
            }
            return sb.toString();
        }
    }

    @SuppressWarnings("serial")
    private class AliasMapStack extends LinkedList<AliasMap>
    {
        @Override
        public String toString()
        {
            StringBuilder sb = new StringBuilder();
            sb.append("[AliasMapStack]\n");
            for (AliasMap aliasMap : _aliasMapStack) {
                sb.append(aliasMap.toString());
            }
            return sb.toString();
        }
    }

    @SuppressWarnings("serial")
    public class TypeList extends LinkedList<IXbfTypeTableChoice>
    {
        @Override
        public String toString()
        {
            StringBuilder sb = new StringBuilder();
            sb.append("[TypeList, element name -> type name]\n");
            for (IXbfTypeTableChoice typeChoice : this) {
                sb.append(XfUtil.getElementName(typeChoice));
                sb.append(" -> ");
                sb.append(typeChoice.getType());
                sb.append("\n");
            }
            return sb.toString();
        }
    }

    public XfTypeManager()
    {
        _typeMap = new TypeMap();
        _symbolMapStack = new SymbolMapStack();
        _aliasMapStack = new AliasMapStack();
        _reverseBasicRefMap = new AliasMap();
    }

    private SymbolMap _getCurrentSymbolMap()
    {
        return _symbolMapStack.peekFirst();
    }

    private AliasMap _getCurrentAliasMap()
    {
        return _aliasMapStack.peekFirst();
    }

    public void enterScope()
    {
        _symbolMapStack.push(new SymbolMap());
        _aliasMapStack.push(new AliasMap());
    }

    public void leaveScope()
    {
        _symbolMapStack.pop();
        _aliasMapStack.pop();
    }

    public void addSymbol(XbfId id)
    {
        XbfName name = id.getName();
        if (name == null) {
            // Ignore invalid symbol name.
            return;
        }

        String symbolName = name.getContent();
        if (symbolName == null) {
            // Ignore invalid symbol name.
            return;
        }

        // Trim key word string.
        symbolName = symbolName.trim();
        if (symbolName.isEmpty() != false) {
            // Ignore invalid symbol name.
            return;
        }

        if (id.getSclass() == null)
            return;
        
        if (XbfId.SCLASS_FTYPE_NAME.equalsIgnoreCase(id.getSclass())) {
            String typeName = id.getType();
            if (XfUtil.isNullOrEmpty(typeName) != false) {
                // Ignore invalid type name.
                return;
            }
            AliasMap aliasMap = _getCurrentAliasMap();
            assert (aliasMap != null);
            aliasMap.put(typeName, symbolName);
        } else if (!XbfId.SCLASS_FCOMMON_NAME.equalsIgnoreCase(id.getSclass())) {
            SymbolMap symbolMap = _getCurrentSymbolMap();
            assert (symbolMap != null);
            symbolMap.put(symbolName, id);
        }
    }

    public XbfId findSymbol(String symbolName)
    {
        XbfId id = null;
        symbolName = symbolName.trim();
        for (SymbolMap symbolMap : _symbolMapStack) {
            id = symbolMap.get(symbolName);
            if (id != null) {
                break;
            }
        }
        return id;
    }

    public void addType(IXbfTypeTableChoice type)
    {
        String typeName = type.getType();
        if (XfUtil.isNullOrEmpty(typeName) != false) {
            // Ignore invalid type name.
            return;
        }

        // Trim key word string.
        typeName = typeName.trim();

        if (typeName.isEmpty() != false) {
            // Ignore invalid type name.
            return;
        }

        _typeMap.put(typeName, type);
        
        if(type instanceof XbfFbasicType && ((XbfFbasicType)type).getRef() != null &&
        		((XbfFbasicType)type).getContent() == null) {
            _reverseBasicRefMap.put(((XbfFbasicType)type).getRef(), type.getType());
        }
    }

    public IXbfTypeTableChoice findType(String typeName)
    {
        if (XfUtil.isNullOrEmpty(typeName) != false) {
            return null;
        }

        // Trim key word string.
        return _typeMap.get(typeName.trim());
    }

    public IXbfTypeTableChoice findType(XbfName nameElem)
    {
        if (nameElem == null) {
            return null;
        }

        String typeName = nameElem.getType();
        if (XfUtil.isNullOrEmpty(typeName) != false) {
            return findTypeFromSymbol(nameElem.getContent());
        }
        return findType(typeName);
    }

    /**
     * Find type element from symbol name.
     * @param symbolName Symbol name.
     * @return  IXbfTypeTableChoice interface or null.<br/>
     *          If null, type is not found.
     */
    public IXbfTypeTableChoice findTypeFromSymbol(String symbolName)
    {
        if (XfUtil.isNullOrEmpty(symbolName) != false) {
            return null;
        }

        XbfId id = findSymbol(symbolName);
        if (id == null) {
            return null;
        }

        return findType(id.getType());
    }

    /**
     * Put tagname of type id.
     * @param typeName
     * @param typeId
     */
    public void putAliasTypeName(String typeId, String typeName)
    {
        if (XfUtil.isNullOrEmpty(typeName)) {
            return;
        }

        if (XfUtil.isNullOrEmpty(typeId)) {
            return;
        }

        AliasMap currentAlias = _getCurrentAliasMap();

        currentAlias.put(typeId, typeName);
    }

    /**
     * Get alias of type name.
     * @param typeName
     * @return When alias not found, return argument type name.
     */
    public String getAliasTypeName(String typeName)
    {
        if (XfUtil.isNullOrEmpty(typeName) != false) {
            return null;
        }

        // Trim key word string.
        typeName = typeName.trim();
        for (AliasMap aliasMap : _aliasMapStack) {
            String aliasName = aliasMap.get(typeName);
            if (aliasName != null) {
                return aliasName;
            }
        }
        
        String inheritName = _reverseBasicRefMap.get(typeName);
        if(inheritName != null) {
            return getAliasTypeName(inheritName);
        }

        throw new IllegalStateException("not found type name of '" + typeName + "'");
    }

    /**
     * Get type reference list.
     * @param typeName Type name.
     * @return First node of list is top level type.
     * @throw XmException thrown if FbasicType has cyclic definition.
     */
    public TypeList getTypeReferenceList(String typeName) throws XmException
    {
        TypeList typeList = new TypeList();

        if (XfUtil.isNullOrEmpty(typeName) != false) {
            // Return empty type list.
            return typeList;
        }

        IXbfTypeTableChoice typeChoice = findType(typeName);
        while (typeChoice != null)
        {
            typeList.addFirst(typeChoice);

            if (typeChoice instanceof XbfFbasicType) {
                XbfFbasicType basicType = (XbfFbasicType)typeChoice;
                String refType = basicType.getRef();

                if(XfType.DERIVED != XfType.getTypeIdFromXcodemlTypeName(refType))
                    break;

                typeChoice = findType(refType);

                if(typeList.contains(typeChoice))
                    throw new XmException("FbasicType" +
                                          basicType.getType() +
                                          "has cyclic definition");


            } else if (typeChoice instanceof XbfFstructType) {
                typeChoice = null;
            } else if (typeChoice instanceof XbfFfunctionType) {
                typeChoice = null;
            } else {
                // Impossible.
                assert (false);
            }
        }

        return typeList;
    }

    /**
     * Gets a type id of the last type of a type reference list.
     *
     * @param typeName the top type of the type reference list.
     * @return if bottomType is not found returns null,
     * else returns the last of the type reference list.
     */
    public String getBottomTypeName(String typeName)
    {
        if (typeName == null)
            throw new IllegalArgumentException();

        if (XfType.DERIVED != XfType.getTypeIdFromXcodemlTypeName(typeName))
            return typeName;

        TypeList typeList = null;

        try {
            typeList = getTypeReferenceList(typeName);
        } catch (XmException e) {
            return null;
        }

        if (typeList == null || typeList.size() == 0) {
            return null;
        }

        IXbfTypeTableChoice typeElem = typeList.getLast();

        if (typeElem instanceof XbfFbasicType) {
            XbfFbasicType basicTypeElem = (XbfFbasicType)typeElem;
            return basicTypeElem.getRef();
        } else if (typeElem instanceof XbfFstructType) {
            XbfFstructType structTypeElem = (XbfFstructType)typeElem;
            return structTypeElem.getType();
        } else if (typeElem instanceof XbfFfunctionType) {
            XbfFfunctionType functionTypeElem = (XbfFfunctionType)typeElem;
            return functionTypeElem.getType();
        }

        // not reached.
        return null;
    }

    /**
     * return if xcodemlTypeName matchs the type includes its reference.
     * 
     * @param xcodemlTypeName
     *      type name
     * @param type
     *      type
     * @return
     *      return true if matchs.
     */
    public boolean isTypeOf(String xcodemlTypeName, XfType type)
    {
        return type.xcodemlName().equals(getBottomTypeName(xcodemlTypeName));
    }

    /**
     * return if xcodemlTypeName is not the type which is able to be decompiled
     * as Fortran token.
     * 
     * @param xcodemlTypeName
     * @return
     */
    public boolean isDecompilableType(String xcodemlTypeName)
    {
        return isTypeOf(xcodemlTypeName, XfType.VOID) == false &&
            isTypeOf(xcodemlTypeName, XfType.NUMERIC) == false &&
            isTypeOf(xcodemlTypeName, XfType.NUMERICALL) == false;
    }

    @Override
    public String toString()
    {
        StringBuilder sb = new StringBuilder();
        sb.append(_typeMap.toString());
        sb.append(_aliasMapStack.toString());
        sb.append(_symbolMapStack.toString());
        return sb.toString();
    }
}

