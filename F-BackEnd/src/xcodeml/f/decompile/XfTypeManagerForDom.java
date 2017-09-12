/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
package xcodeml.f.decompile;

import java.util.*;

import org.w3c.dom.Node;

import xcodeml.util.XmException;
import xcodeml.util.XmDomUtil;

/**
 * Type and Symbol manager, for DOM node.
 */
class XfTypeManagerForDom {
    private TypeMap _typeMap;
    private SymbolMapStack _symbolMapStack;
    private AliasMapStack _aliasMapStack;
    private AliasMap _reverseBasicRefMap;
    private final AliasMapStack _typeToSymbolMapStack;

    /**
     * This map contains node set : { FbasicType, FfunctionType, FstructType }.
     */
    @SuppressWarnings("serial")
    private class TypeMap extends HashMap<String, Node>
    {
        @Override
        public String toString()
        {
            StringBuilder sb = new StringBuilder();
            sb.append("[TypeMap, key(type name) -> value(class)]\n");
            for (Map.Entry<String, Node> entry : entrySet()) {
                sb.append(entry.getKey());
                sb.append(" -> ");
                sb.append(entry.getValue().getNodeName());
                sb.append("\n");
            }
            return sb.toString();
        }
    }

    /** This map contains the node "id". */
    @SuppressWarnings("serial")
    public class SymbolMap extends HashMap<String, Node>
    {
        @Override
        public String toString()
        {
            StringBuilder sb = new StringBuilder();
            sb.append("[SymbolMap, key(symbol name) -> value(type name)]\n");
            for (Map.Entry<String, Node> entry : entrySet()) {
                sb.append(entry.getKey());
                sb.append(" -> ");
                sb.append(XmDomUtil.getAttr(entry.getValue(), "type"));
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

    /**
     * This list contains node set : { FbasicType, FfunctionType, FstructType }.
     */
    @SuppressWarnings("serial")
    public class TypeList extends LinkedList<Node>
    {
        @Override
        public String toString()
        {
            StringBuilder sb = new StringBuilder();
            sb.append("[TypeList, element name -> type name]\n");
            for (Node typeChoice : this) {
                sb.append(typeChoice.getNodeName());
                sb.append(" -> ");
                sb.append(XmDomUtil.getAttr(typeChoice, "type"));
                sb.append("\n");
            }
            return sb.toString();
        }

        public Node findChildNode(String nodeName) {
            Node n;
            for (Node typeChoice : this) {
                n = XmDomUtil.getElement(typeChoice, nodeName);
                if (n != null) {
                    return n;
                }
            }
            return null;
        }
    }

    public XfTypeManagerForDom()
    {
        _typeMap = new TypeMap();
        _symbolMapStack = new SymbolMapStack();
        _aliasMapStack = new AliasMapStack();
        _typeToSymbolMapStack = new AliasMapStack();
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

    private AliasMap _getCurrentTypeToSymbolMap()
    {
        return _typeToSymbolMapStack.peekFirst();
    }

    public void enterScope()
    {
        _symbolMapStack.push(new SymbolMap());
        _aliasMapStack.push(new AliasMap());
        _typeToSymbolMapStack.push(new AliasMap());
    }

    public void leaveScope()
    {
        _symbolMapStack.pop();
        _aliasMapStack.pop();
        _typeToSymbolMapStack.pop();
    }

    public void addSymbol(Node idNode)
    {
      // get name Node
        Node nameNode = XmDomUtil.getElement(idNode, "name");
        if (nameNode == null) {
            // Ignore invalid symbol name.
            return;
        }

        String symbolName = XmDomUtil.getContentText(nameNode);
        if (symbolName == null) {
            // Ignore invalid symbol name.
            return;
        }

        // Trim key word string.
        symbolName = symbolName.trim();
        if (symbolName.isEmpty()) {
            // Ignore invalid symbol name.
            return;
        }

        String sclass = XmDomUtil.getAttr(idNode, "sclass");
        if (sclass == null)
            return;

        if (XfStorageClass.FTYPE_NAME.toXcodeString().equalsIgnoreCase(sclass)) {
            String typeName = XmDomUtil.getAttr(idNode, "type");
            if (XfUtilForDom.isNullOrEmpty(typeName)) {
                // Ignore invalid type name.
                return;
            }
            AliasMap aliasMap = _getCurrentAliasMap();
            assert (aliasMap != null);
            aliasMap.put(typeName, symbolName);
        } else if (!XfStorageClass.FCOMMON_NAME.toXcodeString().equalsIgnoreCase(sclass) &&
                   !XfStorageClass.FNAMELIST_NAME.toXcodeString().equalsIgnoreCase(sclass)) {
            SymbolMap symbolMap = _getCurrentSymbolMap();
            assert (symbolMap != null);
            symbolMap.put(symbolName, idNode);
            AliasMap aliasMap = _getCurrentTypeToSymbolMap();
            String typeName = XmDomUtil.getAttr(idNode, "type");
            if (!XfUtilForDom.isNullOrEmpty(typeName)) {
                aliasMap.put(typeName, symbolName);
            }
        }
    }

    /** @return "id" DOM node. */
    public Node findSymbol(String symbolName)
    {
        Node id = null;
        symbolName = symbolName.trim();
        for (SymbolMap symbolMap : _symbolMapStack) {
            id = symbolMap.get(symbolName);
            if (id != null) {
                break;
            }
        }
        return id;
    }

    public void addType(Node typeNode)
    {
        String typeNameOrg = XmDomUtil.getAttr(typeNode, "type");
        if (XfUtilForDom.isNullOrEmpty(typeNameOrg) != false) {
            // Ignore invalid type name.
            return;
        }

        // Trim key word string.
        String typeName = typeNameOrg.trim();

        if (typeName.isEmpty() != false) {
            // Ignore invalid type name.
            return;
        }

        _typeMap.put(typeName, typeNode);

        String nodeName = typeNode.getNodeName();
        if (nodeName.equals("FbasicType")) {
            String ref = XmDomUtil.getAttr(typeNode, "ref");
            if (ref != null) {
                // Check '(XbfFbasicType)type).getContent() == null'
                ArrayList<Node> contentNodes =
                    XmDomUtil.collectElementsExclude(typeNode, "kind", "coShape");
                if (contentNodes.isEmpty()) {
                    _reverseBasicRefMap.put(ref, typeNameOrg);
                }
            }
        }
    }

    /** @return DOM node one of { FbasicType, FfunctionType, FstructType }. */
    public Node findType(String typeName)
    {
        if (XfUtilForDom.isNullOrEmpty(typeName) != false) {
            return null;
        }

        // Trim key word string.
        return _typeMap.get(typeName.trim());
    }

    public Node findType(Node nameNode)
    {
        if (nameNode == null) {
            return null;
        }

        String typeName = XmDomUtil.getAttr(nameNode, "type");
        if (XfUtilForDom.isNullOrEmpty(typeName) != false) {
            return findTypeFromSymbol(XmDomUtil.getContentText(nameNode));
        }
        return findType(typeName);
    }

    /**
     * Find type element from symbol name.
     * @param symbolName Symbol name.
     * @return  DOM node one of { FbasicType, FfunctionType, FstructType } or null.<br/>
     *          If null, type is not found.
     */
    public Node findTypeFromSymbol(String symbolName)
    {
        if (XfUtilForDom.isNullOrEmpty(symbolName) != false) {
            return null;
        }

        Node id = findSymbol(symbolName);
        if (id == null) {
            return null;
        }

        return findType(XmDomUtil.getAttr(id, "type"));
    }

    public String findNameFromType(String typeId)
    {
        if (typeId == null) {
            return null;
        }

        typeId = typeId.trim();
        for (AliasMap aliasMap : _typeToSymbolMapStack) {
            String aliasName = aliasMap.get(typeId);
            if (aliasName != null) {
                return aliasName;
            }
        }

        return null;
    }

    /**
     * Put tagname of type id.
     * @param typeName
     * @param typeId
     */
    public void putAliasTypeName(String typeId, String typeName)
    {
        if (XfUtilForDom.isNullOrEmpty(typeName)) {
            return;
        }

        if (XfUtilForDom.isNullOrEmpty(typeId)) {
            return;
        }

        AliasMap currentAlias = _getCurrentAliasMap();

        currentAlias.put(typeId, typeName);
    }

    /**
     * Get alias of type name.
     * @param typeId
     * @return When alias not found, return argument type name.
     */
    public String getAliasTypeName(String typeId)
    {
        if (XfUtilForDom.isNullOrEmpty(typeId) != false) {
            return null;
        }

        // Trim key word string.
        typeId = typeId.trim();
        for (AliasMap aliasMap : _aliasMapStack) {
            String aliasName = aliasMap.get(typeId);
            if (aliasName != null) {
                return aliasName;
            }
        }

        String inheritName = _reverseBasicRefMap.get(typeId);
        if (inheritName != null) {
            return getAliasTypeName(inheritName);
        }

        throw new IllegalStateException("not found type name of '" + typeId + "'");
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

        if (XfUtilForDom.isNullOrEmpty(typeName) != false) {
            // Return empty type list.
            return typeList;
        }

        Node typeChoice = findType(typeName);
        while (typeChoice != null) {
            typeList.addFirst(typeChoice);

            String name = typeChoice.getNodeName();
            if ("FbasicType".equals(name)) {
                Node basicType = typeChoice;

                String refType = XmDomUtil.getAttr(basicType, "ref");

                if (XmDomUtil.getAttrBool(basicType, "is_class") && XfUtilForDom.isNullOrEmpty(refType))
                    break;

                if (XmDomUtil.getAttrBool(basicType, "is_pointer") && XfUtilForDom.isNullOrEmpty(refType))
                    break;

                if (XmDomUtil.getAttrBool(basicType, "is_procedure") && XfUtilForDom.isNullOrEmpty(refType))
                    break;

                if (XfType.DERIVED != XfType.getTypeIdFromXcodemlTypeName(refType))
                    break;

                typeChoice = findType(refType);

                if (typeList.contains(typeChoice))
                    throw new XmException("FbasicType" +
                                          XmDomUtil.getAttr(basicType, "type") +
                                          "has cyclic definition");

            } else if ("FstructType".equals(name)) {
                typeChoice = null;
            } else if ("FfunctionType".equals(name)) {
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

        Node typeNode = typeList.getLast();
        String name = typeNode.getNodeName();
        if ("FbasicType".equals(name)) {
            return XmDomUtil.getAttr(typeNode, "ref");
        } else if ("FstructType".equals(name)) {
            return XmDomUtil.getAttr(typeNode, "type");
        } else if ("FfunctionType".equals(name)) {
            return XmDomUtil.getAttr(typeNode, "type");
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

    interface SymbolMatcher {
        boolean match(Node symbol, Node type);
    }

    public Set<String> findSymbolFromCurrentScope(SymbolMatcher matcher) {
        Set<String> set = new HashSet<String>();
        SymbolMap symbolMap = _getCurrentSymbolMap();
        for (String name: symbolMap.keySet()) {
            Node node = symbolMap.get(name);
            String typeName = XmDomUtil.getAttr(node, "type");
            if (typeName == null) {
                continue;
            }
            if (matcher.match(node, findType(typeName))) {
                set.add(name);
            }
        }
        return Collections.unmodifiableSet(set);
    }
}
