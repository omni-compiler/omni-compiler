/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
package xcodeml.f.decompile;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.LinkedList;

import org.w3c.dom.Document;
import org.w3c.dom.Node;
import org.w3c.dom.NodeList;

import java.io.PrintStream;
import java.io.StringWriter;
import javax.xml.transform.OutputKeys;
import javax.xml.transform.TransformerConfigurationException;
import javax.xml.transform.TransformerException;
import javax.xml.transform.TransformerFactory;
import javax.xml.transform.Transformer;
import javax.xml.transform.dom.DOMSource;
import javax.xml.transform.stream.StreamResult;
import com.sun.org.apache.xml.internal.serializer.OutputPropertiesFactory;

import xcodeml.XmException;
import xcodeml.f.util.XmfNodeVisitorMap;
import xcodeml.f.util.XmfWriter;
import xcodeml.f.util.XmfNodeVisitorMap.Pair;
import xcodeml.util.XmDomUtil;
import xcodeml.util.XmTranslationException;

/**
 * Decompiler of XcodeML/F DOM nodes.
 */
public class XfDecompileDomVisitor {
    static final int PRIO_LOW = 0; /* lowest */

    static final int PRIO_DEFINED_BINARY = 1; /* defined binary operation */

    static final int PRIO_EQV = 2; /* EQV, NEQV */
    static final int PRIO_OR = 3; /* .OR. */
    static final int PRIO_AND = 4; /* .AND. */
    static final int PRIO_NOT = 5; /* .NOT. */

    static final int PRIO_COMP = 6; /* <, >,...  */

    static final int PRIO_CONCAT = 7;

    static final int PRIO_PLUS_MINUS = 8;
    static final int PRIO_UNARY_MINUS = 9;
    static final int PRIO_MUL_DIV = 10;
    static final int PRIO_POWER = 11;
    static final int PRIO_DEFINED_UNARY = 12;
    static final int PRIO_HIGH = 13;

    static public String nodeToString(Node n) {
        String ret = null;
        if (n != null) {
            try {
                StringWriter w = new StringWriter();
                Transformer t = 
                    TransformerFactory.newInstance().newTransformer();
                t.setOutputProperty(OutputKeys.INDENT, "yes");
                t.setOutputProperty(
                    OutputPropertiesFactory.S_KEY_INDENT_AMOUNT, "" + 2);
                t.transform(new DOMSource(n),
                            new StreamResult(w));
                ret = w.toString();
            } catch (Exception e) {
                ;
            }
        }
        return ret;
    }

    static public void printNode(PrintStream fd, Node n) {
        String s = nodeToString(n);
        if (s != null) {
            fd.printf("%s\n", s);
        }
    }

    @SuppressWarnings("serial")
    private class InvokeNodeStack extends LinkedList<Node>
    {
        @Override
        public String toString()
        {
            StringBuilder sb = new StringBuilder();
            sb.append("[Invoke Node Stack]\n");
            for (Node node : this.toArray(new Node[0])) {
                sb.append(node.getNodeName());
                sb.append("\n");
            }
            return sb.toString();
        }
    }

    @SuppressWarnings("unchecked")
    public XfDecompileDomVisitor(XmfDecompilerContext context) {
        _context = context;
        _invokeNodeStack = new InvokeNodeStack();
        _validator = new XfRuntimeDomValidator();
        visitorMap = new XmfNodeVisitorMap<XcodeNodeVisitor>(pairs);
    }

    public void invokeEnter(Document doc) throws XmException {
        try {
            _enter(doc.getDocumentElement());
        } catch (XmTranslationException e) {
            throw new XmException(e);
        }
    }

    private void _writeDeclAttr(Node top, Node low) {
        String topName = top.getNodeName();
        String lowName = low.getNodeName();

        if (("FbasicType".equals(topName)) &&
            ("FbasicType".equals(lowName))) {
            _writeBasicTypeAttr(top, low);
            return;
        }

        if ("FbasicType".equals(topName)) {
            _writeBasicTypeAttr(top);
        }

        if ("FbasicType".equals(lowName)) {
           _writeBasicTypeAttr(low);
        }
    }

    private void _writeBasicTypeAttr(Node ... basicTypeNodeArray) {
        if (basicTypeNodeArray == null) {
            return;
        }

        XmfWriter writer = _context.getWriter();

        /* public, private are allowed only in module definition */
        if (_isUnderModuleDef()) {
            for (Node basicTypeNode : basicTypeNodeArray) {
                if (XmDomUtil.getAttrBool(basicTypeNode, "is_public")) {
                    writer.writeToken(", ");
                    writer.writeToken("PUBLIC");
                    break;
                }
            }

            for (Node basicTypeNode : basicTypeNodeArray) {
                if (XmDomUtil.getAttrBool(basicTypeNode, "is_private")) {
                    writer.writeToken(", ");
                    writer.writeToken("PRIVATE");
                    break;
                }
            }
        }

        for (Node basicTypeNode : basicTypeNodeArray) {
            if (XmDomUtil.getAttrBool(basicTypeNode, "is_pointer")) {
                writer.writeToken(", ");
                writer.writeToken("POINTER");
                break;
            }
        }

        for (Node basicTypeNode : basicTypeNodeArray) {
            if (XmDomUtil.getAttrBool(basicTypeNode, "is_target")) {
                writer.writeToken(", ");
                writer.writeToken("TARGET");
                break;
            }
        }

        for (Node basicTypeNode : basicTypeNodeArray) {
            if (XmDomUtil.getAttrBool(basicTypeNode, "is_optional")) {
                writer.writeToken(", ");
                writer.writeToken("OPTIONAL");
                break;
            }
        }

        for (Node basicTypeNode : basicTypeNodeArray) {
            if (XmDomUtil.getAttrBool(basicTypeNode, "is_save")) {
                writer.writeToken(", ");
                writer.writeToken("SAVE222");
                break;
            }
        }

        for (Node basicTypeNode : basicTypeNodeArray) {
            if (XmDomUtil.getAttrBool(basicTypeNode, "is_parameter")) {
                writer.writeToken(", ");
                writer.writeToken("PARAMETER");
                break;
            }
        }

        for (Node basicTypeNode : basicTypeNodeArray) {
            if (XmDomUtil.getAttrBool(basicTypeNode, "is_allocatable")) {
                writer.writeToken(", ");
                writer.writeToken("ALLOCATABLE");
                break;
            }
        }

        for (Node basicTypeNode : basicTypeNodeArray) {  // (ID=60)
            if (XmDomUtil.getAttrBool(basicTypeNode, "is_cray_pointer")) {
                writer.writeToken(", ");
                writer.writeToken("CRAYPOINTER222");
                break;
            }
        }

        for (Node basicTypeNode : basicTypeNodeArray) {
            String intent = XmDomUtil.getAttr(basicTypeNode, "intent");
            if (XfUtilForDom.isNullOrEmpty(intent) == false) {
                writer.writeToken(", ");
                writer.writeToken("INTENT(" + intent.toUpperCase() + ")");
                break;
            }
        }
    }

    private void _writeFunctionSymbol(XfSymbol symbol,
                                      Node funcTypeNode,
                                      Node node) {
        XfTypeManagerForDom typeManager = _context.getTypeManagerForDom();
        XmfWriter writer = _context.getWriter();
        Node lowType = null;

        if (XmDomUtil.getAttrBool(funcTypeNode, "is_intrinsic")) {
            writer.writeToken("INTRINSIC ");
            writer.writeToken(symbol.getSymbolName());
            return;
        }

        boolean isFirstToken = true;
        boolean isPrivateEmit = false;
        boolean isPublicEmit = false;

        /* - always type declaration for SUBROUTINE must not be output.
         * - type declaration for FUNCTION under MODULE must not be output.
         */
        String returnTypeName = XmDomUtil.getAttr(funcTypeNode, "return_type");
        if (typeManager.isDecompilableType(returnTypeName) &&
                (_isUnderModuleDef() == false ||
                 XmDomUtil.getAttrBool(funcTypeNode, "is_external"))) {

            XfType type = XfType.getTypeIdFromXcodemlTypeName(returnTypeName);
            if (type.isPrimitive()) {
                writer.writeToken(type.fortranName());
            } else {
                XfTypeManagerForDom.TypeList typeList = getTypeList(returnTypeName);
                assert typeList != null;

                lowType = typeList.getLast();
                Node topType = typeList.getFirst();

                if ("FbasicType".equals(lowType.getNodeName())) {
                    isPublicEmit = XmDomUtil.getAttrBool(lowType, "is_public");
                    isPrivateEmit = XmDomUtil.getAttrBool(lowType, "is_private");
                }

                String topTypeName = topType.getNodeName();
                if ("FbasicType".equals(topTypeName)) {
                    isPublicEmit = XmDomUtil.getAttrBool(topType, "is_public");
                    isPrivateEmit = XmDomUtil.getAttrBool(topType, "is_private");
                    _writeBasicType(topType, typeList);
                } else if ("FstructType".equals(topTypeName)) {
                    String aliasStructTypeName =
                        typeManager.getAliasTypeName(XmDomUtil.getAttr(topType,
                                                                       "type"));
                    writer.writeToken("TYPE(" + aliasStructTypeName + ")");
                } else {
                    /* topType is FfunctionType. */
                    throw new XmTranslationException(node,
                                                     "Top type must be a FfunctionType.");
                }

                _writeDeclAttr(topType, lowType);
            }
            isFirstToken = false;
        }

        if (_isUnderModuleDef()) {
            if (XmDomUtil.getAttrBool(funcTypeNode, "is_public") && isPublicEmit == false) {
                writer.writeToken((isFirstToken ? "" : ", ") + "PUBLIC");
                isFirstToken = false;
            } else if (XmDomUtil.getAttrBool(funcTypeNode, "is_private") && isPrivateEmit == false) {
                writer.writeToken((isFirstToken ? "" : ", ") + "PRIVATE");
                isFirstToken = false;
            }
        }

        if (isFirstToken == false) {
            writer.writeToken(" :: ");
            writer.writeToken(symbol.getSymbolName());

            if (lowType != null &&
                ("FbasicType".equals(lowType.getNodeName()))) {
                ArrayList<Node> contentNodes =
                    XmDomUtil.collectElementsExclude(lowType,
                                                     "kind", "len", "coShape");
                if (!contentNodes.isEmpty()) {
                    _writeIndexRangeArray(contentNodes);
                }
            }
        }

        if (XmDomUtil.getAttrBool(funcTypeNode, "is_external")) {
            if (isFirstToken == false)
                writer.setupNewLine();
            writer.writeToken("EXTERNAL ");
            writer.writeToken(symbol.getSymbolName());
        }
    }

    private void _writeBasicType(Node basicTypeNode,
                                 XfTypeManagerForDom.TypeList typeList) {
        String refName = XmDomUtil.getAttr(basicTypeNode, "ref");
        XfType refTypeId = XfType.getTypeIdFromXcodemlTypeName(refName);
        assert refTypeId != null;

        if (refTypeId.isPrimitive() == false) {
            _context.debugPrint(
                "Top level type is basic-type, but is not primitive type. (%s)%n", refName);
            if (typeList != null)
                _context.debugPrintLine(typeList.toString());
            _context.setLastErrorMessage(
                XfUtilForDom.formatError(basicTypeNode,
                                         XfError.XCODEML_TYPE_MISMATCH,
                                         "top-level FbasicType",
                                         refName,
                                         "primitive type"));
            fail(basicTypeNode);
        }

        XmfWriter writer = _context.getWriter();
        writer.writeToken(refTypeId.fortranName());

        Node lenNode = XmDomUtil.getElement(basicTypeNode, "len");
        if (lenNode != null) {
            if (refTypeId != XfType.CHARACTER) {
                _context.debugPrint(
                    "A 'len' element is included in a definition of '%s' type.%n",
                    refTypeId.xcodemlName());
                _context.setLastErrorMessage(
                    XfUtilForDom.formatError(lenNode,
                                             XfError.XCODEML_SEMANTICS,
                                             lenNode.getNodeName()));
                fail(lenNode);
            }
        }
        Node kindNode = XmDomUtil.getElement(basicTypeNode, "kind");
        _writeTypeParam(kindNode, lenNode);
    }

    private XfTypeManagerForDom.TypeList getTypeList(String type) {
        XfTypeManagerForDom typeManager = _context.getTypeManagerForDom();
        XfTypeManagerForDom.TypeList typeList = null;

        try {
            typeList = typeManager.getTypeReferenceList(type);
        } catch (XmException e) {
            _context.debugPrintLine(e.toString());
            Node n = _invokeNodeStack.peek();
            _context.setLastErrorMessage(
                XfUtilForDom.formatError(n,
                                         XfError.XCODEML_CYCLIC_TYPE,
                                         type));
            fail(n);
        }

        if (typeList == null || typeList.isEmpty()) {
            _context.debugPrintLine("Type list is empty.");
            Node n = _invokeNodeStack.peek();
            _context.setLastErrorMessage(
                XfUtilForDom.formatError(n,
                                         XfError.XCODEML_TYPE_NOT_FOUND,
                                         type));
            fail(n);
        }

        return typeList;
    }

    /**
     * Write variable declaration.
     *
     * @param symbol
     *            Variable symbol.
     * @example <div class="Example"> PROGRAM main <div class="Indent1"><div
     *          class="Strong"> INTEGER :: int_variable<br/>
     *          TYPE(USER_TYPE) :: derived_variable </div> int_variable = 0
     *          </div> END PROGRAM main </div>
     */
    private void _writeSymbolDecl(XfSymbol symbol, Node node) {
        if (symbol == null) {
            throw new IllegalArgumentException();
        }

        XfType typeId = symbol.getTypeId();
        if (typeId.isPrimitive()) {
            _writeSimplePrimitiveSymbolDecl(symbol);
            return;
        }

        XfTypeManagerForDom typeManager = _context.getTypeManagerForDom();
        XfTypeManagerForDom.TypeList typeList = getTypeList(symbol.getDerivedName());
        assert typeList != null;

        /*
         * The assumption that typeList.size() <= 2 is not valid for now.
         * m-hirano
         */
//        if (typeList.size() > 2) {
//            _context.debugPrintLine("Type list count > 2.");
//            _context.debugPrintLine(typeList.toString());
//            _context.setLastErrorMessage(XfUtil.formatError(_invokeNodeStack.peek(),
//                XfError.XCODEML_SEMANTICS, XfUtil.getElementName(_invokeNodeStack.peek())));
//            return false;
//        }

        Node topTypeChoice = typeList.getFirst();
        Node lowTypeChoice = typeList.getLast();

        XmfWriter writer = _context.getWriter();

        // ================
        // Top type element
        // ================
        String topTypeName = topTypeChoice.getNodeName();
        if ("FbasicType".equals(topTypeName)) {
            _writeBasicType(topTypeChoice, typeList);
        } else if ("FstructType".equals(topTypeName)) {
            String aliasStructTypeName =
                typeManager.getAliasTypeName(XmDomUtil.getAttr(topTypeChoice,
                                                               "type"));
            writer.writeToken("TYPE(" + aliasStructTypeName + ")");
        } else if ("FfunctionType".equals(topTypeName)) {
            _writeFunctionSymbol(symbol, topTypeChoice, node);
        }

        _writeDeclAttr(topTypeChoice, lowTypeChoice);

        // ================
        // Low type element
        // ================
        String lowTypeName = lowTypeChoice.getNodeName();
        if ("FbasicType".equals(lowTypeName)) {
            Node basicTypeNode = lowTypeChoice;
            String refName = XmDomUtil.getAttr(basicTypeNode, "ref");
            XfType refTypeId = XfType.getTypeIdFromXcodemlTypeName(refName);
            assert refTypeId != null;

            writer.writeToken(" :: ");
            writer.writeToken(symbol.getSymbolName());

            ArrayList<Node> contentNodes =
                XmDomUtil.collectElementsExclude(basicTypeNode,
                                                 "kind", "len", "coShape");
            if (!contentNodes.isEmpty()) {
                _writeIndexRangeArray(contentNodes);
            }

            Node coShapeNode = XmDomUtil.getElement(basicTypeNode, "coShape");
            if (coShapeNode != null){
                invokeEnter(coShapeNode);
            }

        } else if ("FstructType".equals(lowTypeName)) {
            writer.writeToken(" :: ");
            writer.writeToken(symbol.getSymbolName());
        }
    }

    /**
     * Call enter method of node.
     *
     * @param nodeArray
     *            IRNode array.
     */
    private void _invokeEnter(ArrayList<Node> nodes) {
        Node currentNode = null;

        if (nodes == null) {
            // Succeed forcibly.
            return;
        }

        for (Node node : nodes) {
            if (_validator.validateAttr(node) == false) {
                _context.debugPrintLine("Detected insufficient attributes");
                _context.setLastErrorMessage(_validator.getErrDesc());
                fail(node);
            }

            _nextNode = node;

            if (currentNode != null) {
                invokeEnter(currentNode);
            }

            currentNode = node;
        }

        _nextNode = null;
        invokeEnter(currentNode);
    }

    /**
     * Call enter method of node, and write delimiter.
     *
     * @param nodeArray
     *            IRNode array.
     * @param delim
     *            Delimiter.
     * @return true/false
     */
    private void _invokeEnterAndWriteDelim(ArrayList<Node> nodes, String delim) {
        if (nodes == null) {
            // Succeed forcibly.
            return;
        }

        XmfWriter writer = _context.getWriter();

        int nodeCount = 0;
        for (Node node : nodes) {
            if (nodeCount > 0) {
                writer.writeToken(delim);
            }
            invokeEnter(node);
            ++nodeCount;
        }
    }

    /**
     * Call enter method of child node, and write delimiter.
     *
     * @param nodeArray
     *            IRNode array.
     * @param delim
     *            Delimiter.
     */
    private void _invokeChildEnterAndWriteDelim(Node node, String delim) {
        if (node == null) {
            // Succeed forcibly.
            return;
        }

        XmfWriter writer = _context.getWriter();

        NodeList list = node.getChildNodes();
        int nodeCount = 0;
        for (int i = 0; i < list.getLength(); i++) {
            Node childNode = list.item(i);
            if (childNode.getNodeType() != Node.ELEMENT_NODE) {
                continue;
            }
            if (nodeCount > 0) {
                writer.writeToken(delim);
            }
            invokeEnter(childNode);
            ++nodeCount;
        }
    }

    /**
     * Call enter method of child node.
     *
     * @param node    DOM node.
     */
    private void _invokeChildEnter(Node node) {
        if (node == null) {
            // Succeed forcibly.
            return;
        }

        Node currentNode = null;

        NodeList list = node.getChildNodes();
        for (int i = 0; i < list.getLength(); i++) {
            Node childNode = list.item(i);
            if (childNode.getNodeType() != Node.ELEMENT_NODE) {
                continue;
            }
            if (_validator.validateAttr(childNode) == false) {
                _context.debugPrintLine("Detected insufficient attributes");
                _context.setLastErrorMessage(_validator.getErrDesc());
                fail(childNode);
            }

            _nextNode = childNode;

            if (currentNode != null) {
                invokeEnter(currentNode);
            }

            currentNode = childNode;
        }

        _nextNode = null;
        invokeEnter(currentNode);
    }

    /**
     * Call enter method of node.
     *
     * @param node    DOM node.
     */
    private void invokeEnter(Node node) {
        if (node == null) {
            // Succeed forcibly.
            return;
        }

        _invokeNodeStack.push(node);
        _preEnter(node);
        _enter(node);
        _postEnter(node);
        _invokeNodeStack.pop();
    }

    /**
     * Get DOM node instance in the invoke stack.
     *
     * @param parentRank
     *            Parent rank.
     *            <ul>
     *            <li>0: Current node.</li>
     *            <li>1: Parent node.</li>
     *            <li>>2: Any ancestor node.</li>
     *            </ul>
     * @return Instance of DOM node or null.
     */
    private Node _getInvokeNode(int parentRank) {
        if (parentRank < 0) {
            throw new IllegalArgumentException();
        }

        if (parentRank >= _invokeNodeStack.size()) {
            return null;
        }

        return _invokeNodeStack.get(parentRank);
    }

    /**
     * Check whether there is the class of designated element to a ancestor of
     * the invoke stack.
     *
     * @param nodeName
     *            Node name.
     * @param parentRank
     *            Parent rank.
     * @return true/false
     */
    private boolean _isInvokeNodeOf(String nodeName, int parentRank) {
        Node node = _getInvokeNode(parentRank);
        return nodeName.equals(node.getNodeName());
    }

    /**
     * return if current context is under FmoduleDefinition's grandchild
     * <br>ex 1) return true
     * <br>FmoduleModuleDefinition
     * <br>+declarations
     * <br> +current
     * <br>
     * <br>ex 2) return false
     * <br>FmoduleModuleDefinition
     * <br>+declarations
     * <br> +FfunctionDefinition
     * <br>  +declarations
     * <br>   +current
     *
     * @return
     *      true if the current context is undef FmoduleDefinition's grandchild
     */
    private boolean _isUnderModuleDef() {
        return _isInvokeNodeOf("FmoduleDefinition", 2);
    }

    /**
     * Check whether there is the class of designated element to a ancestor of
     * the invoke stack.
     *
     * @param nodeName   Name of the node
     * @return true/false
     */
    private boolean _isInvokeAncestorNodeOf(String nodeName) {
        assert nodeName != null;

        Node node = null;
        for (Iterator<Node> it = _invokeNodeStack.iterator(); it.hasNext();) {
            node = it.next();
            if (nodeName.equals(node.getNodeName())) {
                return true;
            }
        }
        return false;
    }

    /**
     * Preprocessing of enter method.
     *
     * @param node    DOM node.
     */
    private void _preEnter(Node node) {
        if (_context.isDebugMode()) {
            _context.debugPrintLine(String.format("%100s", "").subSequence(0,
                (_invokeNodeStack.size() - 1) * 2)
                + "<" + node.getNodeName() + ">");
        }
    }

    /**
     * Postprocessing of enter method.
     *
     * @param node    DOM node.
     */
    private void _postEnter(Node node) {
        if (_context.isDebugMode()) {
            _context.debugPrintLine(String.format("%100s", " ").subSequence(0,
                (_invokeNodeStack.size() - 1) * 2)
                + "</" + node.getNodeName() + ">");
        }
    }

    /**
     * Checks if object represents a constant expression.
     *
     * @param node   DOM node.
     * @return true if node represents a constant expression.
     */
    private boolean _isConstantExpr(Node node) {
        if ((node.getParentNode().getNodeName().equals("unaryMinusExpr") == false) &&
            (node.getNodeName().equals("unaryMinusExpr"))) {
            node = XmDomUtil.collectChildNodes(node).get(0);
        }

        String nodeName = node.getNodeName();
        if ((nodeName.equals("FintConstant")) ||
            (nodeName.equals("FlogicalConstant")) ||
            (nodeName.equals("FcharacterConstant")) ||
            (nodeName.equals("FrealConstant")) ||
            (nodeName.equals("FcomplexConstant")) ||
            (nodeName.equals("value")))
           return true;
        else
            return false;
    }

    /**
     * Make internal symbol from symbol name and type name.
     *
     * @param symbolName
     *            Symbol name.
     * @param typeName
     *            Type name.
     * @return Instance of XfSymbol.
     */
    private XfSymbol _makeSymbol(String symbolName, String typeName) {
        if (XfUtil.isNullOrEmpty(symbolName)) {
            // Symbol name is empty.
            return null;
        }

        XfTypeManagerForDom typeManager = _context.getTypeManagerForDom();

        if (XfUtil.isNullOrEmpty(typeName)) {
            Node idNode = typeManager.findSymbol(symbolName);
            if (idNode == null) {
                // Symbol not found.
                return null;
            }
            typeName = XmDomUtil.getAttr(idNode, "type");
            if (XfUtil.isNullOrEmpty(typeName)) {
                // Type name of symbol is empty.
                return null;
            }
        }

        if (symbolName.equals("**") ||
            symbolName.equals("*") ||
            symbolName.equals("/") ||
            symbolName.equals("+") ||
            symbolName.equals("-") ||
            symbolName.equals("//")) {
            symbolName = "OPERATOR(" + symbolName + ")";
        } else if (symbolName.equals("=")) {
            symbolName = "ASSIGNMENT(" + symbolName + ")";
        } else if (symbolName.startsWith(".") && symbolName.endsWith(".")) {
            symbolName = "OPERATOR(" + symbolName + ")";
        }

        XfSymbol symbol = null;
        XfType typeId = XfType.getTypeIdFromXcodemlTypeName(typeName);
        if (typeId.isPrimitive()) {
            symbol = new XfSymbol(symbolName, typeId);
        } else {
            symbol = new XfSymbol(symbolName, typeId, typeName);
        }

        return symbol;
    }

    /**
     * Make internal symbol from name element.
     *
     * @param nameElem
     *            Instance of XbfName.
     * @return Instance of XfSymbol.
     */
    private XfSymbol _makeSymbol(Node nameNode) {
        if (nameNode == null) {
            // Instance is null.
            return null;
        }

        String symbolName = XmDomUtil.getContentText(nameNode);
        return _makeSymbol(symbolName, XmDomUtil.getAttr(nameNode, "type"));
    }

    private void _writeLineDirective(Node node) {
        _writeLineDirective(XmDomUtil.getAttr(node, "lineno"),
                            XmDomUtil.getAttr(node, "file"));
    }

    /**
     * Write line directive.
     *
     * @param lineNumber
     *            Line number.
     * @param filePath
     *            File path.
     */
    private void _writeLineDirective(String lineNumber, String filePath) {
        if (_context.isOutputLineDirective() &&
            lineNumber != null) {
            XmfWriter writer = _context.getWriter();
            if (filePath == null)
                writer.writeIsolatedLine(String.format("# %s", lineNumber));
            else
                writer.writeIsolatedLine(String.format("# %s \"%s\"", lineNumber, filePath));
        }
    }

    /**
     * Write simple primitive symbol declaration.
     *
     * @param symbol
     *            Instance of XfSymbol.
     */
    private void _writeSimplePrimitiveSymbolDecl(XfSymbol symbol) {
        XmfWriter writer = _context.getWriter();
        writer.writeToken(symbol.getTypeId().fortranName());
        writer.writeToken(" :: ");
        writer.writeToken(symbol.getSymbolName());
    }

    /**
     * Write kind and length in character declaration.
     *
     * @param symbol
     *            Instance of XfSymbol.
     */
    private void _writeTypeParam(Node kindNode, Node lenNode) {
        if (kindNode == null && lenNode == null) {
            // Succeed forcibly.
            return;
        }

        XmfWriter writer = _context.getWriter();
        writer.writeToken("(");

        if (lenNode != null) {
            writer.writeToken("LEN=");
            invokeEnter(lenNode);
        }

        if (kindNode != null) {
            if (lenNode != null) {
                writer.writeToken(", ");
            }
            writer.writeToken("KIND=");
            invokeEnter(kindNode);
        }
        writer.writeToken(")");
    }

    /**
     * Write index ranges of array.
     *
     * @param indexRangeArray
     * @example <div class="Example"> INTEGER value<span class="Strong">(10,
     *          1:20)</span> </div>
     */
    private void _writeIndexRangeArray(ArrayList<Node> indexRangeArray) {
        if (indexRangeArray == null) {
            // Succeed forcibly.
            return;
        }

        XmfWriter writer = _context.getWriter();
        writer.writeToken("(");

        int indexRangeCount = 0;

        for (Node arraySubscriptNode : indexRangeArray) {
            if (indexRangeCount > 0) {
                writer.writeToken(", ");
            }

            String name = arraySubscriptNode.getNodeName();
            if (name.equals("indexRange")
                || name.equals("arrayIndex")) {
                invokeEnter(arraySubscriptNode);
            } else {
                _context
                    .debugPrintLine("Detected discontinuous 'indexRange' or 'arrayIndex' element.");
                _context.setLastErrorMessage(
                    XfUtilForDom.formatError(arraySubscriptNode,
                                             XfError.XCODEML_SEMANTICS, name));
                fail(arraySubscriptNode);
            }
            ++indexRangeCount;
        }
        writer.writeToken(")");
    }

    /**
     * Write unary expression.
     *
     * @param expr
     *            One of defModelExpr's choice.
     * @param operation
     *            Operation string.
     * @param grouping
     *            Grouping flag.
     */
    private void _writeUnaryExpr(Node expr, String operation, boolean grouping) {
        XmfWriter writer = _context.getWriter();

        if (grouping == true) {
            writer.writeToken("(");
        }

        writer.writeToken(operation);

        invokeEnter(expr);

        if (grouping == true) {
            writer.writeToken(")");
        }
    }

    /**
     * Write binary expression.
     *
     * @param leftExpr
     *            One of defModelExpr's choice.
     * @param rightExpr
     *            One of defModelExpr's choice.
     * @param operation
     *            Operation string.
     * @param grouping
     *            Grouping flag.
     */
    private void _writeBinaryExpr(Node leftExpr, Node rightExpr,
                                  String operation, boolean grouping) {
        XmfWriter writer = _context.getWriter();
        boolean need_paren;
        int op_prio = operator_priority(operation);

        if (grouping) writer.writeToken("(");

        need_paren = false;
        if(op_prio > operator_priority(leftExpr))
            need_paren = true;

        if (need_paren) writer.writeToken("(");
        invokeEnter(leftExpr);
        if (need_paren) writer.writeToken(")");

        writer.writeToken(" ");
        writer.writeToken(operation);
        writer.writeToken(" ");

        need_paren = false;
        if(op_prio == PRIO_POWER ||
                op_prio >= operator_priority(rightExpr))
            need_paren = true;

        if (need_paren) writer.writeToken("(");
        invokeEnter(rightExpr);
        if (need_paren) writer.writeToken(")");
        if (grouping) writer.writeToken(")");
    }

    int operator_priority(String operator){

        if (operator.equals("=") || operator.equals("=>"))
            return PRIO_LOW;

        if (operator.equals("-") || operator.equals("+"))
            return PRIO_PLUS_MINUS;
        if (operator.equals("*") || operator.equals("/"))
            return PRIO_MUL_DIV;
        if (operator.equals("**"))
            return PRIO_POWER;

        if (operator.equals("<") || operator.equals(">") ||
                operator.equals("<=") || operator.equals(">=") ||
                operator.equals("/=") || operator.equals("=="))
            return PRIO_COMP;

        if (operator.equals("//")) return PRIO_CONCAT;

        if (operator.equals(".NOT.")) return PRIO_NOT;
        if (operator.equals(".AND.")) return PRIO_AND;
        if (operator.equals(".OR."))  return PRIO_OR;
        if (operator.equals(".NEQV.") || operator.equals(".EQV."))
            return PRIO_EQV;

        if(operator.startsWith(".") && operator.endsWith("."))
            return PRIO_DEFINED_BINARY;

        return PRIO_HIGH;
    }

    int operator_priority(Node expr){

        String name = expr.getNodeName();

        if (name.equals("FassignStatement") ||
                name.equals("FpointerAssignStatement")) return PRIO_LOW;

        if (name.equals("userBinaryExpr")) return PRIO_DEFINED_BINARY;

        if (name.equals("plusExpr") || name.equals("minusExpr"))
            return PRIO_PLUS_MINUS;
        if (name.equals("unaryMinusExpr"))
            return PRIO_UNARY_MINUS;
        if (name.equals("divExpr") || name.equals("mulExpr"))
            return PRIO_MUL_DIV;
        if (name.equals("FpowerExpr"))
            return PRIO_POWER;

        if (name.equals("logLTExpr") || name.equals("logGTExpr") ||
                name.equals("logLEExpr") || name.equals("logGEExpr") ||
                name.equals("logEQExpr") || name.equals("logNEQExpr"))
            return PRIO_COMP;

        if (name.equals("FconcatExpr")) return PRIO_CONCAT;

        if (name.equals("logNotExpr")) return PRIO_NOT;
        if (name.equals("logAndExpr")) return PRIO_AND;
        if (name.equals("logOrExpr"))  return PRIO_OR;
        if (name.equals("logEQVExpr") ||name.equals("logNEQVExpr"))
            return PRIO_EQV;

        if (name.equals("userUnaryExpr")) return PRIO_DEFINED_BINARY;

        return PRIO_HIGH;
    }

    private boolean _checkBinaryExprRequireGrouping(Node expr) {
        Node parent = expr.getParentNode();

        String name = parent.getNodeName();

        if (!(name.equals("unaryMinusExpr"))
            && !(name.equals("userUnaryExpr"))
            && !(name.equals("logNotExpr")))
            return false;

        return operator_priority(parent) > operator_priority(expr);
    }

    /**
     * Write binary expression.
     *
     * @param exprNodes
     *            Nodes these includes left and right expression nodes.
     * @param offset
     *            The offset within exprNodes of the left expression node.
     * @param operation
     *            Operation string.
     * @param grouping
     *            Grouping flag.
     */
    private void _writeBinaryExpr(ArrayList<Node> exprNodes, int offset,
                                  String operation, boolean grouping) {
        _writeBinaryExpr(exprNodes.get(offset),
                         exprNodes.get(offset + 1),
                         operation, grouping);
    }

    /**
     * Dispatch the node to the node visitor.
     *
     * @param node    DOM node.
     */
    private void _enter(Node node) {
        XcodeNodeVisitor visitor = visitorMap.getVisitor(node.getNodeName());
        if (visitor == null) {
            throw new XmTranslationException(node, "Unknown node");
        }
        visitor.enter(node);
    }

    private void fail(Node node) {
        throw new XmTranslationException(node,
                                         _context.getLastErrorMessage(),
                                         _context.getLastCause());
    }

    private XmfDecompilerContext _context;

    private InvokeNodeStack _invokeNodeStack;

    private XfRuntimeDomValidator _validator;

    private Node _nextNode;

    private XmfNodeVisitorMap<XcodeNodeVisitor> visitorMap;


    abstract class XcodeNodeVisitor {
        public abstract void enter(Node n);
    }

    // XcodeProgram
    class XcodeProgramVisitor extends XcodeNodeVisitor {
        /**
         * Decompile 'XcodeProgram' element in XcodeML/F.
         *
         * @see xcodeml.f.binding.gen.RVisitorBase#enter(xcodeml.f.binding.gen.
         *      XbfXcodeProgram)
         */
        @Override public void enter(Node n) {
            XfTypeManagerForDom typeManager = _context.getTypeManagerForDom();

            // for global symbol
            typeManager.enterScope();

            _invokeChildEnter(n);

            typeManager.leaveScope();
        }
    }

    // typeTable
    class TypeTableVisitor extends XcodeNodeVisitor {
        /**
         * Decompile "typeTable" element in XcodeML/F.
         * <p>
         * The decompilation result depends on a child element.
         * </p>
         *
         * @see xcodeml.f.binding.gen.RVisitorBase#enter(xcodeml.f.binding.gen.XbfTypeTable
         *      )
         */
        @Override public void enter(Node n) {
            _invokeChildEnter(n);
        }
    }

    // FbasicType
    class BasicTypeVisitor extends XcodeNodeVisitor {
        /**
         * Decompile "FbasicType" element in XcodeML/F.
         *
         * @see xcodeml.f.binding.gen.RVisitorBase#enter(xcodeml.f.binding
         *      .gen.XbfFbasicType )
         */
        @Override public void enter(Node n) {
            // Note:
            // Because handle it at a upper level element,
            // warn it when this method was called it.
            //assert(_isInvokeAncestorNodeOf(XbfTypeTable.class));
            XfTypeManagerForDom typeManager = _context.getTypeManagerForDom();
            typeManager.addType(n);
        }
    }

    // coShape
    class CoShapeVisitor extends XcodeNodeVisitor {
        /**
         * Decompile child group of "FbasicType" element in XcodeML/F.
         *
         * @see xcodeml.f.binding.gen.RVisitorBase#enter(xcodeml.f.binding.gen.
         *      XbfCoShape)
         */
        @Override public void enter(Node n) {
            _invokeChildEnter(n);
        }
    }

    // FfunctionType
    class FfunctionTypeVisitor extends XcodeNodeVisitor {
        /**
         * Decompile "FfunctionType" element in XcodeML/F.
         *
         * @see xcodeml.f.binding.gen.RVisitorBase#enter(xcodeml.f.binding.gen.
         *      XbfFfunctionType)
         */
        @Override public void enter(Node n) {
            if (_isInvokeAncestorNodeOf("typeTable")) {
                XfTypeManagerForDom typeManager = _context.getTypeManagerForDom();
                typeManager.addType(n);
            } else {
                // Note:
                // Because handle it at a upper level element,
                // warn it when this method was called it.
                assert false;
            }
        }
    }

    // FstructType
    class FstructTypeVisitor extends XcodeNodeVisitor {
        /**
         * Decompile "FstructType" element in XcodeML/F.
         * <p>
         * The decompilation result depends on a child element.
         * </p>
         *
         * @see xcodeml.f.binding.gen.RVisitorBase#enter(xcodeml.f.binding.gen.XbfFstructType
         *      )
         */
        @Override public void enter(Node n) {
            if (_isInvokeAncestorNodeOf("typeTable")) {
                XfTypeManagerForDom typeManager = _context.getTypeManagerForDom();
                typeManager.addType(n);
            } else {
                invokeEnter(XmDomUtil.getElement(n, "symbols"));
            }
        }
    }

    // globalSymbols
    class GlobalSymbolsVisitor extends XcodeNodeVisitor {
        /**
         * Decompile "globalSymbols" element in XcodeML/F.
         *
         * @see xcodeml.f.binding.gen.RVisitorBase#enter(xcodeml.f.binding.gen.
         *      XbfGlobalSymbols)
         */
        @Override public void enter(Node n) {
            XfTypeManagerForDom typeManager = _context.getTypeManagerForDom();
            for (Node idElem : XmDomUtil.collectElements(n, "id")) {
                typeManager.addSymbol(idElem);
            }
        }
    }

    // globalDeclarations
    class GlobalDeclarationsVisitor extends XcodeNodeVisitor {
        /**
         * Decompile "globalDeclarations" element in XcodeML/F.
         * <p>
         * The decompilation result depends on a child element.
         * </p>
         *
         * @see xcodeml.f.binding.gen.RVisitorBase#enter(xcodeml.f.binding.gen.
         *      XbfGlobalDeclarations)
         */
        @Override public void enter(Node n) {
            _invokeChildEnter(n);
        }
    }

    // alloc
    class AllocVisitor extends XcodeNodeVisitor {
        /**
         * Decompile "alloc" element in XcodeML/F.
         *
         * @example <code><div class="Example">
         *      ALLOCATE (<span class="Strong">ptr</span>, <span class="Strong">ptr_array(10)</span>, STAT = error)<br/>
         *      NULLIFY (<span class="Strong">ptr</span>, <span class="Strong">ptr_array</span>)<br/>
         *      DEALLOCATE (<span class="Strong">ptr</span>, <span class="Strong">ptr_array</span>, STAT = error)<br/>
         * </div></code>
         * @see xcodeml.f.binding.gen.RVisitorBase#enter(xcodeml.f.binding.gen.XbfAlloc)
         */
        @Override public void enter(Node n) {
            // Process the content.
            Node contentNode = XmDomUtil.getElement(n, "Var");
            if (contentNode != null) {
                invokeEnter(contentNode);
            } else {
                contentNode = XmDomUtil.getElement(n, "FmemberRef");
                if (contentNode != null) {
                    invokeEnter(contentNode);
                } else {
                    throw new XmTranslationException(n, "Invalid `alloc' content.");
                }
            }

            // Parent node is XbfFallocateStatement?
            if (_isInvokeNodeOf("FallocateStatement", 1)) {
                ArrayList<Node> arraySubscripts =
                    XmDomUtil.collectElements(n, "indexRange", "arrayIndex");
                if ((arraySubscripts != null) && (arraySubscripts.size() > 0)) {
                    _writeIndexRangeArray(arraySubscripts);
                }
            }

            Node coShapeNode = XmDomUtil.getElement(n, "coShape");
            if (coShapeNode != null) {
                invokeEnter(coShapeNode);
            }
        }
    }

    // arguments
    class ArgumentsVisitor extends XcodeNodeVisitor {
        /**
         * Decompile "arguments" element in XcodeML/F.
         *
         * @example <code><div class="Example">
         *      variable = function(<span class="Strong">arg1, arg2</span>)<br/>
         *      call subroutine(<span class="Strong">arg1, arg2</span>)<br/>
         * </div></code>
         *
         * @see xcodeml.f.binding.gen.RVisitorBase#enter(xcodeml.f.binding.gen.XbfArguments
         *      )
         */
        @Override public void enter(Node n) {
            _invokeChildEnterAndWriteDelim(n, ", ");
        }
    }

    // arrayIndex
    class ArrayIndexVisitor extends XcodeNodeVisitor {
        /**
         * Decompile "arrayIndex" element in XcodeML/F.
         *
         * @example <code><div class="Example">
         *      array = int_array_variable(<span class="Strong">10</span>,
         *      1:10, 1:, :)<br/>
         * </div></code>
         * @see xcodeml.f.binding.gen.RVisitorBase#enter(xcodeml.f.binding.gen.XbfArrayIndex
         *      )
         */
        @Override public void enter(Node n) {
            invokeEnter(XmDomUtil.getContent(n));
        }
    }

    // FassignStatement
    class FassignStatementVisitor extends XcodeNodeVisitor {
        /**
         * Decompile "FassignStatement" element in XcodeML/F.
         * <p>
         * The decompilation result depends on a child element.
         * </p>
         *
         * @example <code><div class="Example">
         * PROGRAM main<br/>
         * <div class="Indent1">
         *      INTEGER :: int_variable<br/>
         *      <div class="Strong">
         *      int_variable = 0<br/>
         *      </div>
         *      (any expression statement ...)<br/>
         * </div>
         * END PROGRAM main<br/>
         * </div></code>
         * @see xcodeml.f.binding.gen.RVisitorBase#enter(xcodeml.f.binding.gen.
         *      XbfFassignStatement)
         */
        @Override public void enter(Node n) {
            XmfWriter writer = _context.getWriter();

            _writeLineDirective(n);

            _writeBinaryExpr(XmDomUtil.collectChildNodes(n), 0,
                             "=", false);

            writer.setupNewLine();
        }
    }

    // body
    class BodyVisitor extends XcodeNodeVisitor {
        /**
         * Decompile "body" element in XcodeML/F.
         * <p>
         * The decompilation result depends on a child element.
         * </p>
         *
         * @example <code><div class="Example">
         * PROGRAM main<br/>
         * <div class="Indent1">
         *      INTEGER :: int_variable<br/>
         *      TYPE(USER_TYPE) :: derived_variable<br/>
         *      <div class="Strong">
         *      int_variable = 0<br/>
         *      (any statement...)<br/>
         *      </div>
         * </div>
         * END PROGRAM main<br/>
         * </div></code>
         * @see xcodeml.f.binding.gen.RVisitorBase#enter(xcodeml.f.binding.gen.XbfBody)
         */
        @Override public void enter(Node n) {
            _invokeChildEnter(n);
        }
    }

    // condition
    class ConditionVisitor extends XcodeNodeVisitor {
        /**
         * Decompile "condition" element in XcodeML/F.
         *
         * @example <code><div class="Example">
         * IF <span class="Strong">(variable == 1)</span> THEN<br/>
         * <div class="Indent1">
         *      (any statement...)<br/>
         * </div>
         * ELSE<br/>
         * <div class="Indent1">
         *      (any statement...)<br/>
         * </div>
         * END IF<br/>
         * </div></code>
         * @see xcodeml.f.binding.gen.RVisitorBase#enter(xcodeml.f.binding.gen.XbfCondition
         *      )
         */
        @Override public void enter(Node n) {
            XmfWriter writer = _context.getWriter();
            writer.writeToken("(");

            invokeEnter(XmDomUtil.getContent(n));

            writer.writeToken(")");
        }
    }

    // continueStatement
    class ContinueStatement extends XcodeNodeVisitor {
        /**
         * Decompile "ContinueStatement" element in XcodeML/F.
         *
         * @example <code><div class="Example">
         * DO, variable = 1, 10<br/>
         * <div class="Indent1">
         *      (any statement...)<br/>
         *      <div class="Strong">
         *      CONTINUE<br/>
         *      </div>
         * </div>
         * END DO<br/>
         * </div></code>
         * @see xcodeml.f.binding.gen.RVisitorBase#enter(xcodeml.f.binding.gen.
         *      XbfContinueStatement)
         */
        @Override public void enter(Node n) {
            _writeLineDirective(n);

            XmfWriter writer = _context.getWriter();
            writer.writeToken(" CONTINUE");
            writer.setupNewLine();
        }
    }

    // declarations
    class DeclarationsVisitor extends XcodeNodeVisitor {
        /**
         * Decompile "Declarations" element in XcodeML/F.
         * <p>
         * The decompilation result depends on a child element.
         * </p>
         *
         * @see xcodeml.f.binding.gen.RVisitorBase#enter(xcodeml.f.binding.gen.XbfDeclarations)
         */
        @Override public void enter(Node n) {
            _invokeChildEnter(n);
        }
    }

    // divExpr
    class DivExprVisitor extends XcodeNodeVisitor {
        /**
         * Decompile "divExpr" element in XcodeML/F.
         * <p>
         * The decompilation result depends on a child element.
         * </p>
         *
         * @example <code><div class="Example">
         * variable = <span class="Strong">(&lt;any expression&gt; / &lt;any expression&gt;)</span><br/>
         * </div></code>
         * @see xcodeml.f.binding.gen.RVisitorBase#enter(xcodeml.f.binding.gen.XbfDivExpr
         *      )
         */
        @Override public void enter(Node n) {
            _writeBinaryExpr(XmDomUtil.collectChildNodes(n), 0,
                             "/", _checkBinaryExprRequireGrouping(n));
        }
    }

    // else
    class ElseVisitor extends XcodeNodeVisitor {
        /**
         * Decompile "else" element in XcodeML/F.
         *
         * @example <code><div class="Example">
         * IF (variable == 1) THEN<br/>
         * <div class="Indent1">
         *      (any statement...)<br/>
         * </div>
         * ELSE<br/>
         * <div class="Indent1">
         *      <div class="Strong">
         *      (any statement...)<br/>
         *      </div>
         * </div>
         * END IF<br/>
         * </div></code>
         * @see xcodeml.f.binding.gen.RVisitorBase#enter(xcodeml.f.binding.gen.XbfElse)
         */
        @Override public void enter(Node n) {
            XmfWriter writer = _context.getWriter();
            writer.incrementIndentLevel();

            invokeEnter(XmDomUtil.getElement(n, "body"));

            writer.decrementIndentLevel();
        }
    }

    // exprStatement
    class ExprStatementVisitor extends XcodeNodeVisitor {
        /**
         * Decompile "exprStatement" element in XcodeML/F.
         * <p>
         * The decompilation result depends on a child element.
         * </p>
         *
         * @example <code><div class="Example">
         * PROGRAM main<br/>
         * <div class="Indent1">
         *      INTEGER :: int_variable<br/>
         *      TYPE(USER_TYPE) :: derived_variable<br/>
         *      <div class="Strong">
         *      int_variable = 0<br/>
         *      (any expression statement...)<br/>
         *      </div>
         * </div>
         * END PROGRAM main<br/>
         * </div></code>
         * @see xcodeml.f.binding.gen.RVisitorBase#enter(xcodeml.f.binding.gen.
         *      XbfExprStatement)
         */
        @Override public void enter(Node n) {
            _writeLineDirective(n);

            _invokeChildEnter(n);

            XmfWriter writer = _context.getWriter();
            writer.setupNewLine();
        }
    }

    // externDecl
    class ExternDeclVisitor extends XcodeNodeVisitor {
        /**
         * Decompile "externDecl" element in XcodeML/F.
         *
         * @example <code><div class="Example">
         * <div class="Strong">
         * EXTERNAL function_name<br/>
         * </div>
         * </div></code>
         * @see xcodeml.f.binding.gen.RVisitorBase#enter(xcodeml.f.binding.gen.XbfExternDecl
         *      )
         */
        @Override public void enter(Node n) {
            _writeLineDirective(n);

            Node nameElem = XmDomUtil.getElement(n, "name");
            String externalName = XmDomUtil.getContentText(nameElem);

            XmfWriter writer = _context.getWriter();
            writer.writeToken("EXTERNAL ");
            writer.writeToken(externalName);
            writer.setupNewLine();
        }
    }

    // FallocateStatement
    class FallocateStatement extends XcodeNodeVisitor {
        /**
         * Decompile "FallocateStatement" element in XcodeML/F.
         *
         * @example <code><div class="Example">
         *      <div class="Strong">
         *      ALLOCATE (ptr, ptr_array(10), STAT = error)<br/>
         *      </div>
         * </div></code>
         * @see xcodeml.f.binding.gen.RVisitorBase#enter(xcodeml.f.binding.gen.
         *      XbfFallocateStatement)
         */
        @Override public void enter(Node n) {
            _writeLineDirective(n);

            XmfWriter writer = _context.getWriter();
            writer.writeToken("ALLOCATE (");

            ArrayList<Node> allocNodes = XmDomUtil.collectElements(n, "alloc");
            _invokeEnterAndWriteDelim(allocNodes, ", ");

            String statName = XmDomUtil.getAttr(n, "stat_name");
            if (XfUtil.isNullOrEmpty(statName) == false) {
                writer.writeToken(", ");
                writer.writeToken("STAT = ");
                writer.writeToken(statName);
            }

            writer.writeToken(")");
            writer.setupNewLine();
        }
    }

    // FarrayConstructor
    class FarrayConstructor extends XcodeNodeVisitor {
        /**
         * Decompile "FarrayConstructor" element in XcodeML/F.
         *
         * @example <code><div class="Example">
         *      array = <span class="Strong">(/ 1, 2, (I, I = 1, 10, 2) /)</span><br/>
         * </div></code>
         * @see xcodeml.f.binding.gen.RVisitorBase#enter(xcodeml.f.binding.gen.
         *      XbfFarrayConstructor)
         */
        @Override public void enter(Node n) {
            XmfWriter writer = _context.getWriter();
            writer.writeToken("(/ ");

            _invokeChildEnterAndWriteDelim(n, ", ");

            writer.writeToken(" /)");
        }
    }

    // FarrayRef
    class FarrayRefVisitor extends XcodeNodeVisitor {
        /**
         * Decompile "FarrayRef" element in XcodeML/F.
         *
         * @example <code><div class="Example">
         *      array = <span class="Strong">int_array_variable(10, 1:10, 1:, :)</span><br/>
         * </div></code>
         * @see xcodeml.f.binding.gen.RVisitorBase#enter(xcodeml.f.binding.gen.XbfFarrayRef
         *      )
         */
        @Override public void enter(Node n) {
            invokeEnter(XmDomUtil.getElement(n, "varRef"));

            XmfWriter writer = _context.getWriter();
            writer.writeToken("(");

            ArrayList<Node> contentNodes =
                XmDomUtil.collectElements(n,
                                          "indexRange",
                                          "arrayIndex",
                                          "FarrayConstructor",
                                          "FarrayRef");
            _invokeEnterAndWriteDelim(contentNodes, ", ");

            writer.writeToken(")");
        }
    }

    // FcoArrayRef
    class FcoArrayRefVisitor extends XcodeNodeVisitor {
        /**
         * Decompile "FcoArrayRef" element in XcodeML/F.
         *
         * @example <code><div class="Example">
         *      array = <span class="Strong">int_coarray_variable[10, 1:10, 1:, *]</span><br/>
         * </div></code>
         * @see xcodeml.f.binding.gen.RVisitorBase#enter(xcodeml.f.binding.gen.XbfFcoArrayRef
         *      )
         */
        @Override public void enter(Node n) {
            invokeEnter(XmDomUtil.getElement(n, "varRef"));

            XmfWriter writer = _context.getWriter();
            writer.writeToken("[");

            ArrayList<Node> arrayIndexNodes =
                XmDomUtil.collectElements(n, "arrayIndex");
            _invokeEnterAndWriteDelim(arrayIndexNodes, ", ");

            writer.writeToken("]");
        }
    }

    // FbackspaceStatement
    class FbackspaceStatement extends XcodeNodeVisitor {
        /**
         * Decompile "FbackspaceStatement" element in XcodeML/F.
         *
         * @example <code><div class="Example">
         * <div class="Strong">
         * BACKSPACE (UNIT=1, ...)<br/>
         * </div>
         * </div></code>
         * @see xcodeml.f.binding.gen.RVisitorBase#enter(xcodeml.f.binding.gen.
         *      XbfFbackspaceStatement)
         */
        @Override public void enter(Node n) {
            _writeLineDirective(n);

            XmfWriter writer = _context.getWriter();

            writer.writeToken("BACKSPACE ");
            invokeEnter(XmDomUtil.getElement(n, "namedValueList"));

            writer.setupNewLine();
        }
    }

    // FcaseLabel
    class FcaseLabelVisitor extends XcodeNodeVisitor {
        /**
         * Decompile "FcaseLabel" element in XcodeML/F.
         *
         * @example <code><div class="Example">
         * SELECT CASE (variable)<br/>
         * <div class="Strong">
         * CASE (1)<br/>
         * <div class="Indent1">
         *      (any statement...)<br/>
         * </div>
         * </div>
         * <div class="Strong">
         * CASE (2)<br/>
         * <div class="Indent1">
         *      (any statement...)<br/>
         * </div>
         * </div>
         * <div class="Strong">
         * CASE DEFAULT<br/>
         * <div class="Indent1">
         *      (any statement...)<br/>
         * </div>
         * </div>
         * END SELECT<br/>
         * </div></code>
         * @see xcodeml.f.binding.gen.RVisitorBase#enter(xcodeml.f.binding.gen.XbfFcaseLabel
         *      )
         */
        @Override public void enter(Node n) {
            _writeLineDirective(n);

            XmfWriter writer = _context.getWriter();

            writer.writeToken("CASE");
            ArrayList<Node> caseLabelContents =
                XmDomUtil.collectElementsExclude(n, "body");
            if ((caseLabelContents != null) && (caseLabelContents.size() > 0)) {
                writer.writeToken(" (");
                _invokeEnterAndWriteDelim(caseLabelContents, ", ");
                writer.writeToken(")");
            } else {
                writer.writeToken(" DEFAULT");
            }

            String constuctName = XmDomUtil.getAttr(n, "construct_name");
            if (XfUtil.isNullOrEmpty(constuctName) == false) {
                writer.writeToken(" ");
                writer.writeToken(constuctName);
            }

            writer.setupNewLine();
            writer.incrementIndentLevel();

            invokeEnter(XmDomUtil.getElement(n, "body"));

            writer.decrementIndentLevel();
        }
    }

    // FcharacterConstant
    class FcharacterConstant extends XcodeNodeVisitor {
        /**
         * Decompile "FcharacterConstant" element in XcodeML/F.
         *
         * @example <code><div class="Example">
         * PROGRAM main<br/>
         * <div class="Indent1">
         *      CHARACTER(LEN=10) :: string_variable = <span class="Strong">"text"</span><br/>
         * </div>
         * END PROGRAM main<br/>
         * </div></code>
         *
         * @see xcodeml.f.binding.gen.RVisitorBase#enter(xcodeml.f.binding.gen.
         *      XbfFcharacterConstant)
         */
        @Override public void enter(Node n) {
            XmfWriter writer = _context.getWriter();

            String kind = XmDomUtil.getAttr(n, "kind");
            if (XfUtil.isNullOrEmpty(kind) == false) {
                writer.writeToken(kind + "_");
            }

            writer.writeLiteralString(XmDomUtil.getContentText(n));
        }
    }

    // FcharacterRef
    class FcharacterRef extends XcodeNodeVisitor {
        /**
         * Decompile "FcharacterRef" element in XcodeML/F.
         *
         * @example <code><div class="Example">
         *      substring = <span class="Strong">char_variable(1:10)</span><br/>
         * </div></code>
         * @see xcodeml.f.binding.gen.RVisitorBase#enter(xcodeml.f.binding.gen.
         *      XbfFcharacterRef)
         */
        @Override public void enter(Node n) {
            invokeEnter(XmDomUtil.getElement(n, "varRef"));

            XmfWriter writer = _context.getWriter();
            writer.writeToken("(");

            invokeEnter(XmDomUtil.getElement(n, "indexRange"));

            writer.writeToken(")");
        }
    }

    // FcloseStatement
    class FcloseStatementVisitor extends XcodeNodeVisitor {
        /**
         * Decompile "FcloseStatement" element in XcodeML/F.
         *
         * @example <code><div class="Example">
         * <div class="Strong">
         * CLOSE (UNIT=1, ...)<br/>
         * </div>
         * </div></code>
         * @see xcodeml.f.binding.gen.RVisitorBase#enter(xcodeml.f.binding.gen.
         *      XbfFcloseStatement)
         */
        @Override public void enter(Node n) {
            _writeLineDirective(n);

            XmfWriter writer = _context.getWriter();

            writer.writeToken("CLOSE ");
            invokeEnter(XmDomUtil.getElement(n, "namedValueList"));

            writer.setupNewLine();
        }
    }

    // FcommonDecl
    class FcommonDeclVisitor extends XcodeNodeVisitor {
        /**
         * Decompile "FcommonDecl" element in XcodeML/F.
         *
         * @example <code><div class="Example">
         * <div class="Strong">
         * COMMON /NAME/ variable1, array, // variable3, variable4<br/>
         * </div>
         * </div></code>
         * @see xcodeml.f.binding.gen.RVisitorBase#enter(xcodeml.f.binding.gen.XbfFcommonDecl
         *      )
         */
        @Override public void enter(Node n) {
            _writeLineDirective(n);

            XmfWriter writer = _context.getWriter();
            writer.writeToken("COMMON ");

            _invokeChildEnterAndWriteDelim(n, ", ");

            writer.setupNewLine();
        }
    }

    // FcomplexConstant
    class FcomplexConstantVisitor extends XcodeNodeVisitor {
        /**
         * Decompile "FcomplexConstant" element in XcodeML/F.
         *
         * @example <code><div class="Example">
         * PROGRAM main<br/>
         * <div class="Indent1">
         *      COMPLEX cmp_variable = <span class="Strong">(1.0, 2.0)</span><br/>
         * </div>
         * END PROGRAM main<br/>
         * </div></code>
         * @see xcodeml.f.binding.gen.RVisitorBase#enter(xcodeml.f.binding.gen.
         *      XbfFcomplexConstant)
         */
        @Override public void enter(Node n) {
            ArrayList<Node> childNodes = XmDomUtil.collectChildNodes(n);

            if (childNodes.size() < 2) {
                _context.setLastErrorMessage(
                    XfUtilForDom.formatError(n,
                                       XfError.XCODEML_SEMANTICS,
                                       n.getNodeName()));
                fail(n);
            }
            String typeName = XmDomUtil.getAttr(n, "type");
            if (XfUtil.isNullOrEmpty(typeName) == false) {
                XfTypeManagerForDom typeManager = _context.getTypeManagerForDom();
                String bottomTypeName = typeManager.getBottomTypeName(typeName);
                if (bottomTypeName == null) {
                    _context.setLastErrorMessage(
                        XfUtilForDom.formatError(n,
                                                 XfError.XCODEML_TYPE_NOT_FOUND,
                                                 n.getNodeName(),
                                                 typeName));
                    fail(n);
                }

                XfType typeId = XfType.getTypeIdFromXcodemlTypeName(bottomTypeName);
                if (typeId != XfType.DERIVED && typeId != XfType.COMPLEX) {
                    _context.setLastErrorMessage(
                        XfUtilForDom.formatError(n,
                                                 XfError.XCODEML_TYPE_MISMATCH,
                                                 n.getNodeName(), typeName,
                                                 "Fcomplex"));
                    fail(n);
                }
            }

            XmfWriter writer = _context.getWriter();

            Node realPart = childNodes.get(0);
            Node imaginalPart = childNodes.get(1);

            if ((_isConstantExpr(realPart) == false) ||
               (_isConstantExpr(imaginalPart) == false)) {
                writer.writeToken("CMPLX");
            }

            writer.writeToken("(");
            invokeEnter(realPart);

            writer.writeToken(", ");
            invokeEnter(imaginalPart);

            writer.writeToken(")");
        }
    }

    // FconcatExpr
    class FconcatExprVisitor extends XcodeNodeVisitor {
        /**
         * Decompile "FconcatExpr" element in XcodeML/F.
         * <p>
         * The decompilation result depends on a child element.
         * </p>
         *
         * @example <code><div class="Example">
         * variable = <span class="Strong">(&lt;any expression&gt; // &lt;any expression&gt;)</span><br/>
         * </div></code>
         * @see xcodeml.f.binding.gen.RVisitorBase#enter(xcodeml.f.binding.gen.XbfFconcatExpr
         *      )
         */
        @Override public void enter(Node n) {
            _writeBinaryExpr(XmDomUtil.collectChildNodes(n), 0,
                             "//", _checkBinaryExprRequireGrouping(n));
        }
    }

    // FcontainsStatement
    class FcontainsStatementVisitor extends XcodeNodeVisitor {
        /**
         * Decompile "FcontainsStatement" element in XcodeML/F.
         *
         * @example <code><div class="Example">
         * PROGRAM main<br/>
         * <div class="Indent1">
         *      INTEGER :: int_variable<br/>
         *      (any statement...)<br/>
         * </div>
         * <br/>
         * <div class="Strong">
         * CONTAINS<br/>
         * </div>
         * <div class="Indent1">
         *      SUBROUTINE sub()<br/>
         *      (any statement...)<br/>
         *      END SUBROUTINE sub<br/>
         *      <br/>
         *      FUNCTION func()<br/>
         *      (any statement...)<br/>
         *      END SUBROUTINE sub<br/>
         * </div>
         * <br/>
         * END PROGRAM main<br/>
         * </div></code>
         * @see xcodeml.f.binding.gen.RVisitorBase#enter(xcodeml.f.binding.gen.
         *      XbfFcontainsStatement)
         */
        @Override public void enter(Node n) {
            _writeLineDirective(n);

            XmfWriter writer = _context.getWriter();
            writer.decrementIndentLevel();
            writer.setupNewLine();
            writer.writeToken("CONTAINS");
            writer.setupNewLine();
            writer.incrementIndentLevel();

            _invokeEnter(XmDomUtil.collectChildNodes(n));
        }
    }

    // FcycleStatement
    class FcycleStatementVisitor extends XcodeNodeVisitor {
        /**
         * Decompile "FcycleStatement" element in XcodeML/F.
         *
         * @example <code><div class="Example">
         * DO_NAME: DO, variable = 1, 10<br/>
         * <div class="Indent1">
         *      (any statement...)<br/>
         *      <div class="Strong">
         *      CYCLE DO_NAME<br/>
         *      </div>
         * </div>
         * END DO DO_NAME<br/>
         * </div></code>
         * @see xcodeml.f.binding.gen.RVisitorBase#enter(xcodeml.f.binding.gen.
         *      XbfFcycleStatement)
         */
        @Override public void enter(Node n) {
            _writeLineDirective(n);

            XmfWriter writer = _context.getWriter();
            writer.writeToken("CYCLE");

            String constuctName = XmDomUtil.getAttr(n, "construct_name");
            if (XfUtil.isNullOrEmpty(constuctName) == false) {
                writer.writeToken(" ");
                writer.writeToken(constuctName);
            }

            writer.setupNewLine();
        }
    }

    // FdataDecl
    class FdataDeclVisitor extends XcodeNodeVisitor {
        /**
         * Decompile "FdataDecl" element in XcodeML/F.
         *
         * @example <code><div class="Example">
         * <div class="Strong">
         * DATA variable1, variable2 /2*0/, &<br/>
         *      array1 /10*1/, &<br/>
         *      (array2(i), i = 1, 10, 2) /5*1/<br/>
         * </div>
         * </div></code>
         * @see xcodeml.f.binding.gen.RVisitorBase#enter(xcodeml.f.binding.gen.XbfFdataDecl
         *      )
         */
        @Override public void enter(Node n) {
            _writeLineDirective(n);

            XmfWriter writer = _context.getWriter();
            writer.writeToken("DATA ");

            ArrayList<Node> childNodes = XmDomUtil.collectChildNodes(n);
            int childCount = 0;
            //for (Node childNode : childNodes) {
            for (Iterator<Node> iter = childNodes.iterator(); iter.hasNext(); ) {
                Node varListNode = iter.next();
                if (!"varList".equals(varListNode.getNodeName())) {
                    throw new XmTranslationException(n,
                                                     "Invalid contents");
                }
                if (!iter.hasNext()) {
                    throw new XmTranslationException(n,
                                                     "Invalid contents");
                }
                Node valueListNode = iter.next();
                if (!"valueList".equals(valueListNode.getNodeName())) {
                    throw new XmTranslationException(n,
                                                     "Invalid contents");
                }

                invokeEnter(varListNode);
                writer.writeToken(" /");
                invokeEnter(valueListNode);
                writer.writeToken("/");

                if (childCount > 0) {
                    writer.writeToken(", ");
                }
                ++childCount;
            }
            writer.setupNewLine();
        }
    }

    // FdeallocateStatement
    class FdeallocateStatementVisitor extends XcodeNodeVisitor {
        /**
         * Decompile "FdeallocateStatement" element in XcodeML/F.
         *
         * @example <code><div class="Example">
         *      <div class="Strong">
         *      DEALLOCATE (ptr, ptr_array, STAT = error)<br/>
         *      </div>
         * </div></code>
         * @see xcodeml.f.binding.gen.RVisitorBase#enter(xcodeml.f.binding.gen.
         *      XbfFdeallocateStatement)
         */
        @Override public void enter(Node n) {
            _writeLineDirective(n);

            XmfWriter writer = _context.getWriter();
            writer.writeToken("DEALLOCATE (");

            ArrayList<Node> allocNodes = XmDomUtil.collectElements(n, "alloc");
            _invokeEnterAndWriteDelim(allocNodes, ", ");

            String statName = XmDomUtil.getAttr(n, "stat_name");
            if (XfUtil.isNullOrEmpty(statName) == false) {
                writer.writeToken(", ");
                writer.writeToken("STAT = ");
                writer.writeToken(statName);
            }

            writer.writeToken(")");
            writer.setupNewLine();
        }
    }

    // FdoLoop
    class FdoLoopVisitor extends XcodeNodeVisitor {
        /**
         * Decompile "FdoLoop" element in XcodeML/F.
         *
         * @example <code><div class="Example">
         *      array = (/ 1, 2, <span class="Strong">(I, I = 1, 10, 2)</span> /)<br/>
         * </div></code>
         * @see xcodeml.f.binding.gen.RVisitorBase#enter(xcodeml.f.binding.gen.XbfFdoLoop
         *      )
         */
        @Override public void enter(Node n) {
            XmfWriter writer = _context.getWriter();
            writer.writeToken("(");

            ArrayList<Node> valueNodes = XmDomUtil.collectElements(n, "value");
            _invokeEnterAndWriteDelim(valueNodes, ", ");

            if (valueNodes.size() > 0) {
                writer.writeToken(", ");
            }

            invokeEnter(XmDomUtil.getElement(n, "Var"));

            writer.writeToken(" = ");
            invokeEnter(XmDomUtil.getElement(n, "indexRange"));

            writer.writeToken(")");
        }
    }

    // FdoStatement
    class FdoStatementVisitor extends XcodeNodeVisitor {
        /**
         * Decompile "FdoStatement" element in XcodeML/F.
         *
         * @example <code><div class="Example">
         * <div class="Strong">
         * DO<br/>
         * <div class="Indent1">
         *      (any statement...)<br/>
         * </div>
         * END DO<br/>
         * </div>
         * <br/>
         * <div class="Strong">
         * DO, variable = 1, 10<br/>
         * <div class="Indent1">
         *      (any statement...)<br/>
         * </div>
         * END DO<br/>
         * </div>
         * </div></code>
         * @see xcodeml.f.binding.gen.RVisitorBase#enter(xcodeml.f.binding.gen.
         *      XbfFdoStatement)
         */
        @Override public void enter(Node n) {
            _writeLineDirective(n);

            XmfWriter writer = _context.getWriter();
            String constuctName = XmDomUtil.getAttr(n, "construct_name");
            if (XfUtil.isNullOrEmpty(constuctName) == false) {
                writer.writeToken(constuctName);
                writer.writeToken(": ");
            }

            writer.writeToken("DO");

            Node varNode = XmDomUtil.getElement(n, "Var");
            if (varNode != null) {
                writer.writeToken(" ");
                invokeEnter(varNode);
                writer.writeToken(" = ");
                invokeEnter(XmDomUtil.getElement(n, "indexRange"));
            }

            writer.setupNewLine();
            writer.incrementIndentLevel();

            invokeEnter(XmDomUtil.getElement(n, "body"));

            writer.decrementIndentLevel();

            writer.writeToken("END DO");
            if (XfUtil.isNullOrEmpty(constuctName) == false) {
                writer.writeToken(" ");
                writer.writeToken(constuctName);
            }
            writer.setupNewLine();
        }
    }

    // FdoWhileStatement
    class FdoWhileStatementVisitor extends XcodeNodeVisitor {
        /**
         * Decompile "FdoWhileStatement" element in XcodeML/F.
         *
         * @example <code><div class="Example">
         * <div class="Strong">
         * DO, WHILE (variable > 0)<br/>
         * <div class="Indent1">
         *      (any statement...)<br/>
         * </div>
         * END DO<br/>
         * </div>
         * </div></code>
         * @see xcodeml.f.binding.gen.RVisitorBase#enter(xcodeml.f.binding.gen.
         *      XbfFdoWhileStatement)
         */
        @Override public void enter(Node n) {
            _writeLineDirective(n);

            XmfWriter writer = _context.getWriter();
            String constuctName = XmDomUtil.getAttr(n, "construct_name");
            if (XfUtil.isNullOrEmpty(constuctName) == false) {
                writer.writeToken(constuctName);
                writer.writeToken(": ");
            }

            writer.writeToken("DO, WHILE ");
            invokeEnter(XmDomUtil.getElement(n, "condition"));

            writer.setupNewLine();
            writer.incrementIndentLevel();

            invokeEnter(XmDomUtil.getElement(n, "body"));

            writer.decrementIndentLevel();

            writer.writeToken("END DO");
            if (XfUtil.isNullOrEmpty(constuctName) == false) {
                writer.writeToken(" ");
                writer.writeToken(constuctName);
            }
            writer.setupNewLine();
        }
    }

    // FendFileStatement
    class FendFileStatementVisitor extends XcodeNodeVisitor {
        /**
         * Decompile "FendFileStatement" element in XcodeML/F.
         *
         * @example <code><div class="Example">
         * <div class="Strong">
         * ENDFILE (UNIT=1, ...)<br/>
         * </div>
         * </div></code>
         * @see xcodeml.f.binding.gen.RVisitorBase#enter(xcodeml.f.binding.gen.
         *      XbfFendFileStatement)
         */
        @Override public void enter(Node n) {
            _writeLineDirective(n);

            XmfWriter writer = _context.getWriter();

            writer.writeToken("ENDFILE ");
            invokeEnter(XmDomUtil.getElement(n, "namedValueList"));

            writer.setupNewLine();
        }
    }

    // FentryDecl
    class FentryDeclVisitor extends XcodeNodeVisitor {
        /**
         * Decompile "FentryDecl" element in XcodeML/F.
         *
         * @example <code><div class="Example">
         * FUNCTION func(arg) RESULT(retval)
         * <div class="Indent1">
         *      (any declaration...)<br/>
         * </div>
         * <div class="Strong">
         * ENTRY func_entry(arg) RESULT(retval)<br/>
         * </div>
         * <div class="Indent1">
         *      (any statement...)<br/>
         * </div>
         * END FUNCTION func<br/>
         * </div></code>
         * @see xcodeml.f.binding.gen.RVisitorBase#enter(xcodeml.f.binding.gen.XbfFentryDecl
         *      )
         */
        @Override public void enter(Node n) {
            _writeLineDirective(n);

            XfTypeManagerForDom typeManager = _context.getTypeManagerForDom();
            XmfWriter writer = _context.getWriter();

            Node functionNameNode = XmDomUtil.getElement(n, "name");
            Node typeChoice = typeManager.findType(functionNameNode);
            if (typeChoice == null) {
                _context.setLastErrorMessage(
                    XfUtilForDom.formatError(n,
                                       XfError.XCODEML_TYPE_NOT_FOUND,
                                       XmDomUtil.getAttr(functionNameNode, "type")));
                fail(n);
            } else if ("FfunctionType".equals(typeChoice.getNodeName()) == false) {
                _context.setLastErrorMessage(
                    XfUtilForDom.formatError(n,
                                       XfError.XCODEML_TYPE_MISMATCH,
                                       "function definition",
                                       typeChoice.getNodeName(),
                                       "FfunctionType"));
                fail(n);
            }

            Node functionTypeNode = typeChoice;
            String returnTypeName = XmDomUtil.getAttr(functionTypeNode,
                                                      "return_type");
            XfType typeId = XfType.getTypeIdFromXcodemlTypeName(returnTypeName);
            if (XmDomUtil.getAttrBool(functionTypeNode, "is_program")) {
                // =======
                // PROGRAM
                // =======
                _context.setLastErrorMessage(
                    XfUtilForDom.formatError(n,
                                       XfError.XCODEML_TYPE_MISMATCH,
                                       "function definition",
                                       "PROGRAM",
                                       "FUNCTION or SUBROUTINE"));
                fail(n);
            } else {
                // ======================
                // FUNCTION or SUBROUTINE
                // ======================
                writer.decrementIndentLevel();
                writer.writeToken("ENTRY");
                writer.incrementIndentLevel();
                writer.writeToken(" ");
                writer.writeToken(XmDomUtil.getContentText(functionNameNode));
                writer.writeToken("(");

                invokeEnter(XmDomUtil.getElement(functionTypeNode, "params"));

                writer.writeToken(")");
                if (typeId != XfType.VOID) {
                    String functionResultName =
                        XmDomUtil.getAttr(functionTypeNode, "result_name");
                    if (XfUtil.isNullOrEmpty(functionResultName) == false) {
                        writer.writeToken(" RESULT(" + functionResultName + ")");
                    }
                }
            }

            writer.setupNewLine();
        }
    }

    // FequivalenceDecl
    class FequivalenceDeclVisitor extends XcodeNodeVisitor {
        /**
         * Decompile "FequivalenceDecl" element in XcodeML/F.
         *
         * @example <code><div class="Example">
         * <div class="Strong">
         * EQUIVALENCE (variable1, variable2), (variable3, variable4)<br/>
         * </div>
         * </div></code>
         * @see xcodeml.f.binding.gen.RVisitorBase#enter(xcodeml.f.binding.gen.
         *      XbfFequivalenceDecl)
         */
        @Override public void enter(Node n) {
            _writeLineDirective(n);

            XmfWriter writer = _context.getWriter();
            writer.writeToken("EQUIVALENCE ");

            ArrayList<Node> childNodes = XmDomUtil.collectChildNodes(n);
            for (Iterator<Node> iter = childNodes.iterator(); iter.hasNext(); ) {
                writer.writeToken("(");

                Node varRefNode = iter.next();
                if (!"varRef".equals(varRefNode.getNodeName())) {
                    throw new XmTranslationException(n,
                                                     "Invalid contents");
                }
                if (!iter.hasNext()) {
                    throw new XmTranslationException(n,
                                                     "Invalid contents");
                }
                Node varListNode = iter.next();
                if (!"varList".equals(varListNode.getNodeName())) {
                    throw new XmTranslationException(n,
                                                     "Invalid contents");
                }

                invokeEnter(varRefNode);
                writer.writeToken(", ");
                invokeEnter(varListNode);
                writer.writeToken(")");
            }

            writer.setupNewLine();
        }
    }

    // FexitStatement
    class FexitStatementVisitor extends XcodeNodeVisitor {
        /**
         * Decompile "FexitStatement" element in XcodeML/F.
         *
         * @example <code><div class="Example">
         * DO_NAME: DO, variable = 1, 10<br/>
         * <div class="Indent1">
         *      (any statement...)<br/>
         *      <div class="Strong">
         *      EXIT DO_NAME<br/>
         *      </div>
         * </div>
         * END DO DO_NAME<br/>
         * </div></code>
         * @see xcodeml.f.binding.gen.RVisitorBase#enter(xcodeml.f.binding.gen.
         *      XbfFexitStatement)
         */
        @Override public void enter(Node n) {
            _writeLineDirective(n);

            XmfWriter writer = _context.getWriter();
            writer.writeToken("EXIT");

            String constuctName = XmDomUtil.getAttr(n, "construct_name");
            if (XfUtil.isNullOrEmpty(constuctName) == false) {
                writer.writeToken(" ");
                writer.writeToken(constuctName);
            }

            writer.setupNewLine();
        }
    }

    // FformatDecl
    class FformatDeclVisitor extends XcodeNodeVisitor {
        /**
         * Decompile "FformatDecl" element in XcodeML/F.
         *
         * @example <code><div class="Example">
         * <div class="Strong">
         * 1000 FORMAT (&lt;any format string&gt;)<br/>
         * </div>
         * </div></code>
         * @see xcodeml.f.binding.gen.RVisitorBase#enter(xcodeml.f.binding.gen.XbfFformatStatement
         *      )
         */
        @Override public void enter(Node n) {
            _writeLineDirective(n);

            XmfWriter writer = _context.getWriter();
            writer.writeToken("FORMAT ");
            writer.writeToken(XmDomUtil.getAttr(n, "format"));
            writer.setupNewLine();
        }
    }

    // FfunctionDecl
    class FfunctionDeclVisitor extends XcodeNodeVisitor {
        /**
         * Decompile "FfunctionDecl" element in XcodeML/F.
         *
         * @example <code><div class="Example">
         * INTERFACE interface_name<br/>
         * <div class="Indent1">
         *      <div class="Strong">
         *      SUBROUTINE sub1(arg1)<br/>
         *      <div class="Indent1">
         *          INTEGER arg1<br/>
         *      </div>
         *      END SUBROUTINE<br/>
         *      </div>
         *      MODULE PROCEDURE module_sub<br/>
         *      <br/>
         * </div>
         * <div class="Strong">
         * END INTERFACE<br/>
         * </div>
         * </div></code>
         * @see xcodeml.f.binding.gen.RVisitorBase#enter(xcodeml.f.binding.gen.
         *      XbfFfunctionDecl)
         */
        @Override public void enter(Node n) {
            _writeLineDirective(n);

            XfTypeManagerForDom typeManager = _context.getTypeManagerForDom();
            XmfWriter writer = _context.getWriter();

            Node functionNameNode = XmDomUtil.getElement(n, "name");
            Node typeChoice = typeManager.findType(functionNameNode);
            if (typeChoice == null) {
                _context.setLastErrorMessage(
                    XfUtilForDom.formatError(n,
                                       XfError.XCODEML_TYPE_NOT_FOUND,
                                       XmDomUtil.getAttr(functionNameNode,
                                                         "type")));
                fail(n);
            } else if ("FfunctionType".equals(typeChoice.getNodeName()) == false) {
                _context.setLastErrorMessage(
                    XfUtilForDom.formatError(n,
                                       XfError.XCODEML_TYPE_MISMATCH,
                                       "function definition",
                                       typeChoice.getNodeName(),
                                       "FfunctionType"));
                fail(n);
            }

            Node functionTypeNode = typeChoice;
            String returnTypeName = XmDomUtil.getAttr(functionTypeNode,
                                                      "return_type");
            XfType typeId = XfType.getTypeIdFromXcodemlTypeName(returnTypeName);
            if (XmDomUtil.getAttrBool(functionTypeNode, "is_program")) {
                // =======
                // PROGRAM
                // =======
                _context.setLastErrorMessage(
                    XfUtilForDom.formatError(n,
                                       XfError.XCODEML_TYPE_MISMATCH,
                                       "function definition",
                                       "PROGRAM",
                                       "FUNCTION or SUBROUTINE"));
                fail(n);
            } else if (typeId == XfType.VOID) {
                // ==========
                // SUBROUTINE
                // ==========
                if (XmDomUtil.getAttrBool(functionTypeNode, "is_recursive")) {
                    writer.writeToken("RECURSIVE");
                    writer.writeToken(" ");
                }
                if (XmDomUtil.getAttrBool(functionTypeNode, "is_pure")) {
                    writer.writeToken("PURE");
                    writer.writeToken(" ");
                }
                if (XmDomUtil.getAttrBool(functionTypeNode, "is_elemental")) {
                    writer.writeToken("ELEMENTAL");
                    writer.writeToken(" ");
                }
                writer.writeToken("SUBROUTINE");
                writer.writeToken(" ");
                writer.writeToken(XmDomUtil.getContentText(functionNameNode));
                writer.writeToken("(");

                invokeEnter(XmDomUtil.getElement(functionTypeNode, "params"));

                writer.writeToken(")");
            } else {
                // ========
                // FUNCTION
                // ========

                // Note:
                // In the function definition, do not output return type.

                /*
                 * if (typeId == XfType.DERIVED) {
                 * writer.writeToken(returnTypeName); } else {
                 * writer.writeToken(typeId.fortranName()); }
                 * writer.writeToken(" ");
                 */
                if (XmDomUtil.getAttrBool(functionTypeNode, "is_recursive")) {
                    writer.writeToken("RECURSIVE");
                    writer.writeToken(" ");
                }
                writer.writeToken("FUNCTION");
                writer.writeToken(" ");
                writer.writeToken(XmDomUtil.getContentText(functionNameNode));
                writer.writeToken("(");

                invokeEnter(XmDomUtil.getElement(functionTypeNode, "params"));

                writer.writeToken(")");
                String functionResultName =
                    XmDomUtil.getAttr(functionTypeNode, "result_name");
                if (XfUtil.isNullOrEmpty(functionResultName) == false) {
                    writer.writeToken(" RESULT(" + functionResultName + ")");
                }
            }

            writer.setupNewLine();

            // ========
            // Epilogue
            // ========
            writer.incrementIndentLevel();

            // ======
            // Inside
            // ======
            invokeEnter(XmDomUtil.getElement(n, "declarations"));

            // ========
            // Prologue
            // ========
            writer.decrementIndentLevel();

            assert XmDomUtil.getAttrBool(functionTypeNode, "is_program");
            if (typeId == XfType.VOID) {
                // ==========
                // SUBROUTINE
                // ==========
                writer.writeToken("END SUBROUTINE");
                writer.writeToken(" ");
                writer.writeToken(XmDomUtil.getContentText(functionNameNode));
            } else {
                writer.writeToken("END FUNCTION");
                writer.writeToken(" ");
                writer.writeToken(XmDomUtil.getContentText(functionNameNode));
            }
            writer.setupNewLine();
        }
    }

    // FfunctionDefinition
    class FfunctionDefinitionVisitor extends XcodeNodeVisitor {
        /**
         * Decompile "FfunctionDefinition" element in XcodeML/F.
         *
         * @example <code><div class="Example">
         * <div class="Strong">
         * PROGRAM main<br/>
         * </div>
         * <div class="Indent1">
         *      INTEGER :: int_variable<br/>
         *      (any statement...)<br/>
         * </div>
         * <br/>
         * CONTAINS
         * <div class="Indent1">
         *      <div class="Strong">
         *      SUBROUTINE sub()<br/>
         *      </div>
         *      (any statement...)<br/>
         *      <div class="Strong">
         *      END SUBROUTINE sub<br/>
         *      </div>
         *      <br/>
         *      <div class="Strong">
         *      FUNCTION func()<br/>
         *      </div>
         *      (any statement...)<br/>
         *      <div class="Strong">
         *      END SUBROUTINE sub<br/>
         *      </div>
         * </div>
         * <br/>
         * <div class="Strong">
         * END PROGRAM main<br/>
         * </div>
         * </div></code>
         * @see xcodeml.f.binding.gen.RVisitorBase#enter(xcodeml.f.binding.gen.
         *      XbfFfunctionDefinition)
         */
        @Override public void enter(Node n) {
            _writeLineDirective(n);

            XfTypeManagerForDom typeManager = _context.getTypeManagerForDom();
            XmfWriter writer = _context.getWriter();

            Node functionNameNode = XmDomUtil.getElement(n, "name");
            Node typeChoice = typeManager.findType(functionNameNode);

            if (typeChoice == null) {
                _context.setLastErrorMessage(
                    XfUtilForDom.formatError(n,
                                             XfError.XCODEML_TYPE_NOT_FOUND,
                                             XmDomUtil.getAttr(functionNameNode,
                                                               "type")));
                fail(n);
            } else if ("FfunctionType".equals(typeChoice.getNodeName()) == false) {
                _context.setLastErrorMessage(
                    XfUtilForDom.formatError(n,
                                             XfError.XCODEML_TYPE_MISMATCH,
                                             "function definition",
                                             typeChoice.getNodeName(),
                                             "FfunctionType"));
                fail(n);
            }

            Node functionTypeNode = typeChoice;
            String returnTypeName = XmDomUtil.getAttr(functionTypeNode,
                                                      "return_type");
            XfType typeId = XfType.getTypeIdFromXcodemlTypeName(returnTypeName);
            if (XmDomUtil.getAttrBool(functionTypeNode, "is_program")) {
                // =======
                // PROGRAM
                // =======
                writer.writeToken("PROGRAM");
                writer.writeToken(" ");
                writer.writeToken(XmDomUtil.getContentText(functionNameNode));
            } else if (typeId == XfType.VOID) {
                // ==========
                // SUBROUTINE
                // ==========
                if (XmDomUtil.getAttrBool(functionTypeNode, "is_recursive")) {
                    writer.writeToken("RECURSIVE");
                    writer.writeToken(" ");
                }
                if (XmDomUtil.getAttrBool(functionTypeNode, "is_pure")) {
                    writer.writeToken("PURE");
                    writer.writeToken(" ");
                }
                if (XmDomUtil.getAttrBool(functionTypeNode, "is_elemental")) {
                    writer.writeToken("ELEMENTAL");
                    writer.writeToken(" ");
                }
                writer.writeToken("SUBROUTINE");
                writer.writeToken(" ");
                writer.writeToken(XmDomUtil.getContentText(functionNameNode));
                writer.writeToken("(");

                invokeEnter(XmDomUtil.getElement(functionTypeNode, "params"));

                writer.writeToken(")");
            } else {
                // ========
                // FUNCTION
                // ========

                // Note:
                // In the function definition, do not output return type.

                /*
                 * if (typeId == XfType.DERIVED) {
                 * writer.writeToken(returnTypeName); } else {
                 * writer.writeToken(typeId.fortranName()); }
                 * writer.writeToken(" ");
                 */
                if (XmDomUtil.getAttrBool(functionTypeNode, "is_recursive")) {
                    writer.writeToken("RECURSIVE");
                    writer.writeToken(" ");
                }
                if (XmDomUtil.getAttrBool(functionTypeNode, "is_pure")) {
                    writer.writeToken("PURE");
                    writer.writeToken(" ");
                }
                if (XmDomUtil.getAttrBool(functionTypeNode, "is_elemental")) {
                    writer.writeToken("ELEMENTAL");
                    writer.writeToken(" ");
                }
                writer.writeToken("FUNCTION");
                writer.writeToken(" ");
                writer.writeToken(XmDomUtil.getContentText(functionNameNode));
                writer.writeToken("(");

                invokeEnter(XmDomUtil.getElement(functionTypeNode, "params"));

                writer.writeToken(")");
                String functionResultName =
                    XmDomUtil.getAttr(functionTypeNode, "result_name");
                if (XfUtil.isNullOrEmpty(functionResultName) == false) {
                    writer.writeToken(" RESULT(" + functionResultName + ")");
                }
            }

            writer.setupNewLine();

            // ========
            // Epilogue
            // ========
            writer.incrementIndentLevel();
            typeManager.enterScope();

            // ======
            // Inside
            // ======
            invokeEnter(XmDomUtil.getElement(n, "symbols"));
            invokeEnter(XmDomUtil.getElement(n, "declarations"));

            writer.setupNewLine();

            invokeEnter(XmDomUtil.getElement(n, "body"));

            // ========
            // Prologue
            // ========
            writer.decrementIndentLevel();
            typeManager.leaveScope();

            if (XmDomUtil.getAttrBool(functionTypeNode, "is_program")) {
                // =======
                // PROGRAM
                // =======
                writer.writeToken("END PROGRAM");
                writer.writeToken(" ");
                writer.writeToken(XmDomUtil.getContentText(functionNameNode));
            } else if (typeId == XfType.VOID) {
                // ==========
                // SUBROUTINE
                // ==========
                writer.writeToken("END SUBROUTINE");
                writer.writeToken(" ");
                writer.writeToken(XmDomUtil.getContentText(functionNameNode));
            } else {
                writer.writeToken("END FUNCTION");
                writer.writeToken(" ");
                writer.writeToken(XmDomUtil.getContentText(functionNameNode));
            }
            writer.setupNewLine();
            writer.setupNewLine();
        }
    }

    // FifStatement
    class FifStatementVisitor extends XcodeNodeVisitor {
        /**
         * Decompile "FifStatement" element in XcodeML/F.
         *
         * @example <code><div class="Example">
         * <div class="Strong">
         * IF (variable == 1) THEN<br/>
         * <div class="Indent1">
         *      (any statement...)<br/>
         * </div>
         * ELSE<br/>
         * <div class="Indent1">
         *      (any statement...)<br/>
         * </div>
         * END IF<br/>
         * </div>
         * </div></code>
         * @see xcodeml.f.binding.gen.RVisitorBase#enter(xcodeml.f.binding.gen.
         *      XbfFifStatement)
         */
        @Override public void enter(Node n) {
            _writeLineDirective(n);

            XmfWriter writer = _context.getWriter();
            String constuctName = XmDomUtil.getAttr(n, "construct_name");
            if (XfUtil.isNullOrEmpty(constuctName) == false) {
                writer.writeToken(constuctName);
                writer.writeToken(": ");
            }

            writer.writeToken("IF ");
            invokeEnter(XmDomUtil.getElement(n, "condition"));

            writer.writeToken(" THEN");
            writer.setupNewLine();

            invokeEnter(XmDomUtil.getElement(n, "then"));

            Node elseNode = XmDomUtil.getElement(n, "else");
            if (elseNode != null) {
                writer.writeToken("ELSE");
                if (XfUtil.isNullOrEmpty(constuctName) == false) {
                    writer.writeToken(" ");
                    writer.writeToken(constuctName);
                }
                writer.setupNewLine();

                invokeEnter(elseNode);
            }

            writer.writeToken("END IF");
            if (XfUtil.isNullOrEmpty(constuctName) == false) {
                writer.writeToken(" ");
                writer.writeToken(constuctName);
            }
            writer.setupNewLine();
        }
    }

    // FinquireStatement
    class FinquireStatementVisitor extends XcodeNodeVisitor {
        /**
         * Decompile "FinquireStatement" element in XcodeML/F.
         *
         * @example <code><div class="Example">
         * <div class="Strong">
         * INQUIRE (UNIT=1, ...)<br/>
         * </div>
         * <div class="Strong">
         * INQUIRE (IOLENGTH=variable) out_variable1, out_variable2<br/>
         * </div>
         * </div></code>
         * @see xcodeml.f.binding.gen.RVisitorBase#enter(xcodeml.f.binding.gen.
         *      XbfFinquireStatement)
         */
        @Override public void enter(Node n) {
            _writeLineDirective(n);

            XmfWriter writer = _context.getWriter();

            writer.writeToken("INQUIRE ");
            invokeEnter(XmDomUtil.getElement(n, "namedValueList"));

            writer.writeToken(" ");
            invokeEnter(XmDomUtil.getElement(n, "valueList"));

            writer.setupNewLine();
        }
    }

    // FintConstant
    class FintConstantVisitor extends XcodeNodeVisitor {
        /**
         * Decompile "FintConstant" element in XcodeML/F.
         *
         * @example <code><div class="Example">
         * PROGRAM main<br/>
         * <div class="Indent1">
         *      INTEGER :: int_variable = <span class="Strong">10</span><br/>
         * </div>
         * END PROGRAM main<br/>
         * </div></code>
         * @see xcodeml.f.binding.gen.RVisitorBase#enter(xcodeml.f.binding.gen.
         *      XbfFintConstant)
         */
        @Override public void enter(Node n) {
            String content = XmDomUtil.getContentText(n);
            if (XfUtil.isNullOrEmpty(content)) {
                _context.setLastErrorMessage(
                    XfUtilForDom.formatError(n,
                                             XfError.XCODEML_SEMANTICS,
                                             n.getNodeName()));
                fail(n);
            }

            String typeName = XmDomUtil.getAttr(n, "type");
            if (XfUtil.isNullOrEmpty(typeName) == false) {
                XfTypeManagerForDom typeManager = _context.getTypeManagerForDom();
                String bottomTypeName = typeManager.getBottomTypeName(typeName);
                if (bottomTypeName == null) {
                    _context.setLastErrorMessage(
                        XfUtilForDom.formatError(n,
                                                 XfError.XCODEML_TYPE_NOT_FOUND,
                                                 n.getNodeName(),
                                                 typeName));
                    fail(n);
                }

                XfType typeId = XfType.getTypeIdFromXcodemlTypeName(bottomTypeName);
                if (typeId != XfType.DERIVED && typeId != XfType.INT) {
                    _context.setLastErrorMessage(
                        XfUtilForDom.formatError(n,
                                                 XfError.XCODEML_TYPE_MISMATCH,
                                                 n.getNodeName(),
                                                 typeName,
                                                 "Fint"));
                    fail(n);
                }
            }

            XmfWriter writer = _context.getWriter();

            String kind = XmDomUtil.getAttr(n, "kind");
            if (XfUtil.isNullOrEmpty(kind) == false) {
                writer.writeToken(content + "_" + kind);
            } else {
		/* check minus number */
		if(new Integer(content).intValue() < 0)
		    content = "("+content+")";
                writer.writeToken(content);
            }
        }
    }

    // FinterfaceDecl
    class FinterfaceDeclVisitor extends XcodeNodeVisitor {
        /**
         * Decompile "FinterfaceDecl" element in XcodeML/F.
         *
         * @example <code><div class="Example">
         * <div class="Strong">
         * INTERFACE interface_name<br/>
         * </div>
         * <div class="Indent1">
         *      SUBROUTINE sub1(arg1)<br/>
         *      <div class="Indent1">
         *          INTEGER arg1<br/>
         *      </div>
         *      END SUBROUTINE<br/>
         *      MODULE PROCEDURE module_sub<br/>
         *      <br/>
         * </div>
         * <div class="Strong">
         * END INTERFACE<br/>
         * </div>
         * </div></code>
         * @see xcodeml.f.binding.gen.RVisitorBase#enter(xcodeml.f.binding.gen.
         *      XbfFinterfaceDecl)
         */
        @Override public void enter(Node n) {
            _writeLineDirective(n);

            XmfWriter writer = _context.getWriter();

            String interfaceName = XmDomUtil.getAttr(n, "name");

            writer.writeToken("INTERFACE");
            if (XmDomUtil.getAttrBool(n, "is_assignment")) {
                writer.writeToken(" ASSIGNMENT(=)");
            } else if (XmDomUtil.getAttrBool(n, "is_operator")) {
                writer.writeToken(" OPERATOR(");
                writer.writeToken(interfaceName);
                writer.writeToken(")");
            } else {
                if (XfUtil.isNullOrEmpty(interfaceName) == false) {
                    writer.writeToken(" ");
                    writer.writeToken(interfaceName);
                }
            }

            writer.setupNewLine();
            writer.incrementIndentLevel();

            _invokeChildEnter(n);

            writer.decrementIndentLevel();

            writer.writeToken("END INTERFACE");
            writer.setupNewLine();
        }
    }

    // FlogicalConstant
    class FlogicalConstantVisitor extends XcodeNodeVisitor {
        /**
         * Decompile "FlogicalConstant" element in XcodeML/F.
         *
         * @example <code><div class="Example">
         * PROGRAM main<br/>
         * <div class="Indent1">
         *      LOGICAL log_variable = <span class="Strong">.TRUE.</span><br/>
         * </div>
         * END PROGRAM main<br/>
         * </div></code>
         * @see xcodeml.f.binding.gen.RVisitorBase#enter(xcodeml.f.binding.gen.
         *      XbfFlogicalConstant)
         */
        @Override public void enter(Node n) {
            String content = XmDomUtil.getContentText(n);
            if (XfUtil.isNullOrEmpty(content)) {
                _context.setLastErrorMessage(
                    XfUtilForDom.formatError(n,
                                             XfError.XCODEML_SEMANTICS,
                                             n.getNodeName()));
                fail(n);
            }

            String typeName = XmDomUtil.getAttr(n, "type");
            if (XfUtil.isNullOrEmpty(typeName) == false) {
                XfTypeManagerForDom typeManager = _context.getTypeManagerForDom();
                String bottomTypeName = typeManager.getBottomTypeName(typeName);
                if (bottomTypeName == null) {
                    _context.setLastErrorMessage(
                        XfUtilForDom.formatError(n,
                                                 XfError.XCODEML_TYPE_NOT_FOUND,
                                                 n.getNodeName(),
                                                 typeName));
                    fail(n);
                }

                XfType typeId = XfType.getTypeIdFromXcodemlTypeName(bottomTypeName);
                if (typeId != XfType.DERIVED && typeId != XfType.LOGICAL) {
                    _context.setLastErrorMessage(
                        XfUtilForDom.formatError(n,
                                                 XfError.XCODEML_TYPE_MISMATCH,
                                                 n.getNodeName(),
                                                 typeName,
                                                 "Fint"));
                    fail(n);
                }
            }

            XmfWriter writer = _context.getWriter();

            String kind = XmDomUtil.getAttr(n, "kind");
            if (XfUtil.isNullOrEmpty(kind) == false) {
                writer.writeToken(content + "_" + kind);
            } else {
                writer.writeToken(content);
            }
        }
    }

    // FmemberRef
    class FmemberRefVisitor extends XcodeNodeVisitor {
        /**
         * Decompile "FmemberRef" element in XcodeML/F.
         *
         * @example <code><div class="Example">
         *      variable = <span class="Strong">struct%member</span><br/>
         * </div></code>
         * @see xcodeml.f.binding.gen.RVisitorBase#enter(xcodeml.f.binding.gen.XbfFmemberRef
         *      )
         */
        @Override public void enter(Node n) {
            invokeEnter(XmDomUtil.getElement(n, "varRef"));

            XmfWriter writer = _context.getWriter();
            writer.writeToken("%");
            writer.writeToken(XmDomUtil.getAttr(n, "member"));
        }
    }

    // FmoduleDefinition
    class FmoduleDefinitionVisitor extends XcodeNodeVisitor {
        /**
         * Decompile "FmoduleDefinition" element in XcodeML/F.
         *
         * @example <code><div class="Example">
         * <div class="Strong">
         * MODULE mod<br/>
         * </div>
         * <div class="Indent1">(any statement...)<br/></div>
         * <div class="Strong">
         * END MODULE mod<br/>
         * </div>
         * <br/>
         * PROGRAM main<br/>
         * <div class="Indent1">
         *      INTEGER :: int_variable<br/>
         *      (any statement...)<br/>
         * </div>
         * <br/>
         * CONTAINS
         * <div class="Indent1">
         *      SUBROUTINE sub()<br/>
         *      (any statement...)<br/>
         *      END SUBROUTINE sub<br/>
         *      <br/>
         *      FUNCTION func()<br/>
         *      (any statement...)<br/>
         *      END SUBROUTINE sub<br/>
         * </div>
         * <br/>
         * END PROGRAM main<br/>
         * </div></code>
         * @see xcodeml.f.binding.gen.RVisitorBase#enter(xcodeml.f.binding.gen.
         *      XbfFmoduleDefinition)
         */
        @Override public void enter(Node n) {
            _writeLineDirective(n);

            XfTypeManagerForDom typeManager = _context.getTypeManagerForDom();
            XmfWriter writer = _context.getWriter();

            String name = XmDomUtil.getAttr(n, "name");

            writer.writeToken("MODULE");
            writer.writeToken(" ");
            writer.writeToken(name);
            writer.setupNewLine();

            // ========
            // Epilogue
            // ========
            writer.incrementIndentLevel();
            typeManager.enterScope();

            // ======
            // Inside
            // ======h
            invokeEnter(XmDomUtil.getElement(n, "symbols"));

            invokeEnter(XmDomUtil.getElement(n, "declarations"));

            invokeEnter(XmDomUtil.getElement(n, "FcontainsStatement"));

            // ========
            // Prologue
            // ========
            writer.decrementIndentLevel();
            typeManager.leaveScope();

            writer.writeToken("END MODULE");
            writer.writeToken(" ");
            writer.writeToken(name);
            writer.setupNewLine();
            writer.setupNewLine();
        }
    }

    // FmoduleProcedureDecl
    class FmoduleProcedureDeclVisitor extends XcodeNodeVisitor {
        /**
         * Decompile "FmoduleProcedureDecl" element in XcodeML/F.
         *
         * @example <code><div class="Example">
         * INTERFACE interface_name<br/>
         * <div class="Indent1">
         *      SUBROUTINE sub1(arg1)<br/>
         *      <div class="Indent1">
         *          INTEGER arg1<br/>
         *      </div>
         *      END SUBROUTINE<br/>
         *      <div class="Strong">
         *      MODULE PROCEDURE module_sub<br/>
         *      </div>
         *      <br/>
         * </div>
         * <div class="Strong">
         * END INTERFACE<br/>
         * </div>
         * </div></code>
         * @see xcodeml.f.binding.gen.RVisitorBase#enter(xcodeml.f.binding.gen.
         *      XbfFmoduleProcedureDecl)
         */
        @Override public void enter(Node n) {
            _writeLineDirective(n);

            XmfWriter writer = _context.getWriter();

            if (XmDomUtil.getAttrBool(n, "is_module_specified")) {
                writer.writeToken("MODULE ");
            }
            writer.writeToken("PROCEDURE ");
            int nameCount = 0;
            ArrayList<Node> nameNodes = XmDomUtil.collectElements(n, "name");
            for (Node nameNode : nameNodes) {
                if (nameCount > 0) {
                    writer.writeToken(", ");
                }
                writer.writeToken(XmDomUtil.getContentText(nameNode));
                ++nameCount;
            }
            writer.setupNewLine();
        }
    }

    // FblockDataDefinition
    class FblockDataDefinitionVisitor extends XcodeNodeVisitor {
        /**
         * Decompile "FblockDataDefinition" element in XcodeML/F.
         *
         * @example <code><div class="Example">
         * <div class="Strong">
         * BLOCK DATA dat<br/>
         * </div>
         * <div class="Indent1">(any declaration...)<br/></div>
         * <div class="Strong">
         * END BLOCK DATA dat<br/>
         * </div>
         * <br/>
         * </div></code>
         * @see xcodeml.f.binding.gen.RVisitorBase#enter(xcodeml.f.binding.gen.
         *      XbfFblockDataDefinition)
         */
        @Override public void enter(Node n) {
            _writeLineDirective(n);

            XfTypeManagerForDom typeManager = _context.getTypeManagerForDom();
            XmfWriter writer = _context.getWriter();

            String name = XmDomUtil.getAttr(n, "name");

            writer.writeToken("BLOCK DATA");
            writer.writeToken(" ");
            writer.writeToken(name);
            writer.setupNewLine();

            // ========
            // Epilogue
            // ========
            writer.incrementIndentLevel();
            typeManager.enterScope();

            // ======
            // Inside
            // ======
            invokeEnter(XmDomUtil.getElement(n, "symbols"));

            invokeEnter(XmDomUtil.getElement(n, "declarations"));

            // ========
            // Prologue
            // ========
            writer.decrementIndentLevel();
            typeManager.leaveScope();

            writer.writeToken("END BLOCK DATA");
            writer.writeToken(" ");
            writer.writeToken(name);
            writer.setupNewLine();
            writer.setupNewLine();
        }
    }

    // FnamelistDecl
    class FnamelistDeclVisitor extends XcodeNodeVisitor {
        /**
         * Decompile "FnamelistDecl" element in XcodeML/F.
         *
         * @example <code><div class="Example">
         * <div class="Strong">
         * NAMELIST /NAME1/ variable1, variable2, /NAME2/ variable3, variable4<br/>
         * </div>
         * </div></code>
         * @see xcodeml.f.binding.gen.RVisitorBase#enter(xcodeml.f.binding.gen.
         *      XbfFnamelistDecl)
         */
        @Override public void enter(Node n) {
            _writeLineDirective(n);

            XmfWriter writer = _context.getWriter();
            writer.writeToken("NAMELIST ");

            _invokeChildEnterAndWriteDelim(n, ", ");

            writer.setupNewLine();
        }
    }

    // FnullifyStatement
    class FnullifyStatementVisitor extends XcodeNodeVisitor {
        /**
         * Decompile "FnullifyStatement" element in XcodeML/F.
         *
         * @example <code><div class="Example">
         *      <div class="Strong">
         *      NULLIFY (ptr, ptr_array)<br/>
         *      </div>
         * </div></code>
         * @see xcodeml.f.binding.gen.RVisitorBase#enter(xcodeml.f.binding.gen.
         *      XbfFnullifyStatement)
         */
        @Override public void enter(Node n) {
            _writeLineDirective(n);

            XmfWriter writer = _context.getWriter();
            writer.writeToken("NULLIFY (");
            _invokeChildEnterAndWriteDelim(n, ", ");

            writer.writeToken(")");
            writer.setupNewLine();
        }
    }

    // FopenStatement
    class FopenStatementVisitor extends XcodeNodeVisitor {
        /**
         * Decompile "FopenStatement" element in XcodeML/F.
         *
         * @example <code><div class="Example">
         * <div class="Strong">
         * OPEN (UNIT=1, ...)<br/>
         * </div>
         * </div></code>
         * @see xcodeml.f.binding.gen.RVisitorBase#enter(xcodeml.f.binding.gen.
         *      XbfFopenStatement)
         */
        @Override public void enter(Node n) {
            _writeLineDirective(n);

            XmfWriter writer = _context.getWriter();

            writer.writeToken("OPEN ");
            invokeEnter(XmDomUtil.getElement(n, "namedValueList"));

            writer.setupNewLine();
        }
    }

    // FpointerAssignStatement
    class FpointerAssignStatementVisitor extends XcodeNodeVisitor {
        /**
         * Decompile "FpointerAssignStatement" element in XcodeML/F.
         * <p>
         * The decompilation result depends on a child element.
         * </p>
         *
         * @example <code><div class="Example">
         * PROGRAM main<br/>
         * <div class="Indent1">
         *      INTEGER,POINTER :: int_pointer<br/>
         *      INTEGER :: int_variable<br/>
         *      int_variable = 0<br/>
         *      <div class="Strong">
         *      int_pointer => int_variable
         *      </div>
         *      (any expression statement ...)<br/>
         * </div>
         * END PROGRAM main<br/>
         * </div></code>
         * @see xcodeml.f.binding.gen.RVisitorBase#enter(xcodeml.f.binding.gen.
         *      XbfFpointerAssignStatement)
         */
        @Override public void enter(Node n) {
            _writeLineDirective(n);

            XmfWriter writer = _context.getWriter();

            _writeBinaryExpr(XmDomUtil.collectChildNodes(n), 0,
                             "=>", false);

            writer.setupNewLine();
        }
    }

    // FpowerExpr
    class FpowerExprVisitor extends XcodeNodeVisitor {
        /**
         * Decompile "FpowerExpr" element in XcodeML/F.
         * <p>
         * The decompilation result depends on a child element.
         * </p>
         *
         * @example <code><div class="Example">
         * variable = <span class="Strong">(&lt;any expression&gt; ** &lt;any expression&gt;)</span><br/>
         * </div></code>
         * @see xcodeml.f.binding.gen.RVisitorBase#enter(xcodeml.f.binding.gen.XbfFpowerExpr
         *      )
         */
        @Override public void enter(Node n) {
            _writeBinaryExpr(XmDomUtil.collectChildNodes(n), 0,
                             "**", false);
        }
    }

    // FpragmaStatement
    class FpragmaStatementVisitor extends XcodeNodeVisitor {
        /**
         * Decompile "FpragmaStatement" element in XcodeML/F.
         *
         * @example <code><div class="Example">
         * <div class="Strong">
         * !$OMP &lt;any text&gt;<br/>
         * </div>
         * </div></code>
         * @see xcodeml.f.binding.gen.RVisitorBase#enter(xcodeml.f.binding.gen.
         *      XbfFpragmaStatement)
         */
        @Override public void enter(Node n) {
            _writeLineDirective(n);

            String content = XmDomUtil.getContentText(n);
            if (content.startsWith("!$") == false) {
                content = "!$" + content;
            }

            XmfWriter writer = _context.getWriter();
            writer.writeIsolatedLine(content);
        }
    }
    
    // OMPPragma
    class OMPPragmaVisitor extends XcodeNodeVisitor {
        /**
         * Decompile "OMPPragma" element in XcodeML/F.
         */
        @Override public void enter(Node n) {

            _writeLineDirective(n);
            
            boolean nowaitFlag = false;
            boolean copyprivateFlag = false;
            
            XmfWriter writer = _context.getWriter();

	    XmfWriter.StatementMode prevMode = writer.getStatementMode();
	    writer.setStatementMode(XmfWriter.StatementMode.OMP);

            // directive
            Node dir = n.getFirstChild();
            String dirName = XmDomUtil.getContentText(dir);
            
            if (dirName.equals("FOR")) dirName = "DO";
            writer.writeToken("!$OMP " + dirName);

            if (dirName.equals("THREADPRIVATE") ||
		dirName.equals("FLUSH")){

            	writer.writeToken("(");
            	
            	NodeList varList = dir.getNextSibling().getChildNodes();
        		invokeEnter(varList.item(0));
        		for (int j = 1; j < varList.getLength(); j++){
        			Node var = varList.item(j);
        			writer.writeToken(",");
        			invokeEnter(var);
        		}
        
        		writer.writeToken(")");
			writer.setStatementMode(prevMode);
        		writer.setupNewLine();
        		
        		return;
            }
            else if (dirName.equals("BARRIER")){
	      writer.setStatementMode(prevMode);
	      writer.setupNewLine();
	      return;
	    }

            // clause
            Node clause = dir.getNextSibling();
	    Node copyprivate_arg = null;

            NodeList list0 = clause.getChildNodes();
            for (int i = 0; i < list0.getLength(); i++){          
            	Node childNode = list0.item(i);
                if (childNode.getNodeType() != Node.ELEMENT_NODE) {
                    continue;
                }
                
                String clauseName = XmDomUtil.getContentText(childNode);
                String operator = "";
                if (clauseName.equals("DATA_DEFAULT"))               clauseName = "DEFAULT";
                else if (clauseName.equals("DATA_PRIVATE"))          clauseName = "PRIVATE";
                else if (clauseName.equals("DATA_SHARED"))           clauseName = "SHARED";
                else if (clauseName.equals("DATA_FIRSTPRIVATE"))     clauseName = "FIRSTPRIVATE";
                else if (clauseName.equals("DATA_LASTPRIVATE"))      clauseName = "LASTPRIVATE";
                else if (clauseName.equals("DATA_COPYIN"))           clauseName = "COPYIN";
                else if (clauseName.equals("DATA_REDUCTION_PLUS"))  {clauseName = "REDUCTION"; operator = "+";}
                else if (clauseName.equals("DATA_REDUCTION_MINUS")) {clauseName = "REDUCTION"; operator = "-";}
                else if (clauseName.equals("DATA_REDUCTION_MUL"))   {clauseName = "REDUCTION"; operator = "*";}
                else if (clauseName.equals("DATA_REDUCTION_BITAND")){clauseName = "REDUCTION"; operator = "iand";}
                else if (clauseName.equals("DATA_REDUCTION_BITOR")) {clauseName = "REDUCTION"; operator = "ior";}
                else if (clauseName.equals("DATA_REDUCTION_BITXOR")){clauseName = "REDUCTION"; operator = "ieor";}
                else if (clauseName.equals("DATA_REDUCTION_LOGAND")){clauseName = "REDUCTION"; operator = ".and.";}
                else if (clauseName.equals("DATA_REDUCTION_LOGOR")) {clauseName = "REDUCTION"; operator = ".or.";}
                else if (clauseName.equals("DATA_REDUCTION_MIN"))   {clauseName = "REDUCTION"; operator = "min";}
                else if (clauseName.equals("DATA_REDUCTION_MAX"))   {clauseName = "REDUCTION"; operator = "max";}
                else if (clauseName.equals("DATA_REDUCTION_EQV"))   {clauseName = "REDUCTION"; operator = ".eqv.";}
                else if (clauseName.equals("DATA_REDUCTION_NEQV"))  {clauseName = "REDUCTION"; operator = ".neqv.";}
		else if (clauseName.equals("DATA_COPYPRIVATE"))     {clauseName = "COPYPRIVATE"; copyprivateFlag = true;
		  copyprivate_arg = childNode.getFirstChild().getNextSibling();}
                else if (clauseName.equals("DIR_ORDERED"))           clauseName = "ORDERED";
                else if (clauseName.equals("DIR_IF"))                clauseName = "IF";
                else if (clauseName.equals("DIR_NOWAIT"))           {clauseName = "NOWAIT";    nowaitFlag = true;}
                else if (clauseName.equals("DIR_SCHEDULE"))          clauseName = "SCHEDULE";
            
                if (!clauseName.equals("NOWAIT") && !clauseName.equals("COPYPRIVATE")){
		  writer.writeToken(clauseName);
                
		  Node arg = childNode.getFirstChild().getNextSibling();
		  if (arg != null){
		    writer.writeToken("(");
		    if (operator != "") writer.writeToken(operator + " :");
                    
		    NodeList varList = arg.getChildNodes();

		    if (clauseName.equals("SCHEDULE")){
		      String sched = XmDomUtil.getContentText(varList.item(0));
		      if (sched.equals("0")) sched = "";
		      else if (sched.equals("1")) sched = "STATIC";
		      else if (sched.equals("2")) sched = "DYNAMIC";
		      else if (sched.equals("3")) sched = "GUIDED";
		      else if (sched.equals("4")) sched = "RUNTIME";
		      else if (sched.equals("5")) sched = "AFFINITY";
		      writer.writeToken(sched);
		    }
		    else if (clauseName.equals("DEFAULT")){
		      String attr = XmDomUtil.getContentText(varList.item(0));
		      if (attr.equals("0")) attr = "SHARED";
		      else if (attr.equals("1")) attr = "";
		      else if (attr.equals("2")) attr = "PRIVATE";
		      writer.writeToken(attr);
		    }
		    else {
		      invokeEnter(varList.item(0));
		    }

		    for (int j = 1; j < varList.getLength(); j++){
		      Node var = varList.item(j);
		      writer.writeToken(",");
		      invokeEnter(var);
		    }
                
		    writer.writeToken(")");
		  }
                }
                
            }

	    writer.setStatementMode(prevMode);
            
            writer.setupNewLine();

            // body
            Node body = clause.getNextSibling();

            writer.incrementIndentLevel();

            NodeList list2 = body.getChildNodes();
            for (int i = 0; i < list2.getLength(); i++){
                Node childNode = list2.item(i);
                if (childNode.getNodeType() != Node.ELEMENT_NODE) {
                    continue;
                }
                invokeEnter(childNode);
            }

            writer.decrementIndentLevel();

	    if (!dirName.equals("ATOMIC")){
	      writer.writeToken("!$OMP END " + dirName);
	      if (nowaitFlag) writer.writeToken("NOWAIT");
	      if (copyprivateFlag){
		writer.writeToken("COPYPRIVATE (");
		NodeList varList = copyprivate_arg.getChildNodes();
		invokeEnter(varList.item(0));
		for (int j = 1; j < varList.getLength(); j++){
		  Node var = varList.item(j);
		  writer.writeToken(",");
		  invokeEnter(var);
		}
		writer.writeToken(")");
	      }
	      writer.setupNewLine();
	    }
	    //	    writer.setStatementMode(prevMode);

        }
    }

    // FprintStatement
    class FprintStatementVisitor extends XcodeNodeVisitor {
        /**
         * Decompile "FprintStatement" element in XcodeML/F.
         *
         * @example <code><div class="Example">
         * 1000 FORMAT (&lt;any format string&gt;)<br/>
         * <div class="Strong">
         * PRINT *, "any text", variable1<br/>
         * PRINT 1000, variable2<br/>
         * </div>
         * </div></code>
         *
         * @see xcodeml.f.binding.gen.RVisitorBase#enter(xcodeml.f.binding.gen.
         *      XbfFprintStatement)
         */
        @Override public void enter(Node n) {
            _writeLineDirective(n);

            XmfWriter writer = _context.getWriter();
            writer.writeToken("PRINT ");
            writer.writeToken(XmDomUtil.getAttr(n, "format"));

            Node valueListNode = XmDomUtil.getElement(n, "valueList");
            if (valueListNode != null &&
                XmDomUtil.collectChildNodes(n).size() > 0) {
                writer.writeToken(", ");
            }

            invokeEnter(valueListNode);

            writer.setupNewLine();
        }
    }

    // FreadStatement
    class FreadStatementVisitor extends XcodeNodeVisitor {
        /**
         * Decompile "FreadStatement" element in XcodeML/F.
         *
         * @example <code><div class="Example">
         * <div class="Strong">
         * READ (UNIT=1, ...)<br/>
         * </div>
         * </div></code>
         * @see xcodeml.f.binding.gen.RVisitorBase#enter(xcodeml.f.binding.gen.
         *      XbfFreadStatement)
         */
        @Override public void enter(Node n) {
            _writeLineDirective(n);

            XmfWriter writer = _context.getWriter();

            writer.writeToken("READ ");
            invokeEnter(XmDomUtil.getElement(n, "namedValueList"));

            writer.writeToken(" ");
            invokeEnter(XmDomUtil.getElement(n, "valueList"));

            writer.setupNewLine();
        }
    }

    // FrealConstant
    class FrealConstantVisitor extends XcodeNodeVisitor {
        /**
         * Decompile "FrealConstant" element in XcodeML/F.
         *
         * @example <code><div class="Example">
         * PROGRAM main<br/>
         * <div class="Indent1">
         *      REAL real_variable = <span class="Strong">1.0</span><br/>
         * </div>
         * END PROGRAM main<br/>
         * </div></code>
         * @see xcodeml.f.binding.gen.RVisitorBase#enter(xcodeml.f.binding.gen.
         *      XbfFrealConstant)
         */
        @Override public void enter(Node n) {
            String content = XmDomUtil.getContentText(n);
            if (XfUtil.isNullOrEmpty(content)) {
                _context.setLastErrorMessage(
                    XfUtilForDom.formatError(n,
                                             XfError.XCODEML_SEMANTICS,
                                             n.getNodeName()));
                fail(n);
            }

            String typeName = XmDomUtil.getAttr(n, "type");
            if (XfUtil.isNullOrEmpty(typeName) == false) {
                XfTypeManagerForDom typeManager = _context.getTypeManagerForDom();
                String bottomTypeName = typeManager.getBottomTypeName(typeName);
                if (bottomTypeName == null) {
                    _context.setLastErrorMessage(
                        XfUtilForDom.formatError(n,
                                                 XfError.XCODEML_TYPE_NOT_FOUND,
                                                 n.getNodeName(),
                                                 typeName));
                    fail(n);
                }

                XfType typeId = XfType.getTypeIdFromXcodemlTypeName(bottomTypeName);
                if (typeId != XfType.DERIVED && typeId != XfType.REAL) {
                    _context.setLastErrorMessage(
                        XfUtilForDom.formatError(n,
                                                 XfError.XCODEML_TYPE_MISMATCH,
                                                 n.getNodeName(),
                                                 typeName,
                                                 "Fint"));
                    fail(n);
                }
            }

            XmfWriter writer = _context.getWriter();

            String kind = XmDomUtil.getAttr(n, "kind");
            // gfortran rejects kind with 'd'/'q' exponent
            if (XfUtil.isNullOrEmpty(kind) == false &&
                ((content.toLowerCase().indexOf("d") < 0) &&
                 (content.toLowerCase().indexOf("q") < 0))) {
                writer.writeToken(content + "_" + kind);
            } else {
                writer.writeToken(content);
            }
        }
    }

    // FreturnStatement
    class FreturnStatementVisitor extends XcodeNodeVisitor {
        /**
         * Decompile "FreturnStatement" element in XcodeML/F.
         *
         * @example <code><div class="Example">
         * SUBROUTINE sub()<br/>
         * <div class="Indent1">
         *      (any statement...)<br/>
         *      <div class="Strong">
         *      RETURN<br/>
         *      </div>
         * </div>
         * END SUBROUTINE sub<br/>
         * </div></code>
         * @see xcodeml.f.binding.gen.RVisitorBase#enter(xcodeml.f.binding.gen.
         *      XbfFreturnStatement)
         */
        @Override public void enter(Node n) {
            _writeLineDirective(n);

            XmfWriter writer = _context.getWriter();
            writer.writeToken("RETURN");
            writer.setupNewLine();
        }
    }

    // FrewindStatement
    class FrewindStatementVisitor extends XcodeNodeVisitor {
        /**
         * Decompile "FrewindStatement" element in XcodeML/F.
         *
         * @example <code><div class="Example">
         * <div class="Strong">
         * REWIND (UNIT=1, ...)<br/>
         * </div>
         * </div></code>
         * @see xcodeml.f.binding.gen.RVisitorBase#enter(xcodeml.f.binding.gen.
         *      XbfFrewindStatement)
         */
        @Override public void enter(Node n) {
            _writeLineDirective(n);

            XmfWriter writer = _context.getWriter();

            writer.writeToken("REWIND ");
            invokeEnter(XmDomUtil.getElement(n, "namedValueList"));

            writer.setupNewLine();
        }
    }

    // FselectCaseStatement
    class FselectCaseStatementVisitor extends XcodeNodeVisitor {
        /**
         * Decompile "FselectCaseStatement" element in XcodeML/F.
         *
         * @example <code><div class="Example">
         * <div class="Strong">
         * SELECT CASE (variable)<br/>
         * </div>
         * CASE (1)<br/>
         * <div class="Indent1">
         *      (any statement...)<br/>
         * </div>
         * CASE (2)<br/>
         * <div class="Indent1">
         *      (any statement...)<br/>
         * </div>
         * CASE DEFAULT<br/>
         * <div class="Indent1">
         *      (any statement...)<br/>
         * </div>
         * <div class="Strong">
         * END SELECT<br/>
         * </div>
         * </div></code>
         * @see xcodeml.f.binding.gen.RVisitorBase#enter(xcodeml.f.binding.gen.
         *      XbfFselectCaseStatement)
         */
        @Override public void enter(Node n) {
            _writeLineDirective(n);

            XmfWriter writer = _context.getWriter();
            String constuctName = XmDomUtil.getAttr(n, "construct_name");
            if (XfUtil.isNullOrEmpty(constuctName) == false) {
                writer.writeToken(constuctName);
                writer.writeToken(": ");
            }

            writer.writeToken("SELECT CASE (");
            invokeEnter(XmDomUtil.getElement(n, "value"));

            writer.writeToken(")");
            writer.setupNewLine();

            ArrayList<Node> caseLabelNodes =
                XmDomUtil.collectElements(n, "FcaseLabel");
            _invokeEnter(caseLabelNodes);

            writer.writeToken("END SELECT");
            if (XfUtil.isNullOrEmpty(constuctName) == false) {
                writer.writeToken(" ");
                writer.writeToken(constuctName);
            }
            writer.setupNewLine();
        }
    }

    // FstopStatement
    class FstopStatementVisitor extends XcodeNodeVisitor {
        /**
         * Decompile "FstopStatement" element in XcodeML/F.
         *
         * @example <code><div class="Example">
         * <div class="Strong">
         * STOP "error."<br/>
         * </div>
         * </div></code>
         * @see xcodeml.f.binding.gen.RVisitorBase#enter(xcodeml.f.binding.gen.
         *      XbfFstopStatement)
         */
        @Override public void enter(Node n) {
            _writeLineDirective(n);

            XmfWriter writer = _context.getWriter();
            writer.writeToken("STOP ");

            String code = XmDomUtil.getAttr(n, "code");
            String message = XmDomUtil.getAttr(n, "message");
            if (XfUtil.isNullOrEmpty(code) == false) {
                writer.writeToken(code);
            } else if (XfUtil.isNullOrEmpty(message) == false) {
                writer.writeLiteralString(message);
            }

            writer.setupNewLine();
        }
    }

    // FpauseStatement
    class FpauseStatementVisitor extends XcodeNodeVisitor {
        /**
         * Decompile "FpauseStatement" element in XcodeML/F.
         *
         * @example <code><div class="Example">
         * <div class="Strong">
         * PAUSE 1234<br/>
         * </div>
         * </div></code>
         * @see xcodeml.f.binding.gen.RVisitorBase#enter(xcodeml.f.binding.gen.
         *      XbfFpauseStatement)
         */
        @Override public void enter(Node n) {
            _writeLineDirective(n);

            XmfWriter writer = _context.getWriter();
            writer.writeToken("PAUSE ");

            String code = XmDomUtil.getAttr(n, "code");
            String message = XmDomUtil.getAttr(n, "message");
            if (XfUtil.isNullOrEmpty(code) == false) {
                writer.writeToken(code);
            } else if (XfUtil.isNullOrEmpty(message) == false) {
                writer.writeLiteralString(message);
            } else {
                writer.writeToken("0");
            }

            writer.setupNewLine();
        }
    }

    // FstructConstructor
    class FstructConstructorVisitor extends XcodeNodeVisitor {
        /**
         * Decompile "FstructConstructor" element in XcodeML/F.
         *
         * @example <code><div class="Example">
         *      struct = <span class="Strong">TYPE_NAME(1, 2, "abc")</span><br/>
         * </div></code>
         * @see xcodeml.f.binding.gen.RVisitorBase#enter(xcodeml.f.binding.gen.
         *      XbfFstructConstructor)
         */
        @Override public void enter(Node n) {
            XfTypeManagerForDom typeManager = _context.getTypeManagerForDom();

            Node typeChoice = typeManager.findType(XmDomUtil.getAttr(n, "type"));
            if (typeChoice == null) {
                _context.setLastErrorMessage(
                    XfUtilForDom.formatError(n,
                                             XfError.XCODEML_TYPE_NOT_FOUND,
                                             XmDomUtil.getAttr(n, "type")));
                fail(n);
            } else if (!"FstructType".equals(typeChoice.getNodeName())) {
                _context.setLastErrorMessage(
                    XfUtilForDom.formatError(n,
                                             XfError.XCODEML_TYPE_MISMATCH,
                                             "struct definition",
                                             n.getNodeName(),
                                             "FstructType"));
                fail(n);
            }

            Node structTypeNode = typeChoice;
            String aliasStructTypeName =
                typeManager.getAliasTypeName(XmDomUtil.getAttr(structTypeNode,
                                                               "type"));

            XmfWriter writer = _context.getWriter();
            writer.writeToken(aliasStructTypeName);
            writer.writeToken("(");

            _invokeChildEnterAndWriteDelim(n, ", ");

            writer.writeToken(")");
        }
    }

    // FstructDecl
    class FstructDeclVisitor extends XcodeNodeVisitor {
        /**
         * Decompile "FstructDecl" element in XcodeML/F.
         *
         * @example <code><div class="Example">
         * PROGRAM main<br/>
         * <div class="Indent1">
         *      <div class="Strong">
         *      TYPE derived_type</br>
         *      </div>
         *      <div class="Indent1">
         *      INTEGER :: int_variable
         *      </div>
         *      <div class="Strong">
         *      END TYPE derived_type</br>
         *      </div>
         *      TYPE(derived_type) derived_variable<br/>
         *      (any statement...)<br/>
         * </div>
         * END PROGRAM main<br/>
         * </div></code>
         * @see xcodeml.f.binding.gen.RVisitorBase#enter(xcodeml.f.binding.gen.XbfFstructDecl
         *      )
         */
        @Override public void enter(Node n) {
            _writeLineDirective(n);

            Node nameNode = XmDomUtil.getElement(n, "name");
            if (_validator.validateAttr(nameNode) == false) {
                _context.debugPrintLine("Detected insufficient attributes");
                _context.setLastErrorMessage(_validator.getErrDesc());
                fail(n);
            }

            String typeId = XmDomUtil.getAttr(nameNode, "type");
            XfTypeManagerForDom typeManager = _context.getTypeManagerForDom();
            Node typeChoice = typeManager.findType(typeId);
            if (typeChoice == null) {
                _context.setLastErrorMessage(
                    XfUtilForDom.formatError(n,
                                             XfError.XCODEML_TYPE_NOT_FOUND,
                                             XmDomUtil.getAttr(nameNode, "type")));
                fail(n);
            } else if (!"FstructType".equals(typeChoice.getNodeName())) {
                _context.setLastErrorMessage(
                    XfUtilForDom.formatError(n,
                                             XfError.XCODEML_TYPE_MISMATCH,
                                             "struct definition",
                                             typeChoice.getNodeName(),
                                             "FstructType"));
                fail(n);
            }

            Node structTypeNode = typeChoice;
            String structTypeName = XmDomUtil.getContentText(nameNode);

            typeManager.putAliasTypeName(typeId, structTypeName);

            XmfWriter writer = _context.getWriter();
            writer.writeToken("TYPE");

            if (_isUnderModuleDef()) {
                if (XmDomUtil.getAttrBool(structTypeNode, "is_private")) {
                    writer.writeToken(", PRIVATE");
                } else if (XmDomUtil.getAttrBool(structTypeNode, "is_public")) {
                    writer.writeToken(", PUBLIC");
                }
            }

            writer.writeToken(" :: ");
            writer.writeToken(structTypeName);
            writer.setupNewLine();
            writer.incrementIndentLevel();

            if (_isUnderModuleDef()) {
                if (XmDomUtil.getAttrBool(structTypeNode, "is_internal_private")) {
                    writer.writeToken("PRIVATE");
                    writer.setupNewLine();
                }
            }

            if (XmDomUtil.getAttrBool(structTypeNode, "is_sequence")) {
                writer.writeToken("SEQUENCE");
                writer.setupNewLine();
            }

            invokeEnter(structTypeNode);

            writer.decrementIndentLevel();
            writer.writeToken("END TYPE");
            writer.writeToken(" ");
            writer.writeToken(structTypeName);
            writer.setupNewLine();
        }
    }

    // functionCall
    class FunctionCallVisitor extends XcodeNodeVisitor {
        /**
         * Decompile "functionCall" element in XcodeML/F.
         *
         * @example <code><div class="Example">
         *      variable = <span class="Strong">function(arg1, arg2)</span><br/>
         *      <span class="Strong">call subroutine(arg1, arg2)</span><br/>
         * </div></code>
         * @see xcodeml.f.binding.gen.RVisitorBase#enter(xcodeml.f.binding.gen.
         *      XbfFunctionCall)
         */
        @Override public void enter(Node n) {
            Node functionNameNode = XmDomUtil.getElement(n, "name");
            if (functionNameNode == null) {
                _context.debugPrintLine("Detected a function call without the name element.");
                _context.setLastErrorMessage(
                    XfUtilForDom.formatError(n,
                                             XfError.XCODEML_SEMANTICS,
                                             n.getNodeName()));
                fail(n);
            }

            String functionName = XmDomUtil.getContentText(functionNameNode);
            if (XfUtil.isNullOrEmpty(functionName)) {
                _context.debugPrintLine("Function name is empty.");
                _context.setLastErrorMessage(
                    XfUtilForDom.formatError(functionNameNode,
                                             XfError.XCODEML_SEMANTICS,
                                             functionNameNode.getNodeName()));
                fail(functionNameNode);
            }

            // Note:
            // If it is built-in function, it is not on the type table.
            if (XmDomUtil.getAttrBool(n, "is_intrinsic") == false) {
                XfTypeManagerForDom typeManager = _context.getTypeManagerForDom();
                Node typeChoice = typeManager.findType(functionNameNode);
                if (typeChoice == null) {
                    _context.setLastErrorMessage(
                        XfUtilForDom.formatError(n,
                                                 XfError.XCODEML_TYPE_NOT_FOUND,
                                                 XmDomUtil.getAttr(functionNameNode, "type")));
                    fail(n);
                } else if ("FfunctionType".equals(typeChoice.getNodeName()) == false) {
                    _context.setLastErrorMessage(
                        XfUtilForDom.formatError(n,
                                                 XfError.XCODEML_TYPE_MISMATCH,
                                                 "function definition",
                                                 typeChoice.getNodeName(),
                                                 "FfunctionType"));
                    fail(n);
                }

                Node functionTypeNode = typeChoice;
                if (XmDomUtil.getAttrBool(functionTypeNode, "is_program")) {
                    // =======
                    // PROGRAM
                    // =======
                    _context.setLastErrorMessage(
                        XfUtilForDom.formatError(n,
                                                 XfError.XCODEML_TYPE_MISMATCH,
                                                 "function definition",
                                                 "PROGRAM",
                                                 "FUNCTION or SUBROUTINE"));
                    fail(n);
                }
            }

            XmfWriter writer = _context.getWriter();
            String returnTypeName = XmDomUtil.getAttr(n, "type");
            XfType typeId = XfType.getTypeIdFromXcodemlTypeName(returnTypeName);
            if (typeId == XfType.VOID) {
                // ==========
                // SUBROUTINE
                // ==========
                writer.writeToken("CALL ");
            } else {
                // ========
                // FUNCTION
                // ========
            }

            writer.writeToken(functionName);
            writer.writeToken("(");

            invokeEnter(XmDomUtil.getElement(n, "arguments"));

            writer.writeToken(")");
        }
    }

    // FuseDecl
    class FuseDeclVisitor extends XcodeNodeVisitor {
        /**
         * Decompile "FuseDecl" element in XcodeML/F.
         *
         * @example <code><div class="Example">
         * MODULE mod<br/>
         * <div class="Indent1">
         *      TYPE mod_derived_type<br/>
         *      END TYPE mod_derived_type<br/>
         *      (any statement...)<br/>
         * </div>
         * END MODULE mod<br/>
         * <br/>
         * PROGRAM main<br/>
         * <div class="Indent1">
         *      <div class="Strong">
         *      USE mod, derived_type => mod_derived_type<br/>
         *      </div>
         *      TYPE(derived_type) derived_variable<br/>
         *      (any statement...)<br/>
         * </div>
         * END PROGRAM main<br/>
         * </div></code>
         * @see xcodeml.f.binding.gen.RVisitorBase#enter(xcodeml.f.binding.gen.XbfFuseDecl
         *      )
         */
        @Override public void enter(Node n) {
            _writeLineDirective(n);

            XmfWriter writer = _context.getWriter();
            writer.writeToken("USE ");
            writer.writeToken(XmDomUtil.getAttr(n, "name"));

            ArrayList<Node> renameNodes = XmDomUtil.collectElements(n, "rename");
            for (Node renameNode : renameNodes) {
                if (_validator.validateAttr(renameNode) == false) {
                    _context.debugPrintLine("Detected insufficient attributes");
                    _context.setLastErrorMessage(_validator.getErrDesc());
                    fail(n);
                }
                String localName = XmDomUtil.getAttr(renameNode, "local_name");
                String useName = XmDomUtil.getAttr(renameNode, "use_name");
                writer.writeToken(", ");
                if (XfUtil.isNullOrEmpty(localName) == false) {
                    writer.writeToken(localName);
                    writer.writeToken(" => ");
                }
                if (XfUtil.isNullOrEmpty(useName) == false) {
                    writer.writeToken(useName);
                }
            }
            writer.setupNewLine();
        }
    }

    // FuseOnlyDecl
    class FuseOnlyDeclVisitor extends XcodeNodeVisitor {
        /**
         * Decompile "FuseOnlyDecl" element in XcodeML/F.
         *
         * @example <code><div class="Example">
         * MODULE mod<br/>
         * <div class="Indent1">
         *      TYPE mod_derived_type<br/>
         *      END TYPE mod_derived_type<br/>
         *      (any statement...)<br/>
         * </div>
         * END MODULE mod<br/>
         * <br/>
         * PROGRAM main<br/>
         * <div class="Indent1">
         *      <div class="Strong">
         *      USE mod, ONLY: derived_type => mod_derived_type<br/>
         *      </div>
         *      TYPE(derived_type) derived_variable<br/>
         *      (any statement...)<br/>
         * </div>
         * END PROGRAM main<br/>
         * </div></code>
         * @see xcodeml.f.binding.gen.RVisitorBase#enter(xcodeml.f.binding.gen.
         *      XbfFuseOnlyDecl)
         */
        @Override public void enter(Node n) {
            _writeLineDirective(n);

            XmfWriter writer = _context.getWriter();
            writer.writeToken("USE ");
            writer.writeToken(XmDomUtil.getAttr(n, "name"));
            writer.writeToken(", ONLY: ");

            int renamableCount = 0;
            ArrayList<Node> renamableNodes =
                XmDomUtil.collectElements(n, "renamable");
            for (Node renamableNode : renamableNodes) {
                if (_validator.validateAttr(renamableNode) == false) {
                    _context.debugPrintLine("Detected insufficient attributes");
                    _context.setLastErrorMessage(_validator.getErrDesc());
                    fail(n);
                }

                if (renamableCount > 0) {
                    writer.writeToken(", ");
                }
                String localName = XmDomUtil.getAttr(renamableNode, "local_name");
                String useName = XmDomUtil.getAttr(renamableNode, "use_name");
                if (XfUtil.isNullOrEmpty(localName) == false) {
                    writer.writeToken(localName);
                    writer.writeToken(" => ");
                }
                if (XfUtil.isNullOrEmpty(useName) == false) {
                    writer.writeToken(useName);
                }
                ++renamableCount;
            }
            writer.setupNewLine();
        }
    }

    // FwhereStatement
    class FwhereStatementVisitor extends XcodeNodeVisitor {
        /**
         * Decompile "FwhereStatement" element in XcodeML/F.
         *
         * @example <code><div class="Example">
         * <div class="Strong">
         * WHERE (array > 0)<br/>
         * <div class="Indent1">
         *     array = 0<br/>
         * </div>
         * ELSEWHERE<br/>
         * <div class="Indent1">
         *     array = 1<br/>
         * </div>
         * END WHERE<br/>
         * </div>
         * </div></code>
         * @see xcodeml.f.binding.gen.RVisitorBase#enter(xcodeml.f.binding.gen.
         *      XbfFwhereStatement)
         */
        @Override public void enter(Node n) {
            _writeLineDirective(n);

            XmfWriter writer = _context.getWriter();

            writer.writeToken("WHERE ");
            invokeEnter(XmDomUtil.getElement(n, "condition"));

            writer.setupNewLine();

            invokeEnter(XmDomUtil.getElement(n, "then"));

            Node elseNode = XmDomUtil.getElement(n, "else");
            if (elseNode != null) {
                writer.writeToken("ELSEWHERE");
                writer.setupNewLine();
                invokeEnter(elseNode);
            }

            writer.writeToken("END WHERE");
            writer.setupNewLine();
        }
    }

    // FwriteStatement
    class FwriteStatementVisitor extends XcodeNodeVisitor {
        /**
         * Decompile "FwriteStatement" element in XcodeML/F.
         *
         * @example <code><div class="Example">
         * <div class="Strong">
         * WRITE (UNIT=1, ...)<br/>
         * </div>
         * </div></code>
         * @see xcodeml.f.binding.gen.RVisitorBase#enter(xcodeml.f.binding.gen.
         *      XbfFwriteStatement)
         */
        @Override public void enter(Node n) {
            _writeLineDirective(n);

            XmfWriter writer = _context.getWriter();

            writer.writeToken("WRITE ");
            invokeEnter(XmDomUtil.getElement(n, "namedValueList"));

            writer.writeToken(" ");
            invokeEnter(XmDomUtil.getElement(n, "valueList"));

            writer.setupNewLine();
        }
    }

    // gotoStatement
    class GotoStatementVisitor extends XcodeNodeVisitor {
        /**
         * Decompile "gotoStatement" element in XcodeML/F.
         *
         * @example <code><div class="Example">
         * <div class="Strong">
         * GOTO 1000<br/>
         * </div>
         * 1000 CONTINUE<br/>
         * <br/>
         * <div class="Strong">
         * GOTO (2000, 2001, 2002), variable<br/>
         * </div>
         * 2000 (any statement...)<br/>
         * 2001 (any statement...)<br/>
         * 2002 (any statement...)<br/>
         * </div></code>
         * @see xcodeml.f.binding.gen.RVisitorBase#enter(xcodeml.f.binding.gen.
         *      XbfGotoStatement)
         */
        @Override public void enter(Node n) {
            _writeLineDirective(n);

            XmfWriter writer = _context.getWriter();

            writer.writeToken("GOTO ");
            String labelName = XmDomUtil.getAttr(n, "label_name");
            if (XfUtil.isNullOrEmpty(labelName) == false) {
                writer.writeToken(labelName);
            } else {
                ArrayList<Node> childNodes = XmDomUtil.collectChildNodes(n);
                for (Iterator<Node> iter = childNodes.iterator(); iter.hasNext(); ) {
                    Node paramsNode = iter.next();
                    if (!"params".equals(paramsNode.getNodeName())) {
                        throw new XmTranslationException(n,
                                                         "Invalid contents");
                    }
                    if (!iter.hasNext()) {
                        throw new XmTranslationException(n,
                                                         "Invalid contents");
                    }
                    Node valueNode = iter.next();
                    if (!"value".equals(valueNode.getNodeName())) {
                        throw new XmTranslationException(n,
                                                         "Invalid contents");
                    }

                    writer.writeToken("(");
                    invokeEnter(paramsNode);
                    writer.writeToken("), ");
                    invokeEnter(valueNode);
                }
            }
            writer.setupNewLine();
        }
    }

    // id
    class IdVisitor extends XcodeNodeVisitor {
        /**
         * Decompile "id" element in XcodeML/F.
         *
         * @deprecated Because handle it at a upper level element, warn it when this
         *             method was called it.
         * @see xcodeml.f.binding.gen.RVisitorBase#enter(xcodeml.f.binding.gen.XbfId)
         */
        @Override public void enter(Node n) {
            assert false;
        }
    }

    // indexRange
    class IndexRangeVisitor extends XcodeNodeVisitor {
        /**
         * Decompile "indexRange" element in XcodeML/F.
         *
         * @example <code><div class="Example">
         *      array1 = int_array_variable(<span class="Strong">10</span>,
         *      <span class="Strong">1:10</span>,
         *      <span class="Strong">1:</span>,
         *      <span class="Strong">:</span>)<br/>
         *      array2 = int_array_variable(<span class="Strong">:10:2</span>,
         *      <span class="Strong">1:10:2</span>,
         *      <span class="Strong">1::2</span>,
         *      <span class="Strong">::2</span>)<br/>
         *      array3 = (/ I, I = <span class="Strong">1, 10, 2</span> /)<br/>
         * </div></code>
         * @see xcodeml.f.binding.gen.RVisitorBase#enter(xcodeml.f.binding.gen.XbfIndexRange
         *      )
         */
        @Override public void enter(Node n) {
            XmfWriter writer = _context.getWriter();

            String delim;
            if (_isInvokeNodeOf("FdoLoop", 1)) {
                // Parent node is FdoLoop
                delim = ", ";
            } else if (_isInvokeNodeOf("FdoStatement", 1)) {
                // Parent node is FdoStatement
                delim = ", ";
            } else {
                delim = ":";
            }

            if (XmDomUtil.getAttrBool(n, "is_assumed_shape") &&
                XmDomUtil.getAttrBool(n, "is_assumed_size")) {
                // semantics error.
                _context.debugPrintLine(
                    "'is_assumed_shape' and 'is_assumed_size' are logically exclusize attributes.");
                _context.setLastErrorMessage(
                    XfUtilForDom.formatError(n,
                                             XfError.XCODEML_SEMANTICS,
                                             n.getNodeName()));
                fail(n);
            }

            if (XmDomUtil.getAttrBool(n, "is_assumed_shape")) {
                invokeEnter(XmDomUtil.getElement(n, "lowerBound"));

                writer.writeToken(":");
                return;
            }

            if (XmDomUtil.getAttrBool(n, "is_assumed_size")) {
                Node lowerBound = XmDomUtil.getElement(n, "lowerBound");
                if (lowerBound != null) {
                    invokeEnter(lowerBound);
                    writer.writeToken(":");
                }
                writer.writeToken("*");
                return;
            }

            invokeEnter(XmDomUtil.getElement(n, "lowerBound"));

            writer.writeToken(delim);

            invokeEnter(XmDomUtil.getElement(n, "upperBound"));

            Node step = XmDomUtil.getElement(n, "step");
            if (step != null) {
                writer.writeToken(delim);
                invokeEnter(step);
            }
        }
    }

    // kind
    class KindVisitor extends XcodeNodeVisitor {
        /**
         * Decompile "kind" element in XcodeML/F.
         *
         * @example <code><div class="Example">
         * PROGRAM main<br/>
         * <div class="Indent1">
         *      INTEGER(KIND=<span class="Strong">8</span>) :: i
         * </div>
         * END PROGRAM main<br/>
         * </div></code>
         * @see xcodeml.f.binding.gen.RVisitorBase#enter(xcodeml.f.binding.gen.XbfKind)
         */
        @Override public void enter(Node n) {
            invokeEnter(XmDomUtil.getContent(n));
        }
    }

    // len
    class LenVisitor extends XcodeNodeVisitor {
        /**
         * Decompile "len" element in XcodeML/F.
         *
         * @example <code><div class="Example">
         * PROGRAM main<br/>
         * <div class="Indent1">
         *      CHARACTER(LEN=<span class="Strong">10</span>, KIND=1) :: string_variable
         * </div>
         * END PROGRAM main<br/>
         * </div></code>
         * @see xcodeml.f.binding.gen.RVisitorBase#enter(xcodeml.f.binding.gen.XbfLen)
         */
        @Override public void enter(Node n) {
            Node contentNode = XmDomUtil.getContent(n);
            if (contentNode == null) {
                XmfWriter writer = _context.getWriter();
                writer.writeToken("*");
            } else {
                invokeEnter(contentNode);
            }
        }
    }

    // logAndExpr
    class LogAndExprVisitor extends XcodeNodeVisitor {
        /**
         * Decompile "logAndExpr" element in XcodeML/F.
         * <p>
         * The decompilation result depends on a child element.
         * </p>
         *
         * @example <code><div class="Example">
         * variable = <span class="Strong">(&lt;any expression&gt; .AND. &lt;any expression&gt;)</span><br/>
         * </div></code>
         * @see xcodeml.f.binding.gen.RVisitorBase#enter(xcodeml.f.binding.gen.XbfLogAndExpr
         *      )
         */
        @Override public void enter(Node n) {
            _writeBinaryExpr(XmDomUtil.collectChildNodes(n), 0,
                             ".AND.", _checkBinaryExprRequireGrouping(n));
        }
    }

    // logEQExpr
    class LogEQExprVisitor extends XcodeNodeVisitor {
        /**
         * Decompile "logEQExpr" element in XcodeML/F.
         * <p>
         * The decompilation result depends on a child element.
         * </p>
         *
         * @example <code><div class="Example">
         * variable = <span class="Strong">(&lt;any expression&gt; .EQ. &lt;any expression&gt;)</span><br/>
         * </div></code>
         * @see xcodeml.f.binding.gen.RVisitorBase#enter(xcodeml.f.binding.gen.XbfLogEQExpr
         *      )
         */
        @Override public void enter(Node n) {
            _writeBinaryExpr(XmDomUtil.collectChildNodes(n), 0,
                             "==", _checkBinaryExprRequireGrouping(n));
        }
    }

    // logEQVExpr
    class LogEQVExprVisitor extends XcodeNodeVisitor {
        /**
         * Decompile "logEQVExpr" element in XcodeML/F.
         * <p>
         * The decompilation result depends on a child element.
         * </p>
         *
         * @example <code><div class="Example">
         * variable = <span class="Strong">(&lt;any expression&gt; .EQV. &lt;any expression&gt;)</span><br/>
         * </div></code>
         * @see xcodeml.f.binding.gen.RVisitorBase#enter(xcodeml.f.binding.gen.XbfLogEQVExpr
         *      )
         */
        @Override public void enter(Node n) {
            _writeBinaryExpr(XmDomUtil.collectChildNodes(n), 0,
                             ".EQV.", _checkBinaryExprRequireGrouping(n));
        }
    }

    // logGEExpr
    class LogGEExprVisitor extends XcodeNodeVisitor {
        /**
         * Decompile "logGEExpr" element in XcodeML/F.
         * <p>
         * The decompilation result depends on a child element.
         * </p>
         *
         * @example <code><div class="Example">
         * variable = <span class="Strong">(&lt;any expression&gt; &gt;= &lt;any expression&gt;)</span><br/>
         * </div></code>
         * @see xcodeml.f.binding.gen.RVisitorBase#enter(xcodeml.f.binding.gen.XbfLogGEExpr
         *      )
         */
        @Override public void enter(Node n) {
            _writeBinaryExpr(XmDomUtil.collectChildNodes(n), 0,
                             ">=", _checkBinaryExprRequireGrouping(n));
        }
    }

    // logGTExpr
    class LogGTExprVisitor extends XcodeNodeVisitor {
        /**
         * Decompile "logGTExpr" element in XcodeML/F.
         * <p>
         * The decompilation result depends on a child element.
         * </p>
         *
         * @example <code><div class="Example">
         * variable = <span class="Strong">(&lt;any expression&gt; &gt; &lt;any expression&gt;)</span><br/>
         * </div></code>
         * @see xcodeml.f.binding.gen.RVisitorBase#enter(xcodeml.f.binding.gen.XbfLogGTExpr
         *      )
         */
        @Override public void enter(Node n) {
            _writeBinaryExpr(XmDomUtil.collectChildNodes(n), 0,
                             ">", _checkBinaryExprRequireGrouping(n));
        }
    }

    // logLEExpr
    class LogLEExprVisitor extends XcodeNodeVisitor {
        /**
         * Decompile "logLEExpr" element in XcodeML/F.
         * <p>
         * The decompilation result depends on a child element.
         * </p>
         *
         * @example <code><div class="Example">
         * variable = <span class="Strong">(&lt;any expression&gt; &lt;= &lt;any expression&gt;)</span><br/>
         * </div></code>
         * @see xcodeml.f.binding.gen.RVisitorBase#enter(xcodeml.f.binding.gen.XbfLogLEExpr
         *      )
         */
        @Override public void enter(Node n) {
            _writeBinaryExpr(XmDomUtil.collectChildNodes(n), 0,
                             "<=", _checkBinaryExprRequireGrouping(n));
        }
    }

    // logLTExpr
    class LogLTExprVisitor extends XcodeNodeVisitor {
        /**
         * Decompile "logLTExpr" element in XcodeML/F.
         * <p>
         * The decompilation result depends on a child element.
         * </p>
         *
         * @example <code><div class="Example">
         * variable = <span class="Strong">(&lt;any expression&gt; &lt; &lt;any expression&gt;)</span><br/>
         * </div></code>
         * @see xcodeml.f.binding.gen.RVisitorBase#enter(xcodeml.f.binding.gen.XbfLogLTExpr
         *      )
         */
        @Override public void enter(Node n) {
            _writeBinaryExpr(XmDomUtil.collectChildNodes(n), 0,
                             "<", _checkBinaryExprRequireGrouping(n));
        }
    }

    // logNEQExpr
    class LogNEQExprVisitor extends XcodeNodeVisitor {
        /**
         * Decompile "logNEQExpr" element in XcodeML/F.
         * <p>
         * The decompilation result depends on a child element.
         * </p>
         *
         * @example <code><div class="Example">
         * variable = <span class="Strong">(&lt;any expression&gt; .NEQ. &lt;any expression&gt;)</span><br/>
         * </div></code>
         * @see xcodeml.f.binding.gen.RVisitorBase#enter(xcodeml.f.binding.gen.XbfLogNEQExpr
         *      )
         */
        @Override public void enter(Node n) {
            _writeBinaryExpr(XmDomUtil.collectChildNodes(n), 0,
                             "/=", _checkBinaryExprRequireGrouping(n));
        }
    }

    // logNEQVExpr
    class LogNEQVExprVisitor extends XcodeNodeVisitor {
        /**
         * Decompile "logNEQVExpr" element in XcodeML/F.
         * <p>
         * The decompilation result depends on a child element.
         * </p>
         *
         * @example <code><div class="Example">
         * variable = <span class="Strong">(&lt;any expression&gt; .NEQV. &lt;any expression&gt;)</span><br/>
         * </div></code>
         * @see xcodeml.f.binding.gen.RVisitorBase#enter(xcodeml.f.binding.gen.XbfLogNEQVExpr
         *      )
         */
        @Override public void enter(Node n) {
            _writeBinaryExpr(XmDomUtil.collectChildNodes(n), 0,
                             ".NEQV.", _checkBinaryExprRequireGrouping(n));
        }
    }

    // logNotExpr
    class LogNotExprVisitor extends XcodeNodeVisitor {
        /**
         * Decompile "logNotExpr" element in XcodeML/F.
         * <p>
         * The decompilation result depends on a child element.
         * </p>
         *
         * @example <code><div class="Example">
         * variable = <span class="Strong">(.NOT. &lt;any expression&gt;)</span><br/>
         * </div></code>
         * @see xcodeml.f.binding.gen.RVisitorBase#enter(xcodeml.f.binding.gen.XbfLogNotExpr
         *      )
         */
        @Override public void enter(Node n) {
            _writeUnaryExpr(XmDomUtil.getContent(n), ".NOT.", true);
        }
    }

    // logOrExpr
    class LogOrExprVisitor extends XcodeNodeVisitor {
        /**
         * Decompile "logOrExpr" element in XcodeML/F.
         * <p>
         * The decompilation result depends on a child element.
         * </p>
         *
         * @example <code><div class="Example">
         * variable = <span class="Strong">(&lt;any expression&gt; .OR. &lt;any expression&gt;)</span><br/>
         * </div></code>
         * @see xcodeml.f.binding.gen.RVisitorBase#enter(xcodeml.f.binding.gen.XbfLogOrExpr
         *      )
         */
        @Override public void enter(Node n) {
            _writeBinaryExpr(XmDomUtil.collectChildNodes(n), 0,
                             ".OR.", _checkBinaryExprRequireGrouping(n));
        }
    }

    // lowerBound
    class LowerBoundVisitor extends XcodeNodeVisitor {
        /**
         * Decompile "lowerBound" element in XcodeML/F.
         *
         * @example <code><div class="Example">
         *      array = int_array_variable(10,
         *      <span class="Strong">1</span>:10,
         *      <span class="Strong">1</span>:, :)<br/>
         * </div></code>
         * @see xcodeml.f.binding.gen.RVisitorBase#enter(xcodeml.f.binding.gen.XbfLowerBound
         *      )
         */
        @Override public void enter(Node n) {
            invokeEnter(XmDomUtil.getContent(n));
        }
    }

    // minusExpr
    class MinusExprVisitor extends XcodeNodeVisitor {
        /**
         * Decompile "minusExpr" element in XcodeML/F.
         * <p>
         * The decompilation result depends on a child element.
         * </p>
         *
         * @example <code><div class="Example">
         * variable = <span class="Strong">(&lt;any expression&gt; - &lt;any expression&gt;)</span><br/>
         * </div></code>
         * @see xcodeml.f.binding.gen.RVisitorBase#enter(xcodeml.f.binding.gen.XbfMinusExpr
         *      )
         */
        @Override public void enter(Node n) {
            _writeBinaryExpr(XmDomUtil.collectChildNodes(n), 0,
                             "-", _checkBinaryExprRequireGrouping(n));
        }
    }

    // mulExpr
    class MulExprVisitor extends XcodeNodeVisitor {
        /**
         * Decompile "mulExpr" element in XcodeML/F.
         * <p>
         * The decompilation result depends on a child element.
         * </p>
         *
         * @example <code><div class="Example">
         * variable = <span class="Strong">(&lt;any expression&gt; * &lt;any expression&gt;)</span><br/>
         * </div></code>
         * @see xcodeml.f.binding.gen.RVisitorBase#enter(xcodeml.f.binding.gen.XbfMulExpr
         *      )
         */
        @Override public void enter(Node n) {
            _writeBinaryExpr(XmDomUtil.collectChildNodes(n), 0,
                             "*", _checkBinaryExprRequireGrouping(n));
        }
    }

    // name
    class NameVisitor extends XcodeNodeVisitor {
        /**
         * Decompile "name" element in XcodeML/F.
         *
         * @deprecated Because handle it at a upper level element, warn it when this
         *             method was called it.
         * @see xcodeml.f.binding.gen.RVisitorBase#enter(xcodeml.f.binding.gen.XbfName)
         */
        @Override public void enter(Node n) {
            assert false;
        }
    }

    // namedValue
    class NamedValueVisitor extends XcodeNodeVisitor {
        /**
         * Decompile "namedValue" element in XcodeML/F.
         *
         * @example <code><div class="Example">
         * OPEN (<span class="Strong">UNIT=1</span>, <span class="Strong">...</span>)<br/>
         * </div></code>
         * @see xcodeml.f.binding.gen.RVisitorBase#enter(xcodeml.f.binding.gen.XbfNamedValue
         *      )
         */
        @Override public void enter(Node n) {
            XmfWriter writer = _context.getWriter();
            writer.writeToken(XmDomUtil.getAttr(n, "name"));
            writer.writeToken("=");

            Node contentNode = XmDomUtil.getContent(n);
            if (contentNode == null) {
                writer.writeToken(XmDomUtil.getAttr(n, "value"));
            } else {
                invokeEnter(contentNode);
            }
        }
    }

    // namedValueList
    class NamedValueListVisitor extends XcodeNodeVisitor {
        /**
         * Decompile "namedValueList" element in XcodeML/F.
         *
         * @example <code><div class="Example">
         * OPEN <span class="Strong">(UNIT=1, ...)</span><br/>
         * </div></code>
         * @see xcodeml.f.binding.gen.RVisitorBase#enter(xcodeml.f.binding.gen.
         *      XbfNamedValueList)
         */
        @Override public void enter(Node n) {
            XmfWriter writer = _context.getWriter();
            writer.writeToken("(");

            _invokeChildEnterAndWriteDelim(n, ", ");

            writer.writeToken(")");
        }
    }

    // params
    class ParamsVisitor extends XcodeNodeVisitor {
        /**
         * Decompile "params" element in XcodeML/F.
         *
         * @example <code><div class="Example">
         * PROGRAM main<br/>
         * <div class="Indent1">
         *      INTEGER :: int_variable<br/>
         *      (any statement...)<br/>
         * </div>
         * <br/>
         * CONTAINS
         * <div class="Indent1">
         *      SUBROUTINE sub()<br/>
         *      (any statement...)<br/>
         *      END SUBROUTINE sub<br/>
         *      <br/>
         *      FUNCTION func(<span class="Strong">a, b, c</span>)<br/>
         *      (any statement...)<br/>
         *      END SUBROUTINE sub<br/>
         * </div>
         * <br/>
         * END PROGRAM main<br/>
         * </div></code>
         *
         * @see xcodeml.f.binding.gen.RVisitorBase#enter(xcodeml.f.binding.gen.XbfParams)
         */
        @Override public void enter(Node n) {
            XmfWriter writer = _context.getWriter();

            int paramCount = 0;
            ArrayList<Node> nameNodes = XmDomUtil.collectElements(n, "name");
            for (Node nameNode : nameNodes) {
                if (paramCount > 0) {
                    writer.writeToken(", ");
                }
                writer.writeToken(XmDomUtil.getContentText(nameNode));
                ++paramCount;
            }
        }
    }

    // plusExpr
    class PlusExprVisitor extends XcodeNodeVisitor {
        /**
         * Decompile "plusExpr" element in XcodeML/F.
         * <p>
         * The decompilation result depends on a child element.
         * </p>
         *
         * @example <code><div class="Example">
         * variable = <span class="Strong">(&lt;any expression&gt; + &lt;any expression&gt;)</span><br/>
         * </div></code>
         * @see xcodeml.f.binding.gen.RVisitorBase#enter(xcodeml.f.binding.gen.XbfPlusExpr
         *      )
         */
        @Override public void enter(Node n) {
            _writeBinaryExpr(XmDomUtil.collectChildNodes(n), 0,
                             "+", _checkBinaryExprRequireGrouping(n));
        }
    }

    // renamable (deprecated)
    // rename (deprecated)

    // statementLabel
    class StatementLabelVisitor extends XcodeNodeVisitor {
        /**
         * Decompile "statementLabel" element in XcodeML/F.
         *
         * @example <code><div class="Example">
         * GOTO 1000
         * <span class="Strong">1000 </span>CONTINUE<br/>
         * (any statement...)<br/>
         * </div></code>
         * @see xcodeml.f.binding.gen.RVisitorBase#enter(xcodeml.f.binding.gen.
         *      XbfStatementLabel)
         */
        @Override public void enter(Node n) {
            _writeLineDirective(n);

            XmfWriter writer = _context.getWriter();
            writer.decrementIndentLevel();
            writer.writeToken(XmDomUtil.getAttr(n, "label_name"));
            writer.incrementIndentLevel();
            if ((_nextNode != null) &&
                "statementLabel".equals(_nextNode.getNodeName())) {
                // Note:
                // If next statement is statementLabel,
                // add continue statement.
                //
                // Cauntion!:
                // This change is due to the change of a declaraton.
                // A statement label of the declaration will be move to
                // the body block, thus XcodeML frontend generates
                // the statement label without a statement.
                // (and generate a declaration without a label).
                // To avoid compile errors occurred by this change,
                // the backend add continue statement to the label.
                writer.writeToken(" CONTINUE");
                writer.setupNewLine();
            } else {
                // Note:
                // Handling next statement as continuous line.
                writer.writeToken(" ");
                writer.setupContinuousNewLine();
            }
        }
    }

    // step
    class StepVisitor extends XcodeNodeVisitor {
        /**
         * Decompile "step" element in XcodeML/F.
         *
         * @example <code><div class="Example">
         *      array = int_array_variable(10,
         *      1:10:<span class="Strong">2</span>,
         *      ::<span class="Strong">2</span>, :)<br/>
         * </div></code>
         * @see xcodeml.f.binding.gen.RVisitorBase#enter(xcodeml.f.binding.gen.XbfStep)
         */
        @Override public void enter(Node n) {
            invokeEnter(XmDomUtil.getContent(n));
        }
    }

    // symbols
    class SymbolsVisitor extends XcodeNodeVisitor {
        /**
         * Decompile "symbols" element in XcodeML/F.
         * <p>
         * The decompilation result depends on a child element.
         * </p>
         *
         * @see xcodeml.f.binding.gen.RVisitorBase#enter(xcodeml.f.binding.gen.XbfSymbols
         *      )
         */
        @Override public void enter(Node n) {
            ArrayList<Node> idNodes = XmDomUtil.collectElements(n, "id");
            XfTypeManagerForDom typeManager =
                _context.getTypeManagerForDom();

            if (_isInvokeAncestorNodeOf("FstructDecl") == false) {
                _context.debugPrintLine("Add to symbol table.");

                for (Node idNode : idNodes) {
                    typeManager.addSymbol(idNode);
                }

                // _context.debugPrint(typeManager.toString());
            } else {
                _context.debugPrintLine("Write symbol.");
                for (Node idNode : idNodes) {
                    String typeName;

                    //printNode(System.err, idNode);

                    typeName = XmDomUtil.getAttr(idNode, "type");

                    Node nameNode = XmDomUtil.getElement(idNode, "name");
                    if (typeName == null) {
                        typeName = XmDomUtil.getAttr(nameNode, "type");

                        if (typeName == null) {
                            _context.setLastErrorMessage(
                                XfUtilForDom.formatError(idNode,
                                                         XfError.XCODEML_NEED_ATTR,
                                                         "type",
                                                         n.getNodeName()));
                            fail(n);
                        }
                    }

                    String symbolName = XmDomUtil.getContentText(nameNode);

                    XfSymbol symbol = _makeSymbol(symbolName, typeName);
                    if (symbol == null) {
                        _context.setLastErrorMessage(
                            XfUtilForDom.formatError(idNode,
                                                     XfError.XCODEML_TYPE_NOT_FOUND,
                                                     typeName));
                        fail(n);
                    }
                    _writeSymbolDecl(symbol, n);

                    Node valueNode = XmDomUtil.getElement(idNode, "value");
                    if (valueNode != null) {
                        XmfWriter writer = _context.getWriter();
                        Node tn = typeManager.findType(typeName);

                        if (XmDomUtil.getAttrBool(tn, "is_pointer") == true) {
                            writer.writeToken(" => ");
                        } else {
                            writer.writeToken(" = ");
                        }

                        invokeEnter(valueNode);
                    }

                    _context.getWriter().setupNewLine();
                }
            }
        }
    }

    // then
    class ThenVisitor extends XcodeNodeVisitor {
        /**
         * Decompile "then" element in XcodeML/F.
         *
         * @example <code><div class="Example">
         * IF (variable == 1) THEN<br/>
         * <div class="Indent1">
         *      <div class="Strong">
         *      (any statement...)<br/>
         *      </div>
         * </div>
         * ELSE<br/>
         * <div class="Indent1">
         *      (any statement...)<br/>
         * </div>
         * END IF<br/>
         * </div></code>
         * @see xcodeml.f.binding.gen.RVisitorBase#enter(xcodeml.f.binding.gen.XbfThen)
         */
        @Override public void enter(Node n) {
            XmfWriter writer = _context.getWriter();
            writer.incrementIndentLevel();

            invokeEnter(XmDomUtil.getElement(n, "body"));

            writer.decrementIndentLevel();
        }
    }

    // unaryMinusExpr
    class UnaryMinusExprVisitor extends XcodeNodeVisitor {
        /**
         * Decompile "unaryMinusExpr" element in XcodeML/F.
         * <p>
         * The decompilation result depends on a child element.
         * </p>
         *
         * @example <code><div class="Example">
         * variable = <span class="Strong">(-&lt;any expression&gt;)</span><br/>
         * </div></code>
         * @see xcodeml.f.binding.gen.RVisitorBase#enter(xcodeml.f.binding.gen.
         *      XbfUnaryMinusExpr)
         */
        @Override public void enter(Node n) {
            Node child = XmDomUtil.getContent(n);

            boolean grouping = true;
            if (_isConstantExpr(n.getParentNode()) &&
                _isConstantExpr(child)) {
                grouping = false;
            }

            _writeUnaryExpr(child, "-", grouping);
        }
    }

    // upperBound
    class UpperBoundVisitor extends XcodeNodeVisitor {
        /**
         * Decompile "upperBound" element in XcodeML/F.
         *
         * @example <code><div class="Example">
         *      array = int_array_variable(<span class="Strong">10</span>,
         *      1:<span class="Strong">10</span>,
         *      1:, :)<br/>
         * </div></code>
         * @see xcodeml.f.binding.gen.RVisitorBase#enter(xcodeml.f.binding.gen.XbfUpperBound
         *      )
         */
        @Override public void enter(Node n) {
            invokeEnter(XmDomUtil.getContent(n));
        }
    }

    // userBinaryExpr
    class UserBinaryExprVisitor extends XcodeNodeVisitor {
        /**
         * Decompile "userBinaryExpr" element in XcodeML/F.
         * <p>
         * The decompilation result depends on a child element.
         * </p>
         *
         * @example <code><div class="Example">
         * variable = <span class="Strong">(&lt;any expression&gt; &lt;user defined expr&gt; &lt;any expression&gt;)</span><br/>
         * </div></code>
         *
         * @see xcodeml.f.binding.gen.RVisitorBase#enter(xcodeml.f.binding.gen.
         *      XbfUserBinaryExpr)
         */
        @Override public void enter(Node n) {
            String name = XmDomUtil.getAttr(n, "name");
            if (XfUtil.isNullOrEmpty(name)) {
                _context.setLastErrorMessage(
                    XfUtilForDom.formatError(n,
                                             XfError.XCODEML_NEED_ATTR,
                                             "name",
                                             n.getNodeName()));
                fail(n);
            }

            _writeBinaryExpr(XmDomUtil.collectChildNodes(n), 0,
                             name, _checkBinaryExprRequireGrouping(n));
        }
    }

    // userUnaryExpr
    class UserUnaryExprVisitor extends XcodeNodeVisitor {
        /**
         * Decompile "userUnaryExpr" element in XcodeML/F.
         * <p>
         * The decompilation result depends on a child element.
         * </p>
         *
         * @example <code><div class="Example">
         * variable = <span class="Strong">(&lt;user defined expr&gt;&lt;any expression&gt;)</span><br/>
         * </div></code>
         *
         * @see xcodeml.f.binding.gen.RVisitorBase#enter(xcodeml.f.binding.gen.
         *      XbfUserUnaryExpr)
         */
        @Override public void enter(Node n) {
            String name = XmDomUtil.getAttr(n, "name");
            if (XfUtil.isNullOrEmpty(name)) {
                _context.setLastErrorMessage(
                    XfUtilForDom.formatError(n,
                                             XfError.XCODEML_NEED_ATTR,
                                             "name",
                                             n.getNodeName()));
                fail(n);
            }

            _writeUnaryExpr(XmDomUtil.getContent(n), name, true);
        }
    }

    // value
    class ValueVisitor extends XcodeNodeVisitor {
        /**
         * Decompile "value" element in XcodeML/F.
         * <p>
         * The decompilation result depends on a child element.
         * </p>
         *
         * @see xcodeml.f.binding.gen.RVisitorBase#enter(xcodeml.f.binding.gen.XbfValue)
         */
        @Override public void enter(Node n) {
            Node repeatCountNode = XmDomUtil.getElement(n, "repeat_count");
            if (repeatCountNode != null) {
                XmfWriter writer = _context.getWriter();
                invokeEnter(repeatCountNode);
                writer.writeToken("*");
            }
            ArrayList<Node> contentNodes =
                XmDomUtil.collectElementsExclude(n, "repeat_count");
            if (!contentNodes.isEmpty()) {
                assert contentNodes.size() == 1;
                invokeEnter(contentNodes.get(0));
            }
        }
    }

    // repeat_count
    class RepeatCountVisitor extends XcodeNodeVisitor {
        @Override public void enter(Node n) {
            invokeEnter(XmDomUtil.getContent(n));
        }
    }

    // valueList
    class ValueListVisitor extends XcodeNodeVisitor {
        /**
         * Decompile "valueList" element in XcodeML/F.
         * <p>
         * The decompilation result depends on a child element.
         * </p>
         *
         * @see xcodeml.f.binding.gen.RVisitorBase#enter(xcodeml.f.binding.gen.XbfValueList
         *      )
         */
        @Override public void enter(Node n) {
            _invokeChildEnterAndWriteDelim(n, ", ");
        }
    }

    // Ffunction
    class FfunctionVisitor extends XcodeNodeVisitor {
        /**
         * Decompile "Ffunction" element in XcodeML/F.
         * <p>
         * The decompilation result depends on a child element.
         * </p>
         *
         * @see xcodeml.f.binding.gen.RVisitorBase#enter(xcodeml.f.binding.gen.XbfFfunction)
         */
        @Override public void enter(Node n) {
            XmfWriter writer = _context.getWriter();
            writer.writeToken(XmDomUtil.getContentText(n));
        }
    }

    // Var
    class VarVisitor extends XcodeNodeVisitor {
        /**
         * Decompile "Var" element in XcodeML/F.
         * <p>
         * The decompilation result depends on a child element.
         * </p>
         *
         * @see xcodeml.f.binding.gen.RVisitorBase#enter(xcodeml.f.binding.gen.XbfVar)
         */
        @Override public void enter(Node n) {
            XmfWriter writer = _context.getWriter();
            writer.writeToken(XmDomUtil.getContentText(n));
        }
    }


    // varDecl
    class VarDeclVisitor extends XcodeNodeVisitor {
        /**
         * Decompile "varDecl" element in XcodeML/F.
         * <p>
         * The decompilation result depends on a child element.
         * </p>
         *
         * @example <code><div class="Example">
         * PROGRAM main<br/>
         * <div class="Indent1">
         *      <div class="Strong">
         *      INTEGER :: int_variable<br/>
         *      TYPE(USER_TYPE) :: derived_variable<br/>
         *      (any variant declaration...)<br/>
         *      </div>
         *      int_variable = 0<br/>
         *      (any statement...)<br/>
         * </div>
         * END PROGRAM main<br/>
         * </div></code>
         *
         * @see xcodeml.f.binding.gen.RVisitorBase#enter(xcodeml.f.binding.gen.XbfVarDecl
         *      )
         */
        @Override public void enter(Node n) {
            _writeLineDirective(n);

            Node nameNode = XmDomUtil.getElement(n, "name");
            XfSymbol symbol = _makeSymbol(nameNode);
            if (symbol == null) {
                _context.setLastErrorMessage(
                    XfUtilForDom.formatError(nameNode,
                                             XfError.XCODEML_NAME_NOT_FOUND,
                                             XmDomUtil.getContentText(nameNode)));
                fail(nameNode);
            }
            _writeSymbolDecl(symbol, n);

            Node valueNode = XmDomUtil.getElement(n, "value");
            if (valueNode != null) {
                XmfWriter writer = _context.getWriter();
                writer.writeToken(" = ");
                invokeEnter(valueNode);
            }

            _context.getWriter().setupNewLine();
        }
    }

    // varList
    class VarListVisitor extends XcodeNodeVisitor {
        /**
         * Decompile "varList" element in XcodeML/F.
         * <p>
         * The decompilation result depends on a child element.
         * </p>
         *
         * @see xcodeml.f.binding.gen.RVisitorBase#enter(xcodeml.f.binding.gen.XbfVarList
         *      )
         */
        @Override public void enter(Node n) {
            XmfWriter writer = _context.getWriter();
            String name = XmDomUtil.getAttr(n, "name");

            if (_isInvokeNodeOf("FcommonDecl", 1)) {
                // Parent node is XbfFcommonDecl
                writer.writeToken("/");
                if (XfUtil.isNullOrEmpty(name) == false) {
                    writer.writeToken(name);
                }
                writer.writeToken("/ ");
            } else if (_isInvokeNodeOf("FnamelistDecl", 1)) {
                // Parent node is XbfFnamelistDecl
                if (XfUtil.isNullOrEmpty(name)) {
                    _context.setLastErrorMessage(
                        XfUtilForDom.formatError(n,
                                                 XfError.XCODEML_NEED_ATTR,
                                                 "name",
                                                 n.getNodeName()));
                    fail(n);
                }

                writer.writeToken("/");
                writer.writeToken(name);
                writer.writeToken("/ ");
            } else if (_isInvokeNodeOf("FdataDecl", 1)) {
                // Parent node is FdataDecl
            } else if (_isInvokeNodeOf("FequivalenceDecl", 1)) {
                // Parent node is FequivalenceDecl
            } else {
                assert false;
            }

            _invokeChildEnterAndWriteDelim(n, ", ");
        }
    }

    // varRef
    class VarRefVisitor extends XcodeNodeVisitor {
        /**
         * Decompile 'varRef' element in XcodeML/F.
         * <p>
         * The decompilation result depends on a child element.
         * </p>
         *
         * @see xcodeml.f.binding.gen.RVisitorBase#enter(xcodeml.f.binding.gen.XbfVarRef)
         */
        @Override public void enter(Node n) {
            invokeEnter(XmDomUtil.getContent(n));
        }
    }


    @SuppressWarnings("unchecked")
    private Pair[] pairs = {
        new Pair("XcodeProgram", new XcodeProgramVisitor()),
        new Pair("typeTable", new TypeTableVisitor()),
        new Pair("FbasicType", new BasicTypeVisitor()),
        new Pair("coShape", new CoShapeVisitor()),
        new Pair("FfunctionType", new FfunctionTypeVisitor()),
        new Pair("FstructType", new FstructTypeVisitor()),
        new Pair("globalSymbols", new GlobalSymbolsVisitor()),
        new Pair("globalDeclarations", new GlobalDeclarationsVisitor()),
        new Pair("alloc", new AllocVisitor()),
        new Pair("arguments", new ArgumentsVisitor()),
        new Pair("arrayIndex", new ArrayIndexVisitor()),
        new Pair("FassignStatement", new FassignStatementVisitor()),
        new Pair("body", new BodyVisitor()),
        new Pair("condition", new ConditionVisitor()),
        new Pair("continueStatement", new ContinueStatement()),
        new Pair("declarations", new DeclarationsVisitor()),
        new Pair("divExpr", new DivExprVisitor()),
        new Pair("else", new ElseVisitor()),
        new Pair("exprStatement", new ExprStatementVisitor()),
        new Pair("externDecl", new ExternDeclVisitor()),
        new Pair("FallocateStatement", new FallocateStatement()),
        new Pair("FarrayConstructor", new FarrayConstructor()),
        new Pair("FarrayRef", new FarrayRefVisitor()),
        new Pair("FcoArrayRef", new FcoArrayRefVisitor()),
        new Pair("FbackspaceStatement", new FbackspaceStatement()),
        new Pair("FcaseLabel", new FcaseLabelVisitor()),
        new Pair("FcharacterConstant", new FcharacterConstant()),
        new Pair("FcharacterRef", new FcharacterRef()),
        new Pair("FcloseStatement", new FcloseStatementVisitor()),
        new Pair("FcommonDecl", new FcommonDeclVisitor()),
        new Pair("FcomplexConstant", new FcomplexConstantVisitor()),
        new Pair("FconcatExpr", new FconcatExprVisitor()),
        new Pair("FcontainsStatement", new FcontainsStatementVisitor()),
        new Pair("FcycleStatement", new FcycleStatementVisitor()),
        new Pair("FdataDecl", new FdataDeclVisitor()),
        new Pair("FdeallocateStatement", new FdeallocateStatementVisitor()),
        new Pair("FdoLoop", new FdoLoopVisitor()),
        new Pair("FdoStatement", new FdoStatementVisitor()),
        new Pair("FdoWhileStatement", new FdoWhileStatementVisitor()),
        new Pair("FendFileStatement", new FendFileStatementVisitor()),
        new Pair("FentryDecl", new FentryDeclVisitor()),
        new Pair("FequivalenceDecl", new FequivalenceDeclVisitor()),
        new Pair("FexitStatement", new FexitStatementVisitor()),
        new Pair("FformatDecl", new FformatDeclVisitor()),
        new Pair("FfunctionDecl", new FfunctionDeclVisitor()),
        new Pair("FfunctionDefinition", new FfunctionDefinitionVisitor()),
        new Pair("FifStatement", new FifStatementVisitor()),
        new Pair("FinquireStatement", new FinquireStatementVisitor()),
        new Pair("FintConstant", new FintConstantVisitor()),
        new Pair("FinterfaceDecl", new FinterfaceDeclVisitor()),
        new Pair("FlogicalConstant", new FlogicalConstantVisitor()),
        new Pair("FmemberRef", new FmemberRefVisitor()),
        new Pair("FmoduleDefinition", new FmoduleDefinitionVisitor()),
        new Pair("FmoduleProcedureDecl", new FmoduleProcedureDeclVisitor()),
        new Pair("FblockDataDefinition", new FblockDataDefinitionVisitor()),
        new Pair("FnamelistDecl", new FnamelistDeclVisitor()),
        new Pair("FnullifyStatement", new FnullifyStatementVisitor()),
        new Pair("FopenStatement", new FopenStatementVisitor()),
        new Pair("FpointerAssignStatement", new FpointerAssignStatementVisitor()),
        new Pair("FpowerExpr", new FpowerExprVisitor()),
        new Pair("FpragmaStatement", new FpragmaStatementVisitor()),
        new Pair("OMPPragma", new OMPPragmaVisitor()),
        new Pair("FprintStatement", new FprintStatementVisitor()),
        new Pair("FreadStatement", new FreadStatementVisitor()),
        new Pair("FrealConstant", new FrealConstantVisitor()),
        new Pair("FreturnStatement", new FreturnStatementVisitor()),
        new Pair("FrewindStatement", new FrewindStatementVisitor()),
        new Pair("FselectCaseStatement", new FselectCaseStatementVisitor()),
        new Pair("FstopStatement", new FstopStatementVisitor()),
        new Pair("FpauseStatement", new FpauseStatementVisitor()),
        new Pair("FstructConstructor", new FstructConstructorVisitor()),
        new Pair("FstructDecl", new FstructDeclVisitor()),
        new Pair("functionCall", new FunctionCallVisitor()),
        new Pair("FuseDecl", new FuseDeclVisitor()),
        new Pair("FuseOnlyDecl", new FuseOnlyDeclVisitor()),
        new Pair("FwhereStatement", new FwhereStatementVisitor()),
        new Pair("FwriteStatement", new FwriteStatementVisitor()),
        new Pair("gotoStatement", new GotoStatementVisitor()),
        new Pair("id", new IdVisitor()),
        new Pair("indexRange", new IndexRangeVisitor()),
        new Pair("kind", new KindVisitor()),
        new Pair("len", new LenVisitor()),
        new Pair("logAndExpr", new LogAndExprVisitor()),
        new Pair("logEQExpr", new LogEQExprVisitor()),
        new Pair("logEQVExpr", new LogEQVExprVisitor()),
        new Pair("logGEExpr", new LogGEExprVisitor()),
        new Pair("logGTExpr", new LogGTExprVisitor()),
        new Pair("logLEExpr", new LogLEExprVisitor()),
        new Pair("logLTExpr", new LogLTExprVisitor()),
        new Pair("logNEQExpr", new LogNEQExprVisitor()),
        new Pair("logNEQVExpr", new LogNEQVExprVisitor()),
        new Pair("logNotExpr", new LogNotExprVisitor()),
        new Pair("logOrExpr", new LogOrExprVisitor()),
        new Pair("lowerBound", new LowerBoundVisitor()),
        new Pair("minusExpr", new MinusExprVisitor()),
        new Pair("mulExpr", new MulExprVisitor()),
        new Pair("name", new NameVisitor()),
        new Pair("namedValue", new NamedValueVisitor()),
        new Pair("namedValueList", new NamedValueListVisitor()),
        new Pair("params", new ParamsVisitor()),
        new Pair("plusExpr", new PlusExprVisitor()),
        new Pair("statementLabel", new StatementLabelVisitor()),
        new Pair("step", new StepVisitor()),
        new Pair("symbols", new SymbolsVisitor()),
        new Pair("then", new ThenVisitor()),
        new Pair("unaryMinusExpr", new UnaryMinusExprVisitor()),
        new Pair("upperBound", new UpperBoundVisitor()),
        new Pair("userBinaryExpr", new UserBinaryExprVisitor()),
        new Pair("userUnaryExpr", new UserUnaryExprVisitor()),
        new Pair("value", new ValueVisitor()),
        new Pair("repeat_count", new RepeatCountVisitor()),
        new Pair("valueList", new ValueListVisitor()),
        new Pair("Ffunction", new FfunctionVisitor()),
        new Pair("Var", new VarVisitor()),
        new Pair("varDecl", new VarDeclVisitor()),
        new Pair("varList", new VarListVisitor()),
        new Pair("varRef", new VarRefVisitor()),
    };
}
