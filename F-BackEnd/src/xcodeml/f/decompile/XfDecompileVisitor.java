/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
package xcodeml.f.decompile;

import java.util.HashSet;
import java.util.Iterator;
import java.util.LinkedList;

import xcodeml.XmException;
import xcodeml.binding.IRNode;
import xcodeml.f.binding.gen.*;
import xcodeml.f.util.XmfWriter;

/**
 * Visitor pattern for XcodeML/F
 */
public class XfDecompileVisitor extends RVisitorBase
{
  static final int PRIO_LOW = 0; /* lowest */

  static final int PRIO_EQV = 1; /* EQV, NEQV */
  static final int PRIO_OR = 2; /* .OR. */
  static final int PRIO_AND = 3; /* .AND. */

  static final int PRIO_COMP = 4; /* <, >,...  */
  
  static final int PRIO_CONCAT = 5; 

  static final int PRIO_PLUS_MINUS = 6;
  static final int PRIO_MUL_DIV = 7;
  static final int PRIO_POWER = 8;
  static final int PRIO_HIGH = 10;

    @SuppressWarnings("serial")
    private class InvokeNodeStack extends LinkedList<IRNode>
    {
        @Override
        public String toString()
        {
            StringBuilder sb = new StringBuilder();
            sb.append("[Invoke Node Stack]\n");
            for (IRNode node : this.toArray(new IRNode[0])) {
                sb.append(XfUtil.getElementName(node));
                sb.append("\n");
            }
            return sb.toString();
        }
    }

    private XmfDecompilerContext _context;

    private InvokeNodeStack _invokeNodeStack;

    private XfRuntimeValidator _validator;

    private IRNode _nextNode;

    private HashSet<String> _moduleAllIds;
    private HashSet<String> _declaredIds;

    public XfDecompileVisitor(XmfDecompilerContext context)
    {
        _context = context;
        _invokeNodeStack = new InvokeNodeStack();
        _validator = new XfRuntimeValidator();
        _moduleAllIds = new HashSet<String>();
        _declaredIds = new HashSet<String>();
    }

    /**
     * Get IRNode instance in the invoke stack.
     *
     * @param parentRank
     *            Parent rank.
     *            <ul>
     *            <li>0: Current node.</li>
     *            <li>1: Parent node.</li>
     *            <li>>2: Any ancestor node.</li>
     *            </ul>
     * @return Instance of IRNode or null.
     */
    private IRNode _getInvokeNode(int parentRank)
    {
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
     * @param clazz
     *            Instance of IRNode
     * @param parentRank
     *            Parent rank.
     * @return true/false
     */
    private boolean _isInvokeNodeOf(Class<? extends IRNode> clazz, int parentRank)
    {
        IRNode node = _getInvokeNode(parentRank);
        return clazz.equals(node.getClass());
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
    private boolean _isUnderModuleDef()
    {
        return _isInvokeNodeOf(XbfFmoduleDefinition.class, 2);
    }

    /**
     * Check whether there is the class of designated element to a ancestor of
     * the invoke stack.
     *
     * @param clazz
     *            Instance of IRNode
     * @return true/false
     */
    private boolean _isInvokeAncestorNodeOf(Class<? extends IRNode> clazz)
    {
        IRNode node = null;
        for (Iterator<IRNode> it = _invokeNodeStack.iterator(); it.hasNext();) {
            node = it.next();
            if (clazz.equals(node.getClass())) {
                return true;
            }
        }
        return false;
    }

    /**
     * Preprocessing of enter method.
     *
     * @param visitable
     *            Instance of IRVisitable.
     * @return true/false
     */
    private boolean _preEnter(IRVisitable visitable)
    {
        if (_context.isDebugMode()) {
            _context.debugPrintLine(String.format("%100s", "").subSequence(0,
                (_invokeNodeStack.size() - 1) * 2)
                + "<" + visitable.getClass().getSimpleName() + ">");
        }
        return true;
    }

    /**
     * Postprocessing of enter method.
     *
     * @param visitable
     *            Instance of IRVisitable.
     * @return true/false
     */
    private boolean _postEnter(IRVisitable visitable)
    {
        if (_context.isDebugMode()) {
            _context.debugPrintLine(String.format("%100s", " ").subSequence(0,
                (_invokeNodeStack.size() - 1) * 2)
                + "</" + visitable.getClass().getSimpleName() + ">");
        }
        return true;
    }

    /**
     * Checks if object represents a constant expression.
     *
     * @param node Instance of IRNode
     * @return true if node represents a constant expression.
     */
    private boolean _isConstantExpr(IRNode node)
    {
        if((((node.rGetParentRNode()) instanceof XbfUnaryMinusExpr) == false)&&
           (node instanceof XbfUnaryMinusExpr)) {
           node = (node.rGetRNodes())[0];
        }

        if((node instanceof XbfFintConstant) ||
           (node instanceof XbfFlogicalConstant) ||
           (node instanceof XbfFcharacterConstant) ||
           (node instanceof XbfFrealConstant) ||
           (node instanceof XbfFcomplexConstant) ||
           (node instanceof XbfValue))
           return true;
        else
           return false;
    }

    /**
     * If symbolName match with operaotr, then rename to OPERATOR()/ASSIGNMENT().
     * @param symbolName original symbol name.
     * @return symbol name wrapped with "OPERATOR()" or "ASSIGNMENT()" if required,
     * otherwize return original symbol name.
     */
    private String _toUserOperatorIfRequired(String symbolName)
    {
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
        return symbolName;
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
    private XfSymbol _makeSymbol(String symbolName, String typeName)
    {
        if (XfUtil.isNullOrEmpty(symbolName)) {
            // Symbol name is empty.
            return null;
        }

        XfTypeManager typeManager = _context.getTypeManager();

        if (XfUtil.isNullOrEmpty(typeName)) {
            XbfId idElem = typeManager.findSymbol(symbolName);
            if (idElem == null) {
                // Symbol not found.
                return null;
            }
            typeName = idElem.getType();
            if (XfUtil.isNullOrEmpty(typeName)) {
                // Type name of symbol is empty.
                return null;
            }
        }

        symbolName = _toUserOperatorIfRequired(symbolName);

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
    private XfSymbol _makeSymbol(XbfName nameElem)
    {
        if (nameElem == null) {
            // Instance is null.
            return null;
        }

        String symbolName = nameElem.getContent();
        return _makeSymbol(symbolName, nameElem.getType());
    }

    /**
     * Write line directive.
     *
     * @param lineNumber
     *            Line number.
     * @param filePath
     *            File path.
     */
    private void _writeLineDirective(String lineNumber, String filePath)
    {
        if (_context.isOutputLineDirective() &&
            lineNumber != null) {
            XmfWriter writer = _context.getWriter();
            if(filePath == null)
                writer.writeIsolatedLine(String.format("# %s", lineNumber));
            else
                writer.writeIsolatedLine(String.format("# %s \"%s\"", lineNumber, filePath));
        }
    }

    /**
     * Write unary expression.
     *
     * @param expr
     *            Instance of IXbfDefModelExprChoice.
     * @param operation
     *            Operation string.
     * @param grouping
     *            Grouping flag.
     * @return true/false
     */
    private boolean _writeUnaryExpr(IXbfDefModelExprChoice expr, String operation, boolean grouping)
    {
        XmfWriter writer = _context.getWriter();

        if (grouping == true) {
            writer.writeToken("(");
        }

        writer.writeToken(operation);

        if (invokeEnter(expr) == false) {
            return false;
        }

        if (grouping == true) {
            writer.writeToken(")");
        }

        return true;
    }

    /**
     * Write binary expression.
     *
     * @param leftExpr
     *            Instance of IXbfDefModelExprChoice which expresses a Lvalue.
     * @param rightExpr
     *            Instance of IXbfDefModelExprChoice which expresses a Rvalue.
     * @param operation
     *            Operation string.
     * @param grouping
     *            Grouping flag.
     * @return true/false
     */
    private boolean _writeBinaryExpr(IXbfDefModelExprChoice leftExpr,
				     IXbfDefModelExprChoice rightExpr, 
				     String operator)
    {
        XmfWriter writer = _context.getWriter();
	boolean need_paren;
	int op_prio = operator_priority(operator);

	need_paren = false;
	if(op_prio > operator_priority(leftExpr))
	  need_paren = true;

	if(need_paren) writer.writeToken("(");
        if (invokeEnter(leftExpr) == false) {
            return false;
        }
	if(need_paren) writer.writeToken(")");

        writer.writeToken(" ");
        writer.writeToken(operator);

	need_paren = false;
	if(op_prio == PRIO_POWER ||
	   op_prio >= operator_priority(rightExpr))
	  need_paren = true;
	if(need_paren) writer.writeToken("(");
        if (invokeEnter(rightExpr) == false) {
	  return false;
        }
	if(need_paren) writer.writeToken(")");

        return true;
    }

  int operator_priority(String operator){

    if(operator.equals("=") || operator.equals("=>"))
      return PRIO_LOW;

    if(operator.equals("-") || operator.equals("+")) 
      return PRIO_PLUS_MINUS;
    if(operator.equals("*") || operator.equals("/")) 
      return PRIO_MUL_DIV;
    if(operator.equals("**")) 
      return PRIO_POWER;
    
    if(operator.equals("<") || operator.equals(">") || 
       operator.equals("<=") || operator.equals(">=") ||
       operator.equals("/=") || operator.equals("=="))
      return PRIO_COMP;
    
    if(operator.equals("//")) return PRIO_CONCAT;

    if(operator.equals(".AND.")) return PRIO_AND;
    if(operator.equals(".OR."))  return PRIO_OR;
    if(operator.equals(".NEQV.") || operator.equals(".EQV."))
      return PRIO_EQV;

    return PRIO_HIGH;
  }
  
  int operator_priority(IXbfDefModelExprChoice expr){
    if(expr instanceof XbfFassignStatement ||
       expr instanceof XbfFpointerAssignStatement) return PRIO_LOW;

    if(expr instanceof  XbfPlusExpr || expr instanceof XbfMinusExpr)
      return PRIO_PLUS_MINUS;
    if(expr instanceof XbfDivExpr || expr instanceof XbfMulExpr)
      return PRIO_MUL_DIV;
    if(expr instanceof XbfFpowerExpr)
      return PRIO_POWER;
    
    if(expr instanceof XbfLogLTExpr || expr instanceof XbfLogGTExpr ||
       expr instanceof XbfLogLEExpr || expr instanceof XbfLogGEExpr ||
       expr instanceof XbfLogEQExpr || expr instanceof XbfLogNEQExpr)
      return PRIO_COMP;
    
    if(expr instanceof XbfFconcatExpr) return PRIO_CONCAT;

    if(expr instanceof XbfLogAndExpr) return PRIO_AND;
    if(expr instanceof XbfLogOrExpr)  return PRIO_OR;
    if(expr instanceof XbfLogEQVExpr ||expr instanceof XbfLogNEQVExpr)
      return PRIO_EQV;

    return PRIO_HIGH;
  }

    /**
     * Write simple primitive symbol declaration.
     *
     * @param symbol
     *            Instance of XfSymbol.
     */
    private void _writeSimplePrimitiveSymbolDecl(XfSymbol symbol)
    {
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
    private boolean _writeTypeParam(XbfKind kind, XbfLen lenElem)
    {
        if (kind == null && lenElem == null) {
            // Succeed forcibly.
            return true;
        }

        XmfWriter writer = _context.getWriter();
        writer.writeToken("(");

        if (lenElem != null) {
            writer.writeToken("LEN=");
            if (invokeEnter(lenElem) == false) {
                return false;
            }
        }

        if (kind != null) {
            if (lenElem != null) {
                writer.writeToken(", ");
            }
            writer.writeToken("KIND=");
            if (invokeEnter(kind) == false) {
                return false;
            }
        }
        writer.writeToken(")");

        return true;
    }

    /**
     * Write index ranges of array.
     *
     * @param indexRangeArray
     * @return true/false
     * @example <div class="Example"> INTEGER value<span class="Strong">(10,
     *          1:20)</span> </div>
     */
    private boolean _writeIndexRangeArray(IXbfDefModelArraySubscriptChoice[] indexRangeArray)
    {
        if (indexRangeArray == null) {
            // Succeed forcibly.
            return true;
        }

        XmfWriter writer = _context.getWriter();
        writer.writeToken("(");

        int indexRangeCount = 0;

        for (IXbfDefModelArraySubscriptChoice arraySubscriptChoice : indexRangeArray) {
            if (indexRangeCount > 0) {
                writer.writeToken(", ");
            }

            if ((arraySubscriptChoice instanceof XbfIndexRange)
                || (arraySubscriptChoice instanceof XbfArrayIndex)) {
                if (invokeEnter(arraySubscriptChoice) == false) {
                    return false;
                }
            } else {
                _context
                    .debugPrintLine("Detected discontinuous 'indexRange' or 'arrayIndex' element.");
                _context.setLastErrorMessage(XfUtil.formatError(arraySubscriptChoice,
                    XfError.XCODEML_SEMANTICS, XfUtil.getElementName(arraySubscriptChoice)));
                return false;
            }
            ++indexRangeCount;
        }
        writer.writeToken(")");

        return true;
    }

    /**
     * Write coindex ranges of array.
     *
     * @param indexRangeArray
     * @return true/false
     * @example <div class="Example"> INTEGER value<span class="Strong">[10,
     *          1:*]</span> </div>
     */
    private boolean _writeCoIndexRangeArray(XbfIndexRange[] indexRangeArray)
    {
        if (indexRangeArray == null) {
            // Succeed forcibly.
            return true;
        }

        XmfWriter writer = _context.getWriter();
        writer.writeToken("[");

        int indexRangeCount = 0;

        for (IXbfDefModelArraySubscriptChoice arraySubscriptChoice : indexRangeArray) {
            if (indexRangeCount > 0) {
                writer.writeToken(", ");
            }

            if ((arraySubscriptChoice instanceof XbfIndexRange)
                || (arraySubscriptChoice instanceof XbfArrayIndex)) {
                if (invokeEnter(arraySubscriptChoice) == false) {
                    return false;
                }
            } else {
                _context
                    .debugPrintLine("Detected discontinuous 'indexRange' or 'arrayIndex' element.");
                _context.setLastErrorMessage(XfUtil.formatError(arraySubscriptChoice,
                    XfError.XCODEML_SEMANTICS, XfUtil.getElementName(arraySubscriptChoice)));
                return false;
            }
            ++indexRangeCount;
        }
        writer.writeToken("]");

        return true;
    }

    private void _writeDeclAttr(IXbfTypeTableChoice top, IXbfTypeTableChoice low) 
    {
        if ((top instanceof XbfFbasicType) &&
            (low instanceof XbfFbasicType)) {
            _writeBasicTypeAttr((XbfFbasicType)top, (XbfFbasicType)low);
            return;
        }

        if (top instanceof XbfFbasicType) {
           _writeBasicTypeAttr((XbfFbasicType)top);
        }

        if (low instanceof XbfFbasicType) {
           _writeBasicTypeAttr((XbfFbasicType)low);
        }
    }

    private void _writeBasicTypeAttr(XbfFbasicType... basicTypeArray)
    {
        if (basicTypeArray == null) {
            return;
        }

        XmfWriter writer = _context.getWriter();

        /* public, private are allowed only in module definition */
        if(_isUnderModuleDef()) {
            for (XbfFbasicType basicTypeElem : basicTypeArray) {
                if (basicTypeElem.getIsPublic()) {
                    writer.writeToken(", ");
                    writer.writeToken("PUBLIC");
                    break;
                }
            }            
    
            for (XbfFbasicType basicTypeElem : basicTypeArray) {
                if (basicTypeElem.getIsPrivate()) {
                    writer.writeToken(", ");
                    writer.writeToken("PRIVATE");
                    break;
                }
            }            
        }

        for (XbfFbasicType basicTypeElem : basicTypeArray) {
            if (basicTypeElem.getIsPointer()) {
                writer.writeToken(", ");
                writer.writeToken("POINTER");
                break;
            }
        }

        for (XbfFbasicType basicTypeElem : basicTypeArray) {
            if (basicTypeElem.getIsTarget()) {
                writer.writeToken(", ");
                writer.writeToken("TARGET");
                break;
            }
        }

        for (XbfFbasicType basicTypeElem : basicTypeArray) {
            if (basicTypeElem.getIsOptional()) {
                writer.writeToken(", ");
                writer.writeToken("OPTIONAL");
                break;
            }
        }

        for (XbfFbasicType basicTypeElem : basicTypeArray) {
            if (basicTypeElem.getIsSave()) {
                writer.writeToken(", ");
                writer.writeToken("SAVE");
                break;
            }
        }

        for (XbfFbasicType basicTypeElem : basicTypeArray) {
            if (basicTypeElem.getIsParameter()) {
                writer.writeToken(", ");
                writer.writeToken("PARAMETER");
                break;
            }
        }

        for (XbfFbasicType basicTypeElem : basicTypeArray) {
            if (basicTypeElem.getIsAllocatable()) {
                writer.writeToken(", ");
                writer.writeToken("ALLOCATABLE");
                break;
            }
        }

        for (XbfFbasicType basicTypeElem : basicTypeArray) {
            String intent = basicTypeElem.getIntent();
            if (XfUtil.isNullOrEmpty(intent) == false) {
                writer.writeToken(", ");
                writer.writeToken("INTENT(" + intent.toUpperCase() + ")");
                break;
            }
        }
    }
    
    private boolean _writeFunctionSymbol(XfSymbol symbol, XbfFfunctionType funcType, IRNode visitable)
    {
        XfTypeManager typeManager = _context.getTypeManager();
        XmfWriter writer = _context.getWriter();
        IXbfTypeTableChoice lowType = null;
        
        if(funcType.getIsIntrinsic()) {
            writer.writeToken("INTRINSIC ");
            writer.writeToken(symbol.getSymbolName());
            return true;
        }
        
        boolean isFirstToken = true;
        boolean isPrivateEmit = false;
        boolean isPublicEmit = false;

        /* - always type declaration for SUBROUTINE must not be output.
         * - type declaration for FUNCTION under MODULE must not be output.
         */
        if(typeManager.isDecompilableType(funcType.getReturnType()) &&
            (_isUnderModuleDef() == false || funcType.getIsExternal())) {
            
            XfType type = XfType.getTypeIdFromXcodemlTypeName(funcType.getReturnType());
            if(type.isPrimitive()) {
                writer.writeToken(type.fortranName());
            } else {
                XfTypeManager.TypeList typeList = getTypeList(funcType.getReturnType());
                if(typeList == null)
                    return false;

                lowType = typeList.getLast();
                IXbfTypeTableChoice topType = typeList.getFirst();

                if(lowType instanceof XbfFbasicType) {
                    XbfFbasicType bt = (XbfFbasicType) lowType;
                    if (bt.getIsPublic()) {
                        isPublicEmit = true;
                    }
                    if (bt.getIsPrivate()) {
                        isPrivateEmit = true;
                    }
                }

                if(topType instanceof XbfFbasicType) {
                    XbfFbasicType bt = (XbfFbasicType) topType;
                    if (bt.getIsPublic()) {
                        isPublicEmit = true;
                    }
                    if (bt.getIsPrivate()) {
                        isPrivateEmit = true;
                    }
                }

                if(topType instanceof XbfFbasicType) {
                    if(_writeBasicType((XbfFbasicType)topType, typeList) == false)
                        return false;

                } else if (topType instanceof XbfFstructType) {
                    XbfFstructType structTypeElem = (XbfFstructType)topType;
                    String aliasStructTypeName = typeManager.getAliasTypeName(structTypeElem.getType());
                    writer.writeToken("TYPE(" + aliasStructTypeName + ")");

                } else {
                    /* topType is FfunctionType. */
                    return false;
                }

                _writeDeclAttr(topType, lowType);
            }
            isFirstToken = false;
        }

        if(_isUnderModuleDef()) {
            if (funcType.getIsPublic() && isPublicEmit == false) {
                writer.writeToken((isFirstToken ? "" : ", ") + "PUBLIC");
                isFirstToken = false;
            } else if (funcType.getIsPrivate() && isPrivateEmit == false) {
                writer.writeToken((isFirstToken ? "" : ", ") + "PRIVATE");
                isFirstToken = false;
            }
        }

        if(isFirstToken == false) {
            writer.writeToken(" :: ");
            writer.writeToken(symbol.getSymbolName());

            if(lowType != null &&
               (lowType instanceof XbfFbasicType)) {
                XbfFbasicType basicTypeElem = (XbfFbasicType)lowType;
                IXbfFbasicTypeChoice basicTypeChoice = basicTypeElem.getContent();

                if (basicTypeChoice != null &&
                    (basicTypeChoice instanceof XbfDefModelArraySubscriptSequence1)) {
                    if (invokeEnter(basicTypeChoice) == false) {
                        return false;
                    }

                }
            }
        }
        
        if(funcType.getIsExternal()) {
            if(isFirstToken == false)
                writer.setupNewLine();
            writer.writeToken("EXTERNAL ");
            writer.writeToken(symbol.getSymbolName());
        }
        
        return true;
    }
    
    private boolean _writeBasicType(XbfFbasicType basicTypeElem, XfTypeManager.TypeList typeList)
    {
        String refName = basicTypeElem.getRef();
        XfType refTypeId = XfType.getTypeIdFromXcodemlTypeName(refName);
        assert (refTypeId != null);

        if (refTypeId.isPrimitive() == false) {
            _context.debugPrint(
                "Top level type is basic-type, but is not primitive type. (%s)%n", refName);
            if(typeList != null)
                _context.debugPrintLine(typeList.toString());
            _context.setLastErrorMessage(XfUtil.formatError(basicTypeElem,
                XfError.XCODEML_TYPE_MISMATCH, "top-level FbasicType", refName,
                "primitive type"));
            return false;
        }

        XmfWriter writer = _context.getWriter();
        writer.writeToken(refTypeId.fortranName());

        XbfKind kind = basicTypeElem.getKind();
        XbfLen lenElem = null;

        IXbfFbasicTypeChoice basicTypeChoice = basicTypeElem.getContent();
        if (basicTypeChoice != null) {
            if (basicTypeChoice instanceof XbfLen) {
                if (refTypeId != XfType.CHARACTER) {
                    _context.debugPrint(
                        "A 'len' element is included in a definition of '%s' type.%n",
                        refTypeId.xcodemlName());
                    _context.setLastErrorMessage(XfUtil.formatError(basicTypeChoice,
                        XfError.XCODEML_SEMANTICS, XfUtil.getElementName(basicTypeChoice)));
                    return false;
                }
                lenElem = (XbfLen)basicTypeChoice;
            }
        }
        if (_writeTypeParam(kind, lenElem) == false) {
            return false;
        }
        
        return true;
    }
    
    private XfTypeManager.TypeList getTypeList(String type)
    {
        XfTypeManager typeManager = _context.getTypeManager();
        XfTypeManager.TypeList typeList = null;

        try {
            typeList = typeManager.getTypeReferenceList(type);
        } catch (XmException e) {
            _context.debugPrintLine(e.toString()); 
            _context.setLastErrorMessage(XfUtil.formatError(_invokeNodeStack.peek(),
                XfError.XCODEML_CYCLIC_TYPE, type));
            return null;
        }

        if (typeList == null || typeList.isEmpty()) {
            _context.debugPrintLine("Type list is empty.");
            _context.setLastErrorMessage(XfUtil.formatError(_invokeNodeStack.peek(),
                XfError.XCODEML_TYPE_NOT_FOUND, type));
            return null;
        }
        
        return typeList;
    }

    /**
     * Write variable declaration.
     *
     * @param symbol
     *            Variable symbol.
     * @return true/false
     * @example <div class="Example"> PROGRAM main <div class="Indent1"><div
     *          class="Strong"> INTEGER :: int_variable<br/>
     *          TYPE(USER_TYPE) :: derived_variable </div> int_variable = 0
     *          </div> END PROGRAM main </div>
     */
    private boolean _writeSymbolDecl(XfSymbol symbol, IRNode visitable)
    {
        if (symbol == null) {
            throw new IllegalArgumentException();
        }

        XfType typeId = symbol.getTypeId();
        if (typeId.isPrimitive()) {
            _writeSimplePrimitiveSymbolDecl(symbol);
            return true;
        }

        XfTypeManager typeManager = _context.getTypeManager();
        XfTypeManager.TypeList typeList = getTypeList(symbol.getDerivedName());
        
        if(typeList == null)
            return false;
        
        /*
         * The assumption that typeList.size() <= 2 is not valid for now.
         *	m-hirano
         */
//        if (typeList.size() > 2) {
//            _context.debugPrintLine("Type list count > 2.");
//            _context.debugPrintLine(typeList.toString());
//            _context.setLastErrorMessage(XfUtil.formatError(_invokeNodeStack.peek(),
//                XfError.XCODEML_SEMANTICS, XfUtil.getElementName(_invokeNodeStack.peek())));
//            return false;
//        }

        IXbfTypeTableChoice topTypeChoice = typeList.getFirst();
        IXbfTypeTableChoice lowTypeChoice = typeList.getLast();

        XmfWriter writer = _context.getWriter();

        // ================
        // Top type element
        // ================
        if (topTypeChoice instanceof XbfFbasicType) {
            XbfFbasicType basicTypeElem = (XbfFbasicType)topTypeChoice;
            if(_writeBasicType(basicTypeElem, typeList) == false)
                return false;
        } else if (topTypeChoice instanceof XbfFstructType) {
            XbfFstructType structTypeElem = (XbfFstructType)topTypeChoice;
            String aliasStructTypeName = typeManager.getAliasTypeName(structTypeElem.getType());
            writer.writeToken("TYPE(" + aliasStructTypeName + ")");
        } else if (topTypeChoice instanceof XbfFfunctionType) {
            return _writeFunctionSymbol(symbol, (XbfFfunctionType)topTypeChoice, visitable);
        }

        _writeDeclAttr(topTypeChoice, lowTypeChoice);

        // ================
        // Low type element
        // ================
        if (lowTypeChoice instanceof XbfFbasicType) {
            XbfFbasicType basicTypeElem = (XbfFbasicType)lowTypeChoice;
            String refName = basicTypeElem.getRef();
            XfType refTypeId = XfType.getTypeIdFromXcodemlTypeName(refName);
            assert (refTypeId != null);

            writer.writeToken(" :: ");
            writer.writeToken(symbol.getSymbolName());

            IXbfFbasicTypeChoice basicTypeChoice = basicTypeElem.getContent();
            if (basicTypeChoice != null) {
                if (basicTypeChoice instanceof XbfDefModelArraySubscriptSequence1) {
                    if (invokeEnter(basicTypeChoice) == false) {
                        return false;
                    }
                }
            }

	    XbfCoShape coShape = basicTypeElem.getCoShape();
	    if (coShape != null){
		if (!invokeEnter(coShape)) return false;
	    }

        } else if (lowTypeChoice instanceof XbfFstructType) {
            writer.writeToken(" :: ");
            writer.writeToken(symbol.getSymbolName());
        }

        return true;
    }

    /**
     * Call enter method of node.
     *
     * @param nodeArray
     *            IRNode array.
     * @return true/false
     */
    private boolean _invokeEnter(IRNode[] nodeArray)
    {
        IRNode currentNode = null;

        if (nodeArray == null) {
            // Succeed forcibly.
            return true;
        }

        for (IRNode node : nodeArray) {
            if(_validator.validAttr(node) == false) {
                _context.debugPrintLine("Detected insufficient attributes");
                _context.setLastErrorMessage(_validator.getErrDesc()); 
                return false;
            }
        
            _nextNode = node;

            if (currentNode != null) {
                if (invokeEnter(currentNode) == false) {
                    return false;
                }
            }

            currentNode = node;
        }

        _nextNode = null;
        if (invokeEnter(currentNode) == false) {
            return false;
        }

        return true;
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
    private boolean _invokeEnterAndWriteDelim(IRNode[] nodeArray, String delim)
    {
        if (nodeArray == null) {
            // Succeed forcibly.
            return true;
        }

        XmfWriter writer = _context.getWriter();

        int nodeCount = 0;
        for (IRNode node : nodeArray) {
            if (nodeCount > 0) {
                writer.writeToken(delim);
            }
            if (invokeEnter(node) == false) {
                return false;
            }
            ++nodeCount;
        }
        return true;
    }

    /**
     * Call enter method of child node.
     *
     * @param node
     *            Parent IRNode.
     * @return true/false
     */
    private boolean _invokeChildEnter(IRNode node)
    {
        if (node == null) {
            // Succeed forcibly.
            return true;
        }
        return _invokeEnter(node.rGetRNodes());
    }

    /**
     * Call enter method of node.
     *
     * @param node
     *            IRNode array.
     * @return true/false
     */
    public boolean invokeEnter(IRNode node)
    {
        if (node == null) {
            // Succeed forcibly.
            return true;
        }

        assert (node instanceof IRVisitable);
        IRVisitable visitable = (IRVisitable)node;

        boolean result = true;
        _invokeNodeStack.push(node);
        result = _preEnter(visitable);
        if (result) {
            result = ((IRVisitable)node).enter(this);
        }
        if (result) {
            result = _postEnter(visitable);
        }
        _invokeNodeStack.pop();
        return result;
    }

    /**
     * Decompile 'XcodeProgram' element in XcodeML/F.
     *
     * @see xcodeml.f.binding.gen.RVisitorBase#enter(xcodeml.f.binding.gen.
     *      XbfXcodeProgram)
     */
    @Override
    public boolean enter(XbfXcodeProgram visitable)
    {
        // DONE: XbfXcodeProgram
        XfTypeManager typeManager = _context.getTypeManager();

        // for global symbol
        typeManager.enterScope();

        if (_invokeChildEnter(visitable) == false) {
            return false;
        }

        typeManager.leaveScope();

        return true;
    }

    /**
     * Decompile "typeTable" element in XcodeML/F.
     * <p>
     * The decompilation result depends on a child element.
     * </p>
     *
     * @see xcodeml.f.binding.gen.RVisitorBase#enter(xcodeml.f.binding.gen.XbfTypeTable
     *      )
     */
    @Override
    public boolean enter(XbfTypeTable visitable)
    {
        // DONE: XbfTypeTable
        if (_invokeEnter(visitable.getContent()) == false) {
            return false;
        }

        return true;
    }

    /**
     * Decompile "FbasicType" element in XcodeML/F.
     *
     * @see xcodeml.f.binding.gen.RVisitorBase#enter(xcodeml.f.binding
     *      .gen.XbfFbasicType )
     */
    @Override
    public boolean enter(XbfFbasicType visitable)
    {
        // DONE: XbfFbasicType
        // Note:
        // Because handle it at a upper level element,
        // warn it when this method was called it.
        assert(_isInvokeAncestorNodeOf(XbfTypeTable.class));
        
        XfTypeManager typeManager = _context.getTypeManager();
        typeManager.addType(visitable);

        return true;
    }

    /**
     * Decompile child group of "FbasicType" element in XcodeML/F.
     *
     * @see xcodeml.f.binding.gen.RVisitorBase#enter(xcodeml.f.binding.gen.
     *      XbfDefModelArraySubscriptSequence)
     */
    public boolean enter(XbfDefModelArraySubscriptSequence1 visitable)
    {
        // DONE: XbfDefModelArraySubscriptSequence1
        IXbfDefModelArraySubscriptChoice[] arraySubscriptChoice = visitable
            .getDefModelArraySubscript();
        if (arraySubscriptChoice != null) {
            if (_writeIndexRangeArray(arraySubscriptChoice) == false) {
                return false;
            }
        }

        return true;
    }

    /**
     * Decompile child group of "FbasicType" element in XcodeML/F.
     *
     * @see xcodeml.f.binding.gen.RVisitorBase#enter(xcodeml.f.binding.gen.
     *      XbfCoShape)
     */
    public boolean enter(XbfCoShape visitable)
    {
        // DONE: XbfCoShape
        XbfIndexRange[] indexRange = visitable.getIndexRange();
        if (indexRange != null){
            if (!_writeCoIndexRangeArray(indexRange)) return false;
        }

        return true;
    }

    /**
     * Decompile "FfunctionType" element in XcodeML/F.
     *
     * @see xcodeml.f.binding.gen.RVisitorBase#enter(xcodeml.f.binding.gen.
     *      XbfFfunctionType)
     */
    @Override
    public boolean enter(XbfFfunctionType visitable)
    {
        // DONE: XbfFfunctionType
        if (_isInvokeAncestorNodeOf(XbfTypeTable.class)) {
            XfTypeManager typeManager = _context.getTypeManager();
            typeManager.addType(visitable);
        } else {
            // Note:
            // Because handle it at a upper level element,
            // warn it when this method was called it.
            assert (false);
        }

        return true;
    }

    /**
     * Decompile "FstructType" element in XcodeML/F.
     * <p>
     * The decompilation result depends on a child element.
     * </p>
     *
     * @see xcodeml.f.binding.gen.RVisitorBase#enter(xcodeml.f.binding.gen.XbfFstructType
     *      )
     */
    @Override
    public boolean enter(XbfFstructType visitable)
    {
        // DONE: XbfFstructType
        if (_isInvokeAncestorNodeOf(XbfTypeTable.class)) {
            XfTypeManager typeManager = _context.getTypeManager();
            typeManager.addType(visitable);
        } else {
            if (invokeEnter(visitable.getSymbols()) == false) {
                return false;
            }
        }

        return true;
    }

    /**
     * Decompile "globalSymbols" element in XcodeML/F.
     *
     * @see xcodeml.f.binding.gen.RVisitorBase#enter(xcodeml.f.binding.gen.
     *      XbfGlobalSymbols)
     */
    @Override
    public boolean enter(XbfGlobalSymbols visitable)
    {
        // DONE: XbfGlobalSymbols
        XfTypeManager typeManager = _context.getTypeManager();
        for (XbfId idElem : visitable.getId()) {
            typeManager.addSymbol(idElem);
        }

        return true;
    }

    /**
     * Decompile "globalDeclarations" element in XcodeML/F.
     * <p>
     * The decompilation result depends on a child element.
     * </p>
     *
     * @see xcodeml.f.binding.gen.RVisitorBase#enter(xcodeml.f.binding.gen.
     *      XbfGlobalDeclarations)
     */
    @Override
    public boolean enter(XbfGlobalDeclarations visitable)
    {
        // DONE: XbfGlobalDeclarations
        if (_invokeChildEnter(visitable) == false) {
            return false;
        }

        return true;
    }

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
    @Override
    public boolean enter(XbfAlloc visitable)
    {
        // DONE: XbfAlloc
        if (invokeEnter(visitable.getContent()) == false) {
            return false;
        }

        // Parent node is XbfFallocateStatement?
        if (_isInvokeNodeOf(XbfFallocateStatement.class, 1)) {
            IXbfDefModelArraySubscriptChoice[] arraySubscriptChoice = visitable
                .getDefModelArraySubscript();
            if ((arraySubscriptChoice != null) && (arraySubscriptChoice.length > 0)) {
                if (_writeIndexRangeArray(arraySubscriptChoice) == false) {
                    return false;
                }
            }
        }

	XbfCoShape coShape = visitable.getCoShape();
	if (coShape != null){
	    if (!invokeEnter(coShape)) return false;
	}

        return true;
    }

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
    @Override
    public boolean enter(XbfArguments visitable)
    {
        // DONE: XbfArguments
        if (_invokeEnterAndWriteDelim(visitable.getContent(), ", ") == false) {
            return false;
        }

        return true;
    }

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
    @Override
    public boolean enter(XbfArrayIndex visitable)
    {
        // DONE: XbfArrayIndex
        if (invokeEnter(visitable.getDefModelExpr()) == false) {
            return false;
        }

        return true;
    }

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
    @Override
    public boolean enter(XbfFassignStatement visitable)
    {
        // DONE: XbfFassignStatement
        XmfWriter writer = _context.getWriter();
        
        _writeLineDirective(visitable.getLineno(), visitable.getFile());

        if (_writeBinaryExpr((IXbfDefModelExprChoice)visitable.getDefModelLValue(), visitable.getDefModelExpr(), "=") == false) {
            return false;
        }

        writer.setupNewLine();

        return true;
    }

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
    @Override
    public boolean enter(XbfBody visitable)
    {
        // DONE: XbfBody
        if (_invokeEnter(visitable.getDefModelStatement()) == false) {
            return false;
        }

        return true;
    }

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
    @Override
    public boolean enter(XbfCondition visitable)
    {
        // DONE: XbfCondition
        XmfWriter writer = _context.getWriter();
        writer.writeToken("(");

        if (invokeEnter(visitable.getDefModelExpr()) == false) {
            return false;
        }

        writer.writeToken(")");
        return true;
    }

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
    @Override
    public boolean enter(XbfContinueStatement visitable)
    {
        // DONE: XbfContinueStatement
        _writeLineDirective(visitable.getLineno(), visitable.getFile());

        XmfWriter writer = _context.getWriter();
        writer.writeToken(" CONTINUE");
        writer.setupNewLine();

        return true;
    }

    /**
     * Decompile "Declarations" element in XcodeML/F.
     * <p>
     * The decompilation result depends on a child element.
     * </p>
     *
     * @see xcodeml.f.binding.gen.RVisitorBase#enter(xcodeml.f.binding.gen.XbfDeclarations)
     */
    @Override
    public boolean enter(XbfDeclarations visitable)
    {
        // DONE: XbfDeclarations
        if (_invokeChildEnter(visitable) == false) {
            return false;
        }

        return true;
    }

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
    @Override
    public boolean enter(XbfDivExpr visitable)
    {
        // DONE: XbfDivExpr
        if (_writeBinaryExpr(visitable.getDefModelExpr1(), visitable.getDefModelExpr2(), "/") == false) {
            return false;
        }

        return true;
    }

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
    @Override
    public boolean enter(XbfElse visitable)
    {
        // DONE: XbfElse
        XmfWriter writer = _context.getWriter();
        writer.incrementIndentLevel();

        if (invokeEnter(visitable.getBody()) == false) {
            return false;
        }

        writer.decrementIndentLevel();

        return true;
    }

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
    @Override
    public boolean enter(XbfExprStatement visitable)
    {
        // DONE: XbfExprStatement
        _writeLineDirective(visitable.getLineno(), visitable.getFile());

        if (_invokeChildEnter(visitable) == false) {
            return false;
        }

        XmfWriter writer = _context.getWriter();
        writer.setupNewLine();

        return true;
    }

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
    @Override
    public boolean enter(XbfExternDecl visitable)
    {
        // DONE: XbfExternDecl
        _writeLineDirective(visitable.getLineno(), visitable.getFile());

        XbfName nameElem = visitable.getName();
        String externalName = nameElem.getContent();

        XmfWriter writer = _context.getWriter();
        writer.writeToken("EXTERNAL ");
        writer.writeToken(externalName);
        writer.setupNewLine();

        return true;
    }

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
    @Override
    public boolean enter(XbfFallocateStatement visitable)
    {
        // DONE: XbfFallocateStatement
        _writeLineDirective(visitable.getLineno(), visitable.getFile());

        XmfWriter writer = _context.getWriter();
        writer.writeToken("ALLOCATE (");

        if (_invokeEnterAndWriteDelim(visitable.getAlloc(), ", ") == false) {
            return false;
        }

        String statName = visitable.getStatName();
        if (XfUtil.isNullOrEmpty(statName) == false) {
            writer.writeToken(", ");
            writer.writeToken("STAT = ");
            writer.writeToken(statName);
        }

        writer.writeToken(")");
        writer.setupNewLine();

        return true;
    }

    /**
     * Decompile "FarrayConstructor" element in XcodeML/F.
     *
     * @example <code><div class="Example">
     *      array = <span class="Strong">(/ 1, 2, (I, I = 1, 10, 2) /)</span><br/>
     * </div></code>
     * @see xcodeml.f.binding.gen.RVisitorBase#enter(xcodeml.f.binding.gen.
     *      XbfFarrayConstructor)
     */
    @Override
    public boolean enter(XbfFarrayConstructor visitable)
    {
        // DONE: XbfFarrayConstructor
        XmfWriter writer = _context.getWriter();
        writer.writeToken("(/ ");

        if (_invokeEnterAndWriteDelim(visitable.getDefModelExpr(), ", ") == false) {
            return false;
        }

        writer.writeToken(" /)");

        return true;
    }

    /**
     * Decompile "FarrayRef" element in XcodeML/F.
     *
     * @example <code><div class="Example">
     *      array = <span class="Strong">int_array_variable(10, 1:10, 1:, :)</span><br/>
     * </div></code>
     * @see xcodeml.f.binding.gen.RVisitorBase#enter(xcodeml.f.binding.gen.XbfFarrayRef
     *      )
     */
    @Override
    public boolean enter(XbfFarrayRef visitable)
    {
        // DONE: XbfFarrayRef
        if (invokeEnter(visitable.getVarRef()) == false) {
            return false;
        }

        XmfWriter writer = _context.getWriter();
        writer.writeToken("(");

        if (_invokeEnterAndWriteDelim(visitable.getContent(), ", ") == false) {
            return false;
        }

        writer.writeToken(")");

        return true;
    }

    /**
     * Decompile "FcoArrayRef" element in XcodeML/F.
     *
     * @example <code><div class="Example">
     *      array = <span class="Strong">int_coarray_variable[10, 1:10, 1:, *]</span><br/>
     * </div></code>
     * @see xcodeml.f.binding.gen.RVisitorBase#enter(xcodeml.f.binding.gen.XbfFcoArrayRef
     *      )
     */
    @Override
    public boolean enter(XbfFcoArrayRef visitable)
    {
        // DONE: XbfFcoArrayRef
        if (invokeEnter(visitable.getVarRef()) == false) {
            return false;
        }

        XmfWriter writer = _context.getWriter();
        writer.writeToken("[");

        if (_invokeEnterAndWriteDelim(visitable.getArrayIndex(), ", ") == false) {
            return false;
        }

        writer.writeToken("]");

        return true;
    }

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
    @Override
    public boolean enter(XbfFbackspaceStatement visitable)
    {
        // DONE: XbfFbackspaceStatement
        _writeLineDirective(visitable.getLineno(), visitable.getFile());

        XmfWriter writer = _context.getWriter();

        writer.writeToken("BACKSPACE ");
        if (invokeEnter(visitable.getNamedValueList()) == false) {
            return false;
        }

        writer.setupNewLine();

        return true;
    }

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
    @Override
    public boolean enter(XbfFcaseLabel visitable)
    {
        // DONE: XbfFcaseLabel
        _writeLineDirective(visitable.getLineno(), visitable.getFile());

        XmfWriter writer = _context.getWriter();

        writer.writeToken("CASE");
        IXbfFcaseLabelChoice[] caseLabelChoice = visitable.getContent();
        if ((caseLabelChoice != null) && (caseLabelChoice.length > 0)) {
            writer.writeToken(" (");
            if (_invokeEnterAndWriteDelim(visitable.getContent(), ", ") == false) {
                return false;
            }
            writer.writeToken(")");
        } else {
            writer.writeToken(" DEFAULT");
        }

        String constuctName = visitable.getConstructName();
        if (XfUtil.isNullOrEmpty(constuctName) == false) {
            writer.writeToken(" ");
            writer.writeToken(constuctName);
        }

        writer.setupNewLine();
        writer.incrementIndentLevel();

        if (invokeEnter(visitable.getBody()) == false) {
            return false;
        }

        writer.decrementIndentLevel();

        return true;
    }

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
    @Override
    public boolean enter(XbfFcharacterConstant visitable)
    {
        // DONE: XbfFcharacterConstant
        XmfWriter writer = _context.getWriter();

        String kind = visitable.getKind();
        if (XfUtil.isNullOrEmpty(kind) == false) {
            writer.writeToken(kind + "_");
        }

        writer.writeLiteralString(visitable.getContent());

        return true;
    }

    /**
     * Decompile "FcharacterRef" element in XcodeML/F.
     *
     * @example <code><div class="Example">
     *      substring = <span class="Strong">char_variable(1:10)</span><br/>
     * </div></code>
     * @see xcodeml.f.binding.gen.RVisitorBase#enter(xcodeml.f.binding.gen.
     *      XbfFcharacterRef)
     */
    @Override
    public boolean enter(XbfFcharacterRef visitable)
    {
        // DONE: XbfFcharacterRef
        if (invokeEnter(visitable.getVarRef()) == false) {
            return false;
        }

        XmfWriter writer = _context.getWriter();
        writer.writeToken("(");

        if (invokeEnter(visitable.getIndexRange()) == false) {
            return false;
        }

        writer.writeToken(")");

        return true;
    }

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
    @Override
    public boolean enter(XbfFcloseStatement visitable)
    {
        // DONE: XbfFcloseStatement
        _writeLineDirective(visitable.getLineno(), visitable.getFile());

        XmfWriter writer = _context.getWriter();

        writer.writeToken("CLOSE ");
        if (invokeEnter(visitable.getNamedValueList()) == false) {
            return false;
        }

        writer.setupNewLine();

        return true;
    }

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
    @Override
    public boolean enter(XbfFcommonDecl visitable)
    {
        // DONE: XbfFcommonDecl
        _writeLineDirective(visitable.getLineno(), visitable.getFile());

        XmfWriter writer = _context.getWriter();
        writer.writeToken("COMMON ");

        if (_invokeEnterAndWriteDelim(visitable.getVarList(), ", ") == false) {
            return false;
        }

        writer.setupNewLine();

        return true;
    }

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
    @Override
    public boolean enter(XbfFcomplexConstant visitable)
    {
        // DONE: XbfFcomplexConstant
        IXbfDefModelExprChoice realPart = visitable.getDefModelExpr1();
        IXbfDefModelExprChoice imaginalPart = visitable.getDefModelExpr2();

        if ((realPart == null) || (imaginalPart == null)) {
            _context.setLastErrorMessage(XfUtil.formatError(visitable, XfError.XCODEML_SEMANTICS,
                XfUtil.getElementName(visitable)));
            return false;
        }

        String typeName = visitable.getType();
        if (XfUtil.isNullOrEmpty(typeName) == false) {
            XfTypeManager typeManager = _context.getTypeManager();
            String bottomTypeName = typeManager.getBottomTypeName(typeName);
            if (bottomTypeName == null) {
                _context.setLastErrorMessage(XfUtil.formatError(visitable,
                    XfError.XCODEML_TYPE_NOT_FOUND,
                    XfUtil.getElementName(visitable),
                    typeName));
                return false;
            }

            XfType typeId = XfType.getTypeIdFromXcodemlTypeName(bottomTypeName);
            if (typeId != XfType.DERIVED && typeId != XfType.COMPLEX) {
                _context.setLastErrorMessage(XfUtil.formatError(visitable,
                    XfError.XCODEML_TYPE_MISMATCH, XfUtil.getElementName(visitable), typeName,
                    "Fcomplex"));
                return false;
            }
        }

        XmfWriter writer = _context.getWriter();

        if((_isConstantExpr(realPart) == false) ||
           (_isConstantExpr(imaginalPart) == false)) {
            writer.writeToken("CMPLX");
        }

        writer.writeToken("(");
        if (invokeEnter(realPart) == false) {
            return false;
        }
        writer.writeToken(", ");
        if (invokeEnter(imaginalPart) == false) {
            return false;
        }
        writer.writeToken(")");

        return true;
    }

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
    @Override
    public boolean enter(XbfFconcatExpr visitable)
    {
        // DONE: XbfFconcatExpr
        if (_writeBinaryExpr(visitable.getDefModelExpr1(), visitable.getDefModelExpr2(), "//") == false) {
            return false;
        }

        return true;
    }

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
    @Override
    public boolean enter(XbfFcontainsStatement visitable)
    {
        // DONE: XbfFcontainsStatement
        _writeLineDirective(visitable.getLineno(), visitable.getFile());

        XmfWriter writer = _context.getWriter();
        writer.decrementIndentLevel();
        writer.setupNewLine();
        writer.writeToken("CONTAINS");
        writer.setupNewLine();
        writer.incrementIndentLevel();

        if (_invokeEnter(visitable.getFfunctionDefinition()) == false) {
            return false;
        }

        return true;
    }

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
    @Override
    public boolean enter(XbfFcycleStatement visitable)
    {
        // DONE: XbfFcycleStatement
        _writeLineDirective(visitable.getLineno(), visitable.getFile());

        XmfWriter writer = _context.getWriter();
        writer.writeToken("CYCLE");

        String constuctName = visitable.getConstructName();
        if (XfUtil.isNullOrEmpty(constuctName) == false) {
            writer.writeToken(" ");
            writer.writeToken(constuctName);
        }

        writer.setupNewLine();

        return true;
    }

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
    @Override
    public boolean enter(XbfFdataDecl visitable)
    {
        // DONE: XbfFdataDecl
        _writeLineDirective(visitable.getLineno(), visitable.getFile());

        XmfWriter writer = _context.getWriter();
        writer.writeToken("DATA ");

        if (_invokeEnterAndWriteDelim(visitable.getFdataDeclSequence(), ", ") == false) {
            return false;
        }

        writer.setupNewLine();

        return true;
    }

    /**
     * Decompile child group of "FdataDecl" element in XcodeML/F.
     *
     * @example <code><div class="Example">
     * DATA <span class="Strong">variable1, variable2 /1, 2/</span>, &<br/>
     *      <span class="Strong">array1 /10*1/</span>, &<br/>
     *      <span class="Strong">(array2(i), i = 1, 10, 2) /5*1/</span><br/>
     * </div></code>
     * @see xcodeml.f.binding.gen.RVisitorBase#enter(xcodeml.f.binding.gen.
     *      XbfFdataDeclSequence)
     */
    @Override
    public boolean enter(XbfFdataDeclSequence visitable)
    {
        // DONE: XbfFdataDeclSequence
        XmfWriter writer = _context.getWriter();
        if (invokeEnter(visitable.getVarList()) == false) {
            return false;
        }

        writer.writeToken(" /");

        if (invokeEnter(visitable.getValueList()) == false) {
            return false;
        }

        writer.writeToken("/");

        return true;
    }

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
    @Override
    public boolean enter(XbfFdeallocateStatement visitable)
    {
        // DONE: XbfFdeallocateStatement
        _writeLineDirective(visitable.getLineno(), visitable.getFile());

        XmfWriter writer = _context.getWriter();
        writer.writeToken("DEALLOCATE (");

        if (_invokeEnterAndWriteDelim(visitable.getAlloc(), ", ") == false) {
            return false;
        }

        String statName = visitable.getStatName();
        if (XfUtil.isNullOrEmpty(statName) == false) {
            writer.writeToken(", ");
            writer.writeToken("STAT = ");
            writer.writeToken(statName);
        }

        writer.writeToken(")");
        writer.setupNewLine();

        return true;
    }

    /**
     * Decompile "FdoLoop" element in XcodeML/F.
     *
     * @example <code><div class="Example">
     *      array = (/ 1, 2, <span class="Strong">(I, I = 1, 10, 2)</span> /)<br/>
     * </div></code>
     * @see xcodeml.f.binding.gen.RVisitorBase#enter(xcodeml.f.binding.gen.XbfFdoLoop
     *      )
     */
    @Override
    public boolean enter(XbfFdoLoop visitable)
    {
        // DONE: XbfFdoLoop
        XmfWriter writer = _context.getWriter();
        writer.writeToken("(");

        if (_invokeEnterAndWriteDelim(visitable.getValue(), ", ") == false) {
            return false;
        }

        if (visitable.getValue().length > 0) {
            writer.writeToken(", ");
        }

        XbfVar varElem = visitable.getVar();
        if (invokeEnter(varElem) == false) {
            return false;
        }

        writer.writeToken(" = ");
        if (invokeEnter(visitable.getIndexRange()) == false) {
            return false;
        }

        writer.writeToken(")");

        return true;
    }

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
    @Override
    public boolean enter(XbfFdoStatement visitable)
    {
        // DONE: XbfFdoStatement
        _writeLineDirective(visitable.getLineno(), visitable.getFile());

        XmfWriter writer = _context.getWriter();
        String constuctName = visitable.getConstructName();
        if (XfUtil.isNullOrEmpty(constuctName) == false) {
            writer.writeToken(constuctName);
            writer.writeToken(": ");
        }

        writer.writeToken("DO");

        XbfFdoStatementSequence doStatementSequence = visitable.getFdoStatementSequence();
        if (doStatementSequence != null) {
            writer.writeToken(" ");
            if (invokeEnter(visitable.getFdoStatementSequence()) == false) {
                return false;
            }
        }

        writer.setupNewLine();
        writer.incrementIndentLevel();

        if (invokeEnter(visitable.getBody()) == false) {
            return false;
        }

        writer.decrementIndentLevel();

        writer.writeToken("END DO");
        if (XfUtil.isNullOrEmpty(constuctName) == false) {
            writer.writeToken(" ");
            writer.writeToken(constuctName);
        }
        writer.setupNewLine();

        return true;
    }

    /**
     * Decompile child group of "FdoStatement" element in XcodeML/F.
     *
     * @example <code><div class="Example">
     * DO, <span class="Strong">variable = 1, 10</span><br/>
     * <div class="Indent1">
     *      (any statement...)<br/>
     * </div>
     * END DO<br/>
     * </div></code>
     * @see xcodeml.f.binding.gen.RVisitorBase#enter(xcodeml.f.binding.gen.
     *      XbfFdoStatementSequence)
     */
    @Override
    public boolean enter(XbfFdoStatementSequence visitable)
    {
        // DONE: XbfFdoStatementSequence
        XmfWriter writer = _context.getWriter();

        if (invokeEnter(visitable.getVar()) == false) {
            return false;
        }

        writer.writeToken(" = ");

        if (invokeEnter(visitable.getIndexRange()) == false) {
            return false;
        }

        return true;
    }

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
    @Override
    public boolean enter(XbfFdoWhileStatement visitable)
    {
        // DONE: XbfFdoWhileStatement
        _writeLineDirective(visitable.getLineno(), visitable.getFile());

        XmfWriter writer = _context.getWriter();
        String constuctName = visitable.getConstructName();
        if (XfUtil.isNullOrEmpty(constuctName) == false) {
            writer.writeToken(constuctName);
            writer.writeToken(": ");
        }

        writer.writeToken("DO, WHILE ");
        if (invokeEnter(visitable.getCondition()) == false) {
            return false;
        }

        writer.setupNewLine();
        writer.incrementIndentLevel();

        if (invokeEnter(visitable.getBody()) == false) {
            return false;
        }

        writer.decrementIndentLevel();

        writer.writeToken("END DO");
        if (XfUtil.isNullOrEmpty(constuctName) == false) {
            writer.writeToken(" ");
            writer.writeToken(constuctName);
        }
        writer.setupNewLine();

        return true;
    }

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
    @Override
    public boolean enter(XbfFendFileStatement visitable)
    {
        // DONE: XbfFendFileStatement
        _writeLineDirective(visitable.getLineno(), visitable.getFile());

        XmfWriter writer = _context.getWriter();

        writer.writeToken("ENDFILE ");
        if (invokeEnter(visitable.getNamedValueList()) == false) {
            return false;
        }

        writer.setupNewLine();

        return true;
    }

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
    @Override
    public boolean enter(XbfFentryDecl visitable)
    {
        // DONE: XbfFentryDecl
        _writeLineDirective(visitable.getLineno(), visitable.getFile());

        XfTypeManager typeManager = _context.getTypeManager();
        XmfWriter writer = _context.getWriter();

        XbfName functionNameElem = visitable.getName();
        IXbfTypeTableChoice typeChoice = typeManager.findType(functionNameElem);
        if (typeChoice == null) {
            _context.setLastErrorMessage(XfUtil.formatError(visitable,
                XfError.XCODEML_TYPE_NOT_FOUND, functionNameElem.getType()));
            return false;
        } else if ((typeChoice instanceof XbfFfunctionType) == false) {
            _context.setLastErrorMessage(XfUtil.formatError(visitable,
                XfError.XCODEML_TYPE_MISMATCH, "function definition", XfUtil
                    .getElementName(typeChoice), "FfunctionType"));
            return false;
        }

        XbfFfunctionType functionTypeElem = (XbfFfunctionType)typeChoice;
        String returnTypeName = functionTypeElem.getReturnType();
        XfType typeId = XfType.getTypeIdFromXcodemlTypeName(returnTypeName);
        if (functionTypeElem.getIsProgram()) {
            // =======
            // PROGRAM
            // =======
            _context.setLastErrorMessage(XfUtil.formatError(visitable,
                XfError.XCODEML_TYPE_MISMATCH, "function definition", "PROGRAM",
                "FUNCTION or SUBROUTINE"));
            return false;
        } else {
            // ======================
            // FUNCTION or SUBROUTINE
            // ======================
            writer.decrementIndentLevel();
            writer.writeToken("ENTRY");
            writer.incrementIndentLevel();
            writer.writeToken(" ");
            writer.writeToken(functionNameElem.getContent());
            writer.writeToken("(");

            if (invokeEnter(functionTypeElem.getParams()) == false) {
                return false;
            }

            writer.writeToken(")");
            if (typeId != XfType.VOID) {
                String functionResultName = functionTypeElem.getResultName();
                if (XfUtil.isNullOrEmpty(functionResultName) == false) {
                    writer.writeToken(" RESULT(" + functionResultName + ")");
                }
            }
        }

        writer.setupNewLine();

        return true;
    }

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
    @Override
    public boolean enter(XbfFequivalenceDecl visitable)
    {
        // DONE: XbfFequivalenceDecl
        _writeLineDirective(visitable.getLineno(), visitable.getFile());

        XmfWriter writer = _context.getWriter();
        writer.writeToken("EQUIVALENCE ");

        if (_invokeEnterAndWriteDelim(visitable.getFequivalenceDeclSequence(), ", ") == false) {
            return false;
        }

        writer.setupNewLine();

        return true;
    }

    /**
     * Decompile child group of "FequivalenceDecl" element in XcodeML/F.
     *
     * @example <code><div class="Example">
     * EQUIVALENCE <span class="Strong">(variable1, variable2)</span>, <span class="Strong">(variable3, variable4)</span><br/>
     * </div></code>
     * @see xcodeml.f.binding.gen.RVisitorBase#enter(xcodeml.f.binding.gen.
     *      XbfFequivalenceDeclSequence)
     */
    @Override
    public boolean enter(XbfFequivalenceDeclSequence visitable)
    {
        // DONE: XbfFequivalenceDeclSequence
        XmfWriter writer = _context.getWriter();
        writer.writeToken("(");
        if (invokeEnter(visitable.getVarRef()) == false) {
            return false;
        }

        writer.writeToken(", ");

        if (invokeEnter(visitable.getVarList()) == false) {
            return false;
        }

        writer.writeToken(")");

        return true;
    }

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
    @Override
    public boolean enter(XbfFexitStatement visitable)
    {
        // DONE: XbfFexitStatement
        _writeLineDirective(visitable.getLineno(), visitable.getFile());

        XmfWriter writer = _context.getWriter();
        writer.writeToken("EXIT");

        String constuctName = visitable.getConstructName();
        if (XfUtil.isNullOrEmpty(constuctName) == false) {
            writer.writeToken(" ");
            writer.writeToken(constuctName);
        }

        writer.setupNewLine();

        return true;
    }

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
    @Override
    public boolean enter(XbfFformatDecl visitable)
    {
        // DONE: XbfFformatDecl
        _writeLineDirective(visitable.getLineno(), visitable.getFile());

        XmfWriter writer = _context.getWriter();
        writer.writeToken("FORMAT ");
        writer.writeToken(visitable.getFormat());
        writer.setupNewLine();

        return true;
    }

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
    @Override
    public boolean enter(XbfFfunctionDecl visitable)
    {
        // DONE: XbfFfunctionDecl
        _writeLineDirective(visitable.getLineno(), visitable.getFile());

        XfTypeManager typeManager = _context.getTypeManager();
        XmfWriter writer = _context.getWriter();

        XbfName functionNameElem = visitable.getName();
        IXbfTypeTableChoice typeChoice = typeManager.findType(functionNameElem);
        if (typeChoice == null) {
            _context.setLastErrorMessage(XfUtil.formatError(visitable,
                XfError.XCODEML_TYPE_NOT_FOUND, functionNameElem.getType()));
            return false;
        } else if ((typeChoice instanceof XbfFfunctionType) == false) {
            _context.setLastErrorMessage(XfUtil.formatError(visitable,
                XfError.XCODEML_TYPE_MISMATCH, "function definition", XfUtil
                    .getElementName(typeChoice), "FfunctionType"));
            return false;
        }

        XbfFfunctionType functionTypeElem = (XbfFfunctionType)typeChoice;
        String returnTypeName = functionTypeElem.getReturnType();
        XfType typeId = XfType.getTypeIdFromXcodemlTypeName(returnTypeName);
        if (functionTypeElem.getIsProgram()) {
            // =======
            // PROGRAM
            // =======
            _context.setLastErrorMessage(XfUtil.formatError(visitable,
                XfError.XCODEML_TYPE_MISMATCH, "function definition", "PROGRAM",
                "FUNCTION or SUBROUTINE"));
            return false;
        } else if (typeId == XfType.VOID) {
            // ==========
            // SUBROUTINE
            // ==========
            if (functionTypeElem.getIsRecursive()) {
                writer.writeToken("RECURSIVE");
                writer.writeToken(" ");
            }
            writer.writeToken("SUBROUTINE");
            writer.writeToken(" ");
            writer.writeToken(functionNameElem.getContent());
            writer.writeToken("(");

            if (invokeEnter(functionTypeElem.getParams()) == false) {
                return false;
            }

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
            if (functionTypeElem.getIsRecursive()) {
                writer.writeToken("RECURSIVE");
                writer.writeToken(" ");
            }
            writer.writeToken("FUNCTION");
            writer.writeToken(" ");
            writer.writeToken(functionNameElem.getContent());
            writer.writeToken("(");

            if (invokeEnter(functionTypeElem.getParams()) == false) {
                return false;
            }

            writer.writeToken(")");
            String functionResultName = functionTypeElem.getResultName();
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
        if (invokeEnter(visitable.getDeclarations()) == false) {
            return false;
        }

        // ========
        // Prologue
        // ========
        writer.decrementIndentLevel();

        assert (functionTypeElem.getIsProgram() == false);
        if (typeId == XfType.VOID) {
            // ==========
            // SUBROUTINE
            // ==========
            writer.writeToken("END SUBROUTINE");
            writer.writeToken(" ");
            writer.writeToken(functionNameElem.getContent());
        } else {
            writer.writeToken("END FUNCTION");
            writer.writeToken(" ");
            writer.writeToken(functionNameElem.getContent());
        }
        writer.setupNewLine();

        return true;
    }

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
    @Override
    public boolean enter(XbfFfunctionDefinition visitable)
    {
        // DONE: XbfFfunctionDefinition
        _writeLineDirective(visitable.getLineno(), visitable.getFile());

        XfTypeManager typeManager = _context.getTypeManager();
        XmfWriter writer = _context.getWriter();

        XbfName functionNameElem = visitable.getName();
        IXbfTypeTableChoice typeChoice = typeManager.findType(functionNameElem);
        if (typeChoice == null) {
            _context.setLastErrorMessage(XfUtil.formatError(visitable,
                XfError.XCODEML_TYPE_NOT_FOUND, functionNameElem.getType()));
            return false;
        } else if ((typeChoice instanceof XbfFfunctionType) == false) {
            _context.setLastErrorMessage(XfUtil.formatError(visitable,
                XfError.XCODEML_TYPE_MISMATCH, "function definition", XfUtil
                    .getElementName(typeChoice), "FfunctionType"));
            return false;
        }

        XbfFfunctionType functionTypeElem = (XbfFfunctionType)typeChoice;
        String returnTypeName = functionTypeElem.getReturnType();
        XfType typeId = XfType.getTypeIdFromXcodemlTypeName(returnTypeName);
        if (functionTypeElem.getIsProgram()) {
            // =======
            // PROGRAM
            // =======
            writer.writeToken("PROGRAM");
            writer.writeToken(" ");
            writer.writeToken(functionNameElem.getContent());
        } else if (typeId == XfType.VOID) {
            // ==========
            // SUBROUTINE
            // ==========
            if (functionTypeElem.getIsRecursive()) {
                writer.writeToken("RECURSIVE");
                writer.writeToken(" ");
            }
            writer.writeToken("SUBROUTINE");
            writer.writeToken(" ");
            writer.writeToken(functionNameElem.getContent());
            writer.writeToken("(");

            if (invokeEnter(functionTypeElem.getParams()) == false) {
                return false;
            }

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
            if (functionTypeElem.getIsRecursive()) {
                writer.writeToken("RECURSIVE");
                writer.writeToken(" ");
            }
            writer.writeToken("FUNCTION");
            writer.writeToken(" ");
            writer.writeToken(functionNameElem.getContent());
            writer.writeToken("(");

            if (invokeEnter(functionTypeElem.getParams()) == false) {
                return false;
            }

            writer.writeToken(")");
            String functionResultName = functionTypeElem.getResultName();
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
        if (invokeEnter(visitable.getSymbols()) == false) {
            return false;
        }

        if (invokeEnter(visitable.getDeclarations()) == false) {
            return false;
        }

        writer.setupNewLine();

        if (invokeEnter(visitable.getBody()) == false) {
            return false;
        }

        // ========
        // Prologue
        // ========
        writer.decrementIndentLevel();
        typeManager.leaveScope();

        if (functionTypeElem.getIsProgram()) {
            // =======
            // PROGRAM
            // =======
            writer.writeToken("END PROGRAM");
            writer.writeToken(" ");
            writer.writeToken(functionNameElem.getContent());
        } else if (typeId == XfType.VOID) {
            // ==========
            // SUBROUTINE
            // ==========
            writer.writeToken("END SUBROUTINE");
            writer.writeToken(" ");
            writer.writeToken(functionNameElem.getContent());
        } else {
            writer.writeToken("END FUNCTION");
            writer.writeToken(" ");
            writer.writeToken(functionNameElem.getContent());
        }
        writer.setupNewLine();
        writer.setupNewLine();

        return true;
    }

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
    @Override
    public boolean enter(XbfFifStatement visitable)
    {
        // DONE: XbfFifStatement
        _writeLineDirective(visitable.getLineno(), visitable.getFile());

        XmfWriter writer = _context.getWriter();
        String constuctName = visitable.getConstructName();
        if (XfUtil.isNullOrEmpty(constuctName) == false) {
            writer.writeToken(constuctName);
            writer.writeToken(": ");
        }

        writer.writeToken("IF ");
        if (invokeEnter(visitable.getCondition()) == false) {
            return false;
        }
        writer.writeToken(" THEN");
        writer.setupNewLine();

        if (invokeEnter(visitable.getThen()) == false) {
            return false;
        }

        XbfElse elseElem = visitable.getElse();
        if (elseElem != null) {
            writer.writeToken("ELSE");
            if (XfUtil.isNullOrEmpty(constuctName) == false) {
                writer.writeToken(" ");
                writer.writeToken(constuctName);
            }
            writer.setupNewLine();

            if (invokeEnter(visitable.getElse()) == false) {
                return false;
            }
        }

        writer.writeToken("END IF");
        if (XfUtil.isNullOrEmpty(constuctName) == false) {
            writer.writeToken(" ");
            writer.writeToken(constuctName);
        }
        writer.setupNewLine();

        return true;
    }

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
    @Override
    public boolean enter(XbfFinquireStatement visitable)
    {
        // DONE: XbfFinquireStatement
        _writeLineDirective(visitable.getLineno(), visitable.getFile());

        XmfWriter writer = _context.getWriter();

        writer.writeToken("INQUIRE ");
        if (invokeEnter(visitable.getNamedValueList()) == false) {
            return false;
        }

        if (invokeEnter(visitable.getValueList()) == false) {
            return false;
        }

        writer.setupNewLine();

        return true;
    }

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
    @Override
    public boolean enter(XbfFintConstant visitable)
    {
        // DONE: XbfFintConstant
        String content = visitable.getContent();
        if (XfUtil.isNullOrEmpty(content)) {
            _context.setLastErrorMessage(XfUtil.formatError(visitable, XfError.XCODEML_SEMANTICS,
                XfUtil.getElementName(visitable)));
            return false;
        }

        String typeName = visitable.getType();
        if (XfUtil.isNullOrEmpty(typeName) == false) {
            XfTypeManager typeManager = _context.getTypeManager();
            String bottomTypeName = typeManager.getBottomTypeName(typeName);
            if (bottomTypeName == null) {
                _context.setLastErrorMessage(XfUtil.formatError(visitable,
                    XfError.XCODEML_TYPE_NOT_FOUND,
                    XfUtil.getElementName(visitable),
                    typeName));
                return false;
            }

            XfType typeId = XfType.getTypeIdFromXcodemlTypeName(bottomTypeName);
            if (typeId != XfType.DERIVED && typeId != XfType.INT) {
                _context.setLastErrorMessage(XfUtil.formatError(visitable,
                    XfError.XCODEML_TYPE_MISMATCH, XfUtil.getElementName(visitable), typeName,
                    "Fint"));
                return false;
            }
        }

        XmfWriter writer = _context.getWriter();
        
        String kind = visitable.getKind();
        if (XfUtil.isNullOrEmpty(kind) == false) {
            writer.writeToken(content + "_" + kind);
        } else {
	  /* check minus number */
	  if(new Integer(content).intValue() < 0)
	    content = "("+content+")";
            writer.writeToken(content);
        }

        return true;
    }

    private void _writeInterface(XmfWriter writer, XbfFinterfaceDecl visitable)
    {
        String interfaceName = visitable.getName();
        if (visitable.getIsAssignment()) {
            writer.writeToken("ASSIGNMENT(=)");
        } else if (visitable.getIsOperator()) {
            writer.writeToken("OPERATOR(");
            writer.writeToken(interfaceName);
            writer.writeToken(")");
        } else {
            if (XfUtil.isNullOrEmpty(interfaceName) == false) {
                writer.writeToken(" ");
                writer.writeToken(interfaceName);
            }
        }
    }

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
    @Override
    public boolean enter(XbfFinterfaceDecl visitable)
    {
        // DONE: XbfFinterfaceDecl

        XmfWriter writer = _context.getWriter();
        XfTypeManager typeManager = _context.getTypeManager();

        String interfaceName = visitable.getName();
        if (visitable.getIsAssignment()) {
            interfaceName = "=";
        }

        if (XfUtil.isNullOrEmpty(interfaceName) == false) {
            XbfId interfaceId = typeManager.findSymbol(interfaceName);
            if (interfaceId == null) {
                _context.setLastErrorMessage(XfUtil.formatError(visitable,
                        XfError.XCODEML_NAME_NOT_FOUND, interfaceName));
                return false;
            }

            IXbfTypeTableChoice typeChoice = typeManager.findTypeFromSymbol(interfaceName);
            if (typeChoice == null) {
                _context.setLastErrorMessage(XfUtil.formatError(visitable,
                        XfError.XCODEML_TYPE_NOT_FOUND, interfaceName));
                return false;
            } else if ((typeChoice instanceof XbfFfunctionType) == false) {
                _context.setLastErrorMessage(XfUtil.formatError(visitable,
                        XfError.XCODEML_TYPE_MISMATCH, "function definition", XfUtil
                        .getElementName(typeChoice), "FfunctionType"));
                return false;
            }

            XbfFfunctionType type = (XbfFfunctionType) typeChoice;
            if (type.getIsPublic() || type.getIsPrivate()) {
                if (type.getIsPublic()) {
                    writer.writeToken("PUBLIC :: ");
                } else {
                    writer.writeToken("PRIVATE :: ");
                }
                _writeInterface(writer, visitable);
                _declaredIds.add(interfaceName);
            }
        }

        _writeLineDirective(visitable.getLineno(), visitable.getFile());
        writer.writeToken("INTERFACE ");
        _writeInterface(writer, visitable);

        writer.setupNewLine();
        writer.incrementIndentLevel();

        if (_invokeEnter(visitable.getContent()) == false) {
            return false;
        }

        writer.decrementIndentLevel();

        writer.writeToken("END INTERFACE");
        writer.setupNewLine();

        return true;
    }

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
    @Override
    public boolean enter(XbfFlogicalConstant visitable)
    {
        // DONE: XbfFlogicalConstant
        String content = visitable.getContent();
        if (XfUtil.isNullOrEmpty(content)) {
            _context.setLastErrorMessage(XfUtil.formatError(visitable, XfError.XCODEML_SEMANTICS,
                XfUtil.getElementName(visitable)));
            return false;
        }

        String typeName = visitable.getType();
        if (XfUtil.isNullOrEmpty(typeName) == false) {
            XfTypeManager typeManager = _context.getTypeManager();
            String bottomTypeName = typeManager.getBottomTypeName(typeName);
            if (bottomTypeName == null) {
                _context.setLastErrorMessage(XfUtil.formatError(visitable,
                    XfError.XCODEML_TYPE_NOT_FOUND,
                    XfUtil.getElementName(visitable),
                    typeName));
                return false;
            }

            XfType typeId = XfType.getTypeIdFromXcodemlTypeName(bottomTypeName);
            if (typeId != XfType.DERIVED && typeId != XfType.LOGICAL) {
                _context.setLastErrorMessage(XfUtil.formatError(visitable,
                    XfError.XCODEML_TYPE_MISMATCH, XfUtil.getElementName(visitable), typeName,
                    "Fint"));
                return false;
            }
        }

        XmfWriter writer = _context.getWriter();

        String kind = visitable.getKind();
        if (XfUtil.isNullOrEmpty(kind) == false) {
            writer.writeToken(content + "_" + kind);
        } else {
            writer.writeToken(content);
        }

        return true;
    }

    /**
     * Decompile "FmemberRef" element in XcodeML/F.
     *
     * @example <code><div class="Example">
     *      variable = <span class="Strong">struct%member</span><br/>
     * </div></code>
     * @see xcodeml.f.binding.gen.RVisitorBase#enter(xcodeml.f.binding.gen.XbfFmemberRef
     *      )
     */
    @Override
    public boolean enter(XbfFmemberRef visitable)
    {
        // DONE: XbfFmemberRef
        if (invokeEnter(visitable.getVarRef()) == false) {
            return false;
        }

        XmfWriter writer = _context.getWriter();
        writer.writeToken("%");
        writer.writeToken(visitable.getMember());

        return true;
    }

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
    @Override
    public boolean enter(XbfFmoduleDefinition visitable)
    {
        // DONE: XbfFmoduleDefinition
        _writeLineDirective(visitable.getLineno(), visitable.getFile());

        _moduleAllIds.clear();
        _declaredIds.clear();
        XfTypeManager typeManager = _context.getTypeManager();
        XmfWriter writer = _context.getWriter();

        writer.writeToken("MODULE");
        writer.writeToken(" ");
        writer.writeToken(visitable.getName());
        writer.setupNewLine();

        // ========
        // Epilogue
        // ========
        writer.incrementIndentLevel();
        typeManager.enterScope();

        // ======
        // Inside
        // ======
        if (invokeEnter(visitable.getSymbols()) == false) {
            return false;
        }

        if (invokeEnter(visitable.getDeclarations()) == false) {
            return false;
        }

        _moduleAllIds.removeAll(_declaredIds);
        for (String useAssocId : _moduleAllIds) {
            IXbfTypeTableChoice typeChoice = typeManager.findTypeFromSymbol(useAssocId);
            if(typeChoice == null)
                continue;
            assert(typeChoice.getIsPublic() && typeChoice.getIsPrivate() == false);
            if(typeChoice.getIsPublic()) {
                writer.writeToken("PUBLIC :: ");
                writer.writeToken(_toUserOperatorIfRequired(useAssocId));
                writer.setupNewLine();
            }
            if (typeChoice.getIsPrivate()){
                writer.writeToken("PRIVATE :: ");
                writer.writeToken(_toUserOperatorIfRequired(useAssocId));
                writer.setupNewLine();
            }
        }

        if (invokeEnter(visitable.getFcontainsStatement()) == false) {
            return false;
        }

        // ========
        // Prologue
        // ========
        writer.decrementIndentLevel();
        typeManager.leaveScope();

        writer.writeToken("END MODULE");
        writer.writeToken(" ");
        writer.writeToken(visitable.getName());
        writer.setupNewLine();
        writer.setupNewLine();

        return true;
    }

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
    @Override
    public boolean enter(XbfFmoduleProcedureDecl visitable)
    {
        // DONE: XbfFmoduleProcedureDecl
        _writeLineDirective(visitable.getLineno(), visitable.getFile());

        XmfWriter writer = _context.getWriter();

        writer.writeToken("MODULE PROCEDURE ");
        int nameCount = 0;
        for (XbfName nameElem : visitable.getName()) {
            if (nameCount > 0) {
                writer.writeToken(", ");
            }
            writer.writeToken(nameElem.getContent());
            ++nameCount;
        }
        writer.setupNewLine();

        return true;
    }


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
    @Override
    public boolean enter(XbfFblockDataDefinition visitable)
    {
        // DONE: XbfFblockDataDefinition
        _writeLineDirective(visitable.getLineno(), visitable.getFile());

        XfTypeManager typeManager = _context.getTypeManager();
        XmfWriter writer = _context.getWriter();

        writer.writeToken("BLOCK DATA");
        writer.writeToken(" ");
        writer.writeToken(visitable.getName());
        writer.setupNewLine();

        // ========
        // Epilogue
        // ========
        writer.incrementIndentLevel();
        typeManager.enterScope();

        // ======
        // Inside
        // ======
        if (invokeEnter(visitable.getSymbols()) == false) {
            return false;
        }

        if (invokeEnter(visitable.getDeclarations()) == false) {
            return false;
        }

        // ========
        // Prologue
        // ========
        writer.decrementIndentLevel();
        typeManager.leaveScope();

        writer.writeToken("END BLOCK DATA");
        writer.writeToken(" ");
        writer.writeToken(visitable.getName());
        writer.setupNewLine();
        writer.setupNewLine();

        return true;
    }

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
    @Override
    public boolean enter(XbfFnamelistDecl visitable)
    {
        // DONE: XbfFnamelistDecl


        if (_invokeEnterAndWriteDelim(visitable.getVarList(), ", ") == false) {
            return false;
        }

        return true;
    }

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
    @Override
    public boolean enter(XbfFnullifyStatement visitable)
    {
        // DONE: XbfFnullifyStatement
        _writeLineDirective(visitable.getLineno(), visitable.getFile());

        XmfWriter writer = _context.getWriter();
        writer.writeToken("NULLIFY (");

        if (_invokeEnterAndWriteDelim(visitable.getAlloc(), ", ") == false) {
            return false;
        }

        writer.writeToken(")");
        writer.setupNewLine();

        return true;
    }

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
    @Override
    public boolean enter(XbfFopenStatement visitable)
    {
        // DONE: XbfFopenStatement
        _writeLineDirective(visitable.getLineno(), visitable.getFile());

        XmfWriter writer = _context.getWriter();

        writer.writeToken("OPEN ");
        if (invokeEnter(visitable.getNamedValueList()) == false) {
            return false;
        }

        writer.setupNewLine();

        return true;
    }

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
    @Override
    public boolean enter(XbfFpointerAssignStatement visitable)
    {
        XmfWriter writer = _context.getWriter();
        
        _writeLineDirective(visitable.getLineno(), visitable.getFile());

        if (_writeBinaryExpr((IXbfDefModelExprChoice)visitable.getDefModelLValue(), visitable.getDefModelExpr(), "=>") == false) {
            return false;
        }

        writer.setupNewLine();

        return true;
    }

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
    @Override
    public boolean enter(XbfFpowerExpr visitable)
    {
        // DONE: XbfFpowerExpr
        if (_writeBinaryExpr(visitable.getDefModelExpr1(), visitable.getDefModelExpr2(), "**") == false) {
            return false;
        }

        return true;
    }

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
    @Override
    public boolean enter(XbfFpragmaStatement visitable)
    {
        // DONE: XbfFpragmaStatement
        _writeLineDirective(visitable.getLineno(), visitable.getFile());

        String content;
        content = visitable.getContent();

        if (content.startsWith("!$") == false) {
            content = "!$" + content;
        }

        XmfWriter writer = _context.getWriter();
        writer.writeIsolatedLine(content);

        return true;
    }

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
    @Override
    public boolean enter(XbfFprintStatement visitable)
    {
        // DONE: XbfFprintStatement
        _writeLineDirective(visitable.getLineno(), visitable.getFile());

        XmfWriter writer = _context.getWriter();
        writer.writeToken("PRINT ");
        writer.writeToken(visitable.getFormat());
        
        if(visitable.getValueList() != null && visitable.getValueList().getChildren() != null &&
            visitable.getValueList().getChildren().length > 0) {
            writer.writeToken(", ");
        }

        if (invokeEnter(visitable.getValueList()) == false) {
            return false;
        }

        writer.setupNewLine();

        return true;
    }

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
    @Override
    public boolean enter(XbfFreadStatement visitable)
    {
        // DONE: XbfFreadStatement
        _writeLineDirective(visitable.getLineno(), visitable.getFile());

        XmfWriter writer = _context.getWriter();

        writer.writeToken("READ ");
        if (invokeEnter(visitable.getNamedValueList()) == false) {
            return false;
        }

        if (invokeEnter(visitable.getValueList()) == false) {
            return false;
        }

        writer.setupNewLine();

        return true;
    }

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
    @Override
    public boolean enter(XbfFrealConstant visitable)
    {
        // DONE: XbfFrealConstant
        String content = visitable.getContent();
        if (XfUtil.isNullOrEmpty(content)) {
            _context.setLastErrorMessage(XfUtil.formatError(visitable, XfError.XCODEML_SEMANTICS,
                XfUtil.getElementName(visitable)));
            return false;
        }

        String typeName = visitable.getType();
        if (XfUtil.isNullOrEmpty(typeName) == false) {
            XfTypeManager typeManager = _context.getTypeManager();
            String bottomTypeName = typeManager.getBottomTypeName(typeName);
            if (bottomTypeName == null) {
                _context.setLastErrorMessage(XfUtil.formatError(visitable,
                    XfError.XCODEML_TYPE_NOT_FOUND,
                    XfUtil.getElementName(visitable),
                    typeName));
                return false;
            }

            XfType typeId = XfType.getTypeIdFromXcodemlTypeName(bottomTypeName);
            if (typeId != XfType.DERIVED && typeId != XfType.REAL) {
                _context.setLastErrorMessage(XfUtil.formatError(visitable,
                    XfError.XCODEML_TYPE_MISMATCH, XfUtil.getElementName(visitable), typeName,
                    "Fint"));
                return false;
            }
        }

        XmfWriter writer = _context.getWriter();

        String kind = visitable.getKind();
        // gfortran rejects kind with 'd'/'q' exponent
        if (XfUtil.isNullOrEmpty(kind) == false && 
            ((content.toLowerCase().indexOf("d") < 0) &&
             (content.toLowerCase().indexOf("q") < 0))) {
            writer.writeToken(content + "_" + kind);
        } else {
            writer.writeToken(content);
        }

        return true;
    }

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
    @Override
    public boolean enter(XbfFreturnStatement visitable)
    {
        // DONE: XbfFreturnStatement
        _writeLineDirective(visitable.getLineno(), visitable.getFile());

        XmfWriter writer = _context.getWriter();
        writer.writeToken("RETURN");
        writer.setupNewLine();

        return true;
    }

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
    @Override
    public boolean enter(XbfFrewindStatement visitable)
    {
        // DONE: XbfFrewindStatement
        _writeLineDirective(visitable.getLineno(), visitable.getFile());

        XmfWriter writer = _context.getWriter();

        writer.writeToken("REWIND ");
        if (invokeEnter(visitable.getNamedValueList()) == false) {
            return false;
        }

        writer.setupNewLine();

        return true;
    }

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
    @Override
    public boolean enter(XbfFselectCaseStatement visitable)
    {
        // DONE: XbfFselectCaseStatement
        _writeLineDirective(visitable.getLineno(), visitable.getFile());

        XmfWriter writer = _context.getWriter();
        String constuctName = visitable.getConstructName();
        if (XfUtil.isNullOrEmpty(constuctName) == false) {
            writer.writeToken(constuctName);
            writer.writeToken(": ");
        }

        writer.writeToken("SELECT CASE (");
        if (invokeEnter(visitable.getValue()) == false) {
            return false;
        }

        writer.writeToken(")");
        writer.setupNewLine();

        if (_invokeEnter(visitable.getFcaseLabel()) == false) {
            return false;
        }

        writer.writeToken("END SELECT");
        if (XfUtil.isNullOrEmpty(constuctName) == false) {
            writer.writeToken(" ");
            writer.writeToken(constuctName);
        }
        writer.setupNewLine();

        return true;
    }

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
    @Override
    public boolean enter(XbfFstopStatement visitable)
    {
        // DONE: XbfFstopStatement
        _writeLineDirective(visitable.getLineno(), visitable.getFile());

        XmfWriter writer = _context.getWriter();
        writer.writeToken("STOP ");

        String code = visitable.getCode();
        String message = visitable.getMessage();
        if (XfUtil.isNullOrEmpty(code) == false) {
            writer.writeToken(code);
        } else if (XfUtil.isNullOrEmpty(message) == false) {
            writer.writeLiteralString(message);
        }

        writer.setupNewLine();

        return true;
    }

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
    @Override
    public boolean enter(XbfFpauseStatement visitable)
    {
        // DONE: XbfFpauseStatement
        _writeLineDirective(visitable.getLineno(), visitable.getFile());

        XmfWriter writer = _context.getWriter();
        writer.writeToken("PAUSE ");

        String code = visitable.getCode();
        String message = visitable.getMessage();
        if (XfUtil.isNullOrEmpty(code) == false) {
            writer.writeToken(code);
        } else if (XfUtil.isNullOrEmpty(message) == false) {
            writer.writeLiteralString(message);
        } else {
            writer.writeToken("0");
        }

        writer.setupNewLine();

        return true;
    }

    /**
     * Decompile "FstructConstructor" element in XcodeML/F.
     *
     * @example <code><div class="Example">
     *      struct = <span class="Strong">TYPE_NAME(1, 2, "abc")</span><br/>
     * </div></code>
     * @see xcodeml.f.binding.gen.RVisitorBase#enter(xcodeml.f.binding.gen.
     *      XbfFstructConstructor)
     */
    @Override
    public boolean enter(XbfFstructConstructor visitable)
    {
        // DONE: XbfFstructConstructor
        XfTypeManager typeManager = _context.getTypeManager();

        IXbfTypeTableChoice typeChoice = typeManager.findType(visitable.getType());
        if (typeChoice == null) {
            _context.setLastErrorMessage(XfUtil.formatError(visitable,
                XfError.XCODEML_TYPE_NOT_FOUND, visitable.getType()));
            return false;
        } else if ((typeChoice instanceof XbfFstructType) == false) {
            _context.setLastErrorMessage(XfUtil.formatError(visitable,
                XfError.XCODEML_TYPE_MISMATCH, "struct definition", XfUtil
                    .getElementName(typeChoice), "FstructType"));
            return false;
        }

        XbfFstructType structTypeElem = (XbfFstructType)typeChoice;
        String aliasStructTypeName = typeManager.getAliasTypeName(structTypeElem.getType());

        XmfWriter writer = _context.getWriter();
        writer.writeToken(aliasStructTypeName);
        writer.writeToken("(");

        if (_invokeEnterAndWriteDelim(visitable.getDefModelExpr(), ", ") == false) {
            return false;
        }

        writer.writeToken(")");

        return true;
    }

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
    @Override
    public boolean enter(XbfFstructDecl visitable)
    {
        // DONE: XbfFstructDecl
        _writeLineDirective(visitable.getLineno(), visitable.getFile());

        XbfName nameElem = visitable.getName();
        if(_validator.validAttr(nameElem) == false) {
            _context.debugPrintLine("Detected insufficient attributes");
            _context.setLastErrorMessage(_validator.getErrDesc()); 
            return false;
        }

        String typeId = nameElem.getType();
        XfTypeManager typeManager = _context.getTypeManager();
        IXbfTypeTableChoice typeChoice = typeManager.findType(typeId);
        if (typeChoice == null) {
            _context.setLastErrorMessage(XfUtil.formatError(visitable,
                XfError.XCODEML_TYPE_NOT_FOUND, nameElem.getType()));
            return false;
        } else if ((typeChoice instanceof XbfFstructType) == false) {
            _context.setLastErrorMessage(XfUtil.formatError(visitable,
                XfError.XCODEML_TYPE_MISMATCH, "struct definition", XfUtil
                    .getElementName(typeChoice), "FstructType"));
            return false;
        }

        XbfFstructType structTypeElem = (XbfFstructType)typeChoice;
        String structTypeName = nameElem.getContent();

        typeManager.putAliasTypeName(typeId, structTypeName);

        XmfWriter writer = _context.getWriter();
        writer.writeToken("TYPE");

        if(_isUnderModuleDef()) {
            if (structTypeElem.getIsPrivate()) {
                writer.writeToken(", PRIVATE");
            } else if (structTypeElem.getIsPublic()) {
                writer.writeToken(", PUBLIC");
            }
            _declaredIds.add(structTypeName);
        }

        writer.writeToken(" :: ");
        writer.writeToken(structTypeName);
        writer.setupNewLine();
        writer.incrementIndentLevel();

        if(_isUnderModuleDef()) {
            if (structTypeElem.getIsInternalPrivate()) {
                writer.writeToken("PRIVATE");
                writer.setupNewLine();
            }
        }
        
        if (structTypeElem.getIsSequence()) {
            writer.writeToken("SEQUENCE");
            writer.setupNewLine();
        }

        if (invokeEnter(structTypeElem) == false) {
            return false;
        }

        writer.decrementIndentLevel();
        writer.writeToken("END TYPE");
        writer.writeToken(" ");
        writer.writeToken(structTypeName);
        writer.setupNewLine();

        return true;
    }

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
    @Override
    public boolean enter(XbfFunctionCall visitable)
    {
        // DONE: XbfFunctionCall
        XbfName functionNameElem = visitable.getName();
        if (functionNameElem == null) {
            _context.debugPrintLine("Detected a function call without the name element.");
            _context.setLastErrorMessage(XfUtil.formatError(visitable, XfError.XCODEML_SEMANTICS,
                XfUtil.getElementName(visitable)));
            return false;
        }

        String functionName = functionNameElem.getContent();
        if (XfUtil.isNullOrEmpty(functionName)) {
            _context.debugPrintLine("Function name is empty.");
            _context.setLastErrorMessage(XfUtil.formatError(functionNameElem,
                XfError.XCODEML_SEMANTICS, XfUtil.getElementName(functionNameElem)));
            return false;
        }

        // Note:
        // If it is built-in function, it is not on the type table.
        if(visitable.getIsIntrinsic() == false) {
            XfTypeManager typeManager = _context.getTypeManager();
            IXbfTypeTableChoice typeChoice = typeManager.findType(functionNameElem);
            if (typeChoice == null) {
                _context.setLastErrorMessage(XfUtil.formatError(visitable,
                    XfError.XCODEML_TYPE_NOT_FOUND, functionNameElem.getType()));
                return false;
            } else if ((typeChoice instanceof XbfFfunctionType) == false) {
                _context.setLastErrorMessage(XfUtil.formatError(visitable,
                    XfError.XCODEML_TYPE_MISMATCH, "function definition", XfUtil
                        .getElementName(typeChoice), "FfunctionType"));
                return false;
            }

            XbfFfunctionType functionTypeElem = (XbfFfunctionType)typeChoice;

            if (functionTypeElem.getIsProgram()) {
                // =======
                // PROGRAM
                // =======
                _context.setLastErrorMessage(XfUtil.formatError(visitable,
                    XfError.XCODEML_TYPE_MISMATCH, "function definition", "PROGRAM",
                    "FUNCTION or SUBROUTINE"));
                return false;
            }
        }

        XmfWriter writer = _context.getWriter();
        String returnTypeName = visitable.getType();
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

        if (invokeEnter(visitable.getArguments()) == false) {
            return false;
        }

        writer.writeToken(")");

        return true;
    }

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
    @Override
    public boolean enter(XbfFuseDecl visitable)
    {
        // DONE: XbfFuseDecl
        _writeLineDirective(visitable.getLineno(), visitable.getFile());

        XmfWriter writer = _context.getWriter();
        writer.writeToken("USE ");
        writer.writeToken(visitable.getName());

        for (XbfRename renameElem : visitable.getRename()) {
            if(_validator.validAttr(renameElem) == false) {
                _context.debugPrintLine("Detected insufficient attributes");
                _context.setLastErrorMessage(_validator.getErrDesc()); 
                return false;
            }
        
            String localName = renameElem.getLocalName();
            String useName = renameElem.getUseName();
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

        return true;
    }

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
    @Override
    public boolean enter(XbfFuseOnlyDecl visitable)
    {
        // DONE: XbfFuseOnlyDecl
        _writeLineDirective(visitable.getLineno(), visitable.getFile());

        XmfWriter writer = _context.getWriter();
        writer.writeToken("USE ");
        writer.writeToken(visitable.getName());
        writer.writeToken(", ONLY: ");

        int renamableCount = 0;
        for (XbfRenamable renamableElem : visitable.getRenamable()) {
            if(_validator.validAttr(renamableElem) == false) {
                _context.debugPrintLine("Detected insufficient attributes");
                _context.setLastErrorMessage(_validator.getErrDesc()); 
                return false;
            }
        
            if (renamableCount > 0) {
                writer.writeToken(", ");
            }
            String localName = renamableElem.getLocalName();
            String useName = renamableElem.getUseName();
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

        return true;
    }

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
    @Override
    public boolean enter(XbfFwhereStatement visitable)
    {
        // DONE: XbfFwhereStatement
        _writeLineDirective(visitable.getLineno(), visitable.getFile());

        XmfWriter writer = _context.getWriter();

        writer.writeToken("WHERE ");
        if (invokeEnter(visitable.getCondition()) == false) {
            return false;
        }
        writer.setupNewLine();

        if (invokeEnter(visitable.getThen()) == false) {
            return false;
        }

        XbfElse elseElem = visitable.getElse();
        if (elseElem != null) {
            writer.writeToken("ELSEWHERE");
            writer.setupNewLine();
            if (invokeEnter(visitable.getElse()) == false) {
                return false;
            }
        }

        writer.writeToken("END WHERE");
        writer.setupNewLine();

        return true;
    }

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
    @Override
    public boolean enter(XbfFwriteStatement visitable)
    {
        // DONE: XbfFwriteStatement
        _writeLineDirective(visitable.getLineno(), visitable.getFile());

        XmfWriter writer = _context.getWriter();

        writer.writeToken("WRITE ");
        if (invokeEnter(visitable.getNamedValueList()) == false) {
            return false;
        }

        if (invokeEnter(visitable.getValueList()) == false) {
            return false;
        }

        writer.setupNewLine();

        return true;
    }

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
    @Override
    public boolean enter(XbfGotoStatement visitable)
    {
        // DONE: XbfGotoStatement
        _writeLineDirective(visitable.getLineno(), visitable.getFile());

        XmfWriter writer = _context.getWriter();

        writer.writeToken("GOTO ");
        String labelName = visitable.getLabelName();
        if (XfUtil.isNullOrEmpty(labelName) == false) {
            writer.writeToken(labelName);
        } else {
            if (invokeEnter(visitable.getGotoStatementSequence()) == false) {
                return false;
            }
        }
        writer.setupNewLine();

        return true;
    }

    /**
     * Decompile child group of "gotoStatement" element in XcodeML/F.
     *
     * @example <code><div class="Example">
     * GOTO 1000<br/>
     * 1000 CONTINUE<br/>
     * <br/>
     * GOTO <span class="Strong">(2000, 2001, 2002), variable</span><br/>
     * 2000 (any statement...)<br/>
     * 2001 (any statement...)<br/>
     * 2002 (any statement...)<br/>
     * </div></code>
     * @see xcodeml.f.binding.gen.RVisitorBase#enter(xcodeml.f.binding.gen.
     *      XbfGotoStatementSequence)
     */
    @Override
    public boolean enter(XbfGotoStatementSequence visitable)
    {
        // DONE: XbfGotoStatementSequence
        XmfWriter writer = _context.getWriter();

        writer.writeToken("(");

        if (invokeEnter(visitable.getParams()) == false) {
            return false;
        }

        writer.writeToken("), ");

        if (invokeEnter(visitable.getValue()) == false) {
            return false;
        }

        return true;
    }

    /**
     * Decompile "id" element in XcodeML/F.
     *
     * @deprecated Because handle it at a upper level element, warn it when this
     *             method was called it.
     * @see xcodeml.f.binding.gen.RVisitorBase#enter(xcodeml.f.binding.gen.XbfId)
     */
    @Override
    public boolean enter(XbfId visitable)
    {
        // DONE: XbfId
        assert (false);
        return true;
    }

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
    @Override
    public boolean enter(XbfIndexRange visitable)
    {
        // DONE: XbfIndexRange
        XmfWriter writer = _context.getWriter();

        String delim;
        if (_isInvokeNodeOf(XbfFdoLoop.class, 1)) {
            // Parent node is XbfFdoLoop
            delim = ", ";
        } else if (_isInvokeNodeOf(XbfFdoStatementSequence.class, 1)) {
            // Parent node is XbfFdoStatementSequence
            delim = ", ";
        } else {
            delim = ":";
        }
        
        if(visitable.getIsAssumedShape() &&
           visitable.getIsAssumedSize()) {
           // semantics error.
           _context.debugPrintLine(
                "'is_assumed_shape' and 'is_assumed_size' are logically exclusize attributes.");
           _context.setLastErrorMessage(
                XfUtil.formatError(visitable,
                    XfError.XCODEML_SEMANTICS,
                    XfUtil.getElementName(visitable)));
           return false;
        }

        if (visitable.getIsAssumedShape()) {
            XbfLowerBound lowerBound = visitable.getLowerBound();
            if (invokeEnter(lowerBound) == false) {
                return false;
            }

            writer.writeToken(":");
            return true;
        }

        if (visitable.getIsAssumedSize()) {
            XbfLowerBound lowerBound = visitable.getLowerBound();
            if (lowerBound != null) {
                if (invokeEnter(lowerBound) == false) {
                    return false;
                }
                writer.writeToken(":");
            }
            writer.writeToken("*");
            return true;
        }

        if (invokeEnter(visitable.getLowerBound()) == false) {
            return false;
        }

        writer.writeToken(delim);

        if (invokeEnter(visitable.getUpperBound()) == false) {
            return false;
        }

        XbfStep step = visitable.getStep();
        if (step != null) {
            writer.writeToken(delim);
            if (invokeEnter(step) == false) {
                return false;
            }
        }

        return true;
    }

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
    @Override
    public boolean enter(XbfKind visitable)
    {
        // DONE: XbfKind
        if (invokeEnter(visitable.getDefModelExpr()) == false) {
            return false;
        }

        return true;
    }

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
    @Override
    public boolean enter(XbfLen visitable)
    {
        // DONE: XbfLen
        if (visitable.getDefModelExpr() == null) {
            XmfWriter writer = _context.getWriter();
            writer.writeToken("*");
        } else if (invokeEnter(visitable.getDefModelExpr()) == false) {
            return false;
        }

        return true;
    }

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
    @Override
    public boolean enter(XbfLogAndExpr visitable)
    {
        // DONE: XbfLogAndExpr
        if (_writeBinaryExpr(visitable.getDefModelExpr1(), visitable.getDefModelExpr2(), ".AND.") == false) {
            return false;
        }

        return true;
    }

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
    @Override
    public boolean enter(XbfLogEQExpr visitable)
    {
        // DONE: XbfLogEQExpr
        if (_writeBinaryExpr(visitable.getDefModelExpr1(), visitable.getDefModelExpr2(), "==") == false) {
            return false;
        }

        return true;
    }

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
    @Override
    public boolean enter(XbfLogEQVExpr visitable)
    {
        // DONE: XbfLogEQVExpr
        if (_writeBinaryExpr(visitable.getDefModelExpr1(), visitable.getDefModelExpr2(), ".EQV.") == false) {
            return false;
        }

        return true;
    }

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
    @Override
    public boolean enter(XbfLogGEExpr visitable)
    {
        // DONE: XbfLogGEExpr
        if (_writeBinaryExpr(visitable.getDefModelExpr1(), visitable.getDefModelExpr2(), ">=") == false) {
            return false;
        }

        return true;
    }

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
    @Override
    public boolean enter(XbfLogGTExpr visitable)
    {
        // DONE: XbfLogGTExpr
        if (_writeBinaryExpr(visitable.getDefModelExpr1(), visitable.getDefModelExpr2(), ">") == false) {
            return false;
        }

        return true;
    }

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
    @Override
    public boolean enter(XbfLogLEExpr visitable)
    {
        // DONE: XbfLogLEExpr
        if (_writeBinaryExpr(visitable.getDefModelExpr1(), visitable.getDefModelExpr2(), "<=") == false) {
            return false;
        }

        return true;
    }

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
    @Override
    public boolean enter(XbfLogLTExpr visitable)
    {
        // DONE: XbfLogLTExpr
        if (_writeBinaryExpr(visitable.getDefModelExpr1(), visitable.getDefModelExpr2(), "<") == false) {
            return false;
        }

        return true;
    }

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
    @Override
    public boolean enter(XbfLogNEQExpr visitable)
    {
        // DONE: XbfLogNEQExpr
        if (_writeBinaryExpr(visitable.getDefModelExpr1(), visitable.getDefModelExpr2(), "/=") == false) {
            return false;
        }

        return true;
    }

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
    @Override
    public boolean enter(XbfLogNEQVExpr visitable)
    {
        // DONE: XbfLogNEQVExpr
        if (_writeBinaryExpr(visitable.getDefModelExpr1(), visitable.getDefModelExpr2(), ".NEQV.") == false) {
            return false;
        }

        return true;
    }

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
    @Override
    public boolean enter(XbfLogNotExpr visitable)
    {
        // DONE: XbfLogNotExpr
        if (_writeUnaryExpr(visitable.getDefModelExpr(), ".NOT.", true) == false) {
            return false;
        }

        return true;
    }

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
    @Override
    public boolean enter(XbfLogOrExpr visitable)
    {
        // DONE: XbfLogOrExpr
        if (_writeBinaryExpr(visitable.getDefModelExpr1(), visitable.getDefModelExpr2(), ".OR.") == false) {
            return false;
        }

        return true;
    }

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
    @Override
    public boolean enter(XbfLowerBound visitable)
    {
        // DONE: XbfLowerBound
        if (invokeEnter(visitable.getDefModelExpr()) == false) {
            return false;
        }

        return true;
    }

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
    @Override
    public boolean enter(XbfMinusExpr visitable)
    {
        // DONE: XbfMinusExpr
        if (_writeBinaryExpr(visitable.getDefModelExpr1(), visitable.getDefModelExpr2(), "-") == false) {
            return false;
        }

        return true;
    }

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
    @Override
    public boolean enter(XbfMulExpr visitable)
    {
        // DONE: XbfMulExpr
        if (_writeBinaryExpr(visitable.getDefModelExpr1(), visitable.getDefModelExpr2(), "*") == false) {
            return false;
        }

        return true;
    }

    /**
     * Decompile "name" element in XcodeML/F.
     *
     * @deprecated Because handle it at a upper level element, warn it when this
     *             method was called it.
     * @see xcodeml.f.binding.gen.RVisitorBase#enter(xcodeml.f.binding.gen.XbfName)
     */
    @Override
    public boolean enter(XbfName visitable)
    {
        // DONE: XbfName
        assert (false);
        return true;
    }

    /**
     * Decompile "namedValue" element in XcodeML/F.
     *
     * @example <code><div class="Example">
     * OPEN (<span class="Strong">UNIT=1</span>, <span class="Strong">...</span>)<br/>
     * </div></code>
     * @see xcodeml.f.binding.gen.RVisitorBase#enter(xcodeml.f.binding.gen.XbfNamedValue
     *      )
     */
    @Override
    public boolean enter(XbfNamedValue visitable)
    {
        IXbfDefModelExprChoice defModelExpr;
        defModelExpr = visitable.getDefModelExpr();
        // DONE: XbfNamedValue
        XmfWriter writer = _context.getWriter();
        writer.writeToken(visitable.getName());
        writer.writeToken("=");

        if(defModelExpr == null) {
            writer.writeToken(visitable.getValue());
        } else {
            if (invokeEnter(visitable.getDefModelExpr()) == false) {
                return false;
            }
        }

        return true;
    }

    /**
     * Decompile "namedValueList" element in XcodeML/F.
     *
     * @example <code><div class="Example">
     * OPEN <span class="Strong">(UNIT=1, ...)</span><br/>
     * </div></code>
     * @see xcodeml.f.binding.gen.RVisitorBase#enter(xcodeml.f.binding.gen.
     *      XbfNamedValueList)
     */
    @Override
    public boolean enter(XbfNamedValueList visitable)
    {
        // DONE: XbfNamedValueList
        XmfWriter writer = _context.getWriter();
        writer.writeToken("(");

        if (_invokeEnterAndWriteDelim(visitable.getNamedValue(), ", ") == false) {
            return false;
        }

        writer.writeToken(")");

        return true;
    }

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
    @Override
    public boolean enter(XbfParams visitable)
    {
        // DONE: XbfParams
        XmfWriter writer = _context.getWriter();

        int paramCount = 0;
        for (XbfName nameElem : visitable.getName()) {
            if (paramCount > 0) {
                writer.writeToken(", ");
            }
            writer.writeToken(nameElem.getContent());
            ++paramCount;
        }

        return true;
    }

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
    @Override
    public boolean enter(XbfPlusExpr visitable)
    {
        // DONE: XbfPlusExpr
        if (_writeBinaryExpr(visitable.getDefModelExpr1(), visitable.getDefModelExpr2(), "+") == false) {
            return false;
        }

        return true;
    }

    /**
     * Decompile "renamable" element in XcodeML/F.
     *
     * @deprecated Because handle it at a upper level element, warn it when this
     *             method was called it.
     * @see xcodeml.f.binding.gen.RVisitorBase#enter(xcodeml.f.binding.gen.XbfRenamable
     *      )
     */
    @Deprecated
    @Override
    public boolean enter(XbfRenamable visitable)
    {
        // DONE: XbfRenamable
        assert (false);
        return true;
    }

    /**
     * Decompile "rename" element in XcodeML/F.
     *
     * @deprecated Because handle it at a upper level element, warn it when this
     *             method was called it.
     * @see xcodeml.f.binding.gen.RVisitorBase#enter(xcodeml.f.binding.gen.XbfRename)
     */

    @Deprecated
    @Override
    public boolean enter(XbfRename visitable)
    {
        // DONE: XbfRename
        assert (false);
        return true;
    }

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
    @Override
    public boolean enter(XbfStatementLabel visitable)
    {
        // DONE: XbfStatementLabel
        _writeLineDirective(visitable.getLineno(), visitable.getFile());

        XmfWriter writer = _context.getWriter();
        writer.decrementIndentLevel();
        writer.writeToken(visitable.getLabelName());
        writer.incrementIndentLevel();
        if((_nextNode != null) &&
           (_nextNode instanceof XbfStatementLabel)) {
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

        return true;
    }

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
    @Override
    public boolean enter(XbfStep visitable)
    {
        // DONE: XbfStep
        if (invokeEnter(visitable.getDefModelExpr()) == false) {
            return false;
        }

        return true;
    }

    /**
     * Decompile "symbols" element in XcodeML/F.
     * <p>
     * The decompilation result depends on a child element.
     * </p>
     *
     * @see xcodeml.f.binding.gen.RVisitorBase#enter(xcodeml.f.binding.gen.XbfSymbols
     *      )
     */
    @Override
    public boolean enter(XbfSymbols visitable)
    {
        // DONE: XbfSymbols
        if (_isInvokeAncestorNodeOf(XbfFstructDecl.class) == false) {
            _context.debugPrintLine("Add to symbol table.");
            XfTypeManager typeManager = _context.getTypeManager();
            for (XbfId idElem : visitable.getId()) {
                typeManager.addSymbol(idElem);

                if (_isInvokeNodeOf(XbfFmoduleDefinition.class, 1)) {
                    if(idElem.getName() == null)
                        continue;
                    String name = idElem.getName().getContent();
                    _moduleAllIds.add(name);
                }
            }

            // _context.debugPrint(typeManager.toString());
        } else {
            _context.debugPrintLine("Write symbol.");
            for (XbfId idElem : visitable.getId()) {
                String typeName;

                typeName = idElem.getType();

                XbfName nameElem = idElem.getName();
                if (typeName == null) {
                    typeName = nameElem.getType();

                    if(typeName == null) {
                    _context.setLastErrorMessage(
                         XfUtil.formatError(idElem,
                         XfError.XCODEML_NEED_ATTR,
                         "type",XfUtil.getElementName(visitable)));
                    return false;
                    }
                }

                String symbolName = nameElem.getContent();

                XfSymbol symbol = _makeSymbol(symbolName, typeName);
                if (symbol == null) {
                    _context.setLastErrorMessage(XfUtil.formatError(idElem,
                        XfError.XCODEML_TYPE_NOT_FOUND, typeName));
                    return false;
                }
                if (_writeSymbolDecl(symbol, visitable) == false) {
                    return false;
                }
                _context.getWriter().setupNewLine();
            }
        }

        return true;
    }

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
    @Override
    public boolean enter(XbfThen visitable)
    {
        // DONE: XbfThen
        XmfWriter writer = _context.getWriter();
        writer.incrementIndentLevel();

        if (invokeEnter(visitable.getBody()) == false) {
            return false;
        }

        writer.decrementIndentLevel();

        return true;
    }

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
    @Override
    public boolean enter(XbfUnaryMinusExpr visitable)
    {
        // DONE: XbfUnaryMinusExpr

        boolean grouping = true;
        IXbfDefModelExprChoice child = visitable.getDefModelExpr();

        if (_isConstantExpr(visitable.rGetParentRNode()) &&
            _isConstantExpr(child)) {
            grouping = false;
        }

        if (_writeUnaryExpr(child, "-", grouping) == false) {
            return false;
        }

        return true;
    }

    /**
     * Decompile "lowerBound" element in XcodeML/F.
     *
     * @example <code><div class="Example">
     *      array = int_array_variable(<span class="Strong">10</span>,
     *      1:<span class="Strong">10</span>,
     *      1:, :)<br/>
     * </div></code>
     * @see xcodeml.f.binding.gen.RVisitorBase#enter(xcodeml.f.binding.gen.XbfUpperBound
     *      )
     */
    @Override
    public boolean enter(XbfUpperBound visitable)
    {
        // DONE: XbfUpperBound
        if (invokeEnter(visitable.getDefModelExpr()) == false) {
            return false;
        }

        return true;
    }

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
    @Override
    public boolean enter(XbfUserBinaryExpr visitable)
    {
        // DONE: XbfUserBinaryExpr
        String name = visitable.getName();
        if (XfUtil.isNullOrEmpty(name)) {
            _context.setLastErrorMessage(XfUtil.formatError(visitable, XfError.XCODEML_NEED_ATTR,
                "name", XfUtil.getElementName(visitable)));
            return false;
        }

        if (_writeBinaryExpr(visitable.getDefModelExpr1(), visitable.getDefModelExpr2(), name) == false) {
            return false;
        }

        return true;
    }

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
    @Override
    public boolean enter(XbfUserUnaryExpr visitable)
    {
        // DONE: XbfUserUnaryExpr
        String name = visitable.getName();
        if (XfUtil.isNullOrEmpty(name)) {
            _context.setLastErrorMessage(XfUtil.formatError(visitable, XfError.XCODEML_NEED_ATTR,
                "name", XfUtil.getElementName(visitable)));
            return false;
        }

        if (_writeUnaryExpr(visitable.getDefModelExpr(), name, true) == false) {
            return false;
        }

        return true;
    }

    /**
     * Decompile "value" element in XcodeML/F.
     * <p>
     * The decompilation result depends on a child element.
     * </p>
     *
     * @see xcodeml.f.binding.gen.RVisitorBase#enter(xcodeml.f.binding.gen.XbfValue)
     */
    @Override
    public boolean enter(XbfValue visitable)
    {
        // DONE: XbfValue
        if (visitable.getRepeatCount() != null) {
            XmfWriter writer = _context.getWriter();
            if(invokeEnter(visitable.getRepeatCount()) == false)
                return false;
            writer.writeToken("*");
        }
        if (invokeEnter(visitable.getDefModelExpr()) == false) {
            return false;
        }

        return true;
    }

    @Override
    public boolean enter(XbfRepeatCount visitable)
    {
        return invokeEnter(visitable.getDefModelExpr());
    }

    /**
     * Decompile "valueList" element in XcodeML/F.
     * <p>
     * The decompilation result depends on a child element.
     * </p>
     *
     * @see xcodeml.f.binding.gen.RVisitorBase#enter(xcodeml.f.binding.gen.XbfValueList
     *      )
     */
    @Override
    public boolean enter(XbfValueList visitable)
    {
        // DONE: XbfValueList
        if (_invokeEnterAndWriteDelim(visitable.getValue(), ", ") == false) {
            return false;
        }

        return true;
    }

    /**
     * Decompile "Ffunction" element in XcodeML/F.
     * <p>
     * The decompilation result depends on a child element.
     * </p>
     *
     * @see xcodeml.f.binding.gen.RVisitorBase#enter(xcodeml.f.binding.gen.XbfFfunction)
     */
    @Override
    public boolean enter(XbfFfunction visitable)
    {
        // DONE: XbfFfunction
        XmfWriter writer = _context.getWriter();
        writer.writeToken(visitable.getContent());

        return true;
    }

    /**
     * Decompile "Var" element in XcodeML/F.
     * <p>
     * The decompilation result depends on a child element.
     * </p>
     *
     * @see xcodeml.f.binding.gen.RVisitorBase#enter(xcodeml.f.binding.gen.XbfVar)
     */
    @Override
    public boolean enter(XbfVar visitable)
    {
        // DONE: XbfVar
        XmfWriter writer = _context.getWriter();
        writer.writeToken(visitable.getContent());

        return true;
    }

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
    @Override
    public boolean enter(XbfVarDecl visitable)
    {
        // DONE: XbfVarDecl
        _writeLineDirective(visitable.getLineno(), visitable.getFile());

        XbfName nameElem = visitable.getName();
        XfSymbol symbol = _makeSymbol(nameElem);
        if (symbol == null) {
            _context.setLastErrorMessage(XfUtil.formatError(nameElem,
                XfError.XCODEML_NAME_NOT_FOUND, nameElem.getContent()));
            return false;
        }
        if (_writeSymbolDecl(symbol, visitable) == false) {
            return false;
        }

        XbfValue valueElem = visitable.getValue();
        if (valueElem != null) {
            XmfWriter writer = _context.getWriter();
            if (isPointerType(visitable)) {
                writer.writeToken(" => ");
            } else {
                writer.writeToken(" = ");
            }
            if (invokeEnter(valueElem) == false) {
                return false;
            }
        }

        _context.getWriter().setupNewLine();

        if(_isUnderModuleDef()) {
            _declaredIds.add(symbol.getSymbolName());
        }

        return true;
    }

    private boolean isPointerType(String typesym)
    {
        if(typesym == null)
            return false;
        XfTypeManager typeManager = _context.getTypeManager();
        IXbfTypeTableChoice choice = typeManager.findType(typesym);
        return isPointerType(choice);
    }

    private boolean isPointerType(XbfFbasicType type)
    {
        if (type == null)
            return false;
        if (type.getIsPointer())
            return true;
        else
            return isPointerType(type.getRefAsString());
    }

    private boolean isPointerType(IXbfTypeTableChoice choice)
    {
        if (choice == null)
            return false;
        if (choice instanceof XbfFbasicType)
            return isPointerType((XbfFbasicType)choice);
        return false;
    }

    /**
     * Check if varDecl is a pointer assignment or not.
     * @param lvalue The identifier initialized with rvalue.
     * @return True if varDecl is a pointer assignment.
     */
    private boolean isPointerType(XbfVarDecl lvalue)
    {
        if (lvalue == null)
            return false;
        XbfName name = lvalue.getName();
        if (name == null)
            return false;
        return isPointerType(name.getTypeAsString());
    }

    /**
     * Decompile "varList" element in XcodeML/F.
     * <p>
     * The decompilation result depends on a child element.
     * </p>
     *
     * @see xcodeml.f.binding.gen.RVisitorBase#enter(xcodeml.f.binding.gen.XbfVarList
     *      )
     */
    @Override
    public boolean enter(XbfVarList visitable)
    {
        // DONE: XbfVarList
        XmfWriter writer = _context.getWriter();
        String name = visitable.getName();

        if (_isInvokeNodeOf(XbfFcommonDecl.class, 1)) {
            // Parent node is XbfFcommonDecl
            writer.writeToken("/");
            if (XfUtil.isNullOrEmpty(name) == false) {
                writer.writeToken(name);
            }
            writer.writeToken("/ ");
        } else if (_isInvokeNodeOf(XbfFnamelistDecl.class, 1)) {
            // Parent node is XbfFnamelistDecl
            if (XfUtil.isNullOrEmpty(name)) {
                _context.setLastErrorMessage(XfUtil.formatError(visitable,
                    XfError.XCODEML_NEED_ATTR, "name", XfUtil.getElementName(visitable)));
                return false;
            }

            XfTypeManager typeManager = _context.getTypeManager();
            IXbfTypeTableChoice typeChoice = typeManager.findTypeFromSymbol(name);

            if(typeChoice != null && _isInvokeNodeOf(XbfFmoduleDefinition.class, 3)) {
                assert((typeChoice instanceof XbfFbasicType));
                XbfFbasicType type = (XbfFbasicType) typeChoice;
                assert(type.getIsPublic() && type.getIsPrivate() == false);
                if(type.getIsPublic() || type.getIsPrivate() || type.getIsSave()) {
                    if(type.getIsPublic()) {
                        writer.writeToken("PUBLIC");
                    }
                    if(type.getIsPrivate()) {
                        writer.writeToken("PRIVATE");
                    }
                    if(type.getIsSave()) {
                        if(type.getIsPublic() || type.getIsPrivate()) {
                            writer.writeToken(", ");
                        }
                        writer.writeToken("SAVE");
                    }
                    writer.writeToken(" :: ");
                    writer.writeToken(name);
                    writer.setupNewLine();
                    _declaredIds.add(name);
                }
            }

            XbfFnamelistDecl parent = (XbfFnamelistDecl)_getInvokeNode(1);
            _writeLineDirective(parent.getLineno(), parent.getFile());
            writer.writeToken("NAMELIST ");
            writer.writeToken("/");
            writer.writeToken(name);
            writer.writeToken("/ ");

        } else if (_isInvokeNodeOf(XbfFdataDeclSequence.class, 1)) {
            // Parent node is XbfFdataDeclSequence
        } else if (_isInvokeNodeOf(XbfFequivalenceDeclSequence.class, 1)) {
            // Parent node is XbfFequivalenceDeclSequence
        } else {
            assert (false);
        }

        if (_invokeEnterAndWriteDelim(visitable.getContent(), ", ") == false) {
            return false;
        }

        if (_isInvokeNodeOf(XbfFnamelistDecl.class, 1)) {
            writer.setupNewLine();
        }

        return true;
    }

    /**
     * Decompile 'varRef' element in XcodeML/F.
     * <p>
     * The decompilation result depends on a child element.
     * </p>
     *
     * @see xcodeml.f.binding.gen.RVisitorBase#enter(xcodeml.f.binding.gen.XbfVarRef)
     */
    @Override
    public boolean enter(XbfVarRef visitable)
    {
        // DONE: XbfVarRef
        if (invokeEnter(visitable.getContent()) == false) {
            return false;
        }

        return true;
    }
}
