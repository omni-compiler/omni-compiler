package exc.xcodeml;

import exc.object.*;

public class XcodeMLNameTable_C extends XcodeMLNameTable {
	XcodeMLName table[] = {
		new XcodeMLName(Xcode.IDENT, "name"),
		new XcodeMLName(Xcode.ID_LIST, "symbols"),
		new XcodeMLName(Xcode.LIST, "params"),
		new XcodeMLName(Xcode.LIST, "declarations"),
		new XcodeMLName(Xcode.LIST, "arguments"),
		new XcodeMLName(Xcode.LIST, "body"),

		new XcodeMLName(Xcode.LIST, "list"),
		new XcodeMLName(Xcode.STRING, "string"),
		new XcodeMLName(Xcode.LIST, "value"),
		new XcodeMLName(Xcode.LIST, "codimensions"),        // added ID=284

		new XcodeMLName(Xcode.STRING_CONSTANT, "stringConstant"),
		new XcodeMLName(Xcode.INT_CONSTANT, "intConstant"),
		new XcodeMLName(Xcode.FLOAT_CONSTANT, "floatConstant"),
		new XcodeMLName(Xcode.LONGLONG_CONSTANT, "longlongConstant"),
		new XcodeMLName(Xcode.MOE_CONSTANT, "moeConstant"),

		new XcodeMLName(Xcode.FUNCTION_DEFINITION, "functionDefinition"),
		new XcodeMLName(Xcode.VAR_DECL, "varDecl"),
		new XcodeMLName(Xcode.FUNCTION_DECL, "functionDecl"),

		new XcodeMLName(Xcode.COMPOUND_STATEMENT, "compoundStatement"),
		new XcodeMLName(Xcode.EXPR_STATEMENT, "exprStatement"),
		new XcodeMLName(Xcode.WHILE_STATEMENT, "whileStatement"),
		new XcodeMLName(Xcode.DO_STATEMENT, "doStatement"),
		new XcodeMLName(Xcode.FOR_STATEMENT, "forStatement"),
		new XcodeMLName(Xcode.IF_STATEMENT, "ifStatement"),
		new XcodeMLName(Xcode.SWITCH_STATEMENT, "switchStatement"),
		new XcodeMLName(Xcode.BREAK_STATEMENT, "breakStatement"),
		new XcodeMLName(Xcode.RETURN_STATEMENT, "returnStatement"),
		new XcodeMLName(Xcode.GOTO_STATEMENT, "gotoStatement"),
		new XcodeMLName(Xcode.CONTINUE_STATEMENT, "continueStatement"),
		new XcodeMLName(Xcode.STATEMENT_LABEL, "statementLabel"),
		new XcodeMLName(Xcode.CASE_LABEL, "caseLabel"),
		new XcodeMLName(Xcode.DEFAULT_LABEL, "defaultLabel"),

		new XcodeMLName(Xcode.CONDITIONAL_EXPR, "condExpr"),
		new XcodeMLName(Xcode.COMMA_EXPR, "commaExpr"),
		new XcodeMLName(Xcode.ASSIGN_EXPR, "assignExpr"),
		new XcodeMLName(Xcode.PLUS_EXPR, "plusExpr"),
		new XcodeMLName(Xcode.ASG_PLUS_EXPR, "asgPlusExpr"),
		new XcodeMLName(Xcode.MINUS_EXPR, "minusExpr"),
		new XcodeMLName(Xcode.ASG_MINUS_EXPR, "asgMinusExpr"),
		new XcodeMLName(Xcode.UNARY_MINUS_EXPR, "unaryMinusExpr"),
		new XcodeMLName(Xcode.MUL_EXPR, "mulExpr"),
		new XcodeMLName(Xcode.ASG_MUL_EXPR, "asgMulExpr"),
		new XcodeMLName(Xcode.DIV_EXPR, "divExpr"),
		new XcodeMLName(Xcode.ASG_DIV_EXPR, "asgDivExpr"),
		new XcodeMLName(Xcode.MOD_EXPR, "modExpr"),
		new XcodeMLName(Xcode.ASG_MOD_EXPR, "asgModExpr"),
		new XcodeMLName(Xcode.LSHIFT_EXPR, "LshiftExpr"),
		new XcodeMLName(Xcode.ASG_LSHIFT_EXPR, "asgLshiftExpr"),
		new XcodeMLName(Xcode.RSHIFT_EXPR, "RshiftExpr"),
		new XcodeMLName(Xcode.ASG_RSHIFT_EXPR, "asgRshiftExpr"),
		new XcodeMLName(Xcode.BIT_AND_EXPR, "bitAndExpr"),
		new XcodeMLName(Xcode.ASG_BIT_AND_EXPR, "asgBitAndExpr"),
		new XcodeMLName(Xcode.BIT_OR_EXPR, "bitOrExpr"),
		new XcodeMLName(Xcode.ASG_BIT_OR_EXPR, "asgBitOrExpr"),
		new XcodeMLName(Xcode.BIT_XOR_EXPR, "bitXorExpr"),
		new XcodeMLName(Xcode.ASG_BIT_XOR_EXPR, "asgBitXorExpr"),
		new XcodeMLName(Xcode.BIT_NOT_EXPR, "bitNotExpr"),
		new XcodeMLName(Xcode.DESIGNATED_VALUE, "designatedValue"),
		new XcodeMLName(Xcode.COMPOUND_VALUE, "compoundValue"),
		new XcodeMLName(Xcode.COMPOUND_VALUE_ADDR, "compoundValueAddr"),

		new XcodeMLName(Xcode.LOG_EQ_EXPR, "logEQExpr"),
		new XcodeMLName(Xcode.LOG_NEQ_EXPR, "logNEQExpr"),
		new XcodeMLName(Xcode.LOG_GE_EXPR, "logGEExpr"),
		new XcodeMLName(Xcode.LOG_GT_EXPR, "logGTExpr"),
		new XcodeMLName(Xcode.LOG_LE_EXPR, "logLEExpr"),
		new XcodeMLName(Xcode.LOG_LT_EXPR, "logLTExpr"),
		new XcodeMLName(Xcode.LOG_AND_EXPR, "logAndExpr"),
		new XcodeMLName(Xcode.LOG_OR_EXPR, "logOrExpr"),
		new XcodeMLName(Xcode.LOG_NOT_EXPR, "logNotExpr"),

		new XcodeMLName(Xcode.FUNCTION_CALL, "functionCall"),
		new XcodeMLName(Xcode.POINTER_REF, "pointerRef"),
		new XcodeMLName(Xcode.SIZE_OF_EXPR, "sizeOfExpr"),
		new XcodeMLName(Xcode.CAST_EXPR, "castExpr"),
		new XcodeMLName(Xcode.PRE_INCR_EXPR, "preIncrExpr"),
		new XcodeMLName(Xcode.PRE_DECR_EXPR, "preDecrExpr"),
		new XcodeMLName(Xcode.POST_INCR_EXPR, "postIncrExpr"),
		new XcodeMLName(Xcode.POST_DECR_EXPR, "postDecrExpr"),
		new XcodeMLName(Xcode.ADDR_OF, "addrOf"), // In XmcXobjectToXmObjTranslator, name is not used.
		new XcodeMLName(Xcode.ADDR_OF, "addrOfExpr"),
		new XcodeMLName(Xcode.TYPE_NAME, "typeName"),

		new XcodeMLName(Xcode.VAR, "Var"),
		new XcodeMLName(Xcode.VAR_ADDR, "varAddr"),
		new XcodeMLName(Xcode.ARRAY_REF, "arrayRef"),
		new XcodeMLName(Xcode.ARRAY_ADDR, "arrayAddr"),
		new XcodeMLName(Xcode.FUNC_ADDR, "funcAddr"),
		new XcodeMLName(Xcode.MEMBER_REF, "memberRef"),
		new XcodeMLName(Xcode.MEMBER_ARRAY_REF, "memberArrayRef"),
		new XcodeMLName(Xcode.MEMBER_ADDR, "memberAddr"),
		new XcodeMLName(Xcode.MEMBER_ARRAY_ADDR, "memberArrayAddr"),

		new XcodeMLName(Xcode.PRAGMA_LINE, "pragma"),
		new XcodeMLName(Xcode.TEXT, "text"),

		new XcodeMLName(Xcode.BUILTIN_OP, "builtin_op"),
		new XcodeMLName(Xcode.GCC_ATTRIBUTES, "gccAttributes"),
		new XcodeMLName(Xcode.GCC_ATTRIBUTE, "gccAttribute"),
		new XcodeMLName(Xcode.GCC_ASM, "gccAsm"),
		new XcodeMLName(Xcode.GCC_ASM_DEFINITION, "gccAsmDefinition"),
		new XcodeMLName(Xcode.GCC_ASM_STATEMENT, "gccAsmStatement"),
		new XcodeMLName(Xcode.GCC_ASM_OPERANDS, "gccAsmOperands"),
		new XcodeMLName(Xcode.GCC_ASM_OPERAND, "gccAsmOperand"),
		new XcodeMLName(Xcode.GCC_ASM_CLOBBERS, "gccAsmClobbers"),
		new XcodeMLName(Xcode.GCC_ALIGN_OF_EXPR, "gccAlignOfExpr"),
		new XcodeMLName(Xcode.GCC_MEMBER_DESIGNATOR, "gccMemberDesignator"),
		new XcodeMLName(Xcode.GCC_LABEL_ADDR, "gccLabelAddr"),
		new XcodeMLName(Xcode.GCC_COMPOUND_EXPR, "gccCompoundExpr"),
		new XcodeMLName(Xcode.GCC_RANGED_CASE_LABEL, "gccRangedCaseLabel"),

		new XcodeMLName(Xcode.XMP_DESC_OF, "xmpDescOf"),
		new XcodeMLName(Xcode.SUB_ARRAY_REF, "subArrayRef"),
 		new XcodeMLName(Xcode.INDEX_RANGE, "indexRange"),
 		//new XcodeMLName(Xcode.LOWER_BOUND, "lowerBound"),
		new XcodeMLName(Xcode.LOWER_BOUND, "base"),
 		//new XcodeMLName(Xcode.UPPER_BOUND, "upperBound"),
		new XcodeMLName(Xcode.UPPER_BOUND, "length"),
 		new XcodeMLName(Xcode.STEP, "step"),
		new XcodeMLName(Xcode.CO_ARRAY_REF, "coArrayRef"),
		new XcodeMLName(Xcode.CO_ARRAY_ASSIGN_EXPR, "coArrayAssignExpr"),

		new XcodeMLName(Xcode.OMP_PRAGMA, "OMPPragma"),
		new XcodeMLName(Xcode.XMP_PRAGMA, "XMPPragma"),
		new XcodeMLName(Xcode.ACC_PRAGMA, "ACCPragma"),
	};

	// constructor
	public XcodeMLNameTable_C() {
		initHTable(table);
		initOpcodeToNameTable(table);
	}
}
