package exc.xcodeml;

import java.util.Arrays;

import exc.object.*;

public class XcodeMLNameTable_F extends XcodeMLNameTable {
	XcodeMLName table[] = {
			new XcodeMLName(Xcode.IDENT, "name"),
			new XcodeMLName(Xcode.ID_LIST, "symbols"),
			new XcodeMLName(Xcode.LIST, "params"),
			new XcodeMLName(Xcode.LIST, "declarations"),
			new XcodeMLName(Xcode.LIST, "arguments"),
			new XcodeMLName(Xcode.LIST, "kind"),

			new XcodeMLName(Xcode.LIST, "list"),
			new XcodeMLName(Xcode.STRING, "string"),

			new XcodeMLName(Xcode.FUNCTION_DEFINITION, "FfunctionDefinition"),
			new XcodeMLName(Xcode.VAR_DECL, "varDecl"),
			new XcodeMLName(Xcode.F_STRUCT_DECL, "FstructDecl"),
			new XcodeMLName(Xcode.F_EXTERN_DECL, "externDecl"),
			new XcodeMLName(Xcode.F_MODULE_DEFINITION, "FmoduleDefinition"),
			new XcodeMLName(Xcode.F_USE_DECL, "FuseDecl"),
			new XcodeMLName(Xcode.F_USE_ONLY_DECL, "FuseOnlyDecl"),
			new XcodeMLName(Xcode.F_INTERFACE_DECL, "FinterfaceDecl"),
			new XcodeMLName(Xcode.F_MODULE_PROCEDURE_DECL,
					"FmoduleProcedureDecl"),
			new XcodeMLName(Xcode.F_BLOCK_DATA_DEFINITION,
					"FblockDataDefinition"),
			new XcodeMLName(Xcode.FUNCTION_DECL, "FfunctionDecl"),

			new XcodeMLName(Xcode.F_VALUE, "value"),
			new XcodeMLName(Xcode.F_STATEMENT_LIST, "body"),
			new XcodeMLName(Xcode.F_RENAME, "rename"),
			new XcodeMLName(Xcode.F_RENAMABLE, "renamable"),
			new XcodeMLName(Xcode.F_ARRAY_INDEX, "arrayIndex"),
			new XcodeMLName(Xcode.F_INDEX_RANGE, "indexRange"),
			new XcodeMLName(Xcode.F_CO_SHAPE, "coShape"),           // #060
			new XcodeMLName(Xcode.F_DO_LOOP, "FdoLoop"),
			new XcodeMLName(Xcode.F_NAMED_VALUE, "namedValue"),
			new XcodeMLName(Xcode.F_NAMED_VALUE_LIST, "namedValueList"),
			new XcodeMLName(Xcode.F_VALUE_LIST, "valueList"),
			new XcodeMLName(Xcode.F_VAR_LIST, "varList"),
			new XcodeMLName(Xcode.INT_CONSTANT, "FintConstant"),
			new XcodeMLName(Xcode.FLOAT_CONSTANT, "FrealConstant"),
			new XcodeMLName(Xcode.F_COMPLEX_CONSTATNT, "FcomplexConstant"),
			new XcodeMLName(Xcode.F_CHARACTER_CONSTATNT, "FcharacterConstant"),
			new XcodeMLName(Xcode.F_LOGICAL_CONSTATNT, "FlogicalConstant"),
			new XcodeMLName(Xcode.F_ARRAY_CONSTRUCTOR, "FarrayConstructor"),
			new XcodeMLName(Xcode.F_STRUCT_CONSTRUCTOR, "FstructConstructor"),
			new XcodeMLName(Xcode.VAR, "Var"),

			new XcodeMLName(Xcode.F_ARRAY_REF, "FarrayRef"),
			new XcodeMLName(Xcode.F_CHARACTER_REF, "FcharacterRef"),
			new XcodeMLName(Xcode.MEMBER_REF, "FmemberRef"),
			new XcodeMLName(Xcode.CO_ARRAY_REF, "FcoArrayRef"),      // #060
			new XcodeMLName(Xcode.FUNC_ADDR, "Ffunction"),
			new XcodeMLName(Xcode.F_VAR_REF, "varRef"),
			new XcodeMLName(Xcode.FUNCTION_CALL, "functionCall"),
			new XcodeMLName(Xcode.PLUS_EXPR, "plusExpr"),
			new XcodeMLName(Xcode.MINUS_EXPR, "minusExpr"),
			new XcodeMLName(Xcode.MUL_EXPR, "mulExpr"),
			new XcodeMLName(Xcode.DIV_EXPR, "divExpr"),
			new XcodeMLName(Xcode.F_POWER_EXPR, "FpowerExpr"),
			new XcodeMLName(Xcode.F_CONCAT_EXPR, "FconcatExpr"),
			new XcodeMLName(Xcode.LOG_EQ_EXPR, "logEQExpr"),
			new XcodeMLName(Xcode.LOG_NEQ_EXPR, "logNEQExpr"),
			new XcodeMLName(Xcode.LOG_GE_EXPR, "logGEExpr"),
			new XcodeMLName(Xcode.LOG_GT_EXPR, "logGTExpr"),
			new XcodeMLName(Xcode.LOG_LE_EXPR, "logLEExpr"),
			new XcodeMLName(Xcode.LOG_LT_EXPR, "logLTExpr"),
			new XcodeMLName(Xcode.LOG_AND_EXPR, "logAndExpr"),
			new XcodeMLName(Xcode.LOG_OR_EXPR, "logOrExpr"),
			new XcodeMLName(Xcode.F_LOG_EQV_EXPR, "logEQVExpr"),
			new XcodeMLName(Xcode.F_LOG_NEQV_EXPR, "logNEQVExpr"),
			new XcodeMLName(Xcode.UNARY_MINUS_EXPR, "unaryMinusExpr"),
			new XcodeMLName(Xcode.LOG_NOT_EXPR, "logNotExpr"),
			new XcodeMLName(Xcode.F_USER_BINARY_EXPR, "userBinaryExpr"),
			new XcodeMLName(Xcode.F_USER_UNARY_EXPR, "userUnaryExpr"),

			new XcodeMLName(Xcode.F_ASSIGN_STATEMENT, "FassignStatement"),
			new XcodeMLName(Xcode.F_POINTER_ASSIGN_STATEMENT,
					"FpointerAssignStatement"),
			new XcodeMLName(Xcode.EXPR_STATEMENT, "exprStatement"),
			new XcodeMLName(Xcode.F_IF_STATEMENT, "FifStatement"),
			new XcodeMLName(Xcode.F_DO_STATEMENT, "FdoStatement"),
			new XcodeMLName(Xcode.F_DO_WHILE_STATEMENT, "FdoWhileStatement"),
			new XcodeMLName(Xcode.F_CONTINUE_STATEMENT, "continueStatement"),
			new XcodeMLName(Xcode.F_CYCLE_STATEMENT, "FcycleStatement"),
			new XcodeMLName(Xcode.F_EXIT_STATEMENT, "FexitStatement"),
			new XcodeMLName(Xcode.RETURN_STATEMENT, "FreturnStatement"),
			new XcodeMLName(Xcode.GOTO_STATEMENT, "gotoStatement"),
			new XcodeMLName(Xcode.STATEMENT_LABEL, "statementLabel"),
			new XcodeMLName(Xcode.F_SELECT_CASE_STATEMENT,
					"FselectCaseStatement"),
			new XcodeMLName(Xcode.SELECT_TYPE_STATEMENT,
					"selectTypeStatement"),
      new XcodeMLName(Xcode.TYPE_GUARD, "typeGuard"),
			new XcodeMLName(Xcode.F_CASE_LABEL, "FcaseLabel"),
			new XcodeMLName(Xcode.F_WHERE_STATEMENT, "FwhereStatement"),
			new XcodeMLName(Xcode.F_STOP_STATEMENT, "FstopStatement"),
			new XcodeMLName(Xcode.F_PAUSE_STATEMENT, "FpauseStatement"),
			new XcodeMLName(Xcode.F_READ_STATEMENT, "FreadStatement"),
			new XcodeMLName(Xcode.F_WRITE_STATEMENT, "FwriteStatement"),
			new XcodeMLName(Xcode.F_PRINT_STATEMENT, "FprintStatement"),
			new XcodeMLName(Xcode.F_REWIND_STATEMENT, "FrewindStatement"),
			new XcodeMLName(Xcode.F_END_FILE_STATEMENT, "FendFileStatement"),
			new XcodeMLName(Xcode.F_BACKSPACE_STATEMENT, "FbackspaceStatement"),
			new XcodeMLName(Xcode.F_OPEN_STATEMENT, "FopenStatement"),
			new XcodeMLName(Xcode.F_CLOSE_STATEMENT, "FcloseStatement"),
			new XcodeMLName(Xcode.F_INQUIRE_STATEMENT, "FinquireStatement"),
			new XcodeMLName(Xcode.F_FORMAT_DECL, "FformatDecl"),
			new XcodeMLName(Xcode.F_DATA_DECL, "FdataDecl"),
			new XcodeMLName(Xcode.F_DATA_STATEMENT, "FdataStatement"),
			new XcodeMLName(Xcode.F_NAMELIST_DECL, "FnamelistDecl"),
			new XcodeMLName(Xcode.F_EQUIVALENCE_DECL, "FequivalenceDecl"),
			new XcodeMLName(Xcode.F_COMMON_DECL, "FcommonDecl"),
			new XcodeMLName(Xcode.F_ENTRY_DECL, "FentryDecl"),
			new XcodeMLName(Xcode.F_ALLOCATE_STATEMENT, "FallocateStatement"),
			new XcodeMLName(Xcode.F_DEALLOCATE_STATEMENT,
					"FdeallocateStatement"),
			new XcodeMLName(Xcode.F_NULLIFY_STATEMENT, "FnullifyStatement"),
			new XcodeMLName(Xcode.F_CONTAINS_STATEMENT, "FcontainsStatement"),
			new XcodeMLName(Xcode.F_ALLOC, "alloc"),

			new XcodeMLName(Xcode.PRAGMA_LINE, "FpragmaStatement"),

			new XcodeMLName(Xcode.OMP_PRAGMA, "OMPPragma"),
			new XcodeMLName(Xcode.XMP_PRAGMA, "XMPPragma"),
			new XcodeMLName(Xcode.ACC_PRAGMA, "ACCPragma"),

			new XcodeMLName(Xcode.F_SYNCALL_STATEMENT, "syncAllStatement"),
			new XcodeMLName(Xcode.F_SYNCIMAGE_STATEMENT, "syncImagesStatement"),
			new XcodeMLName(Xcode.F_SYNCMEMORY_STATEMENT, "syncMemoryStatement"),
			new XcodeMLName(Xcode.F_CRITICAL_STATEMENT, "criticalStatement"),
			new XcodeMLName(Xcode.F_LOCK_STATEMENT, "lockStatement"),
			new XcodeMLName(Xcode.F_UNLOCK_STATEMENT, "unlockStatement"),
			new XcodeMLName(Xcode.F_SYNC_STAT, "syncStat"),
			new XcodeMLName(Xcode.F_BLOCK_STATEMENT, "blockStatement"),
			new XcodeMLName(Xcode.F_TYPE_PARAM, "typeParam"),
			new XcodeMLName(Xcode.F_TYPE_PARAMS, "typeParams"),
			new XcodeMLName(Xcode.F_TYPE_PARAM_VALUES, "typeParamValues"),
			new XcodeMLName(Xcode.F_LEN, "len"),
      new XcodeMLName(Xcode.F_IMPORT_STATEMENT, "FimportDecl"),
			new XcodeMLName(Xcode.F_TYPE_BOUND_PROCEDURES, "typeBoundProcedures"),
			new XcodeMLName(Xcode.F_TYPE_BOUND_PROCEDURE , "typeBoundProcedure" ),
			new XcodeMLName(Xcode.F_TYPE_BOUND_GENERIC_PROCEDURE , "typeBoundGenericProcedure"),
			new XcodeMLName(Xcode.F_BINDING, "binding"),
			new XcodeMLName(Xcode.F_MODULE_PROCEDURE_DEFINITION, "FmoduleProcedureDefinition"),
			new XcodeMLName(Xcode.F_FORALL_STATEMENT, "forallStatement"),
			new XcodeMLName(Xcode.F_CONDITION, "condition"),
	};

	// constructor
	public XcodeMLNameTable_F() {
		initHTable(table);

        // For Xobject -> XcodeML translation.
        XcodeMLName[] additionalTable = {
            new XcodeMLName(Xcode.WHILE_STATEMENT, "FdoWhileStatement"),
            new XcodeMLName(Xcode.IF_STATEMENT, "FifStatement"),
            new XcodeMLName(Xcode.FUNC_ADDR, "Ffunction"),
            new XcodeMLName(Xcode.LONGLONG_CONSTANT, "FintConstant"),
            new XcodeMLName(Xcode.STRING_CONSTANT, "stringConstant"),
        };

        XcodeMLName newTable[] = Arrays.copyOf(table,
                                               table.length + additionalTable.length);
        System.arraycopy(additionalTable, 0,
                         newTable, table.length, additionalTable.length);
        initOpcodeToNameTable(newTable);
	}

	// test
	public static void main(String args[]) {
		XcodeMLNameTable_F table = new XcodeMLNameTable_F();
		System.out.println("test res=" + table.getXcode("minusExpr"));
	}
}
