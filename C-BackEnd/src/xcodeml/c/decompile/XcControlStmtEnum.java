/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
package xcodeml.c.decompile;

/**
 * An enumerator of the control statement object.
 */
public enum XcControlStmtEnum
{
    /**
     *   ifStatement, forStatement, whileStatement, doStatement, switchStatement,
     *   returnStatement, continueStatement, breakStatement, gotoStatement,
     *   caseLabel, defaultLabel, statementLabel
     */
    
    IF,
    FOR,
    WHILE,
    DO,
    SWITCH,
    RETURN,
    CONTINUE,
    BREAK,
    GOTO,
    CASE_LABEL,
    GCC_RANGED_CASE_LABEL,
    DEFAULT_LABEL,
    STMT_LABEL,
    ;
}
