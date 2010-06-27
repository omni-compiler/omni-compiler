/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
package xcodeml.c.decompile;

import xcodeml.c.obj.XcNode;

/**
 * Internal object represents following elements:
 *   declarations,
 *   exprStatement, compoundStatement, ifStatement, whileStatment,
 *   doStatement, forStatement, breakStatement, continueStatement,
 *   returnStatment, gotoStatement, statementLabel, switchStatement,
 *   caseLabel, defaultLabel
 */
public interface XcStAndDeclObj extends XcAppendable, XcNode, XcSourcePositioned
{
}
