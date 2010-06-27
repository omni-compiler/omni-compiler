/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
package exc.block;

import java.util.Iterator;

public interface StatementIterator extends Iterator<Statement>
{
    boolean hasMoreStatement();

    Statement nextStatement();
}
