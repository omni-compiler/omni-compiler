/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
package exc.block;

public class forwardStatementIterator implements StatementIterator
{
    Statement current_statement;

    public forwardStatementIterator(Statement s)
    {
        current_statement = s;
    }

    public boolean hasMoreStatement()
    {
        return current_statement != null;
    }

    public Statement nextStatement()
    {
        Statement s = current_statement;
        current_statement = current_statement.getNext();
        return s;
    }

    @Override
    public boolean hasNext()
    {
        return hasMoreStatement();
    }

    @Override
    public Statement next()
    {
        return nextStatement();
    }

    @Override
    public void remove()
    {
        throw new UnsupportedOperationException();
    }
}
