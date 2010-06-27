/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
package exc.block;

public class backwardStatementIterator implements StatementIterator
{
    private Statement current_statement;

    public backwardStatementIterator(Statement s)
    {
        current_statement = s;
    }

    @Override
    public boolean hasMoreStatement()
    {
        return current_statement != null;
    }

    @Override
    public Statement nextStatement()
    {
        Statement s = current_statement;
        current_statement = current_statement.getPrev();
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
