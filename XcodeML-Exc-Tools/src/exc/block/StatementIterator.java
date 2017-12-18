package exc.block;

import java.util.Iterator;

public interface StatementIterator extends Iterator<Statement>
{
    boolean hasMoreStatement();

    Statement nextStatement();
}
