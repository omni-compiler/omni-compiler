/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
package exc.block;

import exc.object.*;

public class BasicBlockExprIterator
{
    BasicBlockIterator i;
    protected StatementIterator is;
    protected Statement current_statement;
    protected BasicBlock current_bblock;

    // constructor
    public BasicBlockExprIterator(Block b)
    {
        init(b);
    }

    public BasicBlockExprIterator(BlockList body)
    {
        init(body);
    }

    public BasicBlockExprIterator()
    {
    }

    public void init(Block b)
    {
        i = new BasicBlockIterator(b);
        initialize();
    }

    public void init(BlockList body)
    {
        i = new BasicBlockIterator(body);
        initialize();
    }

    public void init()
    {
        i.init();
        initialize();
    }

    // initialize internal iterators
    void initialize()
    {
        current_bblock = i.getBasicBlock();
        if(current_bblock == null)
            return;
        is = current_bblock.statements();
        next();
    }

    public void next()
    {
        current_statement = null; // unset current_statment
        do {
            if(is != null) { // if statement iterator is active,
                if(is.hasMoreStatement()) {
                    current_statement = is.nextStatement();
                    return;
                } else {
                    is = null; // inactive
                    if(current_bblock.getExpr() != null)
                        return;
                }
            }
            // move to the next basic block.
            nextBasicBlock();
            if(current_bblock == null)
                return;
            is = current_bblock.statements();
        } while(true);
    }

    public void nextBasicBlock()
    {
        i.next();
        if(i.end())
            current_bblock = null;
        else
            current_bblock = i.getBasicBlock();
    }

    public boolean end()
    {
        return current_bblock == null;
    }

    public Statement getStatement()
    {
        return current_statement;
    }

    public BasicBlock getBasicBlock()
    {
        return current_bblock;
    }

    public Xobject getExpr()
    {
        if(current_statement != null)
            return current_statement.getExpr();
        else
            return current_bblock.getExpr();
    }

    public void setExpr(Xobject x)
    {
        if(current_statement != null)
            current_statement.setExpr(x);
        else
            current_bblock.setExpr(x);
    }

    public void insertStatement(Xobject s)
    {
        if(current_statement != null)
            current_statement.insert(s);
        else
            current_bblock.add(s);
    }

    public void addStatement(Xobject s)
    {
        if(current_statement != null)
            current_statement.add(s);
        else
            current_bblock.add(s);
    }

    public LineNo getLineNo()
    {
        if(current_statement != null)
            return current_statement.getLineNo();
        else
            return current_bblock.getParent().getLineNo();
    }
}
