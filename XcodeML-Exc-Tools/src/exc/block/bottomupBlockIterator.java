/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
package exc.block;

import java.util.Vector;

public class bottomupBlockIterator extends BlockIterator
{
    Vector<Block> blocks;
    int index;

    public bottomupBlockIterator(Block b)
    {
        super(b);
        init(b);
    }

    public bottomupBlockIterator(BlockList body)
    {
        super(body);
        init(body);
    }

    @Override
    public void init()
    {
        index = 0;
        currentBlock = blocks.elementAt(0);
    }

    @Override
    public void init(Block b)
    {
        blocks = new Vector<Block>();
        bottomupTraverse(b);
        index = 0;
        currentBlock = blocks.elementAt(0);
    }

    public void init(BlockList body)
    {
        blocks = new Vector<Block>();
        bottomupTraverse(body);
        index = 0;
        currentBlock = blocks.elementAt(0);
    }

    void bottomupTraverse(Block b)
    {
        if(b == null)
            return;
        if(b instanceof IfBlock) {
            bottomupTraverse(b.getThenBody());
            bottomupTraverse(b.getElseBody());
        } else {
            bottomupTraverse(b.getBody());
        }
        blocks.addElement(b);
    }

    void bottomupTraverse(BlockList b_list)
    {
        if(b_list == null)
            return;
        for(Block b = b_list.getHead(); b != null; b = b.getNext()) {
            bottomupTraverse(b);
        }
    }

    @Override
    public void next()
    {
        if(++index >= blocks.size())
            currentBlock = null;
        else
            currentBlock = blocks.elementAt(index);
    }

    @Override
    public boolean end()
    {
        return index >= blocks.size();
    }

    @Override
    public int size()
    {
        return blocks.size();
    }
}
