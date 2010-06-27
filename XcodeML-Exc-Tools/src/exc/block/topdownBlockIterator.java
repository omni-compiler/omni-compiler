/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
package exc.block;

import java.util.Vector;

public class topdownBlockIterator extends BlockIterator
{
    Vector<Block> blocks;
    int index;

    public topdownBlockIterator(Block b)
    {
        super(b);
        init(b);
    }

    public topdownBlockIterator(BlockList body)
    {
        super(body);
        init(body);
    }

    @Override
    public void init()
    {
        index = 0;
        if(blocks.size() > 0)
            currentBlock = blocks.elementAt(0);
    }

    @Override
    public void init(Block b)
    {
        blocks = new Vector<Block>();
        topdownTraverse(b);
        index = 0;
        if(blocks.size() > 0)
            currentBlock = blocks.elementAt(0);
    }

    public void init(BlockList body)
    {
        blocks = new Vector<Block>();
        topdownTraverse(body);
        index = 0;
        if(blocks.size() > 0)
            currentBlock = blocks.elementAt(0);
    }

    void topdownTraverse(Block b)
    {
        if(b == null)
            return;
        blocks.addElement(b);
        if(b instanceof IfBlock) {
            topdownTraverse(b.getThenBody());
            topdownTraverse(b.getElseBody());
        } else if(b instanceof FmoduleBlock){
            topdownTraverse(b.getBody());
            topdownTraverse(((FmoduleBlock)b).getFunctionBlocks());
        } else {
            topdownTraverse(b.getBody());
        }
    }

    void topdownTraverse(BlockList b_list)
    {
        if(b_list == null)
            return;
        for(Block b = b_list.getHead(); b != null; b = b.getNext()) {
            topdownTraverse(b);
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
