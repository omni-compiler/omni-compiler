package exc.block;

import java.util.Vector;

public class topdownBlockIterator extends BlockIterator
{
    public topdownBlockIterator(Block b)
    {
        super(b);
    }

    public topdownBlockIterator(BlockList body)
    {
        super(body);
    }

    @Override
    public void init()
    {
        index = 0;
	if(blocks == null) return;
        if(blocks.size() > 0) currentBlock = blocks.elementAt(0);
    }

    @Override
    public void init(Block b)
    {
	if(b == null) return;
        blocks = new Vector<Block>();
        topdownTraverse(b);
        index = 0;
        if(blocks.size() > 0)
            currentBlock = blocks.elementAt(0);
    }

    public void init(BlockList body)
    {
	if(body == null) return;
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
	if(currentBlock == null) return;
        if(++index >= blocks.size())
            currentBlock = null;
        else
            currentBlock = blocks.elementAt(index);
    }

    @Override
    public boolean end()
    {
	if(currentBlock == null) return true;
        return index >= blocks.size();
    }

    @Override
    public int size()
    {
	if(blocks == null) return 0;
        return blocks.size();
    }
}
