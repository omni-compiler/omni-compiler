package exc.block;

import java.util.Vector;

/**
 * super class for Block iterator
 */
public abstract class BlockIterator
{
    Vector<Block> blocks;
    int index;

    Block currentBlock = null;

    /** constructor with block */
    public BlockIterator(Block b)
    {
        currentBlock = b; // set defaults
        init(b);
    }

    public BlockIterator(BlockList body)
    {
        init(body);
    }

    public abstract void init();

    public abstract void init(Block b);

    public abstract void init(BlockList body);

    public abstract void next();

    public abstract boolean end();

    public abstract int size();

    /** Return the current block */
    public Block getBlock()
    {
        return currentBlock;
    }

    /* + replace the current block with Block b */
    public void setBlock(Block b)
    {
        currentBlock.replace(b);
    }

    public void setAside(Block b)
    {
        if (blocks.removeElement(b))
            index = blocks.lastIndexOf(currentBlock);
    }

    public void setAside(Vector<Block> vb)
    {
        if (blocks.removeAll(vb))
            index = blocks.lastIndexOf(currentBlock);
    }

    public Vector<Block> getContainer()
    {
        return blocks;
    }
}
