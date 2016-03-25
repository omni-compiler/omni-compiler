/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
package exc.block;

/**
 * super class for Block iterator
 */
public abstract class BlockIterator
{
    Block currentBlock = null;

    /** constructor with block */
    public BlockIterator(Block b)
    {
        currentBlock = b; // set defaults
    }

    public BlockIterator(BlockList body)
    {
        // nothing
    }

    public abstract void init();

    public abstract void init(Block b);

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
}
