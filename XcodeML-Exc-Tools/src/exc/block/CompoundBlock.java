/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
package exc.block;

import exc.object.*;

/**
 * Compound block with code COMPOUND_STATEMENT.
 */
public class CompoundBlock extends Block
{
    BlockList body;

    /** constructor with BlockList */
    public CompoundBlock(BlockList body)
    {
        this(Xcode.COMPOUND_STATEMENT, body, null);
    }

    /** constructor with code and BlockList */
    public CompoundBlock(Xcode code, BlockList body)
    {
        this(code, body, null);
    }
    
    /** constructor with code and BlockList */
    public CompoundBlock(Xcode code, BlockList body, String construct_name)
    {
        super(code, null, construct_name);
        this.body = body;
        if(body != null) {
            body.parent = this;
            setLineNo(body.getHeadLineNo());
        }
    }

    /** make clone */
    public CompoundBlock(CompoundBlock b)
    {
        super(b);
        if(b.body != null)
            this.body = b.body.copy();
    }

    /** make clone */
    @Override
    public Block copy()
    {
        return new CompoundBlock(this);
    }

    /** return body */
    @Override
    public BlockList getBody()
    {
        return body;
    }

    /** set body */
    @Override
    public void setBody(BlockList s)
    {
        body = s;
        s.parent = this;
    }

    /** convert to Xobject representation */
    @Override
    public Xobject toXobject()
    {
        if(body == null)
            return null;
        Xobject x = body.toXobject();
        if(x != null)
            x.setLineNo(getLineNo());
        return x;
    }

    @Override
    public String toString()
    {
        StringBuilder s = new StringBuilder(256);
        s.append("(CompoundBlock ");
        s.append(Opcode());
        s.append(" ");
        s.append(body);
        s.append(" ");
        s.append(getBasicBlock());
        s.append(")");
        return s.toString();
    }
}
