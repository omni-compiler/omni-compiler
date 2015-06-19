/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
package exc.block;

import exc.object.*;

/**
 * abstract class of a statement block
 */
public class Block extends PropObject implements IVarContainer
{
    /** the nubmer of gnerated block. */
    protected static int BlockCounter = 0;

    Xcode code; // opcode for this block
    BasicBlock bblock; // for expressions, statements

    Block prev, next; // double linked list in BlockList
    private BlockList parent; // block list which this block belongs to.

    private LineNo lineno; // lineno information
    
    private String construct_name; // construct name for Fortran

    /** contructor for simple block whick contains a basic block */
    public Block(Xcode code, BasicBlock bblock, String constructName)
    {
        this.code = code;
        this.bblock = bblock;
        if(bblock != null)
            bblock.parent = this;
        this.construct_name = constructName;
        id = BlockCounter++;
    }

    public Block(Xcode code, BasicBlock bblock)
    {
        this(code, bblock, null);
    }
    
    /** constructor to make a copy of block b */
    public Block(Block b)
    {
        this.code = b.code;
        this.lineno = b.lineno;
        if(b.bblock != null) {
            this.bblock = b.bblock.copy();
            this.bblock.parent = this;
        }
        this.construct_name = b.construct_name;
        id = BlockCounter++;
    }

    /** copy this block */
    public Block copy()
    {
        return new Block(this);
    }

    /** get Opcode for this block */
    public Xcode Opcode()
    {
        return code;
    }

    /** get the BasicBlock */
    public BasicBlock getBasicBlock()
    {
        return bblock;
    }

    /** get the next block of this block */
    public Block getNext()
    {
        return next;
    }

    /** get the previous block */
    public Block getPrev()
    {
        return prev;
    }

    /** get the parent BlockList which this block belongs to. */
    public BlockList getParent()
    {
        return parent;
    }
    
    public void setParent(BlockList parent)
    {
        this.parent = parent;
    }

    /** get the parent Block with which this block belongs to. */
    public Block getParentBlock()
    {
        return parent == null ? null : parent.getParent();
    }

    /** set the line number information */
    public void setLineNo(LineNo ln)
    {
        lineno = ln;
    }

    /** get the line number information */
    public LineNo getLineNo()
    {
        return lineno;
    }
    
    /** get the construct name for Fortran */
    public XobjString getConstructNameObj()
    {
        if(construct_name == null)
            return null;
        return Xcons.Symbol(Xcode.IDENT, construct_name);
    }

    /** convert to the printable string */
    @Override
    public String toString()
    {
        return "Block." + code.toString() + "#" + id;
    }

    /** apply BasicBlockVisitor v */
    public void visitBasicBlock(BasicBlockVisitor v)
    {
        v.visit(bblock);
    }

    /** apply BasicBlockVisitor for the body */
    public void visitBody(BasicBlockVisitor v)
    {
        v.visit(getBody());
    }

    //
    // abstract member function
    //
    /** convert to Xobject */
    public Xobject toXobject()
    {
        throw new UnsupportedOperationException(toString());
    }

    /** get the body */
    public BlockList getBody()
    {
        return null;
    }

    /** set the body */
    public void setBody(BlockList s)
    {
        throw new UnsupportedOperationException(toString());
    }

    /** get BasicBlock in cond part */
    public BasicBlock getCondBBlock()
    {
        throw new UnsupportedOperationException(toString());
    }

    /** get the BlockList of "then" part */
    public BlockList getThenBody()
    {
        throw new UnsupportedOperationException(toString());
    }

    /** get the BlockList of "else" part */
    public BlockList getElseBody()
    {
        throw new UnsupportedOperationException(toString());
    }

    /** set the BlockList of "then" part */
    public void setThenBody(BlockList s)
    {
        throw new UnsupportedOperationException(toString());
    }

    /** set the BlockList of "else" part */
    public void setElseBody(BlockList s)
    {
        throw new UnsupportedOperationException(toString());
    }

    /** get loop initialization expression block. */
    public BasicBlock getInitBBlock()
    {
        throw new UnsupportedOperationException(toString());
    }

    /** set loop initialization expression block. */
    public void setInitBBlock(BasicBlock s)
    {
        throw new UnsupportedOperationException(toString());
    }

    /** get loop iteration expression block. */
    public BasicBlock getIterBBlock()
    {
        throw new UnsupportedOperationException(toString());
    }

    /** set loop iteration expression block. */
    public void setIterBBlock(BasicBlock s)
    {
        throw new UnsupportedOperationException(toString());
    }

    /** get label name. */
    public Xobject getLabel()
    {
        throw new UnsupportedOperationException(toString());
    }

    /** set label name. */
    public void setLabel(Xobject x)
    {
        throw new UnsupportedOperationException(toString());
    }

    /** get optional information expression. */
    public Xobject getInfoExpr()
    {
        return null;
    }

    /** add Block s */
    public Block add(Block s)
    {
        s.parent = parent;
        s.next = next;
        next = s;
        s.prev = this;
        if(s.next == null) { // tail of list
            parent.tail = s;
        } else {
            s.next.prev = s;
        }
        return s;
    }

    public Block add(BasicBlock bb)
    {
        return add(Bcons.BasicBlock(bb));
    }

    public Block add(Xobject s)
    {
        if(code == Xcode.LIST) {
            bblock.add(s);
            return this;
        }
        return add(Bcons.Statement(s));
    }

    public Block insert(Block s)
    {
        s.parent = parent;
        s.prev = prev;
        prev = s;
        s.next = this;
        if(s.prev == null) { // head of list
            parent.head = s;
        } else {
            s.prev.next = s;
        }
        return s;
    }

    public Block insert(BasicBlock bb)
    {
        return insert(Bcons.BasicBlock(bb));
    }

    public Block insert(Xobject s)
    {
        if(code == Xcode.LIST) {
            bblock.insert(s);
            return this;
        }
        return insert(Bcons.Statement(s));
    }

    public void replace(Block b)
    {
        b.parent = parent;
        b.prev = prev;
        b.next = next;
        if(prev != null)
            prev.next = b;
        else
            parent.head = b;
        if(next != null)
            next.prev = b;
        else
            parent.tail = b;
    }

    // remove block from list
    public Block remove()
    {
        if(prev != null)
            prev.next = next;
        else
            parent.head = next;
        if(next != null)
            next.prev = prev;
        else
            parent.tail = prev;
        parent = null;
        next = null;
        prev = null;
        return this;
    }

    // find Id in this context
    @Override
    public Ident findVarIdent(String name)
    {
        BlockList b_list;
        Ident id;
        for(b_list = parent; b_list != null; b_list = b_list.getParentList()) {
            if((id = b_list.findLocalIdent(name)) != null)
                return id;
            if(b_list.getParent() instanceof FunctionBlock) {
                return ((FunctionBlock)b_list.getParent()).getEnv().findVarIdent(name);
            }
        }
        return null; // not found
    }
    @Override
    public Ident findCommonIdent(String name)
    {
	return null;
    }
    public Boolean removeVarIdent(String name){

      BlockList b_list;
      Ident id;
      for (b_list = parent; b_list != null; b_list = b_list.getParentList()){
	if (b_list.removeIdent(name)) return true;
      }
      return false;

    }

    public static final int numberOfBlock()
    {
        return Block.BlockCounter;
    }

    public static final int numberOfBasicBlock()
    {
        return BasicBlock.BasicBlockCounter;
    }
}
