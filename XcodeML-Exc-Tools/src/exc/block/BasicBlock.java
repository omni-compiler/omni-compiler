/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
package exc.block;

import java.util.*;

import xcodeml.util.XmOption;
import exc.object.*;

/**
 * Basic block. It contains statements without control flow.
 */
public class BasicBlock extends PropObject implements Iterable<Statement>
{
    /** counter for generated BasicBlock */
    protected static int BasicBlockCounter = 0;

    /** expression at tail for branch */
    Xobject exp;
    /** enclosed statement block */
    Block parent;
    /** head and tail of child statements */
    Statement head, tail;

    /** vector of BasicBlock for flow computation */
    Vector<BasicBlock> cflow_in;
    /** vector of BasicBlock for flow computation */
    Vector<BasicBlock> cflow_out;
    /** depth first order number */
    int depth_first_num;
    BasicBlock top_next;
    BasicBlock top_prev;
    boolean mark;

    public BasicBlock()
    {
        id = BasicBlockCounter++; // not used
        cflow_in = new Vector<BasicBlock>();
        cflow_out = new Vector<BasicBlock>();
    }

    public BasicBlock(BasicBlock b)
    {
        for(Statement s = b.head; s != null; s = s.getNext())
            this.add(s.getExpr().copy());
        if(b.exp != null)
            this.exp = b.exp.copy();
    }

    public BasicBlock copy()
    {
        return new BasicBlock(this);
    }

    public final Block getParent()
    {
        return parent;
    }

    public final void setParent(Block b)
    {
        parent = b;
    }

    public final Statement getHead()
    {
        return head;
    }

    public final Statement getTail()
    {
        return tail;
    }

    public Xobject getExpr()
    {
        return exp;
    }

    public void setExpr(Xobject x)
    {
        exp = x;
    }

    public StatementIterator statements()
    {
        return new forwardStatementIterator(getHead());
    }

    @Override
    public Iterator<Statement> iterator()
    {
        return statements();
    }

    public Vector<BasicBlock> getCflowIn()
    {
        return cflow_in;
    }

    public Vector<BasicBlock> getCflowOut()
    {
        return cflow_out;
    }

    public BasicBlock getCflowIn(int i)
    {
        return (cflow_in.elementAt(i));
    }

    public BasicBlock getCflowOut(int i)
    {
        return (cflow_out.elementAt(i));
    }

    public void addCflowTo(BasicBlock b)
    {
        cflow_out.addElement(b);
        b.cflow_in.addElement(this);
    }

    public void removeCflowTo(BasicBlock b)
    {
        cflow_out.removeElement(b);
        b.cflow_in.removeElement(this);
    }

    public BasicBlock topNext()
    {
        return top_next;
    }

    public BasicBlock topPrev()
    {
        return top_prev;
    }

    public void setTopNext(BasicBlock bb)
    {
        top_next = bb;
    }

    public void setTopPrev(BasicBlock bb)
    {
        top_prev = bb;
    }

    public void resetMark()
    {
        mark = false;
    }

    public void setMark()
    {
        mark = true;
    }

    public boolean isMarked()
    {
        return mark;
    }

    public int DFN()
    {
        return depth_first_num;
    }

    public void setDFN(int n)
    {
        depth_first_num = n;
    }

    // add at tail
    public void add(Statement s)
    {
        s.parent = this;
        if(tail == null) {
            head = tail = s;
            return;
        }
        tail.next = s;
        s.prev = tail;
        tail = s;
    }

    public void add(Xobject statement)
    {
        add(new Statement(statement));
    }

    public void addStatement(Statement s)
    {
        BasicBlock parent_bb;
        Block block;

        parent_bb = this;
        block = parent_bb.parent;
        if(block instanceof ForBlock) {
            if(block.prev == null) {
                BlockList bl = block.getParent();
                Block b = new SimpleBlock(Xcode.LIST, new BasicBlock());
                if(bl.head != block) {
                    System.err.println("ForBlock is not first Block in BlockList");
                    System.exit(1);
                }
                bl.head = b;
                b.next = block;
                block.prev = b;
            }
            parent_bb = block.prev.getBasicBlock();
        }
        s.parent = parent_bb;
        if(parent_bb.tail == null) {
            parent_bb.head = parent_bb.tail = s;
            return;
        }
        parent_bb.tail.next = s;
        s.prev = parent_bb.tail;
        parent_bb.tail = s;
    }

    public void addStatement(Xobject statement)
    {
        addStatement(new Statement(statement));
    }

    public void insert(Xobject statement)
    {
        insert(new Statement(statement));
    }

    public void insert(Statement s)
    {
        s.parent = this;
        if(tail == null) {
            head = tail = s;
            return;
        }
        head.prev = s;
        s.next = head;
        head = s;
    }

    public void insertStatement(Xobject statement)
    {
        insertStatement(new Statement(statement));
    }

    public void insertStatement(Statement s)
    {
        BasicBlock parent_bb;
        Block block;

        parent_bb = this;
        block = parent_bb.parent;
        if(block instanceof ForBlock || block instanceof IfBlock) {
            if(block.getParent() instanceof BlockList) {
                BlockList bl = block.getParent();
                Block b = new SimpleBlock(Xcode.LIST, new BasicBlock());
                if(bl.head == block) {
                    bl.head = b;
                    b.next = block;
                    block.prev = b;
                } else if(bl.tail == block) {
                    b.prev = block.prev;
                    block.prev.next = b;
                    b.next = block;
                } else {
                    b.prev = block.prev;
                    block.prev.next = b;
                    b.next = block;
                    block.prev = b;
                }
                parent_bb = b.getBasicBlock();
            }
        }
        s.parent = parent_bb;
        if(parent_bb.tail == null) {
            parent_bb.head = parent_bb.tail = s;
            return;
        }
        parent_bb.head.prev = s;
        s.next = parent_bb.head;
        parent_bb.head = s;
    }

    public boolean isEmpty()
    {
        return exp == null && head == null;
    }

    public boolean isSingle()
    {
        return exp == null && head != null && head == tail;
    }

    //
    // static member function for constructor
    //
    public static BasicBlock Cond(Xobject x)
    {
        BasicBlock bb = new BasicBlock();
        bb.exp = x;
        return bb;
    }

    public static BasicBlock Statement(Xobject x)
    {
        BasicBlock bb = new BasicBlock();
        // expand a top level of COMMA_EXPR
        if(x == null)
            return bb;
        if(x.Opcode() == Xcode.COMMA_EXPR) {
            for(XobjArgs a = x.getArgs(); a != null; a = a.nextArgs())
                bb.add(a.getArg());
        } else
            bb.add(x);
        return bb;
    }
    
    public static BasicBlock Expr(Block parent, Xobject x)
    {
        BasicBlock bb = new BasicBlock();
        bb.parent = parent;
        bb.exp = x;
        return bb;
    }

    public Xobject toXobject()
    {
        if(head == null)
            return exp;
        if(head == tail && exp == null) {
            Xobject expr = head.getExpr();
            if(XmOption.isLanguageF()) {
                if(!expr.Opcode().isFstatement() &&
		   expr.Opcode() != Xcode.EXPR_STATEMENT){
		  expr = Xcons.List(Xcode.EXPR_STATEMENT, expr);
		}
            }
            return expr;
        }
        
        Xobject l = Xcons.List(XmOption.isLanguageC() ? 
			       Xcode.COMMA_EXPR : Xcode.F_STATEMENT_LIST);
        
        for(Statement s = head; s != null; s = s.getNext())
            if(s.getExpr() != null)
                l.add(s.getExpr());
        if(exp != null)
            l.add(exp);
        
        if(XmOption.isLanguageC())
            l.setType(l.getTail().Type());
        
        return l;
    }

    @Override
    public String toString()
    {
        StringBuilder b = new StringBuilder(256);
        b.append("(BB:" + id);
        for(Statement s = head; s != null; s = s.getNext())
            b.append(s.toString());
        b.append(")");
        return b.toString();
    }
}
