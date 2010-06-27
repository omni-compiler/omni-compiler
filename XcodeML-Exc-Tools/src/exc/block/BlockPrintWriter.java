/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
package exc.block;

import exc.object.*;
import java.io.*;

/*
 * Printer for debug
 */
public class BlockPrintWriter extends XobjectPrintWriter implements BasicBlockVisitor
{
    int indent_level;

    public BlockPrintWriter(OutputStream out)
    {
        super(out);
    }

    public BlockPrintWriter(Writer out)
    {
        super(out);
    }

    public void print(Block b)
    {
        indent_level = 0;
        visit(b);
        println();
        flush();
    }

    public void print(BlockList body)
    {
        indent_level = 0;
        for(Block b = body.getHead(); b != null; b = b.getNext()) {
            visit(b);
            println();
        }
        flush();
    }

    void indent()
    {
        for(int i = 0; i < indent_level; i++)
            print("  ");
    }

    public void visit(BasicBlock bb)
    {
        if(bb == null) {
            // println("BB:*");
            return;
        }
        indent();
        println(bb.toString());
        for(Statement s = bb.getHead(); s != null; s = s.getNext()) {
            indent();
            println("-: " + s.getExpr());
        }
        if(bb.getExpr() != null) {
            indent();
            println("@: " + bb.getExpr());
        }
    }

    public void visit(BlockList b_list)
    {
        if(b_list == null)
            return;
        if(b_list.getIdentList() != null || b_list.getDecls() != null) {
            indent();
            println(" id_list=" + b_list.getIdentList());
            indent();
            println(" decls=" + b_list.getDecls());
        }
        for(Block b = b_list.getHead(); b != null; b = b.getNext())
            visit(b);
    }

    public void visit(Block b)
    {
        if(b == null) {
            // println("#null");
            return;
        }

        indent();
        println(b.toString());

        indent_level++;
        if(b instanceof PragmaBlock) {
            PragmaBlock pb = (PragmaBlock)b;
            indent();
            println(" pramga=" + pb.getPragma());
            indent();
            println(" args=" + pb.getClauses());
        } else if(b instanceof FunctionBlock) {
            indent();
            println(" fname=" + ((FunctionBlock)b).getName());
        }

        b.visitBasicBlock(this);
        b.visitBody(this);
        indent_level--;
    }
}
