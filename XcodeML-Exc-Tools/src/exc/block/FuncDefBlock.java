/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
package exc.block;

import java.io.*;

import exc.object.*;

/**
 * A object to represent a function definition. Note that it is not a kind of
 * "Block".
 */

public class FuncDefBlock
{
    protected XobjectDef def;
    protected FunctionBlock fblock;
    final static String funcBlockProp = "prop.FuncDefBlock";

    /** contructor to make FuncDefBlock from XobjectDef def */
    public FuncDefBlock(XobjectDef def)
    {
        this.def = def;
        fblock = (FunctionBlock)def.getProp(funcBlockProp); // check cache
        if(fblock == null) {
            def.setDef(canonicalizeExpr(def.getDef()));
            fblock = Bcons.buildFunctionBlock(def);
            def.setProp(funcBlockProp, fblock);
        }
    }

    /** make clone */
    public FuncDefBlock(FuncDefBlock fd)
    {
        this.def = fd.def;
        this.fblock = fd.fblock;
    }

    /** constructor with fucntion name, id_list, decls, body and env. */
    public FuncDefBlock(Xobject name, Xobject id_list, Xobject decls, BlockList body,
        Xobject gcc_attrs, XobjectFile env)
    {
        // make dummy XobjectDef
        this.def = XobjectDef.Func(name, id_list, decls, body.toXobject());
        this.def.setParent(env);
	this.fblock = new FunctionBlock(name, id_list, decls,
					Bcons.COMPOUND(body), gcc_attrs, env);
	def.setProp(funcBlockProp,this.fblock);
    }

    /** return its FucntionBlock */
    public FunctionBlock getBlock()
    {
        return fblock;
    }

    /** returns XobjectDef associated with this */
    public XobjectDef getDef()
    {
        return def;
    }

    /** returns XobjectFile env asscoiated with this */
    public XobjectFile getFile()
    {
        return def.getFile();
    }

    public void Finalize()
    {
        // System.out.println("-- Finalize:"); print();
        def.setDef(fblock.toXobject());
        def.remProp(funcBlockProp); // flush cache
    }

    /** remove initializer for local variables. */
    public void removeDeclInit()
    {
        BlockIterator i = new topdownBlockIterator(fblock);
        for(i.init(); !i.end(); i.next()) {
            BlockList body = i.getBlock().getBody();
            if(body != null)
                body.removeDeclInit();
        }
    }

    /** rewrite VarAddr -> (ADDR_OF Var) */
    Xobject canonicalizeExpr(Xobject e)
    {
        XobjectIterator i = new bottomupXobjectIterator(e);
        for(i.init(); !i.end(); i.next()) {
            Xobject x = i.getXobject();
            if(x == null)
                continue;
            // if type is null, don't care
            if(x.isVarAddr() && x.Type() != null) {
                i.setXobject(Xcons.AddrOf(Xcons.PointerRef(x)));
            } else if(x.Opcode() == Xcode.POINTER_REF && x.left().Opcode() == Xcode.ADDR_OF)
                i.setXobject(x.left().left());
        }
        return i.topXobject();
    }

    /** print out this object to System.out (for debug) */
    public void print()
    {
        print(System.out);
    }

    /** print out this object to out (for debug) */
    public void print(OutputStream out)
    {
        BlockPrintWriter debug_out = new BlockPrintWriter(out);
        debug_out.print(fblock);
        debug_out.flush();
    }

    /** print out this object to out (for debug) */
    public void print(Writer out)
    {
        BlockPrintWriter debug_out = new BlockPrintWriter(out);
        debug_out.print(fblock);
        debug_out.flush();
    }
}
