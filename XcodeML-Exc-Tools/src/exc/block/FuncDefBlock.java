/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
package exc.block;

import java.io.*;
import java.util.HashMap;
import java.util.Map;
import java.util.Vector;

import exc.object.*;
import exc.openmp.OMPanalyzeDecl;
import exc.openmp.OMPpragma;

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

    public void searchCommonMember(String name,OMPanalyzeDecl env,XobjectDef d)
    {
	Map <String,XobjList> common_db= new HashMap<String,XobjList>();
	Vector <Xobject> xobjVector= new Vector<Xobject>();
	XobjList nameList = new XobjList();

	topdownBlockIterator bi = new topdownBlockIterator(this.fblock);
	for(bi.init();!bi.end();bi.next())
	    {
		XobjectIterator xi = new topdownXobjectIterator(bi.getBlock().toXobject());
		for(xi.init();!xi.end();xi.next())
		    {
			Xobject xobj = xi.getXobject();
			Xobject x=null;
			if(xobj!=null)
			    switch (xobj.Opcode())
				{
				case F_COMMON_DECL:
				    {
					String keyIdent =xobj.getArgOrNull(0).getArgOrNull(0).getName().toString();
					for(XobjArgs args = xobj.getArgOrNull(0).getArgOrNull(1).getArgs(); args!=null ;args= args.nextArgs())
					    {
						x = args.getArg().getArgOrNull(0);
						nameList.add(args.getArg().getArgOrNull(0));
						xobjVector.add(x);
					    }
					common_db.put(keyIdent, nameList);
					nameList = new XobjList();
				    }
				default:
				}
		    }

	    }
	topdownBlockIterator bi2 = new topdownBlockIterator(this.fblock);
	for(bi2.init();!bi2.end();bi2.next())
	    {
		if(bi2.getBlock().toXobject().Opcode()==Xcode.OMP_PRAGMA ){
		    {
			XobjectIterator xi2 = new topdownXobjectIterator(bi2.getBlock().toXobject());
			for(xi2.init();!xi2.end();xi2.next())
			    {
				if(xi2.getXobject()!=null  && xi2.getXobject().Opcode()==Xcode.LIST && xi2.getXobject().getArg(0).Opcode()==Xcode.STRING && OMPpragma.valueOf(xi2.getXobject().getArg(0)) == OMPpragma.DATA_COPYIN)
				    {
					for(int i=0;i<xobjVector.size();i++)
					    {
						for(XobjArgs args2 = xi2.getXobject().getArg(1).getArgs();args2!=null ;args2=args2.nextArgs())
						    if(xobjVector.get(i).getName().equals(args2.getArg().getName()))
							env.addThdprvVars(d.findIdent(args2.getArg().getName()));
					    }
				    }
			    }
		    }
		}
	    }
    }
}
