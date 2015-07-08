/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
package exc.block;

import xcodeml.util.XmLog;
import xcodeml.util.XmOption;
import exc.object.*;

/**
 * a class for static class Block constructor.
 */
public class Bcons
{
    private static Xcode statement_list_code()
    {
        return XmOption.isLanguageC() ? Xcode.LIST : Xcode.F_STATEMENT_LIST;
    }
    
    /** create SimpleBlock for Statement x. */
    public static Block Statement(Xobject x)
    {
        return new SimpleBlock(statement_list_code(), BasicBlock.Statement(x));
    }

    /** create SimpleBlock for condition */
    public static Block Cond(Xobject x)
    {
        return new SimpleBlock(Xcode.LIST, BasicBlock.Cond(x));
    }

    /** creates SimpleBlock with BasicBlock bb. */
    public static Block BasicBlock(BasicBlock bb)
    {
        return new SimpleBlock(statement_list_code(), bb);
    }

    /** create empty block */
    public static Block emptyBlock()
    {
        return new SimpleBlock(statement_list_code(), new BasicBlock());
    }

    /** create empty block list */
    public static BlockList emptyBody()
    {
        return new BlockList();
    }

    /** create block list by specified blocks */
    public static BlockList blockList(Block ... blocks)
    {
        BlockList bl = new BlockList();
        for(Block b : blocks)
            bl.add(b);
        return bl;
    }

    /** create empty block list with specified ids and decls */
    public static BlockList emptyBody(Xobject id_list, Xobject decls)
    {
        return new BlockList(id_list, decls);
    }

    /** create compound statement (or statement list) block */
    public static Block COMPOUND(BlockList b_list)
    {
        if(XmOption.isLanguageC())
            return new CompoundBlock(b_list);
        else
            return new CompoundBlock(Xcode.F_STATEMENT_LIST, b_list);
    }

    /** create 'pragma' statement block */
    public static Block PRAGMA(Xcode code, String pragma, Xobject args, BlockList body)
    {
        return new PragmaBlock(code, pragma, args, body);
    }

    /** create 'if' statement block */
    public static Block IF(Xcode code, BasicBlock cond, BlockList then_part, BlockList else_part, String construct_name)
    {
        /* to fix, the problem of two-body IF */
        if(then_part != null && then_part.getIdentList() != null)
            then_part = new BlockList(Bcons.COMPOUND(then_part));
        if(else_part != null && else_part.getIdentList() != null)
            else_part = new BlockList(Bcons.COMPOUND(else_part));
        return new IfBlock(code, cond, then_part, else_part, construct_name);
    }

    /** create 'if' statement block */
    public static Block IF(BasicBlock cond, BlockList then_part, BlockList else_part)
    {
        return IF(Xcode.IF_STATEMENT, cond, then_part, else_part, null);
    }
    
    /** create if statement block */
    public static Block IF(Xobject cond, Block then_part, Block else_part)
    {
        return IF(BasicBlock.Cond(cond), new BlockList(then_part), new BlockList(else_part));
    }

    /** create 'if' statement block */
    public static Block IF(Xobject cond, Xobject then_part, Xobject else_part)
    {
        return IF(cond, buildBlock(then_part), else_part != null ? buildBlock(else_part) : null);
    }
    
    /** create Fortran 'where' statement block */
    public static Block Fwhere(BasicBlock cond, BlockList then_part, BlockList else_part)
    {
        return IF(Xcode.F_WHERE_STATEMENT, cond, then_part, else_part, null);
    }
    
    /** create 'for' statement block */
    public static Block FOR(BasicBlock init, BasicBlock cond, BasicBlock iter, BlockList body, String construct_name)
    {
        return new CforBlock(init, cond, iter, body, null);
    }

    /** create 'for' statement block */
    public static Block FOR(BasicBlock init, BasicBlock cond, BasicBlock iter, BlockList body)
    {
        return FOR(init, cond, iter, body, null);
    }
    
    /** create 'for' statement block */
    public static Block FOR(Xobject init, Xobject cond, Xobject iter, Block body, String construct_name)
    {
        return new CforBlock(BasicBlock.Statement(init), BasicBlock.Cond(cond),
			     BasicBlock.Statement(iter), new BlockList(body), construct_name);
    }
    
    /** create 'for' statement block */
    public static Block FOR(Xobject init, Xobject cond, Xobject iter, Block body)
    {
        return FOR(init, cond, iter, body, null);
    }

    /** create 'for' statement block */
    public static Block FORall(Xobject ind_var, Xobject lb, Xobject ub, Xobject step,
			       Xcode checkOp, BlockList body)
    {
        return Bcons.FOR(Xcons.Set(ind_var, lb),
			 Xcons.binaryOp(checkOp, ind_var, ub),
			 Xcons.asgOp(Xcode.ASG_PLUS_EXPR, ind_var, step),
			 Bcons.COMPOUND(body));
    }
    
    /** create Fortran 'do' statement block */
    public static Block Fdo(Xobject var, Xobject idx_range, BlockList body, String construct_name)
    {
        return new FdoBlock(var, idx_range, body, construct_name);
    }

    /** create 'while' statement block */
    public static Block WHILE(Xcode code, BasicBlock cond, BlockList body, String constructName)
    {
        return new CondBlock(code, cond, body, constructName);
    }

    /** create 'do while' statement block */
    public static Block WHILE(BasicBlock cond, BlockList body)
    {
        return WHILE(Xcode.WHILE_STATEMENT, cond, body, null);
    }

    /** create 'do while' statement block */
    public static Block WHILE(Xobject cond, Block body)
    {
        return WHILE(BasicBlock.Cond(cond), new BlockList(body));
    }
     
    /** create 'while' statement block */
    public static Block DO_WHILE(BasicBlock cond, BlockList body)
    {
    	return WHILE(Xcode.F_DO_WHILE_STATEMENT, cond, body, null);
    }

    /** create 'while' statement block */
    public static Block DO_WHILE(Xobject cond, Block body)
    {
    	return DO_WHILE(BasicBlock.Cond(cond), new BlockList(body));
    }

    /** create C 'do' statement block */
    public static Block DO(BlockList body, BasicBlock cond, String construct_name)
    {
        return new CondBlock(Xcode.DO_STATEMENT, cond, body, construct_name);
    }

    /** create C 'do' statement block */
    public static Block DO(BlockList body, BasicBlock cond)
    {
        return DO(body, cond, null);
    }
    
    /** create C 'do' statement block */
    public static Block DO(Block body, Xobject cond)
    {
        return DO(new BlockList(body), BasicBlock.Cond(cond));
    }

    /** create 'switch' statement block */
    public static Block SWITCH(BasicBlock cond, BlockList body)
    {
        return new CondBlock(Xcode.SWITCH_STATEMENT, cond, body, null);
    }
    
    /** create 'switch' statement block */
    public static Block SWITCH(Xobject cond, Block body)
    {
        return SWITCH(BasicBlock.Cond(cond), new BlockList(body));
    }

    /** create Fortran 'select case' statement block */
    public static Block FselectCase(Xobject cond, BlockList body, String construct_name)
    {
        if(cond.Opcode() != Xcode.F_VALUE)
            cond = Xcons.List(Xcode.F_VALUE, cond);
        return new CondBlock(Xcode.F_SELECT_CASE_STATEMENT,
			     BasicBlock.Cond(cond), body, construct_name);
    }

    /** create 'break' statement block */
    public static Block BREAK()
    {
        return new SimpleBlock(Xcode.BREAK_STATEMENT);
    }

    /** create 'continue' statement block */
    public static Block CONTINUE()
    {
        return new SimpleBlock(Xcode.CONTINUE_STATEMENT);
    }

    /** create Fortran 'cycle' statement block */
    public static Block Fcycle()
    {
        return new SimpleBlock(Xcode.F_CYCLE_STATEMENT);
    }

    /** create 'goto' statement block */
    public static Block GOTO(Xobject label)
    {
        return new LabelBlock(Xcode.GOTO_STATEMENT, label);
    }

    /** create 'goto' statement block */
    public static Block GOTO(Xobject label, Xobject value, Xobject params)
    {
        return new LabelBlock(Xcode.GOTO_STATEMENT, label, value, params);
    }

    /** create statement label block */
    public static Block LABEL(Xobject label)
    {
        return new LabelBlock(Xcode.STATEMENT_LABEL, label);
    }

    /** create C case label block */
    public static Block CASE(Xobject value)
    {
        return new LabelBlock(Xcode.CASE_LABEL, value);
    }
    
    /** create Fortran case label block */
    public static Block FcaseLabel(XobjList values, BlockList body, String construct_name)
    {
        return new FcaseLabelBlock(values, body, construct_name);
    }

    /** create C default case label block */
    public static Block DEFAULT_LABEL()
    {
        return new SimpleBlock(Xcode.DEFAULT_LABEL);
    }

    /** create 'return' statement block */
    public static Block RETURN()
    {
        return new SimpleBlock(Xcode.RETURN_STATEMENT);
    }

    /** create 'return' statement block */
    public static Block RETURN(BasicBlock ret)
    {
        return new SimpleBlock(Xcode.RETURN_STATEMENT, ret);
    }

    /** create 'return' statement block */
    public static Block RETURN(Xobject ret)
    {
        return new SimpleBlock(Xcode.RETURN_STATEMENT, BasicBlock.Cond(ret));
    }

    //
    // build routine from XobjectDef and Xobject
    //
    /** create FuctionBlock for XobjectDef */
    public static FunctionBlock buildFunctionBlock(XobjectDef d)
    {
	return new FunctionBlock(d.getDef().Opcode(),d.getNameObj(),
				 d.getFuncIdList(), d.getFuncDecls(),
				 buildBlock(d.getFuncBody()), 
				 d.getFuncGccAttributes(), d.getParentEnv());
    }

    /** create block list for Xobject */
    static BlockList buildList(Xobject v)
    {
        Block b;
        BlockList b_list = new BlockList();
        if(v == null)
            return b_list;
        
        switch(v.Opcode()) {
        case LIST: /* (LIST statement ....) */
        case F_STATEMENT_LIST: /* (F_STATEMENT_LIST statement ....) */
            break;
        case COMPOUND_STATEMENT:
            /* (COMPOUND_STATEMENT id-list decl statement-list) */
            b_list.id_list = v.getArg(0);
            b_list.decls = v.getArg(1);
            v = v.getArg(2);
            break;
        default:
            /* BlockList, which has one statement */
            b = buildBlock(v);
            b.setLineNo(v.getLineNo());
            b_list.add(b);
            return b_list;
        }
        addBlocks(v, b_list);
        return b_list;
    }

    private static void addBlocks(Xobject v, BlockList b_list)
    {
        Block b;
        if(v == null)
            return;
        switch(v.Opcode()) {
        case LIST:
        case F_STATEMENT_LIST:
            for(Xobject a : (XobjList)v) {
                addBlocks(a, b_list); // call recursively
            }
            break;
        case EXPR_STATEMENT: /* just statement */
            b = b_list.getTail();
            if(b == null || b.Opcode() != Xcode.LIST) {
                b = new SimpleBlock(statement_list_code());
                b.setLineNo(v.getLineNo());
                b_list.add(b);
            }
            //Statement s = new Statement(v.getArg(0));
	    Statement s = null;
	    if (v.getArg(0).Opcode() == Xcode.FUNCTION_CALL)
		s = new Statement(v);
	    else
		s = new Statement(v.getArg(0));
            s.setLineNo(v.getLineNo());
            b.getBasicBlock().add(s);
            break;
        default:
            b = buildBlock(v);
            if(b.getLineNo() == null)
                b.setLineNo(v.getLineNo());
            b_list.add(b);
        }
    }

    /** create block for Xobject */
    public static Block buildBlock(Xobject v)
    {
	if(v == null) return null;

	Xcode code = v.Opcode();
        switch(code) {
        default:
            if(code.isFstatement()) {
                return Statement(v);
            }
            XmLog.fatal("build: unknown code: " + v.OpcodeName());
            return null;

        case GCC_ASM_STATEMENT:
            return Statement(v);

        case NULL:
            return new NullBlock();

        case LIST: /* (LIST statement ....) */
            XmLog.fatal("LIST is appear in non-compound statement");
            return null;

        case F_STATEMENT_LIST:
        case COMPOUND_STATEMENT:
            return COMPOUND(buildList(v));
            
        case OMP_PRAGMA:
            return PRAGMA(Xcode.OMP_PRAGMA, v.getArg(0).getString(), v.getArgOrNull(1),
			  buildList(v.getArgOrNull(2)));

        case XMP_PRAGMA:
            return PRAGMA(Xcode.XMP_PRAGMA, v.getArg(0).getString(), v.getArg(1),
			  buildList(v.getArgOrNull(2)));

        case ACC_PRAGMA:
	    return PRAGMA(Xcode.ACC_PRAGMA, v.getArg(0).getString(), v.getArgOrNull(1),
			  buildList(v.getArgOrNull(2)));

        case IF_STATEMENT: /* (IF_STATMENT cond then-part else-part) */
            return IF(BasicBlock.Cond(v.getArg(0)), buildList(v.getArg(1)),
		      buildList(v.getArg(2)));
            
        case FOR_STATEMENT: /* (FOR init cond iter body) */
            return FOR(BasicBlock.Statement(v.getArg(0)), BasicBlock.Cond(v.getArg(1)),
		       BasicBlock.Statement(v.getArg(2)), buildList(v.getArg(3)));

        case WHILE_STATEMENT: /* (WHILE_STATEMENT cond body) */
            return WHILE(BasicBlock.Cond(v.getArg(0)), buildList(v.getArg(1)));
            
        case DO_STATEMENT: /* (DO_STATEMENT body cond) */
            return DO(buildList(v.getArg(0)), BasicBlock.Cond(v.getArg(1)));

        case SWITCH_STATEMENT: /* (SWITCH_STATEMENT value body) */
            return SWITCH(BasicBlock.Cond(v.getArg(0)), buildList(v.getArg(1)));

        case BREAK_STATEMENT: /* (BREAK_STATEMENT) */
            return BREAK();

        case CONTINUE_STATEMENT: /* (CONTINUE_STATEMENT) */
            return CONTINUE();

        case GOTO_STATEMENT: /* (GOTO_STATEMENT label) */
            return GOTO(v.getArg(0), v.getArgOrNull(1), v.getArgOrNull(2));

        case STATEMENT_LABEL: /* (STATEMENT_LABEL label_ident) */
            return LABEL(v.getArg(0));

        case CASE_LABEL: /* (CASE_LABEL value) */
            return CASE(v.getArg(0));

        case DEFAULT_LABEL: /* (DEFAULT_LABEL) */
            return DEFAULT_LABEL();

        case RETURN_STATEMENT: /* (RETURN_STATEMENT value) */
            return RETURN(BasicBlock.Cond(v.getArgOrNull(0)));

        case EXPR_STATEMENT:
            /* signle statement Basic Block */
            //return Statement(v.getArg(0));
	    if (v.getArg(0).Opcode() == Xcode.FUNCTION_CALL)
		return Statement(v);
	    else
		return Statement(v.getArg(0));

        case PRAGMA_LINE:
	    return PRAGMA(Xcode.PRAGMA_LINE, v.getArg(0).getString(), null, null);
        case TEXT:
            return new SimpleBlock(statement_list_code(), BasicBlock.Statement(v));
            
        case F_IF_STATEMENT: /* (F_IF_STATMENT construct_name cond then-part else-part) */
            return IF(code, BasicBlock.Cond(v.getArg(1)),
		      buildList(v.getArg(2)), buildList(v.getArg(3)), getArg0Name(v));
            
        case F_WHERE_STATEMENT: /* (F_WHERE_STATMENT () cond then-part else-part) */
            return Fwhere(BasicBlock.Cond(v.getArg(1)),
			  buildList(v.getArg(2)), buildList(v.getArg(3)));
            
        case F_DO_STATEMENT: /* (F_DO_STATEMENT construct_name var index_range body) */
            return Fdo(v.getArg(1), v.getArg(2), buildList(v.getArg(3)), getArg0Name(v));
            
        case F_DO_WHILE_STATEMENT: /* (F_DO_WHILE_STATEMENT construct_name cond */
            return WHILE(code, BasicBlock.Cond(v.getArg(1)),
			 buildList(v.getArg(2)), getArg0Name(v));
            
        case F_SELECT_CASE_STATEMENT: /* (F_SELECT_CASE_STATEMENT construct_name case_value body ) */
            return FselectCase(v.getArg(1), buildList(v.getArg(2)), getArg0Name(v));
            
        case F_CASE_LABEL: /* (F_CASE_LABEL construct_name case_value body */
            return FcaseLabel((XobjList)v.getArg(1), buildList(v.getArgOrNull(2)), getArg0Name(v));
        }
    }
    
    private static String getArg0Name(Xobject v)
    {
        Xobject x = v.getArgOrNull(0);
        if(x == null)
            return null;
        return x.getName();
    }
}
