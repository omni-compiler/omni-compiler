/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
package exc.openmp;

import xcodeml.XmException;
import xcodeml.util.XmOption;
import exc.object.*;
import exc.object.PragmaLexer.Result;
import static exc.object.PragmaLexer.*;
import exc.xcodeml.XmSymbol;
import exc.xcodeml.XmSymbolUtil;


/**
 * OpenMP pragma lexer
 */
public class OMPpragmaLexer implements ExternalPragmaLexer
{
    private PragmaLexer _lexer;
    
    public OMPpragmaLexer(PragmaLexer lexer)
    {
        _lexer = lexer;
    }

    @Override
    public Result continueLex() throws XmException
    {
        Xobject[] retArgs = { null };
        PragmaSyntax[] retSyntax = { null };
        OMPpragma pragmaDir = lexOMPDirective(retSyntax, retArgs);
        
        if(pragmaDir == null) {
            return new Result(_lexer, null);
        }
        
        Xobject xobj = Xcons.List(Xcode.OMP_PRAGMA,
                Xcons.String(retSyntax[0].toString()),
                Xcons.String(pragmaDir.toString()),
                retArgs[0]);
        return new Result(_lexer, xobj);
    }
    
    private boolean pg_is_ident(String name)
    {
        return _lexer.pg_is_ident(name);
    }
    
    private void pg_get_token()
    {
        _lexer.pg_get_token();
    }
    
    private char pg_tok()
    {
        return _lexer.pg_tok();
    }
    
    private String pg_tok_buf()
    {
        return _lexer.pg_tok_buf();
    }
    
    private void error(String s)
    {
        _lexer.error(s);
    }
    
    private boolean is_dir_nowait(Xobject x)
    {
        if(x.Nargs() != 1)
            return false;
        Xobject xx = x.getArg(0);
        if(xx == null || !(xx instanceof XobjList) || xx.Nargs() != 2)
            return false;
        Xobject dir = xx.getArg(0);
        if(!OMPpragma.DIR_NOWAIT.toString().equals(dir.getString()))
            return false;
        if(xx.getArgOrNull(1) != null)
            return false;
        return true;
    }
    
    private OMPpragma lexOMPDirective(PragmaSyntax[] retSyntax, Xobject[] retArgs) throws XmException
    {
        PragmaSyntax syntax = XmOption.isLanguageC() ? PragmaSyntax.SYN_PREFIX : PragmaSyntax.SYN_START;
        Xobject args = null;
        OMPpragma pragmaDir = null;
        pg_get_token();
        boolean err = false;
        
        do {
            if(pg_tok() != PG_IDENT) {
                err = true;
                break;
            }
            // parallel block directive
            if(pg_is_ident("parallel")) {
                pg_get_token();
                if(pg_tok() == PG_IDENT) {
                    if((is_lang_c() && pg_is_ident("for")) ||
                        !is_lang_c() && pg_is_ident("do")) {
                        // parallel for
                        pragmaDir = OMPpragma.PARALLEL_FOR;
                        pg_get_token();
                        err = ((args = parse_OMP_clause()) == null);
                        break;
                    }
                    if(pg_is_ident("sections")) {
                        // parallel sections
                        pragmaDir = OMPpragma.PARALLEL_SECTIONS;
                        pg_get_token();
                        err = ((args = parse_OMP_clause()) == null);
                        break;
                    }
                }
                pragmaDir = OMPpragma.PARALLEL;
                err = ((args = parse_OMP_clause()) == null);
                break;
            }
            
            if(!is_lang_c() && pg_is_ident("paralleldo")) {
                // parallel for
                pragmaDir = OMPpragma.PARALLEL_FOR;
                pg_get_token();
                err = ((args = parse_OMP_clause()) == null);
                break;
            }
            
            if(!is_lang_c() && pg_is_ident("parallelsections")) {
                // parallel sections
                pragmaDir = OMPpragma.PARALLEL_SECTIONS;
                pg_get_token();
                err = ((args = parse_OMP_clause()) == null);
                break;
            }
            
            if((is_lang_c() && pg_is_ident("for")) ||
                (!is_lang_c() && pg_is_ident("do"))) {
                pragmaDir = OMPpragma.FOR;
                pg_get_token();
                err = ((args = parse_OMP_clause()) == null);
                break;
            }
            
            if(pg_is_ident("sections")) {
                pragmaDir = OMPpragma.SECTIONS;
                pg_get_token();
                if(!(err = ((args = parse_OMP_clause()) == null)) && !args.isEmpty()) {
                    pg_get_token();
                }
                break;
            }
            
            if(pg_is_ident("single")) {
                pragmaDir = OMPpragma.SINGLE;
                pg_get_token();
                err = ((args = parse_OMP_clause()) == null);
                break;
            }
            
            if(pg_is_ident("master")) {
                pragmaDir = OMPpragma.MASTER;
                pg_get_token();
                break;
            }
            
            if(pg_is_ident("critical")) {
                pragmaDir = OMPpragma.CRITICAL;
                pg_get_token();
                if(pg_tok() == '(') {
                    if((err = (args = parse_OMP_namelist_as_not_symbol()) == null))
                        break;
                } else {
                    args = null;
                }
                break;
            }
            
            if(pg_is_ident("ordered")) {
                pragmaDir = OMPpragma.ORDERED;
                pg_get_token();
                break;
            }
            
            if(pg_is_ident("section")) {
                pragmaDir = OMPpragma.SECTION;
                syntax = PragmaSyntax.SYN_SECTION;
                pg_get_token();
                break;
            }
            
            if(pg_is_ident("barrier")) {
                pragmaDir = OMPpragma.BARRIER;
                syntax = PragmaSyntax.SYN_EXEC;
                pg_get_token();
                break;
            }

            if(pg_is_ident("atomic")) {
                pragmaDir = OMPpragma.ATOMIC;
                if(XmOption.isLanguageF())
                    syntax = PragmaSyntax.SYN_PREFIX;
                pg_get_token();
                break;
            }
            
            if(pg_is_ident("flush")) {
                pragmaDir = OMPpragma.FLUSH;
                syntax = PragmaSyntax.SYN_EXEC;
                pg_get_token();
                if(pg_tok() == '(') {
                    if((err = (args = parse_OMP_namelist_as_symbol()) == null))
                        break;
                } else {
                    args = null;
                }
                break;
            }

            if(pg_is_ident("threadprivate")) {
                pragmaDir = OMPpragma.THREADPRIVATE;
                syntax = PragmaSyntax.SYN_DECL;
                pg_get_token();
                if((err = (args = parse_OMP_namelist_as_symbol()) == null))
                    break;
                break;
            }
            
            if(!is_lang_c() && pg_is_ident("end")) {
                pg_get_token();
                if(pg_is_ident("parallel")) {
                    pragmaDir = OMPpragma.PARALLEL;
                    syntax = PragmaSyntax.SYN_POSTFIX;
                    pg_get_token();
                    if(pg_tok() == 0)
                        break;
                    if(pg_is_ident("do")) {
                        pragmaDir = OMPpragma.PARALLEL_FOR;
                        pg_get_token();
                        break;
                    } else if(pg_is_ident("sections")) {
                        pragmaDir = OMPpragma.PARALLEL_SECTIONS;
                        pg_get_token();
                        break;
                    }
                } else if(pg_is_ident("paralleldo")) {
                    pragmaDir = OMPpragma.PARALLEL_FOR;
                    syntax = PragmaSyntax.SYN_POSTFIX;
                    pg_get_token();
                    break;
                } else if(pg_is_ident("parallelsections")) {
                    pragmaDir = OMPpragma.PARALLEL_SECTIONS;
                    syntax = PragmaSyntax.SYN_POSTFIX;
                    pg_get_token();
                    break;
                } else if(pg_is_ident("do")) {
                    pragmaDir = OMPpragma.FOR;
                    syntax = PragmaSyntax.SYN_POSTFIX;
                    pg_get_token();
                    if((err = (args = parse_OMP_clause()) == null) || args.isEmpty())
                        break;
                    if(is_dir_nowait(args))
                        break;
                } else if(pg_is_ident("sections")) {
                    pragmaDir = OMPpragma.SECTIONS;
                    syntax = PragmaSyntax.SYN_POSTFIX;
                    pg_get_token();
                    if((err = (args = parse_OMP_clause()) == null) || args.isEmpty())
                        break;
                    if(is_dir_nowait(args))
                        break;
                } else if(pg_is_ident("single")) {
                    pragmaDir = OMPpragma.SINGLE;
                    syntax = PragmaSyntax.SYN_POSTFIX;
                    pg_get_token();
                    err = ((args = parse_OMP_clause()) == null);
                    //accept copypivate
                    break;
                } else if(pg_is_ident("critical")) {
                    pragmaDir = OMPpragma.CRITICAL;
                    syntax = PragmaSyntax.SYN_POSTFIX;
                    pg_get_token();
                    if(pg_tok() == '(') {
                        if((err = (args = parse_OMP_namelist_as_not_symbol()) == null))
                            break;
                    } else {
                        args = null;
                    }
                    break;
                } else if(pg_is_ident("master")) {
                    pragmaDir = OMPpragma.MASTER;
                    syntax = PragmaSyntax.SYN_POSTFIX;
                    pg_get_token();
                    break;
                } else if(pg_is_ident("ordered")) {
                    pragmaDir = OMPpragma.ORDERED;
                    syntax = PragmaSyntax.SYN_POSTFIX;
                    pg_get_token();
                    break;
                }
            }
            
            if(!is_lang_c() && pg_is_ident("endparallel")) {
                pragmaDir = OMPpragma.PARALLEL;
                syntax = PragmaSyntax.SYN_POSTFIX;
                pg_get_token();
                if(pg_tok() == 0)
                    break;
                if(pg_is_ident("do")) {
                    pragmaDir = OMPpragma.PARALLEL_FOR;
                    pg_get_token();
                    break;
                } else if(pg_is_ident("sections")) {
                    pragmaDir = OMPpragma.PARALLEL_SECTIONS;
                    pg_get_token();
                    break;
                }
            }
            
            if(!is_lang_c() && pg_is_ident("enddo")) {
                pragmaDir = OMPpragma.FOR;
                syntax = PragmaSyntax.SYN_POSTFIX;
                pg_get_token();
                if((err = (args = parse_OMP_clause()) == null) || args.isEmpty())
                    break;
                if(is_dir_nowait(args))
                    break;
            }
            
            if(!is_lang_c() && pg_is_ident("endsections")) {
                pragmaDir = OMPpragma.SECTIONS;
                syntax = PragmaSyntax.SYN_POSTFIX;
                pg_get_token();
                if((err = (args = parse_OMP_clause()) == null) || args.isEmpty())
                    break;
                if(is_dir_nowait(args))
                    break;
            }
        
            if(!is_lang_c() && pg_is_ident("endsingle")) {
                pragmaDir = OMPpragma.SINGLE;
                syntax = PragmaSyntax.SYN_POSTFIX;
                pg_get_token();
                if((err = (args = parse_OMP_clause()) == null) || args.isEmpty())
                    break;
                //accept copypivate
            }
            
            if(!is_lang_c() && pg_is_ident("endcritical")) {
                pragmaDir = OMPpragma.CRITICAL;
                syntax = PragmaSyntax.SYN_POSTFIX;
                pg_get_token();
                if(pg_tok() == '(') {
                    if((err = (args = parse_OMP_namelist_as_not_symbol()) == null))
                        break;
                } else {
                    args = null;
                }
                break;
            }
            
            if(!is_lang_c() && pg_is_ident("endmaster")) {
                pragmaDir = OMPpragma.MASTER;
                syntax = PragmaSyntax.SYN_POSTFIX;
                pg_get_token();
                break;
            }
            
            if(!is_lang_c() && pg_is_ident("endordered")) {
                pragmaDir = OMPpragma.ORDERED;
                syntax = PragmaSyntax.SYN_POSTFIX;
                pg_get_token();
                break;
            }
            
            error("unknown OpenMP directive, '" + pg_tok_buf() + "'");
            err = true;
            
        } while(false);
        
        if(err)
            return null;
        
        if(pg_tok() != 0 || pragmaDir == null) {
            error("extra arguments for OpenMP directive");
            return null;
        }
        
        retSyntax[0] = syntax;
        retArgs[0] = args;
        
        return pragmaDir;
    }

    private Xobject omp_kwd(OMPpragma pg)
    {
        return Xcons.String(pg.toString());
    }

    private Xobject omp_pg_list(OMPpragma pg, Xobject args)
    {
        return Xcons.List(Xcode.LIST, Xcons.String(pg.toString()), args);
    }
    
    private Xobject parse_OMP_clause() throws XmException
    {
        Xobject args, v, c;
        boolean err = false;
        args = Xcons.List();

        while(pg_tok() == PG_IDENT || pg_tok() == ',') {
            if(pg_tok() == ',') {
                pg_get_token();
                continue;
            } else if(pg_is_ident("private")) {
                pg_get_token();
                if((err = (v = parse_OMP_namelist_as_symbol()) == null))
                    break;
                c = omp_pg_list(OMPpragma.DATA_PRIVATE, v);
            } else if(pg_is_ident("shared")) {
                pg_get_token();
                if((err = (v = parse_OMP_namelist_as_symbol()) == null))
                    break;
                c = omp_pg_list(OMPpragma.DATA_SHARED, v);
            } else if(pg_is_ident("firstprivate")) {
                pg_get_token();
                if((err = (v = parse_OMP_namelist_as_symbol()) == null))
                    break;
                c = omp_pg_list(OMPpragma.DATA_FIRSTPRIVATE, v);
            } else if(pg_is_ident("lastprivate")) {
                pg_get_token();
                if((err = (v = parse_OMP_namelist_as_symbol()) == null))
                    break;
                c = omp_pg_list(OMPpragma.DATA_LASTPRIVATE, v);
            } else if(pg_is_ident("copyprivate")) {
                pg_get_token();
                if((err = (v = parse_OMP_namelist_as_symbol()) == null))
                    break;
                c = omp_pg_list(OMPpragma.DATA_COPYPRIVATE, v);
            } else if(pg_is_ident("copyin")) {
                pg_get_token();
                if((err = (v = parse_OMP_namelist_as_symbol()) == null))
                    break;
                c = omp_pg_list(OMPpragma.DATA_COPYIN, v);
            } else if(pg_is_ident("num_threads")) {
                pg_get_token();
                if((err = (pg_tok() != '(')))
                    break;
                pg_get_token();
                if((err = (v = pg_parse_expr()) == null))
                    break;
                if((err = (pg_tok() != ')')))
                    break;
                pg_get_token();
                c = omp_pg_list(OMPpragma.DATA_NUM_THREADS, v);
            } else if(pg_is_ident("reduction")) {
                pg_get_token();
                OMPpragma clause[] = {null};
                if((err = (v = parse_OMP_reduction_namelist(clause)) == null))
                    break;
                c = omp_pg_list(clause[0], v);
            } else if(pg_is_ident("default")) {
                pg_get_token();
                if((err = (pg_tok() != '(')))
                    break;
                pg_get_token();
                if((err = (pg_tok() != PG_IDENT)))
                    break;
                if(pg_is_ident("shared")) {
                    c = omp_pg_list(OMPpragma.DATA_DEFAULT,
                        omp_kwd(OMPpragma.DEFAULT_SHARED));
                } else if(pg_is_ident("none")) {
                    c = omp_pg_list(OMPpragma.DATA_DEFAULT,
                        omp_kwd(OMPpragma.DEFAULT_NONE));
                } else if(pg_is_ident("private")) {
                    c = omp_pg_list(OMPpragma.DATA_DEFAULT,
                        omp_kwd(OMPpragma.DEFAULT_PRIVATE));
                } else {
                    err = true;
                    break;
                }
                pg_get_token();
                if((err = (pg_tok() != ')')))
                    break;
                pg_get_token();
            } else if(pg_is_ident("if")) {
                pg_get_token();
                if((err = (pg_tok() != '(')))
                    break;
                pg_get_token();
                if((err = (v = pg_parse_expr()) == null))
                    break;
                if((err = (pg_tok() != ')')))
                    break;
                pg_get_token();
                c = omp_pg_list(OMPpragma.DIR_IF, v);
            } else if(pg_is_ident("schedule")) {
                pg_get_token();
                if((err = (pg_tok() != '(')))
                    break;
                pg_get_token();
                if((err = (pg_tok() != PG_IDENT)))
                    break;
                OMPpragma sched = null;
                if(pg_is_ident("static"))
                    sched = OMPpragma.SCHED_STATIC;
                else if(pg_is_ident("dynamic"))
                    sched = OMPpragma.SCHED_DYNAMIC;
                else if(pg_is_ident("guided"))
                    sched = OMPpragma.SCHED_GUIDED;
                else if(pg_is_ident("runtime"))
                    sched = OMPpragma.SCHED_RUNTIME;
                else if(pg_is_ident("affinity"))
                    sched = OMPpragma.SCHED_AFFINITY;
                else {
                    err = true;
                    break;
                }
                pg_get_token();
                
                if(sched == OMPpragma.SCHED_AFFINITY) {
                    error("affinity schedule clause is not supported");
                    err = true;
                    break;
                }
                 
                if(pg_tok() == ',') {
                    pg_get_token();
                    if((err = (v = _lexer.pg_parse_expr()) == null))
                        break;
                    v = omp_pg_list(sched, v);
                } else {
                    v = omp_pg_list(sched, null);
                }
                if((err = (pg_tok() != ')')))
                    break;
                pg_get_token();
                c = omp_pg_list(OMPpragma.DIR_SCHEDULE, v);
            } else if(pg_is_ident("ordered")) {
                pg_get_token();
                c = omp_pg_list(OMPpragma.DIR_ORDERED, null);
            } else if(pg_is_ident("nowait")) {
                pg_get_token();
                c = omp_pg_list(OMPpragma.DIR_NOWAIT, null);
            } else {
                error("unknown OpenMP directive clause '" + pg_tok_buf() + "'");
                err = true;
                break;
            }
            
            args.add(c);
        }
        
        if(err) {
            if(!_lexer.has_error())
                error("syntax error in OpenMP pragma clause");
            return null;
        }

        return args;
    }
    
    private Xobject parse_OMP_namelist_as_symbol()
    {
        return parse_OMP_namelist(true);
    }
    
    private Xobject parse_OMP_namelist_as_not_symbol()
    {
        return parse_OMP_namelist(false);
    }
    
    private boolean pg_get_common_name()
    {
        if(is_lang_c())
            return false;
        
        if(pg_tok() != '/')
            return false;

        pg_get_token();
        
        if(pg_tok() != PG_IDENT)
            return false;

        pg_get_token();

        if(pg_tok() != '/')
            return false;
        
        return true;
    }
    
    private Xobject pg_parse_expr() throws XmException
    {
        return _lexer.pg_parse_expr();
    }
    
    private Xobject parse_OMP_namelist(boolean asSymbol)
    {
        Xobject args = Xcons.List();
        
        if(pg_tok() != '(') {
            error("OpenMP directive clause requires name list");
            return null;
        }
        
        while(true) {
            pg_get_token();
            
            if(pg_tok() == '/') {
                if(!pg_get_common_name()) {
                    error("OpenMP directive clause requires name list");
                    return null;
                }
            } else if(pg_tok() != PG_IDENT) {
                error("empty name list in OpenMP directive clause");
                return null;
            }
            
            if(asSymbol) {
                XmSymbol sym = XmSymbolUtil.lookupSymbol(_lexer.getContext(), pg_tok_buf());
                if(sym == null) {
                    error("undefined identifier in OpenMP directive clause : " + pg_tok_buf());
                    return null;
                } else if(!sym.isIdent() &&
                    !sym.getSclass().equals(StorageClass.FCOMMON_NAME)) {
                    error("bad identifier in OpenMP directive clause : " + pg_tok_buf());
                    return null;
                }
            }
            args.add(Xcons.Symbol(Xcode.IDENT, pg_tok_buf()));
            pg_get_token();
            if(pg_tok() == ')') {
                pg_get_token();
                return args;
            }
            if(pg_tok() != ',') {
                break;
            }
        };
    
        error("syntax error in OpenMP pragma clause");
        return null;
    }

    Xobject parse_OMP_reduction_namelist(OMPpragma[] result)
    {
        Xobject args = Xcons.List();
        OMPpragma clause = null;
        
        if(pg_tok() != '(') {
        	error("OpenMP reduction clause requires name list");
        	return null;
        }
        pg_get_token();
        
        clause = get_OMP_reduction_clause();
        
        if(clause == null)
            return null;
        
        pg_get_token();
        if(pg_tok() != ':')
            return null;
        pg_get_token();

        if(pg_tok() != PG_IDENT) {
        	error("empty name list in OpenMP reduction clause");
        	return null;
        }

        while(true) {
            XmSymbol sym = XmSymbolUtil.lookupSymbol(_lexer.getContext(), pg_tok_buf());
            if(sym == null) {
                error("undefined identifier in OpenMP directive clause : " + pg_tok_buf());
                return null;
            }
            if(!sym.isIdent()) {
            	error("bad identifier in OpenMP directive clause");
            	return null;
            }
            args.add(Xcons.Symbol(Xcode.IDENT, sym.getName()));
            pg_get_token();
            if(pg_tok() == ',') {
            	pg_get_token();
            	continue;
            } else if(pg_tok() == ')') {
            	pg_get_token();
            	result[0] = clause;
            	return args;
            }
            break;
        }
    
        error("syntax error in OpenMP directive clause");
        return null;
    }
    
    private boolean is_lang_c()
    {
        return _lexer.is_lang_c();        
    }
    
    private OMPpragma get_OMP_reduction_clause()
    {
        if(is_lang_c()) {
            switch(pg_tok()) {
            case '+':
                return OMPpragma.DATA_REDUCTION_PLUS;
            case '-':
                return OMPpragma.DATA_REDUCTION_MINUS;
            case '*':
                return OMPpragma.DATA_REDUCTION_MUL;
            case '&':
                return OMPpragma.DATA_REDUCTION_BITAND;
            case '|':
                return OMPpragma.DATA_REDUCTION_BITOR;
            case '^':
                return OMPpragma.DATA_REDUCTION_BITXOR;
            case PG_ANDAND:
                return OMPpragma.DATA_REDUCTION_LOGAND;
            case PG_OROR:
                return OMPpragma.DATA_REDUCTION_LOGOR;
            case PG_IDENT:
                if(pg_is_ident("max")) {
                    return OMPpragma.DATA_REDUCTION_MAX;
                }
                if(pg_is_ident("min")) {
                    return OMPpragma.DATA_REDUCTION_MIN;
                }
            }
        } else {
            switch(pg_tok()) {
            case '+':
                return OMPpragma.DATA_REDUCTION_PLUS;
            case '-':
                return OMPpragma.DATA_REDUCTION_MINUS;
            case '*':
                return OMPpragma.DATA_REDUCTION_MUL;
            case PG_FAND:
                return OMPpragma.DATA_REDUCTION_LOGAND;
            case PG_FOR:
                return OMPpragma.DATA_REDUCTION_LOGOR;
            case PG_FEQV:
                return OMPpragma.DATA_REDUCTION_LOGEQV;
            case PG_FNEQV:
                return OMPpragma.DATA_REDUCTION_LOGNEQV;
            case PG_IDENT:
                if(pg_is_ident("max")) {
                    return OMPpragma.DATA_REDUCTION_MAX;
                }
                if(pg_is_ident("min")) {
                    return OMPpragma.DATA_REDUCTION_MIN;
                }
                if(pg_is_ident("iand")) {
                    return OMPpragma.DATA_REDUCTION_IAND;
                }
                if(pg_is_ident("ior")) {
                    return OMPpragma.DATA_REDUCTION_IOR;
                }
                if(pg_is_ident("ieor")) {
                    return OMPpragma.DATA_REDUCTION_IEOR;
                }
            }
        }
        
        return null; // syntax error
    }
}
