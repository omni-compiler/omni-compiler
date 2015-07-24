/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
package exc.openmp;

import xcodeml.IXobject;
import xcodeml.XmException;
import xcodeml.util.XmLog;
import xcodeml.util.XmOption;
import exc.object.ExternalPragmaParser;
import exc.object.Ident;
import exc.object.PragmaParser;
import exc.object.PragmaSyntax;
import exc.object.StorageClass;
import exc.object.Xcode;
import exc.object.Xcons;
import exc.object.XobjArgs;
import exc.object.XobjList;
import exc.object.Xobject;

import java.util.ArrayList;

/**
 * OpenMP pragma parser
 */
public class OMPpragmaParser implements ExternalPragmaParser
{
    /** base parser */
    private PragmaParser _parser;
    
    public OMPpragmaParser(PragmaParser parser)
    {
        _parser = parser;
    }

    private class ResultClause
    {
        final Xobject pclause;
        final Xobject dclause;
        
        ResultClause(Xobject pclause, Xobject dclause)
        {
            this.pclause = pclause;
            this.dclause = dclause;
        }
    }
    
    private XmException exception(String s)
    {
        return new XmException("[OpenMP] " + s);
    }

    private Xobject omp_pg_list(OMPpragma pg, Xobject args)
    {
        return Xcons.List(Xcode.LIST, Xcons.String(pg.toString()), args);
    }
    
    private Xobject omp_kwd(OMPpragma pg)
    {
        return Xcons.String(pg.toString());
    }
    
    @Override
    public Xobject parse(Xobject x) throws XmException
    {
        Xobject v, c;
        ResultClause rc;
        
        if(x.getArgOrNull(0) == null)
            XmLog.fatal("pragma kind is null");
        OMPpragma pragma = OMPpragma.valueOf(x.getArg(0));

        switch(pragma) {
        case PARALLEL:
            /* parallel <clause_list> */
            rc = compile_OMP_pragma_clause(x.getArg(1), OMPpragma.PARALLEL, true);
            return Xcons.List(Xcode.OMP_PRAGMA, x.getArg(0), rc.pclause, x.getArg(2));
    
        case PARALLEL_FOR:
            /* parallel for <clause_list> */
            rc = compile_OMP_pragma_clause(x.getArg(1), OMPpragma.FOR, true);
            return Xcons.List(Xcode.OMP_PRAGMA,
                omp_kwd(OMPpragma.PARALLEL), rc.pclause,
                Xcons.List(Xcode.OMP_PRAGMA,
                    omp_kwd(OMPpragma.FOR),
                    rc.dclause, x.getArg(2)));
    
        case PARALLEL_SECTIONS:
            /* parallel sections <clause_list> */
            rc = compile_OMP_pragma_clause(x.getArg(1), OMPpragma.SECTIONS, true);
            v = compile_OMP_SECTIONS_statement(x.getArg(2));
            return Xcons.List(Xcode.OMP_PRAGMA,
                omp_kwd(OMPpragma.PARALLEL), rc.pclause,
                Xcons.List(Xcode.OMP_PRAGMA,
                    omp_kwd(OMPpragma.SECTIONS), rc.dclause, v));
    
        case FOR:
            /* for <clause_list> */
        	rc = compile_OMP_pragma_clause(x.getArg(1), OMPpragma.FOR, false);
            return Xcons.List(Xcode.OMP_PRAGMA, x.getArg(0), rc.dclause, x.getArg(2));

        case SECTIONS:
            /* sections <clause_list> */
            rc = compile_OMP_pragma_clause(x.getArg(1), OMPpragma.SECTIONS, false);
            if((v = compile_OMP_SECTIONS_statement(x.getArgOrNull(2))) == null)
                break;
            return Xcons.List(Xcode.OMP_PRAGMA, x.getArg(0), rc.dclause, v);
    
        case SINGLE:
            /* single <clause list> */
            rc = compile_OMP_pragma_clause(x.getArg(1), OMPpragma.SINGLE, false);
            return Xcons.List(Xcode.OMP_PRAGMA, x.getArg(0), rc.dclause, x.getArg(2));
    
        case MASTER:
            /* master */
        case ORDERED:
            /* ordered */
            return Xcons.List(Xcode.OMP_PRAGMA, x.getArg(0), null, x.getArg(2));
            
        case CRITICAL:
            /* critical <name> */
            c = x.getArg(1);
            if(c != null && c.Nargs() > 1) {
                throw exception("bad critical section name");
            }
            return Xcons.List(Xcode.OMP_PRAGMA, x.getArg(0), c, x.getArg(2));
    
        case ATOMIC:
            /* atomic */
            /* should check next statment */
            if((v = x.getArgOrNull(2)) == null) 
                break;
            if((XmOption.isLanguageC() && v.Opcode() != Xcode.EXPR_STATEMENT) ||
                XmOption.isLanguageF() && !v.Opcode().isFstatement()) {
                throw exception("bad statement for OMP atomic directive");
            }
            return Xcons.List(Xcode.OMP_PRAGMA, x.getArg(0), null, v);

        case SECTION:
            /* section */
            /* section directive must appear in section block */
            throw exception("'section' directive must be in sections block");
    
        case BARRIER:
            /* barrier */
            return Xcons.List(Xcode.OMP_PRAGMA, x.getArg(0), null, null);
    
        case FLUSH:
            /* flush <namelist> */
            c = x.getArg(1);
            compile_OMP_name_list(c);
            return Xcons.List(Xcode.OMP_PRAGMA, x.getArg(0), c, null);
    
        case THREADPRIVATE:
            /* threadprivate <namelist> */
            c = x.getArg(1);
            compile_OMP_name_list(c);
            return Xcons.List(Xcode.OMP_PRAGMA, x.getArg(0), c, null);
    
        default:
            XmLog.fatal("unknown pragma " + pragma.toString().toLowerCase());
        }
        return null;
    }
    
    private Xobject compile_OMP_SECTIONS_statement(Xobject x) throws XmException
    {
        Xobject section_list, current_section;
        
        if(x == null) {
            throw exception("sections directive must be followed by compound statement block");
        }
        
        Xobject body = null;
        
        if(XmOption.isLanguageC()) {
            if(x.Opcode() != Xcode.COMPOUND_STATEMENT)
                throw exception("sections directive must be followed by compound statement block");
            Xobject id_list = x.getArg(0);
            if(id_list != null && !id_list.isEmpty()) {
                throw exception("declarations in sections block");
            }
            body = x.getArg(2);
        } else {
            body = x;
        }
    
        section_list = Xcons.List();
        current_section = null;
        for(Xobject a : (XobjList)body) {
            
            // child SECTION Xobject is not parsed yet here.
            //   ... (OMP_PRAGMA (PragmaSyntax) (OMPpragma))
            if(a.Opcode() == Xcode.OMP_PRAGMA &&
                OMPpragma.valueOf(a.getArg(1)) == OMPpragma.SECTION) {
                current_section = Xcons.CompoundStatement(
                    Xcons.IDList(), Xcons.List(), Xcons.List());
                section_list.add(current_section);
            } else  if(current_section != null) {
                current_section.getArg(2).add(a);
            } else {
                // no section. create one section.
                current_section = Xcons.CompoundStatement(
                    Xcons.IDList(), Xcons.List(), Xcons.List(a));
                section_list.add(current_section);
            }
        }
        
        return section_list;
    }
    
    /* PARALLEL - private,firstprivate,reduction,default,shared,copyin,if
     * FOR      - private,firstprivate,lastprivate,reduction,ordered,shed,nowait
     * SECTIONS - private,firstprivate,lastprivate,reduction,nowait
     * SINGLE   - private,firstprivate,nowait
     */
    private ResultClause compile_OMP_pragma_clause(Xobject x, OMPpragma pragma,
        boolean is_parallel) throws XmException
    {
        Xobject v;
        Xobject pclause = null;
        Xobject dclause = Xcons.List();
    
        if(is_parallel)
            pclause = Xcons.statementList();
        for(Xobject c : (XobjList)x) {
            OMPpragma p = OMPpragma.valueOf(c.getArg(0));
            switch(p) {
            case DATA_DEFAULT:  /* default(shared|none) */
                if(!is_parallel) {
                    throw exception("'default' clause must be in PARALLEL");
                }
                pclause.add(c);
                break;
                
            case DATA_SHARED:
                compile_OMP_name_list(c.getArg(1));
                if(!is_parallel) {
                    throw exception("'shared' clause must be in PARALLEL");
                }
                pclause.add(c);
                break;
                
            case DATA_COPYIN:
                compile_OMP_name_list(c.getArg(1));
                if(!is_parallel) {
                    throw exception("'copyin' clause must be in PARALLEL");
                }
                pclause.add(c);
                break;
                
            case DIR_NUM_THREADS:
                if(pclause != null)
                    pclause.add(c);
                else
                    dclause.add(c);
                break;
                
            case DIR_IF:
                if(!is_parallel) {
                    throw exception("'if' clause must be in PARALLEL");
                }
                v = c.getArg(1);
                pclause.add(Xcons.List(c.getArg(0), v));
                break;
                
            case DATA_PRIVATE:
            case DATA_FIRSTPRIVATE:
                /* all pragma can have these */
                compile_OMP_name_list(c.getArg(1));
                if(pragma == OMPpragma.PARALLEL)
                    pclause.add(c);
                else     
                    dclause.add(c);
                break;
        
            case DATA_LASTPRIVATE:
                compile_OMP_name_list(c.getArg(1));
                if(pragma != OMPpragma.FOR && pragma != OMPpragma.SECTIONS) {
                    if(XmOption.isLanguageC())
                        throw exception("'lastprivate' clause must be in FOR or SECTIONS");
                    else
                        throw exception("'lastprivate' clause must be in DO or SECTIONS");
                }
                dclause.add(c);
                break;
                
            case DATA_COPYPRIVATE:
                compile_OMP_name_list(c.getArg(1));
                if(pragma != OMPpragma.SINGLE) {
                    throw exception("'copyprivate' clause must be in SINGLE");
                }
                dclause.add(c);
                break;
                
            case DIR_ORDERED:
                if(pragma != OMPpragma.FOR) {
                    if(XmOption.isLanguageC())
                        throw exception("'ordered' clause must be in FOR");
                    else
                        throw exception("'ordered' clause must be in DO");
                }
                dclause.add(c);
                break;
        
            case DIR_SCHEDULE:
                if(pragma != OMPpragma.FOR) {
                    if(XmOption.isLanguageC())
                        throw exception("'schedule' clause must be in FOR");
                    else
                        throw exception("'schedule' clause must be in DO");
                }
                v = c.getArg(1).getArg(1);
                if(v != null && 
                    OMPpragma.valueOf(c.getArg(1).getArg(0)) != OMPpragma.SCHED_AFFINITY) {
                    c = Xcons.List(c.getArg(0), Xcons.List(c.getArg(1).getArg(0), v));
                }
                dclause.add(c);
                break;
        
            case DIR_NOWAIT:
                if(is_parallel) {
                    throw exception("'nowait' clause must not be in PARALLEL");
                }
                dclause.add(c);
                break;
        
            default:
                if(p.isDataReduction()) {
                    compile_OMP_name_list(c.getArg(1));
                    if(pragma == OMPpragma.PARALLEL)
                        pclause.add(c);
                    else if(pragma == OMPpragma.FOR || pragma == OMPpragma.SECTIONS)
                        dclause.add(c);
                    else 
                        throw exception("'reduction' clause must not be in SINGLE");
                } else {
                    XmLog.fatal(c.getArg(0).toString());
                }
                break;
            }
        }
    
        /* combination with PARALLEL, don't have to wait */
        if(is_parallel && (pragma != OMPpragma.PARALLEL))
            dclause.add(omp_pg_list(OMPpragma.DIR_NOWAIT, null));
    
        return new ResultClause(pclause, dclause);
    }
    
    private void compile_OMP_name_list(Xobject name_list) throws XmException
    {
        if(name_list == null)
            return;
        
        ArrayList<Ident> addIdList = new ArrayList<Ident>();
        
        for(XobjArgs a = name_list.getArgs(); a != null; a = a.nextArgs()) {
            Xobject v = a.getArg();
            Ident id = _parser.findIdent(v.getName(), IXobject.FINDKIND_VAR);
            
            if(id == null) {
                if(XmOption.isLanguageC()) {
                    id = _parser.getXobjectFile().findVarIdent(v.getName());
                } else {
                    id = _parser.findIdent(v.getName(), IXobject.FINDKIND_COMMON);
                    if(id.getStorageClass() != StorageClass.FCOMMON_NAME)
                        id = null;
                }
                
                if (id == null) {
                    throw exception("undefined variable, " + v.getName() + " in pragma");
                }
            }
            
            switch(id.getStorageClass()) {
            case AUTO:
            case PARAM:
            case EXTERN:
            case EXTDEF:
            case STATIC:
            case REGISTER:
            case FLOCAL:
            case FCOMMON:
            case FPARAM:
            case FSAVE:
            case FFUNC:
                if(id.Type() == null)
                    XmLog.fatal("type is null");
                if(id.Type().isFunction()) {
                    throw exception("function name, " + v.getName() + " in pragma");
                }
                break;
            case FCOMMON_NAME:
                addIdList.addAll(id.getFcommonVars());
                name_list.removeArgs(a);
                break;
            default:
                throw exception("identifer, " + v.getName() + " is not variable in pragma");
            }
        }
        
        for(Ident id : addIdList) {
            name_list.add(id);
        }
    }

    public boolean isPrePostPair(Xobject x1, Xobject x2)
    {
        OMPpragma p1 = OMPpragma.valueOf(x1.getArg(0).getString());
        OMPpragma p2 = OMPpragma.valueOf(x2.getArg(1).getString());

        if(p1 != p2)
            return false;
        
        if(p1 == OMPpragma.CRITICAL) {        
            Xobject a1 = x1.getArgOrNull(1);
            Xobject a2 = x2.getArgOrNull(2);
            if(a1 == null && a2 == null)
                return true;
            if((a1 != null && a2 == null) || (a1 == null && a2 != null))
                return false;
            Xobject c1 = a1.getArgOrNull(0);
            Xobject c2 = a2.getArgOrNull(0);
            if(c1 == null && c2 == null)
                return true;
            if((c1 != null && c2 == null) || (c1 == null && c2 != null))
                return false;
            if(c1.Opcode() != Xcode.IDENT && c2.Opcode() != Xcode.IDENT)
                return false;
            return c1.getName().equals(c2.getName());
        }
        
        return true;
    }

    public void completePragmaEnd(Xobject prefix, Xobject body)
    {
        OMPpragma prefixp = OMPpragma.valueOf(prefix.getArg(0).getString());
        if(prefixp != OMPpragma.PARALLEL)
            return;
        int order = 0;
        int hasFor = 0, hasSections = 0;
        boolean hasEndFor = false, hasEndSections = false;
        
        for(Xobject x : (XobjList)body) {
            if(x.Opcode() != Xcode.OMP_PRAGMA)
                continue;
            PragmaSyntax s = PragmaSyntax.valueOf(x.getArg(0).getString());
            OMPpragma p = OMPpragma.valueOf(x.getArg(1).getString());
            switch(p) {
            case FOR:
                if(s == PragmaSyntax.SYN_START)
                    hasFor = ++order;
                else
                    hasEndFor = true;
                break;
            case SECTIONS:
                if(s == PragmaSyntax.SYN_START)
                    hasSections = ++order;
                else
                    hasEndSections = true;
                break;
            }
        }

        for(int i = 1; i <= 2; ++i) {
            if(i == hasFor && !hasEndFor) {
                body.add(Xcons.List(Xcode.OMP_PRAGMA,
                    Xcons.String(PragmaSyntax.SYN_POSTFIX.toString()),
                    Xcons.String(OMPpragma.FOR.toString())));
            } else if(i == hasSections && !hasEndSections) {
                body.add(Xcons.List(Xcode.OMP_PRAGMA,
                    Xcons.String(PragmaSyntax.SYN_POSTFIX.toString()),
                    Xcons.String(OMPpragma.SECTIONS.toString())));
            }
        }
    }

    public XobjArgs getAbbrevPostfix(XobjArgs prefixArgs)
    {
        Xobject prefix = prefixArgs.getArg();
        OMPpragma prefixp = OMPpragma.valueOf(prefix.getArg(0).getString());
        if(prefixp == OMPpragma.FOR || prefixp == OMPpragma.PARALLEL_FOR ||
            prefixp == OMPpragma.SECTIONS || prefixp == OMPpragma.PARALLEL_SECTIONS) {
            
            XobjArgs nArg = prefixArgs.nextArgs();
            XobjArgs nnArg = nArg.nextArgs();
            Xobject postfix = Xcons.List(Xcode.OMP_PRAGMA,
                Xcons.String(prefixp.toString()),
                Xcons.String(PragmaSyntax.SYN_POSTFIX.toString()),
                Xcons.List());
            XobjArgs postfixArgs = new XobjArgs(postfix, nnArg);
            nArg.setNext(postfixArgs);
            return postfixArgs;
        }
        
        return null;
    }

    public void mergeStartAndPostfixArgs(Xobject start, Xobject postfix)
    {
        if(OMPpragma.valueOf(start.getArg(0).getString()) == OMPpragma.CRITICAL)
            return;
        
        Xobject postClause = postfix.getArgOrNull(2);
        
        if(postClause != null) {
            Xobject startClause = start.getArgOrNull(1);
            if(startClause == null) {
                startClause = Xcons.List();
                start.add(startClause);
            }
            for(Xobject aa : (XobjList)postClause)
                startClause.add(aa);
        }
    }
}
