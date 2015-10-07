/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
package exc.openmp;

import exc.object.*;
import exc.block.*;
import java.util.LinkedList;
import java.util.List;
import java.util.Vector;

import xcodeml.util.XmOption;

public class OMPanalyzeDecl implements OMPfileEnv
{
  private XobjectFile env;
  private Vector<Ident> thdprv_vars = new Vector<Ident>();
  private static final String PROP_KEY = "OMPanalyzeDecl";
  private OMPanalyzeDecl parent;
  private List<String> list = new LinkedList<String>();

  public OMPanalyzeDecl(XobjectFile env)
  {
    this.env = env;
  }
    
    @Override
    public XobjectFile getFile()
    {
        return env;
    }

    private void setToXobject(PropObject o)
    {
        o.setProp(PROP_KEY, this);
    }
    
    private OMPanalyzeDecl getParent(XobjectDef def)
    {
        if(parent != null)
            return parent;
        
        PropObject p = def.getParent();
        if(p == null)
            p = def.getParentEnv();
        if(p == null)
            throw new IllegalStateException();
        OMPanalyzeDecl a = (OMPanalyzeDecl)p.getProp(PROP_KEY);
        if(a == null) {
            a = new OMPanalyzeDecl(def.getFile());
            p.setProp(PROP_KEY, a);
        }
        parent = a;
        return a;
    }
    
    public void analyze(XobjectDef def)
    {
        Xobject x = def.getDef();
        
        if(x.Opcode() == Xcode.OMP_PRAGMA) {
            // (OMP_PRAGMA (STRING PragmaSyntax) (STRING OMPPragma) (LIST ...))
            if(OMPpragma.valueOf(x.getArg(1)) != OMPpragma.THREADPRIVATE)
                OMP.fatal("not threadprivate in decl");
            getParent(def).declThreadPrivate(x, def, x.getArg(2));
            def.setDef(null);
        } else if(def.isFuncDef() || def.isFmoduleDef()) {
            setToXobject(def);
            getParent(def); // to set parent
            topdownXobjectIterator ite = def.getDef().topdownIterator();
            for(ite.init(); !ite.end(); ite.next()) {
                x = ite.getXobject();
                if(x == null || x.Opcode() == null)
                    continue;
                switch(x.Opcode()) {
                case OMP_PRAGMA:
                    // (OMP_PRAGMA (STRING OMPPragma) (LIST ...))
                    if(OMPpragma.valueOf(x.getArg(0)) == OMPpragma.THREADPRIVATE) {
                        if(def.isFmoduleDef()) {
                            OMP.error((LineNo)def.getLineNo(),
                                "threadprivate for module variable is not supported");
                        } else
                            declThreadPrivate(x, def, x.getArg(1));
                        ite.setXobject(null);
			//                        ite.setXobject(Xcons.List(Xcode.NULL));
                    }
                    break;
                }
            }
        }
    }
    
    // declare threadprivate variables
    @Override
    public void declThreadPrivate(Xobject x, IVarContainer vc, Xobject args)
    {
        Xtype voidP_t = Xtype.Pointer(Xtype.voidType);
        for(XobjArgs a = args.getArgs(); a != null; a = a.nextArgs()) {
            String name = a.getArg().getName();
	    Ident id = (vc != null) ? (vc.findCommonIdent(name)==null ? vc.findVarIdent(name):vc.findCommonIdent(name) ) : (x.findCommonIdent(name)==null ? x.findVarIdent(name):x.findCommonIdent(name));
            if(id == null) {
                OMP.fatal("undefined variable '" + name
                    + "' in threadprivate directive");
                continue;
            }

            if(isThreadPrivate(id))
                continue; // already defined as threadprivate
            thdprv_vars.addElement(id);

            OMP.setThreadPrivate(id);
            
            if(OMP.leaveThreadPrivateFlag)
                continue;

            switch(id.getStorageClass()) {
            case FCOMMON_NAME:
              list.add(id.getName());
            case EXTDEF:
            case EXTERN:
            case STATIC:
            case FCOMMON:
            case FSAVE:
                break;
            default:
                if(XmOption.isLanguageC()) {
                    OMP.error(x.getLineNo(), "variable '" + id.getName()
                        + "' is not extern or static variable.");
                } else {
                    OMP.error(x.getLineNo(), "variable '" + id.getName()
                        + "' does not have common or save attribute.");
                }
                return;
            }
            
            if(XmOption.isLanguageC()) {
                // declare threadprivate pointer table
                String thdprv_name = OMPtransPragma.THDPRV_STORE_PREFIX + id.getName();
                switch(id.getStorageClass()) {
                case EXTDEF:
                    env.declGlobalIdent(thdprv_name, voidP_t);
                    break;
                case EXTERN:
                    env.declExternIdent(thdprv_name, voidP_t);
                    break;
                case STATIC:
                    env.declStaticIdent(thdprv_name, voidP_t);
                    break;
                default:
                    OMP.fatal("declThreadPrivate: bad class, " + id.getName());
                }
            }
        }
    }

    public void addThdprvVars(Ident id)
    {
	thdprv_vars.addElement(id);
    }
    
    @Override
    public boolean isThreadPrivate(Ident id)
    {
	if(thdprv_vars.contains(id)||OMP.isThreadPrivate(id))
            return true;
        return (parent != null) ? parent.isThreadPrivate(id) : false;
    }

    @Override
    public Ident findThreadPrivate(Block b, String name)
    {
        for(Ident id : thdprv_vars) {
            if(id.getName().equals(name))
                return id;
        }
        
        return (parent != null) ? parent.findThreadPrivate(b, name) : null;
    }
    
    public List getCommonName()
    {
     return list;      
    }

}
