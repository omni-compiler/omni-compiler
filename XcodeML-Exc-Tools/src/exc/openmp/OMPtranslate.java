/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
package exc.openmp;

import java.util.List;

import xcodeml.util.XmOption;
import exc.object.*;
import exc.xcodeml.XcodeMLtools;
import exc.block.*;

/**
 * OpenMP AST translator
 */
public class OMPtranslate implements XobjectDefVisitor
{
    BlockPrintWriter debug_out;
    XobjectFile env;

    // alorithms
    OMPanalyzePragma anaPragma = new OMPanalyzePragma();
    OMPrewriteExpr rewriteExpr = new OMPrewriteExpr();
    OMPtransPragma transPragma = new OMPtransPragma();
    
    public OMPtranslate()
    {
    }
    
    public OMPtranslate(XobjectFile env)
    {
        init(env);
    }

    public void init(XobjectFile env)
    {
        this.env = env;
        if(OMP.debugFlag)
            debug_out = new BlockPrintWriter(System.out);
    }
    
    public void finish()
    {
        FmoduleBlock mod = (FmoduleBlock)env.getProp(OMPtransPragma.PROP_KEY_FINTERNAL_MODULE);
        if(mod != null) {
            XobjectDef lastMod = null;
            for(XobjectDef d : env) {
                if(d.isFmoduleDef())
                    lastMod = d;
            }
            if(lastMod == null)
                env.insert(mod.toXobjectDef());
            else
                lastMod.addAfterThis(mod.toXobjectDef());
        }
        env.collectAllTypes();
        env.fixupTypeRef();
    }
    
    private void replace_main(XobjectDef d)
    {
        String name = d.getName();
        Ident id = env.findVarIdent(name);
        if(id == null)
            OMP.fatal("'" + name + "' not in id_list");
        id.setName(transPragma.mainFunc);
        d.setName(transPragma.mainFunc);
    }

    // do transform takes three passes
    public void doDef(XobjectDef d)
    {
        OMP.resetError();

        OMPanalyzeDecl anaDecl = new OMPanalyzeDecl(env);
        anaDecl.analyze(d);
        
        if(OMP.hasError())
            return;
        
        if(!d.isFuncDef()) {
            return;
        }
        
        if(XmOption.isLanguageC()) {
            if(d.getName().equals("main"))
                replace_main(d);
        } else {
            Xtype ft = d.getFuncType();
            if(ft != null && ft.isFprogram()) {
                ft.setIsFprogram(false);
                replace_main(d);
            }
        }
        
        FuncDefBlock fd = new FuncDefBlock(d);

	List list = anaDecl.getCommonName();
        for(int i=0;i<list.size();i++)
              fd.searchCommonMember(list.get(i).toString(),anaDecl,d);

        if(XmOption.isLanguageC())
            fd.removeDeclInit();

        if(OMP.hasError())
            return;
        
        OMP.debug("3");

        anaPragma.run(fd, anaDecl);
        if(OMP.hasError())
            return;

        OMP.debug("4");

        rewriteExpr.run(fd, anaDecl);
        if(OMP.hasError())
            return;
        OMP.debug("5");

        transPragma.run(fd);
        if(OMP.hasError())
            return;
        
        // finally, replace body
        fd.Finalize();
    }

    // not used?
    boolean haveOMPpragma(Xobject x)
    {
        XobjectIterator i = new topdownXobjectIterator(x);
        for(i.init(); !i.end(); i.next()) {
            Xobject xx = i.getXobject();
            if(xx != null && xx.Opcode() == Xcode.OMP_PRAGMA)
                return true;
        }
        return false;
    }
}
