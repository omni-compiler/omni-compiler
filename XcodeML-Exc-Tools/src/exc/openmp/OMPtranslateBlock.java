/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
package exc.openmp;

import java.io.*;

import xcodeml.util.XmOption;
import exc.xcodeml.XcodeMLtools;
import exc.object.*;
import exc.block.*;

public class OMPtranslateBlock implements XobjectDefVisitor
{
    BlockPrintWriter debug_out;
    XobjectFile env;

    // alorithms
    OMPanalyzePragma anaPragma = new OMPanalyzePragma();
    OMPrewriteExpr rewriteExpr = new OMPrewriteExpr();
    OMPtransPragmaBlock transPragma = new OMPtransPragmaBlock();
    boolean finalize_mode;

    public void init(XobjectFile env, boolean finalize_mode)
    {
        this.env = env;
        this.finalize_mode = finalize_mode;
        if(finalize_mode)
            return;

        if(OMP.debugFlag) {
            OutputStream fout;
            fout = System.out;
            try {
                fout = new FileOutputStream("debug.OMPtranlateBlock");
            } catch(IOException e) {
            }
            ;
            debug_out = new BlockPrintWriter(fout);
        }
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
        if(finalize_mode) {
            if(d.isFuncDef()) {
                FuncDefBlock fd = new FuncDefBlock(d);
                if(OMP.debugFlag)
                    debug_out.print(fd.getBlock());
                transPragma.transBlock(fd);
                fd.Finalize();
            }
            return;
        }

        OMPanalyzeDecl anaDecl = new OMPanalyzeDecl(env);
        anaDecl.analyze(d);
        
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

        OMP.resetError();

        FuncDefBlock fd = new FuncDefBlock(d);
        fd.removeDeclInit();

        anaPragma.run(fd, anaDecl);
        if(OMP.hasError())
            return;

        rewriteExpr.run(fd, anaDecl);
        if(OMP.hasError())
            return;

        transPragma.run(fd);

        // no Finialize
    }
}
