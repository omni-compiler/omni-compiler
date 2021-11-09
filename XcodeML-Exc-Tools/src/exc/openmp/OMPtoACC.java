package exc.openmp;

import java.util.List;
import xcodeml.util.XmOption;
import exc.object.*;
import exc.xcodeml.XcodeMLtools;
import exc.block.*;
import exc.openacc.ACCpragma;
import java.util.Iterator;

public class OMPtoACC extends OMPtranslate {
    private static String MAIN_ARGC_NAME = "argc";
    private static String MAIN_ARGV_NAME = "argv";

    private static String ACC_INIT_FUNC_NAME = "_ACC_init";
    private static String ACC_FINAL_FUNC_NAME = "_ACC_finalize";

    private static final String ACC_TRAVERSE_INIT_FUNC_NAME = "acc_traverse_init";
    private static final String ACC_TRAVERSE_FINAL_FUNC_NAME = "acc_traverse_finalize";

    // ompc_main.
    private String ompcMainFunc = transPragma.mainFunc;
    // ompc_main_org.
    private String ompcMainOrgFunc = transPragma.mainFunc + "_org";

    private boolean isConverted = false;

    public OMPtoACC(XobjectFile env) {
        super(env);

        if (!XmOption.isLanguageC()) {
            OMP.fatal("current version only supports C language.");
        }
    }

    public void finish(){
        OMP.resetError();

        if (isConverted()) {
            setupMain();
            if (OMP.hasError()) {
                return;
            }
        }

        super.finish();
    }

    public void doDef(XobjectDef d) {
        OMP.resetError();

        convert(d);
        if (OMP.hasError()) {
            return;
        }

        super.doDef(d);
    }

    public boolean isConverted() {
        return isConverted;
    }

    private void setIsConverted(boolean b) {
       isConverted = b;
    }

    private XobjList convertFromMap(Xobject xobj,
                                    XobjList clause) {
        if (clause.Nargs() != 2) {
            OMP.error((LineNo)xobj.getLineNo(),
                      "Number of clauses is too small.");
            return null;
        }

        XobjList mapArgs = (XobjList) clause.getArg(1);
        if (mapArgs.Opcode() != Xcode.LIST) {
            OMP.error((LineNo)xobj.getLineNo(),
                      "Clause list does not exist");
            return null;
        }

        XobjString mapType = (XobjString) mapArgs.getArg(0);
        if (mapType.Opcode() != Xcode.STRING) {
            OMP.error((LineNo)xobj.getLineNo(),
                      "Map type does not exist.");
            return null;
        }

        // check map-type-modifier.
        if (mapType.getName().split(" ").length > 1) {
            OMP.error((LineNo)xobj.getLineNo(),
                      "'map-type-modifier (" +
                      mapType.getName().split(" ")[0] +
                      ")' is not supported.");
            return null;
        }

        XobjList mapValues = (XobjList) mapArgs.getArg(1);

        // NOTE: OpenMP xcode has an empty list
        //       if the array range specification is omitted.
        //       So, remove the empty list.
        if (mapValues.getArg(0).getArg(1).isEmpty()) {
            mapValues = Xcons.List(mapValues.getArg(0).getArg(0));
        }

        XobjList list = null;
        switch (mapType.getName()) {
        case "alloc":
            list = Xcons.List(Xcons.String("CREATE"),
                              mapValues);
            break;
        case "to":
            list = Xcons.List(Xcons.String("COPYIN"),
                              mapValues);
            break;
        case "from":
            list = Xcons.List(Xcons.String("COPYOUT"),
                              mapValues);
            break;
        case "tofrom":
            list = Xcons.List(Xcons.String("COPY"),
                              mapValues);
            break;
        default:
            OMP.error((LineNo)xobj.getLineNo(),
                      "'" + mapType.getName() + "'" +
                      " cannot be specified for map.");
            return null;
        }
        return list;
    }

    private void convertFromTargetData(Xobject xobj,
                                       XobjArgs args) {
        if (xobj.getArg(1) == null ||
            xobj.getArg(1).Opcode() != Xcode.LIST) {
            OMP.error((LineNo)xobj.getLineNo(),
                      "Not found clause list.");
            return;
        }

        XobjList clauses = (XobjList) xobj.getArg(1);
        XobjList newClauses = Xcons.List();
        for (Iterator<Xobject> it = clauses.iterator(); it.hasNext();) {
            XobjList clause = (XobjList) it.next();
            if (clause.Opcode() != Xcode.LIST ||
                clause.Nargs() < 1) {
                OMP.error((LineNo)xobj.getLineNo(),
                          "Clause list does not exist or number of clauses is too small.");
                return;
            }

            switch (OMPpragma.valueOf(clause.getArg(0))) {
            case TARGET_DATA_MAP:
                XobjList l = convertFromMap(xobj, clause);
                if (OMP.hasError()) {
                    return;
                }
                newClauses.add(l);
                break;
            }

        }

        XobjList list = Xcons.List(Xcode.ACC_PRAGMA, xobj.Type());
        list.setLineNo(xobj.getLineNo());
        list.add(Xcons.String(ACCpragma.PARALLEL.toString()));
        list.add(newClauses);
        args.setArg(list);
        setIsConverted(true);
    }

    private void ompToAccForDirective(Xobject directive,
                                      Xobject xobj, XobjArgs args) {
        switch (OMPpragma.valueOf(directive)) {
        // sample.
        /*
        case SINGLE:
            XobjList list = Xcons.List(Xcode.ACC_PRAGMA);
            list.add(Xcons.String(ACCpragma.PARALLEL.toString()));
            list.add(Xcons.List());
            list.add(xobj.getArg(2)); // copy structured-block.
            args.setArg(list);

            setIsConverted(true); // Always call it when it is converted to OpenACC.
            break;
        */
        case TARGET_DATA:
            convertFromTargetData(xobj, args);
            setIsConverted(true);
            break;
        }
    }

    private void ompToAcc(Xobject xobj, XobjArgs args) {
        if (xobj == null || !(xobj instanceof XobjList)) {
            return;
        }

        if (xobj.Opcode() == Xcode.OMP_PRAGMA) {
            Xobject directive = xobj.left();

            if (directive.Opcode() == Xcode.STRING) {
                ompToAccForDirective(directive, xobj, args);
            } else {
                OMP.error((LineNo)xobj.getLineNo(),
                          "directive is not specified.");
                return;
            }
        }

        for (XobjArgs a = xobj.getArgs(); a != null; a = a.nextArgs()) {
            ompToAcc(a.getArg(), a);
            if (OMP.hasError()) {
                return;
            }
        }

        return;
    }

    private void convert(XobjectDef d) {
        if (!d.isFuncDef()) {
            return;
        }

        Xobject xobj = d.getDef();

        OMP.debug("OMPtoACC: Before convert: " +  xobj);

        ompToAcc(xobj, null);

        OMP.debug("OMPtoACC: After convert:  " +  xobj);
    }


    private XobjList getRefs(XobjList ids){
        XobjList refs = Xcons.List();
        for (Xobject x : ids) {
            Ident id = (Ident)x;
            refs.add(id.Ref());
        }
        return refs;
    }

    private void replaceOmpcMain(XobjectDef mainXobjDef) {
        Ident mainId = env.findVarIdent(ompcMainFunc);
        Xtype mainType = mainId.Type().getBaseRefType();

        XobjList mainIdList = (XobjList)mainXobjDef.getFuncIdList();
        Xobject mainDecls = mainXobjDef.getFuncDecls();
        Xobject mainBody = mainXobjDef.getFuncBody();

        Ident mainFunc = env.declExternIdent(ompcMainFunc,
                                             Xtype.Function(mainType));
        Ident mainOrgFunc = env.declStaticIdent(ompcMainOrgFunc,
                                                Xtype.Function(mainType));
        Ident accInit = env.declExternIdent(ACC_INIT_FUNC_NAME,
                                            Xtype.Function(Xtype.voidType));
        Ident accFinal = env.declExternIdent(ACC_FINAL_FUNC_NAME,
                                             Xtype.Function(Xtype.voidType));
        Ident accTraverseInit = env.declExternIdent(ACC_TRAVERSE_INIT_FUNC_NAME,
                                                    Xtype.Function(Xtype.voidType));
        Ident accTraverseFinal = env.declExternIdent(ACC_TRAVERSE_FINAL_FUNC_NAME,
                                                     Xtype.Function(Xtype.voidType));

        // Add ompc_main_org().
        env.add(XobjectDef.Func(mainOrgFunc, mainIdList,
                                mainDecls, mainBody));

        // Create new ompc_main().
        BlockList newMainBody = Bcons.emptyBody();
        newMainBody.setIdentList(Xcons.IDList());
        newMainBody.setDecls(Xcons.List());

        XobjList args = getRefs(mainIdList);
        newMainBody.add(accInit.Call(args));
        newMainBody.add(accTraverseInit.Call());

        if (mainType.equals(Xtype.voidType)) {
            newMainBody.add(mainOrgFunc.Call(args));
            newMainBody.add(accTraverseFinal.Call());
            newMainBody.add(accFinal.Call(null));
        } else {
            Ident r = Ident.Local("r", mainType);
            newMainBody.addIdent(r);
            newMainBody.add(Xcons.Set(r.Ref(), mainOrgFunc.Call(args)));
            newMainBody.add(accTraverseFinal.Call());
            newMainBody.add(accFinal.Call(null));
            newMainBody.add(Xcons.List(Xcode.RETURN_STATEMENT,
                                       r.Ref()));
        }

        XobjList newMain = Xcons.List(Xcode.FUNCTION_DEFINITION, mainFunc,
                                      mainIdList, mainDecls,
                                      newMainBody.toXobject());

        // replace ompc_main().
        mainXobjDef.setDef(newMain);
    }

    private void checkFirstArg(Xobject arg) {
        if (!arg.Type().isBasic()) {
             OMP.error((LineNo)arg.getLineNo(),
                       "Type of first argument in main() must be an interger.");
        }
        if (arg.Type().getBasicType() != BasicType.INT) {
            OMP.error((LineNo)arg.getLineNo(),
                      "Type of first argument in main() must be an interger.");
        }
    }

    private void checkSecondArg(Xobject arg) {
      if(!arg.Type().isPointer()){
          OMP.error((LineNo)arg.getLineNo(),
                    "Type of second argument in main() must be char **.");
      }

      boolean flag = false;
      if(arg.Type().getRef().isPointer() && arg.Type().getRef().getRef().isBasic()){
          if(arg.Type().getRef().getRef().getBasicType() == BasicType.CHAR){
              flag = true;
          }
      }

      if(!flag){
          OMP.error((LineNo)arg.getLineNo(),
                    "Type of second argument in main() must be char **.");
      }
    }

    private void addArgsIntoOmpcMain(XobjectDef mainXobjDef) {
        Xobject args = mainXobjDef.getFuncIdList();
        int numArgs = args.Nargs();
        Ident argc = Ident.Param(MAIN_ARGC_NAME,
                                 Xtype.intType);
        Ident argv = Ident.Param(MAIN_ARGV_NAME,
                                 Xtype.Pointer(Xtype.Pointer(Xtype.charType)));
        Ident funcId = env.findVarIdent(ompcMainFunc);

        if (numArgs == 1) {
            args.add(argv);
            ((FunctionType)funcId.Type()).setFuncParamIdList(args);
        } else if(numArgs == 0) {
            args.add(argc);
            args.add(argv);
            ((FunctionType)funcId.Type()).setFuncParamIdList(args);
        }

        // Check arguments.
        checkFirstArg(args.getArgOrNull(0));
        checkSecondArg(args.getArgOrNull(1));
    }

    private void setupMain() {
        topdownXobjectDefIterator ite = new topdownXobjectDefIterator(env);
        for(ite.init(); !ite.end(); ite.next()){
            XobjectDef d = ite.getDef();
            if(d.isFuncDef() && d.getName().equals(ompcMainFunc)){
                // If the arguments of ompc_main() are short,
                // add them(argc, argv).
                addArgsIntoOmpcMain(d);
                if (OMP.hasError()) {
                    return;
                }

                // Rename ompc_main() => ompc_main_org().
                // And create new ompc_main().
                replaceOmpcMain(d);
                if (OMP.hasError()) {
                    return;
                }
            }
        }
    }
}
