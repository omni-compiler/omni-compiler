package exc.openmp;

import java.util.List;
import java.util.Arrays;
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

    private XobjList createAccPragma(ACCpragma directive, XobjList clauses,
                                     Xobject xobj, int addArgsHeadPos) {
        XobjList accPragma = Xcons.List(Xcode.ACC_PRAGMA, xobj.Type());
        accPragma.setLineNo(xobj.getLineNo());
        accPragma.add(Xcons.String(directive.toString()));
        accPragma.add(clauses);

        int pos = addArgsHeadPos;
        Xobject arg = null;
        while ((arg = xobj.getArgOrNull(pos)) != null) {
            accPragma.add(arg);
            pos++;
        }

        return accPragma;
    }

    private XobjList convertFromMap(Xobject xobj,
                                    XobjList clause) {
        if (clause.Nargs() != 2) {
            OMP.error((LineNo)xobj.getLineNo(),
                      "Number of clauses is large or small.");
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

        // create COPY()/CREATE().
        XobjList list = null;
        switch (mapType.getName()) {
        case "alloc":
            list = Xcons.List(Xcons.String(ACCpragma.CREATE.toString()),
                              mapValues);
            break;
        case "to":
            list = Xcons.List(Xcons.String(ACCpragma.COPYIN.toString()),
                              mapValues);
            break;
        case "from":
            list = Xcons.List(Xcons.String(ACCpragma.COPYOUT.toString()),
                              mapValues);
            break;
        case "tofrom":
            list = Xcons.List(Xcons.String(ACCpragma.COPY.toString()),
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

    private XobjList convertFromIf(Xobject xobj,
                                   XobjList clause,
                                   OMPpragma[] modifiers) {
        return convertFromIf(xobj, clause, modifiers, new OMPpragma[]{});
    }

    private XobjList convertFromIf(Xobject xobj,
                                   XobjList clause,
                                   OMPpragma[] modifiers,
                                   OMPpragma[] ignoreModifiers) {
        if (clause.Nargs() != 3) {
            OMP.error((LineNo)xobj.getLineNo(),
                      "Number of clauses is large or small.");
            return null;
        }

        // check modifier.
        if (!clause.getArg(1).isEmpty()) {
            String modifierStr = clause.getArg(1).getArg(0).getName();

            if (Arrays.asList(ignoreModifiers).
                contains(OMPpragma.valueOf(modifierStr))) {
                OMP.warning((LineNo)xobj.getLineNo(),
                            "modifier('" +  modifierStr.replace("_", " ").
                            toLowerCase() + "') cannot be specified. ignore.");
            } else if (!Arrays.asList(modifiers).
                       contains(OMPpragma.valueOf(modifierStr))) {
                OMP.error((LineNo)xobj.getLineNo(),
                          "modifier('" + modifierStr.replace("_", " ").
                          toLowerCase() + "') cannot be specified.");
                return null;
            }
        }

        // create IF().
        return Xcons.List(Xcons.String(ACCpragma.IF.toString()),
                          clause.getArg(2));
    }

    private void convertFromTargetData(Xobject xobj,
                                       XobjArgs currentArgs) {
        if (xobj.getArg(1) == null ||
            xobj.getArg(1).Opcode() != Xcode.LIST) {
            OMP.error((LineNo)xobj.getLineNo(),
                      "Not found clause list.");
            return;
        }

        XobjList ompClauses = (XobjList) xobj.getArg(1);
        XobjList accClauses = Xcons.List();
        for (Iterator<Xobject> it = ompClauses.iterator(); it.hasNext();) {
            XobjList clause = (XobjList) it.next();
            if (clause.Opcode() != Xcode.LIST ||
                clause.Nargs() < 1) {
                OMP.error((LineNo)xobj.getLineNo(),
                          "Clause list does not exist or number of clauses is too small.");
                return;
            }

            XobjList l = null;
            switch (OMPpragma.valueOf(clause.getArg(0))) {
            case TARGET_DATA_MAP:
                l = convertFromMap(xobj, clause);
                break;
            case DIR_IF:
                l = convertFromIf(xobj, clause,
                                  new OMPpragma[]{OMPpragma.TARGET_DATA});
                break;
            }

            if (OMP.hasError()) {
                return;
            }
            accClauses.add(l);
        }


        currentArgs.setArg(createAccPragma(ACCpragma.PARALLEL,
                                           accClauses, xobj, 2));
    }

    private void convertFromTargetTeamsDistributeParallelLoop(Xobject xobj,
                                                              XobjArgs currentArgs) {
        if (xobj.getArg(1) == null ||
            xobj.getArg(1).Opcode() != Xcode.LIST) {
            OMP.error((LineNo)xobj.getLineNo(),
                      "Not found clause list.");
            return;
        }

        XobjList ompClauses = (XobjList) xobj.getArg(1);
        XobjList gangClause = Xcons.List(Xcons.String(ACCpragma.GANG.toString()));
        XobjList accClauses = Xcons.List(gangClause);

        for (Iterator<Xobject> it = ompClauses.iterator(); it.hasNext();) {
            XobjList clause = (XobjList) it.next();
            if (clause.Opcode() != Xcode.LIST ||
                clause.Nargs() < 1) {
                OMP.error((LineNo)xobj.getLineNo(),
                          "Clause list does not exist or number of clauses is too small.");
                return;
            }

            XobjList l = null;
            switch (OMPpragma.valueOf(clause.getArg(0))) {
            case TARGET_DATA_MAP:
                l = convertFromMap(xobj, clause);
                break;
            case DIR_IF:
                l = convertFromIf(xobj, clause,
                                  new OMPpragma[]{OMPpragma.TARGET_DATA},
                                  new OMPpragma[]{OMPpragma.PARALLEL_FOR});
                break;
            }

            if (OMP.hasError()) {
                return;
            }
            accClauses.add(l);
        }

        currentArgs.setArg(createAccPragma(ACCpragma.PARALLEL_LOOP,
                                           accClauses, xobj, 2));
    }


    private Xobject ompToAccForDirective(Xobject directive,
                                         Xobject xobj, XobjArgs currentArgs) {
        switch (OMPpragma.valueOf(directive)) {
        // sample.
        /*
        case SINGLE:
            XobjList list = Xcons.List(Xcode.ACC_PRAGMA);
            list.add(Xcons.String(ACCpragma.PARALLEL.toString()));
            list.add(Xcons.List());
            list.add(xobj.getArg(2)); // copy structured-block.
            currentArgs.setArg(list);

            setIsConverted(true); // Always call it when it is converted to OpenACC.
            break;
        */
        case TARGET_DATA:
            convertFromTargetData(xobj, currentArgs);
            setIsConverted(true); // Always call it when it is converted to OpenACC.
            break;
        case TARGET_TEAMS_DISTRIBUTE_PARALLEL_LOOP:
            convertFromTargetTeamsDistributeParallelLoop(xobj, currentArgs);
            setIsConverted(true); // Always call it when it is converted to OpenACC.
            break;
        }

        // return converted xobject.
        return currentArgs.getArg();
    }

    // NOTE: OMP xcodeML is paralleled if it is written in parallel.
    //       ACC xcodeML is nested if it is written in parallel.
    //       So, written in parallel will be converted to nesting.
    //
    //       - OMP xcodeML: paralleled.
    //         <OMPPragma ...>
    //           <string>TARGET_DATA</string>
    //           <list>
    //           </list>
    //         </OMPPragma>
    //         <linemarker ...>
    //         <OMPPragma ...>
    //           <string>TARGET_TEAMS_DISTRIBUTE_PARALLEL_LOOP</string>
    //
    //       - ACC xcodeML: nested.
    //         <ACCPragma ...>
    //           <string>PARALLEL</string>
    //           <list>
    //           </list>
    //           <ACCPragma ...>
    //             <string>PARALLEL_LOOP</string>
    //
    private void convertToNest(Xobject directive,
                               Xobject xobj, XobjArgs currentArgs) {
        switch (OMPpragma.valueOf(directive)) {
        case TARGET_DATA:
            XobjArgs xargsHead = null;
            for (XobjArgs a = currentArgs.nextArgs(); a != null; a = a.nextArgs()) {
                Xobject x = a.getArg();
                if (x.Opcode() == Xcode.OMP_PRAGMA ||
                    x.Opcode() == Xcode.LINEMARKER) {
                    if (x.Opcode() == Xcode.OMP_PRAGMA) {
                        XobjArgs as = new XobjArgs(x, null);

                        if (xargsHead == null) {
                            xargsHead = as;
                        } else {
                            xargsHead.tail().setNext(as);
                        }
                    }
                } else {
                    currentArgs.setNext(a);
                    break;
                }
            }

            if (xargsHead != null) {
                xobj.getArgs().tail().setNext(xargsHead);
            }
            break;
        }

        return;
    }

    private void ompToAcc(Xobject xobj, XobjArgs currentArgs) {
        if (xobj == null || !(xobj instanceof XobjList)) {
            return;
        }

        if (xobj.Opcode() == Xcode.OMP_PRAGMA) {
            Xobject directive = xobj.left();

            if (directive.Opcode() == Xcode.STRING) {
                convertToNest(directive, xobj, currentArgs);
                xobj = ompToAccForDirective(directive, xobj, currentArgs);
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
