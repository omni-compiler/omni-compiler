package exc.openacc;

import exc.object.*;
import java.util.*;

public class ACCglobalDecl{
  private XobjectFile   _env;
  private Map<String, FuncInfo> funcInfoMap;
  private XobjList _globalConstructorFuncBody;
  private XobjList _globalDestructorFuncBody;
  
  public ACCglobalDecl(XobjectFile env) {
    _env = env;
    funcInfoMap = new HashMap<String, FuncInfo>();
    _globalConstructorFuncBody = Xcons.List();
    _globalDestructorFuncBody = Xcons.List();
  }
  
  public XobjectFile getEnv() {
    return _env;
  }
  
  public void setCalleeInfo(String funcName, int paramNum, Xcode type, boolean mustBeAllocated){
    FuncInfo funcInfo = funcInfoMap.get(funcName);
    if(funcInfo == null){
      funcInfo = new FuncInfo(funcName);
      funcInfoMap.put(funcName, funcInfo);
    }
    funcInfo.setParamInfo(paramNum, type, mustBeAllocated);
  }
  public void setCallerInfo(String funcName, int argNum, Xcode type, boolean isAllocated){
    FuncInfo funcInfo = funcInfoMap.get(funcName);
    if(funcInfo == null){
      funcInfo = new FuncInfo(funcName);
      funcInfoMap.put(funcName, funcInfo);
    }
    funcInfo.setArgInfo(argNum, type, isAllocated);
  }
  public void checkPresentData(){
    for(FuncInfo funcInfo : funcInfoMap.values()){
      for(FuncParam funcParam : funcInfo.funcParams.values()){
        System.out.println("check:func=" + funcInfo.funcName + ", arg=" + funcParam.paramNum);
        if(funcParam.mustBeAllocated){
          FuncArg funcArg = funcInfo.funcArgs.get(funcParam.paramNum);
          if(funcArg == null){
            ACC.fatal("func=" + funcInfo.funcName + ", arg=" + funcParam.paramNum + " is not allocated");
          }else{
            if(funcParam.type != funcArg.type){
              ACC.fatal("param type not equals to arg type");
            }
            if(! funcArg.isAllocated){
              ACC.fatal("func=" + funcInfo.funcName + ", arg=" + funcParam.paramNum + " is not allocated");
            }
          }
        }
      }
    }
  }
  public void setupGlobalConstructor() {
    _globalConstructorFuncBody.cons(Xcons.List(Xcode.EXPR_STATEMENT, declExternFunc("_ACC_gpu_init").Call(null)));

    Ident argv = Ident.Param("argv", Xtype.Pointer(Xtype.Pointer(Xtype.charType)));   // create "int argc" & "char **argv"
    Ident argc = Ident.Param("argc", Xtype.intType);
    XobjList params = Xcons.IDList();
    params.add(argc);
    params.add(argv);
    
    XobjList args = Xcons.List(argc.Ref(), argv.Ref());
    
    _globalConstructorFuncBody.cons(Xcons.List(Xcode.EXPR_STATEMENT, declExternFunc("_ACC_init").Call(args)));
    Xtype funcType = Xtype.Function(Xtype.voidType);

    funcType.setGccAttributes(Xcons.List(Xcode.GCC_ATTRIBUTES,
                                         Xcons.List(Xcode.GCC_ATTRIBUTE,
                                                    new Ident("constructor", null, null, null, null),
                                                    Xcons.List())));
    Ident funcId = _env.declStaticIdent("_ACC_constructor", funcType);

    _env.add(XobjectDef.Func(funcId, params, null, Xcons.List(Xcode.COMPOUND_STATEMENT,
                                                            (Xobject)null, null, _globalConstructorFuncBody)));
  }

  public void setupGlobalDestructor() {
    _globalDestructorFuncBody.add(Xcons.List(Xcode.EXPR_STATEMENT, declExternFunc("_ACC_gpu_finalize").Call(null)));

    _globalDestructorFuncBody.add(Xcons.List(Xcode.EXPR_STATEMENT, declExternFunc("_ACC_finalize").Call(null)));

    Xtype funcType = Xtype.Function(Xtype.voidType);
    funcType.setGccAttributes(Xcons.List(Xcode.GCC_ATTRIBUTES,
                                         Xcons.List(Xcode.GCC_ATTRIBUTE,
                                                    new Ident("destructor", null, null, null, null),
                                                    Xcons.List())));
    Ident funcId = _env.declStaticIdent("_ACC_destructor", funcType);
    _env.add(XobjectDef.Func(funcId, null, null, Xcons.List(Xcode.COMPOUND_STATEMENT,
                             (Xobject)null, null, _globalDestructorFuncBody)));
  }

  public Ident declExternFunc(String funcName) {
    return ACCutil.getMacroFuncId(funcName, Xtype.voidType);
  }
  public String genSym(String prefix) {
    return _env.genSym(prefix);
  }
  public Ident declExternIdent(String name, Xtype t) {
    return _env.declExternIdent(name, t);
  }
  public void finalize() {
    _env.collectAllTypes();
    _env.fixupTypeRef();
  }
}

class FuncInfo{
  String funcName;
  Map<Integer, FuncArg> funcArgs;
  Map<Integer, FuncParam> funcParams;
  FuncInfo(String funcName){
    this.funcName = funcName;
    this.funcArgs = new HashMap<Integer, FuncArg>();
    this.funcParams = new HashMap<Integer, FuncParam>();
  }
  public void setParamInfo(int paramNum, Xcode type, boolean mustBeAllocated){ //for collee
    FuncParam funcParam = funcParams.get(paramNum);
    if(funcParam == null){
      funcParam = new FuncParam(paramNum, type, mustBeAllocated);
      funcParams.put(paramNum, funcParam);
    }else{
      if(funcParam.type != type){
        ACC.fatal("param type collision "+ funcParam.type + ":" + type);
      }else{
        funcParam.mustBeAllocated = funcParam.mustBeAllocated || mustBeAllocated;
      }
    }
  }
  public void setArgInfo(int argNum, Xcode type, boolean isAllocated){ //for coller
    FuncArg funcArg = funcArgs.get(argNum);
    if(funcArg == null){
      funcArg = new FuncArg(argNum, type, isAllocated);
      funcArgs.put(argNum, funcArg);
    }else{
      if(funcArg.type != type){
        ACC.fatal("arg type collision "+ funcArg.type + ":" + type);
      }else{
        funcArg.isAllocated = funcArg.isAllocated && isAllocated;
      }
    }
  }
}

class FuncParam{
  int paramNum;
  Xcode type;
  boolean mustBeAllocated;
  FuncParam(int paramNum, Xcode type, boolean mustBeAllocated){
    this.paramNum = paramNum;
    this.type = type;
    this.mustBeAllocated = mustBeAllocated;
  }
}

class FuncArg{
  int argNum;
  Xcode type;
  boolean isAllocated;
  FuncArg(int argNum, Xcode type, boolean isAllocated){
    this.argNum = argNum;
    this.type = type;
    this.isAllocated = isAllocated;
  }
  
  
  
}
