package exc.openacc;

import exc.object.*;

import java.util.*;

public class AccInformation {
  private final ACCpragma pragma; /* directive */
  private final EnumSet<ACCpragma> boolSet = EnumSet.noneOf(ACCpragma.class);
  private final Map<ACCpragma, Xobject> exprMap = new LinkedHashMap<ACCpragma, Xobject>(); //contain order
  private final Map<ACCpragma, List<ACCvar>> varMap = new LinkedHashMap<ACCpragma, List<ACCvar>>(); //contains order
  private final Set<String> declaredSymbolSet = new HashSet<String>();
  public static final String prop = "_ACC_information";

  AccInformation(ACCpragma pragma, Xobject arg) throws ACCexception {
    this.pragma = pragma;
    if(pragma == ACCpragma.WAIT || pragma == ACCpragma.CACHE){
      setClause(pragma, arg);
      return;
    }

    for (Xobject o : (XobjList)arg) {
      XobjList clause = (XobjList) o;
      ACCpragma clauseKind = ACCpragma.valueOf(clause.getArg(0));
      Xobject clauseArg = clause.getArgOrNull(1);

      if (isAcceptableClause(pragma, clauseKind)) {
        setClause(clauseKind, clauseArg);
      } else {
        throw new ACCexception(clauseKind.getName() + " clause is not allowed");
      }
    }

    System.out.println(this);
  }

  void setBool(ACCpragma clauseKind) throws ACCexception {
    if(boolSet.contains(clauseKind)){
      throw new ACCexception(clauseKind.getName() + " clause is already specified");
    }
    switch (clauseKind){
    case ASYNC:
    case GANG:
    case WORKER:
    case VECTOR:
      setIntExpr(clauseKind, null);
      break;
    case INDEPENDENT:
    case SEQ:
      boolSet.add(clauseKind);
      break;
    default:
      throw new ACCexception(clauseKind.getName() + " is not allowed");
    }
  }

  void setIntExpr(ACCpragma clauseKind, Xobject clauseArg) throws ACCexception {
    //if (clauseArg != null && !clauseArg.Type().isIntegral()){
    //  throw new ACCexception(clauseArg + " is not integer expression");
    //}

    switch(clauseKind){
    case IF:
    case NUM_GANGS:
    case GANG:
    case NUM_WORKERS:
    case WORKER:
    case VECT_LEN:
    case VECTOR:
    case COLLAPSE:
    case WAIT:  //it will be int expr list
    case ASYNC: //it will be int expr list
      if(exprMap.containsKey(clauseKind)){
        throw new ACCexception(clauseKind.getName() + " clause is already specified");
      }
      exprMap.put(clauseKind, clauseArg);
      break;
    default:
      throw new ACCexception(clauseKind.getName() + " is not allowed");
    }
  }

  void setVarList(ACCpragma clauseKind, Xobject clauseArg) throws ACCexception {
    if(clauseArg == null) return;
    for(Xobject x : (XobjList)clauseArg){
      setVar(clauseKind, x);
    }
  }

  void checkDuplication(ACCvar var) throws ACCexception {
    String symbol = var.getSymbol();
    if(declaredSymbolSet.contains(symbol)){
      throw new ACCexception("symbol '" + symbol + "' is already specified");
    }
    declaredSymbolSet.add(symbol);
  }

  void setVar(ACCpragma clauseKind, Xobject arg) throws ACCexception {
    ACCvar var = new ACCvar(arg, clauseKind);
    if(isDeclarativeClause(clauseKind)){
      checkDuplication(var);
    }

    List<ACCvar> list;
    if(varMap.containsKey(clauseKind)){
      list = varMap.get(clauseKind);
    }else {
      list = new ArrayList<ACCvar>();
      varMap.put(clauseKind, list);
    }
    list.add(var);
  }

  @Override
  public String toString(){
    StringBuilder sb = new StringBuilder();
    sb.append("#pragma acc");

    if(pragma != ACCpragma.CACHE && pragma != ACCpragma.WAIT) {
      sb.append(' ');
      sb.append(pragma.getName());
    }

    for(ACCpragma clauseKind : boolSet){
      sb.append(' ');
      sb.append(clauseKind.getName());
    }

    for(ACCpragma clauseKind : exprMap.keySet()){
      sb.append(' ');
      sb.append(clauseKind.getName());
      Xobject expr = exprMap.get(clauseKind);
      if(expr == null) continue;
      sb.append('(');
      sb.append(expr);
      sb.append(')');
    }

    for(ACCpragma clause : varMap.keySet()){
      List<ACCvar> list = varMap.get(clause);
      sb.append(' ');
      sb.append(clause.getName());
      sb.append("(");
      sb.append(list.get(0));
      for(int i = 1; i < list.size(); i++){
        sb.append(',');
        sb.append(list.get(i));
      }
      sb.append(')');
    }

    return new String(sb);
  }

  private void setClause(ACCpragma kind, Xobject arg) throws ACCexception {
    switch (kind) {
    case INDEPENDENT:
    case SEQ:
      if (arg != null) {
        throw new ACCexception("unnessesary arg");
      }
      setBool(kind);
      break;
    case IF:
    case NUM_GANGS:
    case GANG:
    case NUM_WORKERS:
    case WORKER:
    case VECT_LEN:
    case VECTOR:
    case COLLAPSE:
    case WAIT:
    case ASYNC:
      setIntExpr(kind, arg);
      break;
    default:
      setVarList(kind, arg);
    }
  }

  private boolean isAcceptableClause(ACCpragma directive, ACCpragma clause) throws ACCexception {
    switch (directive){
    case PARALLEL:
      return AccParallel.isAcceptableClause(clause);
    case LOOP:
      return AccLoop.isAcceptableClause(clause);
    case DATA:
      return AccData.isAcceptableClause(clause);
    case DECLARE:
      return AccDeclare.isAcceptableClause(clause);
    case KERNELS:
      return AccKernels.isAcceptableClause(clause);
    case UPDATE:
      return AccUpdate.isAcceptableClause(clause);
    case HOST_DATA:
      return AccHostData.isAcceptableClause(clause);
    case PARALLEL_LOOP:
      return AccParallelLoop.isAcceptableClause(clause);
    case KERNELS_LOOP:
      return AccKernelsLoop.isAcceptableClause(clause);
    case ENTER_DATA:
      return AccEnterData.isAcceptableClause(clause);
    case EXIT_DATA:
      return AccExitData.isAcceptableClause(clause);
    default:
      throw new ACCexception("'" + directive.getName() + "' directive is not supported yet");
    }
  }

  Xobject toXobject(){
    if(pragma == ACCpragma.CACHE){
      XobjList varList = Xcons.List();
      for(ACCvar var : varMap.get(pragma)){
        varList.add(var.toXobject());
      }
      return varList;
    }else if(pragma == ACCpragma.WAIT){
      return exprMap.get(pragma);
    }

    XobjList clauses = Xcons.List();

    for(ACCpragma clauseKind : boolSet){
      clauses.add(Xcons.List(Xcons.String(clauseKind.toString())));
    }

    for(ACCpragma clauseKind : exprMap.keySet()){
      Xobject expr = exprMap.get(clauseKind);
      clauses.add(Xcons.List(Xcons.String(clauseKind.toString()), expr));
    }

    for(ACCpragma clauseKind : varMap.keySet()){
      XobjList varList = Xcons.List();
      for(ACCvar var : varMap.get(clauseKind)){
        varList.add(var.toXobject());
      }
      clauses.add(Xcons.List(Xcons.String(clauseKind.toString()), varList));
    }

    return clauses;
  }

  private boolean isDeclarativeClause(ACCpragma clauseKind){
    switch (clauseKind){
    case PRIVATE:
    case FIRSTPRIVATE:
    case DEVICE_RESIDENT:
    case USE_DEVICE:
    case CACHE:
      return true;
    default:
      return clauseKind.isDataClause();
    }
  }

  boolean isDeclared(String symbol){
    return declaredSymbolSet.contains(symbol);
  }

  List<ACCvar> getACCvarList(){
    List<ACCvar> list = new ArrayList<ACCvar>();
    for(ACCpragma clauseKind : varMap.keySet()){
      list.addAll(varMap.get(clauseKind));
    }
    return list;
  }
  List<ACCvar> getDeclarativeACCvarList(){
    List<ACCvar> list = new ArrayList<ACCvar>();
    for(ACCpragma clauseKind : varMap.keySet()){
      if(! isDeclarativeClause(clauseKind)) continue;
      list.addAll(varMap.get(clauseKind));
    }
    return list;
  }
  public Xobject getIntExpr(ACCpragma clauseKind){
    return exprMap.get(clauseKind);
  }
  ACCvar findACCvar(String symbol){
    for(ACCpragma clauseKind : varMap.keySet()){
      ACCvar var = findACCvar(clauseKind, symbol);
      if(var != null) return var;
    }
    return null;
  }
  ACCvar findACCvar(ACCpragma clauseKind, String symbol){
    List<ACCvar> list = varMap.get(clauseKind);
    if(list == null ) return null;

    for(ACCvar var : list){
      if(var.getSymbol().equals(symbol)){
        return var;
      }
    }
    return null;
  }
  ACCvar findReductionACCvar(String symbol){
    for(ACCpragma clauseKind : varMap.keySet()){
      if(! clauseKind.isReduction()) continue;
      ACCvar var = findACCvar(clauseKind, symbol);
      if(var != null) return var;
    }
    return null;
  }

  public ACCpragma getPragma(){
    return pragma;
  }
  boolean hasClause(ACCpragma clauseKind){
    return exprMap.containsKey(clauseKind) ||
            boolSet.contains(clauseKind) ||
            varMap.containsKey(clauseKind);
  }
}

/*
class Var{
  private final String symbol;
  private final XobjList subscripts;
  Var(Xobject x){
    XobjList subscripts = null;
    if(x.Opcode() == Xcode.LIST){
      Xobject var = x.getArg(0);
      symbol = var.getName();
      subscripts = (XobjList)x.copy();
      subscripts.removeFirstArgs();
    }else{
      symbol = x.getName();
    }
    this.subscripts = subscripts;
  }
  Var(String symbol, XobjList subscripts){
    this.symbol = symbol;
    this.subscripts = subscripts;
  }
  public String toString(){
    StringBuilder sb = new StringBuilder();
    sb.append(symbol);
    if(subscripts != null){
      for(Xobject subscript : subscripts){
        sb.append('[');
        sb.append(subscript.getArg(0).getName());
        sb.append(':');
        sb.append(subscript.getArg(1).getName());
        sb.append(']');
      }
    }
    return new String(sb);
  }
  public Xobject toXobject(){
    Xobject var = Xcons.Symbol(Xcode.VAR, symbol);
    if(subscripts == null) {
      return var;
    }else{
      XobjList l = Xcons.List(var);
      l.mergeList(subscripts);
      return l;
    }
  }
  public String getSymbol(){
    return symbol;
  }
}
*/