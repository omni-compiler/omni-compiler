package exc.openacc;

import exc.object.*;

import java.util.*;

class AccInformation {
  private final ACCpragma _pragma; /* directive kind*/
  private final List<Clause> _clauseList = new ArrayList<Clause>();

  //for redundant check
  private final Set<String> _declaredSymbolSet = new HashSet<String>();
  private final Set<ACCpragma> _singleClauseSet = new HashSet<ACCpragma>();

  class Clause {
    final ACCpragma _clauseKind;

    Clause(ACCpragma clauseKind) throws ACCexception{
      _clauseKind = clauseKind;
      if(isSingle()){
        if(_singleClauseSet.contains(clauseKind)){
          throw new ACCexception("'" + clauseKind.getName() + "' is already specified");
        }
        _singleClauseSet.add(clauseKind);
      }
    }
    @Override
    public String toString(){
      return " " + _clauseKind.getName();
    }
    Xobject toXobject(){
      return Xcons.List(Xcons.String(_clauseKind.toString()));
    }
    List<ACCvar> getVarList(){
      return null;
    }
    ACCvar findVar(String s){
      return null;
    }
    void validate(AccDirective directive) throws ACCexception {
      if (!directive.isAcceptableClause(_clauseKind)) {
        throw new ACCexception(_clauseKind.getName() + " clause is not allowed");
      }
    }
    Xobject getIntExpr(){
      return null;
    }
    boolean isSingle(){
      return true;
    }
  }

  class IntExprClause extends Clause {
    private final Xobject arg;
    IntExprClause(ACCpragma clauseKind, Xobject arg) throws ACCexception{
      super(clauseKind);
      if(arg == null) throw new ACCexception("null expr");
      this.arg = arg;
    }
    @Override
    public String toString(){
      return super.toString() + '(' + arg + ')';
    }
    @Override
    Xobject toXobject(){
      Xobject clauseXobj = super.toXobject();
      clauseXobj.add(arg);
      return clauseXobj;
    }
    void validate(AccDirective directive) throws ACCexception {
      super.validate(directive);
      if(! directive.isIntExpr(arg)){
        throw new ACCexception("'" + arg + "' is not int expr");
      }
    }
    @Override
    Xobject getIntExpr(){
      return arg;
    }
  }

  class VarListClause extends Clause{
    final List<ACCvar> varList = new ArrayList<ACCvar>();

    VarListClause(ACCpragma clauseKind, XobjList args) throws ACCexception{
      super(clauseKind);

      if(args == null) return;
      for(Xobject x : args){
        addVar(x);
      }
    }
    VarListClause(ACCpragma clauseKind) throws ACCexception{
      this(clauseKind, null);
    }
    void addVar(Xobject arg) throws ACCexception{
      ACCvar var = new ACCvar(arg, _clauseKind);
      if (_clauseKind.isDeclarativeClause()) {
        checkDuplication(var);
      }
      varList.add(var);
    }
    @Override
    ACCvar findVar(String symbol) {
      for (ACCvar var : varList) {
        if (var.getSymbol().equals(symbol)) {
          return var;
        }
      }
      return null;
    }
    @Override
    public String toString(){
      StringBuilder sb = new StringBuilder();
      sb.append(super.toString());
      sb.append('(');
      if(varList.size() > 0) {
        sb.append(varList.get(0));
        for (int i = 1; i < varList.size(); i++) {
          sb.append(',');
          sb.append(varList.get(i));
        }
      }
      sb.append(')');
      return new String(sb);
    }

    @Override
    Xobject toXobject(){
      XobjList varXobjList = Xcons.List();
      for(ACCvar var : varList){
        varXobjList.add(var.toXobject());
      }
      Xobject clauseXobj = super.toXobject();
      clauseXobj.add(varXobjList);
      return  clauseXobj;
    }
    @Override
    List<ACCvar> getVarList(){
      return varList;
    }
    @Override
    void validate(AccDirective directive) throws ACCexception{
      super.validate(directive);
      for(ACCvar var : varList){
        directive.setVarIdent(var);
      }
    }
    @Override
    boolean isSingle(){
      return false;
    }
  }

  private void addClause(Clause clause){
    _clauseList.add(clause);
  }

  AccInformation(ACCpragma pragma, Xobject arg) throws ACCexception {
    _pragma = pragma;
    if(pragma == ACCpragma.WAIT || pragma == ACCpragma.CACHE){
      addClause(makeClause(pragma, arg));
      return;
    }

    for (Xobject o : (XobjList)arg) {
      XobjList clause = (XobjList) o;
      ACCpragma clauseKind = ACCpragma.valueOf(clause.getArg(0));
      Xobject clauseArg = clause.getArgOrNull(1);

      addClause(makeClause(clauseKind, clauseArg));
    }
  }

  Clause makeClause(ACCpragma clauseKind, Xobject arg) throws ACCexception{
    switch (clauseKind) {
    case INDEPENDENT:
    case SEQ:
      return new Clause(clauseKind);
    case IF:
    case NUM_GANGS:
    case NUM_WORKERS:
    case VECT_LEN:
    case COLLAPSE:
      return new IntExprClause(clauseKind, arg);
    case GANG:
    case WORKER:
    case VECTOR:
    case WAIT: //it will be int expr list
    case ASYNC: //it will be int expr list
      if(arg == null){
        return new Clause(clauseKind);
      }else {
        return new IntExprClause(clauseKind, arg);
      }
    default:
      return new VarListClause(clauseKind, (XobjList)arg);
    }
  }

  void checkDuplication(ACCvar var) throws ACCexception {
    String symbol = var.getSymbol();
    if(_declaredSymbolSet.contains(symbol)){
      throw new ACCexception("symbol '" + symbol + "' is already specified");
    }
    _declaredSymbolSet.add(symbol);
  }

  void addVar(ACCpragma clauseKind, Xobject varXobj) throws ACCexception {
    VarListClause clause = null;
    if(clauseKind != ACCpragma.HOST && clauseKind != ACCpragma.DEVICE){
      clause = (VarListClause)findClause(clauseKind);
    }
    if(clause == null){
      clause = new VarListClause(clauseKind);
      addClause(clause);
    }
    clause.addVar(varXobj);
  }

  void addClause(ACCpragma clauseKind) throws ACCexception{
    addClause(makeClause(clauseKind, null));
  }

  @Override
  public String toString(){
    StringBuilder sb = new StringBuilder();
    sb.append("#_pragma acc");

    if(_pragma != ACCpragma.CACHE && _pragma != ACCpragma.WAIT) {
      sb.append(' ');
      sb.append(_pragma.getName());
    }

    for(Clause c : _clauseList) {
      sb.append(c);
    }

    return new String(sb);
  }

  List<Clause> findAllClauses(ACCpragma clauseKind){
    List<Clause> clauses = new ArrayList<Clause>();
    for(Clause c : _clauseList){
      if(c._clauseKind == clauseKind){
        clauses.add(c);
      }
    }
    return clauses;
  }

  Clause findClause(ACCpragma clauseKind){
    List<Clause> clauses = findAllClauses(clauseKind);
    return clauses.isEmpty() ? null : clauses.get(0);
  }

  Xobject toXobject(){
    if(_pragma == ACCpragma.CACHE || _pragma == ACCpragma.WAIT){
      Clause c = findClause(_pragma);
      return c.toXobject();
    }

    XobjList xobjList = Xcons.List();
    for(Clause c : _clauseList){
      xobjList.add(c.toXobject());
    }
    return xobjList;
  }

  boolean isDeclared(String symbol){
    return _declaredSymbolSet.contains(symbol);
  }

  List<ACCvar> getACCvarList(){
    List<ACCvar> list = new ArrayList<ACCvar>();
    for(Clause c : _clauseList){
      List<ACCvar> varList = c.getVarList();
      if(varList == null) continue;
      list.addAll(varList);
    }
    return list;
  }
  List<ACCvar> getDeclarativeACCvarList(){
    List<ACCvar> list = new ArrayList<ACCvar>();
    for(Clause c : _clauseList){
      if(c._clauseKind.isDeclarativeClause()) {
        list.addAll(c.getVarList());
      }
    }
    return list;
  }
  Xobject getIntExpr(ACCpragma clauseKind){
    Clause clause = findClause(clauseKind);
    if(clause == null) {
      return null;
    }
    return clause.getIntExpr();
  }

  ACCvar findACCvar(List<Clause> clauseList, String symbol){
    for(Clause clause : clauseList){
      ACCvar var = clause.findVar(symbol);
      if(var != null){
        return var;
      }
    }
    return null;
  }
  ACCvar findACCvar(String symbol){
    return findACCvar(_clauseList, symbol);
  }
  ACCvar findACCvar(ACCpragma clauseKind, String symbol){
    return findACCvar(findAllClauses(clauseKind), symbol);
  }
  ACCvar findReductionACCvar(String symbol){
    List<Clause> reductionClauses = new ArrayList<Clause>();
    for(Clause clause : _clauseList){
      if(clause._clauseKind.isReduction()){
        reductionClauses.add(clause);
      }
    }
    return findACCvar(reductionClauses, symbol);
  }

  ACCpragma getPragma(){
    return _pragma;
  }
  boolean hasClause(ACCpragma clauseKind){
    return ! findAllClauses(clauseKind).isEmpty();
  }

  void validate(AccDirective directive) throws ACCexception{
    for(Clause clause : _clauseList){
      clause.validate(directive);
    }
  }
}
