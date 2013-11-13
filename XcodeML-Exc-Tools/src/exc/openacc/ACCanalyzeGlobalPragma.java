package exc.openacc;
import exc.block.*;
import exc.object.*;


public class ACCanalyzeGlobalPragma {
  private ACCglobalDecl   _globalDecl;
  private XobjectDef    currentDef;

  public ACCanalyzeGlobalPragma(ACCglobalDecl globalDecl) {
    _globalDecl = globalDecl;
  }
  
  public void analyze(XobjectDef def) {
    Xobject x = def.getDef();
    if(x.Opcode() == Xcode.ACC_PRAGMA){
      try{
        analyzePragma(x);
      }catch(ACCexception e){
        ACC.error(x.getLineNo(), e.getMessage());
      }
      //System.out.println(x);
    }else if(x.Opcode() == Xcode.PRAGMA_LINE){
      ACC.error(x.getLineNo(), "unknown pragma : " + x);
    }
    
  }
  
  void analyzePragma(Xobject x) throws ACCexception{
    String pragmaName = x.getArg(0).getString();
    switch(ACCpragma.valueOf(pragmaName)){
    case DECLARE:
      analyzeDeclare(x);
      break;
    default:
      throw new ACCexception("'" + pragmaName.toLowerCase() + "' directive is not supported yet");
    }
  }
  
  void analyzeDeclare(Xobject x) throws ACCexception{
    XobjList clauseList = (XobjList)x.getArg(1);
    //XXX need to save ACCinfo in globalDecl
    ACCinfo accInfo = new ACCinfo(ACCpragma.DECLARE,x,_globalDecl);
    x.setProp(ACC.prop, accInfo);
    
    ACC.debug("declare directive : " + clauseList);
    
    for(Xobject o : clauseList){
      XobjList clause = (XobjList)o;
      ACCpragma clauseName = ACCpragma.valueOf(clause.getArg(0));
      Xobject clauseArgs = (clause.Nargs() > 1)? clause.getArg(1) : null;
            
      if(clauseName.isDataClause() || clauseName == ACCpragma.DEVICE_RESIDENT){
        //XXX need to implement
        for(Xobject var : (XobjList)clauseArgs) accInfo.declACCvar(var.getName(), clauseName);
      }else{
        ACC.fatal("'" + clauseName + "' clause is not allowed in 'declare' directive");
      }
    }
  }
  
  

}
