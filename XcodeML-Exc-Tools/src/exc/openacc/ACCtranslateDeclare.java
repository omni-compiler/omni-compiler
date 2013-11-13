package exc.openacc;

import java.util.*;

import xcodeml.IXobject;

import exc.block.*;
import exc.object.*;

public class ACCtranslateDeclare {
  private translateDeclare translator;

  ACCtranslateDeclare(PragmaBlock pb){
    translator = new translateLocalDeclare(pb);
  }
  
  ACCtranslateDeclare(Xobject x){
    translator = new translateGlobalDeclare(x);
  }

  public void translate() throws ACCexception {
    translator.translate();
  }
  public void rewrite() {
    translator.rewrite();
  }
}

abstract class translateDeclare {
  protected ACCinfo declareInfo;
  translateDeclare(PropObject o){
    declareInfo = (ACCinfo)o.getProp(ACC.prop);
    if(declareInfo == null){
      ACC.fatal("cannot get accinfo");
    }
  }
  abstract public void translate() throws ACCexception;
  abstract public void rewrite();
}

class translateGlobalDeclare extends translateDeclare{
  private Xobject px;
  translateGlobalDeclare(Xobject px) {
    super(px);
    this.px = px;
  }
  public void translate() throws ACCexception{
    ACC.debug("translate global declare");
    
    ACCtranslateData data = new ACCtranslateData(px);
    data.translate();
  }
  public void rewrite(){
    ACC.debug("rewrite global declare");
    
    Block beginBlock = declareInfo.getBeginBlock();
    Block endBlock = declareInfo.getEndBlock();
    
    ACCglobalDecl globalDecl = declareInfo.getGlobalDecl();
    XobjList idList = (XobjList)globalDecl.getEnv().getGlobalIdentList();
    
    XobjList id_list = declareInfo.getIdList();
    
    idList.mergeList(id_list);
    globalDecl.getEnv().setIdentList(idList);
    
    globalDecl.addGlobalConstructor(beginBlock.toXobject());
    globalDecl.addGlobalDestructor(endBlock.toXobject());
  }
}

class translateLocalDeclare extends translateDeclare{
  private PragmaBlock pb;
  translateLocalDeclare(PragmaBlock pb) {
    super(pb);
    this.pb = pb;
  }  
  public void translate() throws ACCexception{
    ACC.debug("translate local declare");
    
    ACCtranslateData data = new ACCtranslateData(pb);
    data.translate();
    
  }
  public void rewrite(){
    ACC.debug("rewrite local declare");
    
    Block beginBlock = declareInfo.getBeginBlock();
    Block endBlock = declareInfo.getEndBlock();
    
    BlockList parentBody = pb.getParent();
    parentBody.insert(beginBlock);
    parentBody.add(endBlock);
    
    XobjList idList = declareInfo.getIdList();
    XobjList parentIdList = parentBody.getIdentList();
    parentIdList.mergeList(idList);
    parentBody.setIdentList(parentIdList);
    
    pb.remove();
  }
}