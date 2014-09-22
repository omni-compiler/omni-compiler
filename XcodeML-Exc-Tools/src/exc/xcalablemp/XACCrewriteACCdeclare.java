package exc.xcalablemp;

import exc.object.*;
import exc.block.*;

public class XACCrewriteACCdeclare extends XACCrewriteACCdata{
  //common
  //private XobjList clauses;
  //private XMPglobalDecl _globalDecl;
  //private XACCdevice device = null;
  //protected XACClayout layout = null;
  //static final String XACC_DESC_PREFIX = "_XACC_DESC_";
  //private boolean isGlobal;
  
  //global
  
  //local
  //private PragmaBlock pb;
  //private BlockList mainBody;
  //private XMPsymbolTable localSymbolTable;
  //protected Block replaceBlock;
  
  //global pragma
  public XACCrewriteACCdeclare(XMPglobalDecl decl, Xobject p) {
    _globalDecl = decl;
    clauses = (XobjList)p.getArg(1);
    isGlobal = true;
    mainBody = Bcons.emptyBody();
    XACCtranslatePragma trans = new XACCtranslatePragma(_globalDecl);
    if (clauses != null){
      device = trans.getXACCdevice((XobjList)clauses);
      layout = trans.getXACClayout((XobjList)clauses);
    }
  }
  
  public XACCrewriteACCdeclare(XMPglobalDecl decl, PragmaBlock pb){
    isGlobal = false;
    
    // TODO Auto-generated constructor stub
  }
  
  public Block makeReplaceBlock()
  {
    if(isGlobal){
      if(device == null) return null;
      
      
      XobjList createArgs = Xcons.List();
      XobjList updateDeviceArgs = Xcons.List();
      XobjList updateHostArgs = Xcons.List();
      XobjList deleteArgs = Xcons.List();
      analyzeClause(createArgs, updateDeviceArgs, updateHostArgs, deleteArgs);
      Block beginDeviceLoopBlock = makeBeginDeviceLoop(createArgs, updateDeviceArgs);
      Block endDeviceLoopBlock = makeEndDeviceLoop(deleteArgs, updateHostArgs);
      
      mainBody.add(beginDeviceLoopBlock);
      
      //_globalDecl.addXACCconstructor(beginDeviceLoopBlock.toXobject());
      _globalDecl.addXACCconstructor(mainBody.toXobject());
      _globalDecl.insertXACCdestructor(endDeviceLoopBlock.toXobject());
      
      return null;
    }else{
      return null;  
    }
  }
}


