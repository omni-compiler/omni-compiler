package exc.xcalablemp;

import java.util.Vector;

import exc.block.*;
import exc.object.*;

public class XMPon {
  public final static int DUPLICATION = 100;
  public final static int BLOCK       = 101;
  
  //private Ident _descId;
  private XMPalignedArray _alignedArray;
  private Vector<Xobject>     _onVector;
  
  public XMPon(XobjList arrayRef, XMPalignedArray alignedArray) {
    _onVector = new Vector<Xobject>();
    //_descId = 
    _alignedArray = alignedArray;
    
    
    XobjArgs arg = arrayRef.getArgs();
    Xobject arrayname = arg.getArg();
    arg = arg.nextArgs();
    
    
    for(;arg != null; arg = arg.nextArgs()){
      _onVector.add(arg.getArg());
    }    
  }
  public Ident getArrayDesc(){
    return _alignedArray.getDescId();//_descId;
  }
  public Xobject getLoopVarAt(int index) throws XMPexception {
    return _onVector.get(index);
  }
  public int getCorrespondingDim(String loopVarName) {
    for(int i = 0; i < _onVector.size(); i++){
      Xobject x = _onVector.elementAt(i);
      if(x == null) continue;
      if(x.getSym().equals(loopVarName)){
        return i;
      }
    }

    return -1;
  }
  public XMPlayout getLayout(){
    return _alignedArray.getLayout();
  }
}
