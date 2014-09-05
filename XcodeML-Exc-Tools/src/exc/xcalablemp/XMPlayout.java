package exc.xcalablemp;

import java.util.Vector;
import exc.object.*;

public class XMPlayout {
  public final static int DUPLICATION = 100;
  public final static int BLOCK       = 101;
  
  private Vector<Integer>     _layoutMannerVector;
  private Vector<Xobject>     _layoutWidthVector;
  //private Vector<Xobject>       _sizeVector;
  
  public XMPlayout(XobjList layout) {
    //super(XMPobject.LAYOUT, name, dim, descId);
    _layoutMannerVector = new Vector<Integer>();
    _layoutWidthVector = new Vector<Xobject>();
    //_sizeVector = new Vector<Xobject>();
    
    for(Xobject x : layout){
      _layoutMannerVector.add(x.left().getInt());
      _layoutWidthVector.add(x.right());
    }
  }

  public void setDistMannerAt(int manner, int index) {
    _layoutMannerVector.setElementAt(new Integer(manner), index);
  }
  public int getDistMannerAt(int index) throws XMPexception {
//    if (!_isDistributed) {
//      throw new XMPexception("template " + getName() + " is not distributed");
//    }
    return _layoutMannerVector.get(index).intValue();
  }
  
  public static String getDistMannerString(int manner) throws XMPexception {
    switch (manner) {
      case DUPLICATION:
        return new String("DUPLICATION");
      case BLOCK:
        return new String("BLOCK");
      default:
        throw new XMPexception("unknown distribute manner");
    }
  }
}
