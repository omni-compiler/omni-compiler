package exc.xcalablemp;

import java.util.Vector;

import exc.object.*;

public class XMPlayout {
  public final static int DUPLICATION = 100;
  public final static int BLOCK       = 101;
  
  public final static int SHADOW_NONE = 200;
  public final static int SHADOW_NORMAL = 201;
  public final static int SHADOW_FULL   = 202;
  
  private Vector<Integer>     _layoutMannerVector;
  private Vector<Xobject>     _layoutWidthVector;
  private Vector<Integer>     _shadowTypeVector;
  private Vector<Xobject>     _shadowLoVector;
  private Vector<Xobject>     _shadowHiVector;
  //private Vector<Xobject>       _sizeVector;
  private boolean hasShadow = false;
  
  public XMPlayout(XobjList layout) {
    //super(XMPobject.LAYOUT, name, dim, descId);
    _layoutMannerVector = new Vector<Integer>();
    _layoutWidthVector = new Vector<Xobject>();
    //_sizeVector = new Vector<Xobject>();
    _shadowTypeVector = new Vector<Integer>();
    _shadowLoVector = new Vector<Xobject>();
    _shadowHiVector = new Vector<Xobject>();
    
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
  
  public void setShadow(Xobject shadow){
    hasShadow = true;
    for(Xobject x : (XobjList)shadow){
      int type = x.getArg(0).getInt();
      Xobject widthList = (XobjList)x.getArg(1);
      _shadowTypeVector.add(new Integer(type));
      _shadowLoVector.add(widthList.getArg(0));
      _shadowHiVector.add(widthList.getArg(1));
    }
  }
  public int getShadowTypeAt(int i){
    return _shadowTypeVector.get(i).intValue();
  }
  public Xobject getShadowLoAt(int i){
    return _shadowLoVector.get(i);
  }
  public Xobject getShadowHiAt(int i){
    return _shadowHiVector.get(i);
  }

  public static String getShadowTypeString(int shadowType) throws XMPexception {
    switch (shadowType) {
    case SHADOW_NONE:
      return new String("NONE");
    case SHADOW_NORMAL:
      return new String("NORMAL");
    case SHADOW_FULL:
      return new String("FULL");
    default:
      throw new XMPexception("unknown shadow type");
    }
  }
  public boolean hasShadow(){
    return hasShadow;
  }
}
