package exc.xcalablemp;

import exc.object.Ident;

public class XACCdeviceArray {
  private XMPlayout _deviceLayout;
  private XMPalignedArray _alignedArray;
  private Ident _descId;
  
  public XACCdeviceArray(Ident descId, XMPalignedArray alignedArray, XMPlayout layout)
  {
    _alignedArray = alignedArray;
    _deviceLayout = layout;
    _descId = descId;//alignedArray.getDescId();
  }
  
  public String getName(){
    return _alignedArray.getName();
  }
  public void setLayout(XMPlayout layout){
    _deviceLayout = layout;
  }
  public XMPlayout getLayout(){
    return _deviceLayout;
  }
  public Ident getDescId(){
    return _descId;
  }
}
