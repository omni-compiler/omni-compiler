/*
 * $TSUKUBA_Release: $
 * $TSUKUBA_Copyright:
 *  $
 */

package exc.xcalablemp;

import exc.block.*;
import exc.object.*;

import java.util.Vector;

import xcodeml.IXobject;

public class XACCdevice extends XMPobject {

  Ident _accDevice;
  Xobject _stride;

  public XACCdevice(String name, Ident descId, Ident deviceRefId, Xobject lower, Xobject upper, Xobject stride) {
    super(XMPobject.DEVICE, name, 1, descId);
    _accDevice = deviceRefId;
    addLower(lower);
    addUpper(upper);
    _stride = stride;
  }

  public Ident getAccDevice(){
    return _accDevice;
  }

  public Xobject getLower(){
    return this.getLowerAt(0);
  }

  public Xobject getUpper(){
    return this.getUpperAt(0);
  }

  public Xobject getStride(){
    return _stride;
  }

  public static void translateDevice(XobjList deviceDecl, XMPglobalDecl globalDecl,
				     boolean isLocalPragma, PragmaBlock pb) throws XMPexception {

    // local parameters
    XMPsymbolTable localXMPsymbolTable = null;
    Block parentBlock = null;
    if (isLocalPragma){
      parentBlock = pb.getParentBlock();
      localXMPsymbolTable = XMPlocalDecl.declXMPsymbolTable2(parentBlock);
    }

    // check name collision
    String deviceName = deviceDecl.getArg(0).getString();
    if (isLocalPragma){
      XMPlocalDecl.checkObjectNameCollision(deviceName, parentBlock.getBody(), localXMPsymbolTable);
    }
    else {
      globalDecl.checkObjectNameCollision(deviceName);
    }

    // declare device descriptor
    Ident deviceDescId = null;
    if (isLocalPragma){
      deviceDescId = XMPlocalDecl.addObjectId2(XMP.DESC_PREFIX_ + deviceName, parentBlock);
    }
    else {
      deviceDescId = globalDecl.declStaticIdent(XMP.DESC_PREFIX_ + deviceName, Xtype.voidPtrType);
    }

    XobjList inheritDecl = (XobjList)deviceDecl.getArg(1);

    // inheritedDevice
    String inheritedDeviceName = inheritDecl.getArg(0).getString();
    Ident inheritedDeviceId = null;
    if (isLocalPragma){
      inheritedDeviceId = XMPlocalDecl.findLocalIdent(pb, inheritedDeviceName);
    }
    else {
      inheritedDeviceId = globalDecl.getEnv().findIdent(inheritedDeviceName, IXobject.FINDKIND_ANY);
//      inheritedDeviceId = globalDecl.findVarIdent(inheritedDeviceName);
      
    }
    if (inheritedDeviceId == null) throw new XMPexception("'" + inheritedDeviceName + "' is not declared");

    // create function call
    XobjList deviceArgs = Xcons.List(deviceDescId.getAddr(), getDeviceRef(inheritedDeviceId));

    XobjList subscript = (XobjList)inheritDecl.getArg(1);

    // lower
    Xobject lower = subscript.getArg(0);
    deviceArgs.add(Xcons.Cast(Xtype.intType, lower));

    // upper
    Xobject upper = subscript.getArg(1);
    deviceArgs.add(Xcons.Cast(Xtype.intType, upper));

    // stride
    Xobject stride = subscript.getArg(2);
    deviceArgs.add(Xcons.Cast(Xtype.intType, stride));

    // add constructor call
    if (isLocalPragma){
      XMPlocalDecl.addConstructorCall2("_XACC_init_device", deviceArgs, globalDecl, parentBlock);
      XMPlocalDecl.insertDestructorCall2("_XACC_finalize_device", Xcons.List(deviceDescId.Ref()),
					 globalDecl, parentBlock);
    }
    else {
      globalDecl.addGlobalInitFuncCall("_XACC_init_device", deviceArgs);
    }

    // declare device object
    XACCdevice deviceObject = new XACCdevice(deviceName, deviceDescId, inheritedDeviceId, lower, upper, stride);
    if (isLocalPragma){
      localXMPsymbolTable.putXMPobject(deviceObject);
    }
    else {
      globalDecl.putXMPobject(deviceObject);
    }

  }
  
  public static Xobject getDeviceRef(Ident id)
  {
    if(id.getStorageClass() == StorageClass.MOE){
      return Xcons.SymbolRef(id);
    }else{
      return id.Ref();
    }
  }

  public Xobject getDeviceRef() {
    return getDeviceRef(_accDevice);
  }
}
