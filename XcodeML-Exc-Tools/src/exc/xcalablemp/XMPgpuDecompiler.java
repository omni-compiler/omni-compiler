/*
 * $TSUKUBA_Release: $
 * $TSUKUBA_Copyright:
 *  $
 */

package exc.xcalablemp;

import exc.object.*;
import java.io.*;

public class XMPgpuDecompiler {
  public static final String GPU_FUNC_CONF = "XCALABLEMP_GPU_FUNC_CONF_PROP";

  private static XMPgpuDecompileWriter out = null;
  private static final int BUFFER_SIZE = 4096;
  private static final String GPU_SRC_EXTENSION = ".cu";

  public static void decompile(Ident id, XobjList paramIdList, Xobject deviceBodyObj, XobjectFile env) throws XMPexception {
    try {
      if (out == null) {
        Writer w = new BufferedWriter(new FileWriter(getSrcName(env.getSourceFileName()) + GPU_SRC_EXTENSION), BUFFER_SIZE);
        out = new XMPgpuDecompileWriter(w, env);
      }

      // decompile device function
      XobjectDef deviceDef = XobjectDef.Func(id, paramIdList, null, deviceBodyObj);
      out.printDeviceFunc(deviceDef, id);
      out.println();

      // generate wrapping function
      Ident hostFuncId = XMP.getMacroId(id.getName() + "_DEVICE");
      Xobject hostFuncCall = hostFuncId.Call(genFuncArgs(paramIdList));
      // FIXME
      hostFuncCall.setProp(GPU_FUNC_CONF, (Object)Xcons.List());

      Xobject hostBodyObj = Xcons.CompoundStatement(hostFuncCall);
      XobjectDef hostDef = XobjectDef.Func(id, paramIdList, null, hostBodyObj);
      out.printHostFunc(hostDef);
      out.println();

      out.flush();
    } catch (IOException e) {
      throw new XMPexception("error in gpu decompiler: " + e.getMessage());
    }
  }

  private static String getSrcName(String srcName) {
    String name = "";
    String[] buffer = srcName.split("\\.");
    for (int i = 0; i < buffer.length - 1; i++) {
      name += buffer[i];
    }

    return name;
  }

  private static XobjList genFuncArgs(XobjList paramIdList) {
    XobjList funcArgs = Xcons.List();
    for (XobjArgs i = paramIdList.getArgs(); i != null; i = i.nextArgs()) {
      Ident id = (Ident)i.getArg();
      funcArgs.add(id.Ref());
    }

    return funcArgs;
  }
}
