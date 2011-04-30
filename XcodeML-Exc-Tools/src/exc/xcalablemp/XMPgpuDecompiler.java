/*
 * $TSUKUBA_Release: $
 * $TSUKUBA_Copyright:
 *  $
 */

package exc.xcalablemp;

import exc.object.*;
import java.io.*;

public class XMPgpuDecompiler {
  private static final int BUFFER_SIZE = 4096;
  private static final String GPU_SRC_EXTENSION = ".cu";

  public static void decompile(XobjectDef def, XobjectFile env) throws XMPexception {
    try {
      Writer w = new BufferedWriter(new FileWriter("__omni_xmpgpu_" + env.getSourceFileName() + GPU_SRC_EXTENSION), BUFFER_SIZE);
      XMPgpuDecompileWriter out = new XMPgpuDecompileWriter(w, env);

      out.print(def);
      out.println();
      out.flush();

      w.close();
    } catch (IOException e) {
      throw new XMPexception("error in gpu decompiler: " + e.getMessage());
    }
  }
}
