package exc.block;

import exc.object.*;

public interface METAXdecl {
  public abstract void runDecl(BlockList bl, XobjList clauses);
  public abstract void run(BlockList bl, XobjList clauses);
}
