/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
package exc.block;

import exc.object.*;

public class FcaseLabelBlock extends CompoundBlock
{
    private XobjList values;
    
    public FcaseLabelBlock(XobjList values, BlockList body, String construct_name)
    {
        super(Xcode.F_CASE_LABEL, body, construct_name);
        this.values = values;
    }
    
    /** get case values */
    public XobjList getValues()
    {
        return values;
    }
    
    @Override
    public Xobject toXobject()
    {
        return Xcons.List(Opcode(), getConstructNameObj(), values, body.toXobject());
    }
}
