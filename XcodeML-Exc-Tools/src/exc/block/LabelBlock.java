/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
package exc.block;

import exc.object.*;

public class LabelBlock extends Block
{
    Xobject label, value, params;

    public LabelBlock(Xcode code, Xobject label)
    {
        this(code, label, null, null);
    }
    
    public LabelBlock(Xcode code, Xobject label, Xobject value, Xobject params)
    {
        super(code, new BasicBlock(), null);
        this.label = label;
        this.value = value;
        this.params = params;
    }

    @Override
    public Xobject getLabel()
    {
        return label;
    }

    @Override
    public void setLabel(Xobject x)
    {
        label = x;
    }

    @Override
    public Xobject toXobject()
    {
        Xobject x = new XobjList(Opcode(), label);
        x.setLineNo(getLineNo());
        if(value != null)
            x.add(value);
        if(params != null)
            x.add(params);
        
        return x;
    }

    @Override
    public String toString()
    {
        StringBuilder s = new StringBuilder(256);
        s.append("(LabelBlock ");
        s.append(Opcode());
        s.append(" ");
        s.append(label);
        if(value != null) {
            s.append(value);
            s.append(" ");
        }
        if(params != null) {
            s.append(params);
            s.append(" ");
        }
        s.append(" ");
        s.append(getBasicBlock());
        s.append(")");
        return s.toString();
    }
    
    @Override
    public LabelBlock copy()
    {
      return new LabelBlock(this.code, this.label, this.value, this.params);
    }
}
