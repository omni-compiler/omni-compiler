package exc.block;

import exc.object.Xcode;
import exc.object.Xcons;
import exc.object.Xobject;

/**
 * No operation block
 */
public class NullBlock extends Block
{
    public NullBlock()
    {
        super(null, null);
    }

    @Override
    public Xobject toXobject()
    {
        return Xcons.List(Xcode.NULL);
    }
}
