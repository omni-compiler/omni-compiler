package exc.object;

import xcodeml.util.XmException;

/**
 * Pragma parser which processes pragma statement in XcodeML.
 */
public interface ExternalPragmaParser
{
    /**
     * Parse pragma element.
     * 
     * @param x
     *      pragma element
     * @return
     *      parsed pragma element
     */
    public Xobject parse(Xobject x) throws XmException;
}
